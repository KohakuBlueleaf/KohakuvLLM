"""One NVFP4 global scale per group of linears vLLM fuses, in a compressed-tensors checkpoint.

vLLM runs fused linears (q/k/v, gate/up, GDN qkv/z and b/a) as one GEMM with one weight global
scale, the largest of the shards'; a shard stored with another global scale is then dequantized
off by the ratio. This rewrites each such shard to the shared scale: a weight's value is
``code * block_scale / global``, so its E4M3 block scales are multiplied by ``shared / own``.
``--to min`` (the default) shares the group's smallest stored scale, so block scales only shrink:
a power-of-two ratio is then exact except for scales pushed into E4M3's subnormals; ``--to max``
grows them and clips at 448. The share of block scales that changed inexactly and the largest
relative change are reported. Input global scales of a group are set to the group's largest
(vLLM uses that one already).

    python tools/share_fp4_scale.py SRC DST [--to min]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E4M3 = torch.float8_e4m3fn
E4M3_MAX = torch.finfo(E4M3).max
FUSE = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
    ("in_proj_qkv", "in_proj_z"),
    ("in_proj_b", "in_proj_a"),
)


def groups(names: set[str]) -> list[list[str]]:
    """Linear names (prefix before ``.weight_global_scale``) grouped as vLLM fuses them."""
    out, seen = [], set()
    for n in sorted(names):
        if n in seen:
            continue
        group = [n]
        for parts in FUSE:
            last = n.rsplit(".", 1)[-1]
            if last in parts:
                stem = n[: -len(last)]
                group = [stem + p for p in parts if stem + p in names]
                break
        seen.update(group)
        out.append(group)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--to", choices=("min", "max"), default="min")
    args = p.parse_args()
    pick = min if args.to == "min" else max
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy(f, dst / f.name)

    shards = sorted(src.glob("*.safetensors"))
    where: dict[str, Path] = {}
    scales: dict[str, float] = {}
    inputs: dict[str, float] = {}
    for shard in shards:
        with safe_open(shard, "pt") as s:
            for k in list(s.keys()):
                where[k] = shard
                if k.endswith(".weight_global_scale"):
                    scales[k.removesuffix(".weight_global_scale")] = s.get_tensor(k).float().item()
                if k.endswith(".input_global_scale"):
                    inputs[k.removesuffix(".input_global_scale")] = s.get_tensor(k).float().item()

    # Per linear: (the factor its block scales take, the shared weight and input scales)
    plan: dict[str, tuple[float, float, float | None]] = {}
    for group in groups(set(scales)):
        shared = pick(scales[n] for n in group)
        shared_in = max((inputs[n] for n in group if n in inputs), default=None)
        for n in group:
            plan[n] = (shared / scales[n], shared, shared_in)

    changed, scales_total, scales_inexact, worst = 0, 0, 0, 0.0
    for shard in shards:
        out = {}
        with safe_open(shard, "pt") as s:
            meta = s.metadata()
            for k in list(s.keys()):
                out[k] = s.get_tensor(k)
            for name, (factor, shared, shared_in) in plan.items():
                if where.get(name + ".weight_scale") != shard:
                    continue
                if shared_in is not None:
                    out[name + ".input_global_scale"] = torch.tensor([shared_in])
                out[name + ".weight_global_scale"] = torch.tensor([shared])
                if factor == 1.0:
                    continue
                old = out[name + ".weight_scale"].float()
                new = (old * factor).clamp(max=E4M3_MAX).to(E4M3)
                ratio = torch.where(old > 0, new.float() / (old * factor), torch.ones_like(old))
                worst = max(worst, (ratio - 1).abs().max().item())
                out[name + ".weight_scale"] = new
                changed += 1
                scales_total += old.numel()
                scales_inexact += int((new.float() != old * factor).sum())
        save_file(out, dst / shard.name, metadata=meta)
    report = {
        "linears": len(plan),
        "rescaled": changed,
        "inexact_share": scales_inexact / max(scales_total, 1),
        "max_rel": worst,
    }
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
