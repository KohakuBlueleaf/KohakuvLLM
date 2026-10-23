"""Quantize linears of a checkpoint to a compressed-tensors W8A8 scheme with dynamic per-token
activations and per-output-channel weights.

``--scheme fp8``: weights FP8 E4M3, scale = amax(row) / 448 (``float-quantized``; tensor cores on
sm89 and up). ``--scheme int8``: weights INT8 symmetric, scale = amax(row) / 127 (``int-quantized``;
INT8 tensor cores on sm75 and up, where FP8 is weight-only). The scale is rounded to its stored
dtype before the weight is divided by it, so the stored weight is exact for the stored scale.

Which BF16 weights: names ending with one of ``--suffixes``, or with ``--like REF`` every weight
REF stores with a ``weight_scale`` (REF's ``ignore`` list is taken with it); ``--keep`` suffixes
stay BF16 and join ``ignore`` (keep a fused group whole: q/k/v, gate/up, qkv/z). The quantization
config is the source's own, or REF's, with the scheme's type and format.

    python tools/quantize.py SRC DST --scheme fp8 --suffixes linear_attn.in_proj_qkv.weight \
        linear_attn.in_proj_z.weight linear_attn.out_proj.weight
    python tools/quantize.py qwen35-4b-bf16 qwen35-4b-int8 --scheme int8 --like qwen35-4b-fp8-gdn
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCHEMES = {
    "fp8": (torch.float8_e4m3fn, torch.finfo(torch.float8_e4m3fn).max, "float"),
    "int8": (torch.int8, 127.0, "int"),
}
DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def quantize(w: torch.Tensor, scheme: str, scale_dtype: torch.dtype):
    qdtype, qmax, _ = SCHEMES[scheme]
    w32 = w.float()
    scale = (w32.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-12).to(scale_dtype)
    q = (w32 / scale.float()).clamp(-qmax, qmax)
    return (q.round() if qdtype == torch.int8 else q).to(qdtype), scale


def ref_names(ref: Path) -> set[str]:
    names = json.loads((ref / "model.safetensors.index.json").read_text())["weight_map"]
    return {k.removesuffix("_scale") for k in names if k.endswith(".weight_scale")}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--scheme", choices=sorted(SCHEMES), required=True)
    pick = p.add_mutually_exclusive_group(required=True)
    pick.add_argument("--suffixes", nargs="+")
    pick.add_argument("--like")
    p.add_argument("--keep", nargs="*", default=[], help="weight-name suffixes left in BF16")
    p.add_argument("--scale-dtype", choices=sorted(DTYPES), default="BF16")
    args = p.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    ref = Path(args.like) if args.like else None
    wanted = ref_names(ref) if ref else None
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name != "config.json":
            shutil.copy(f, dst / f.name)

    kept: set[str] = set()

    def chosen(k: str) -> bool:
        if any(k.endswith(x) for x in args.keep):
            kept.add(k.removesuffix(".weight"))
            return False
        return k in wanted if wanted is not None else any(k.endswith(x) for x in args.suffixes)

    weight_map, done = {}, []
    sdt = DTYPES[args.scale_dtype]
    for shard in sorted(src.glob("*.safetensors")):
        out = {}
        with safe_open(shard, "pt") as s:
            meta = s.metadata()
            for k in list(s.keys()):
                t = s.get_tensor(k)
                if chosen(k) and t.dtype == torch.bfloat16:
                    q, scale = quantize(
                        t.cuda() if torch.cuda.is_available() else t, args.scheme, sdt
                    )
                    out[k], out[k + "_scale"] = q.cpu(), scale.cpu()
                    done.append(k.removesuffix(".weight"))
                else:
                    out[k] = t
        save_file(out, dst / shard.name, metadata=meta)
        weight_map.update(dict.fromkeys(out, shard.name))

    index = src / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
        data["weight_map"] = weight_map
        (dst / index.name).write_text(json.dumps(data, indent=2))
    config = json.loads((src / "config.json").read_text())
    if ref:
        config["quantization_config"] = json.loads((ref / "config.json").read_text())[
            "quantization_config"
        ]
    q = config["quantization_config"]
    kind = SCHEMES[args.scheme][2]
    q["format"] = f"{kind}-quantized"
    for group in q["config_groups"].values():
        group["format"] = q["format"]
        for side in ("weights", "input_activations"):
            group[side]["type"] = kind
    gone = set(done)
    q["ignore"] = [n for n in q["ignore"] if n not in gone] + sorted(kept - set(q["ignore"]))
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    print(f"quantized {len(done)} linears to {args.scheme}; {len(q['ignore'])} left in ignore")
    return 0


if __name__ == "__main__":
    sys.exit(main())
