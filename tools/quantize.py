"""Quantize linears of a BF16 checkpoint to compressed-tensors schemes, one scheme per group of
linears, so one checkpoint can mix them (the MLP in NVFP4, the projections in FP8).

Schemes (``SCHEMES``):

- ``fp8``: weights FP8 E4M3, one scale per output channel (amax / 448), activations dynamic per
  token (``float-quantized``; tensor cores on sm89 and up).
- ``int8``: the same in INT8 (amax / 127; ``int-quantized``; sm75 and up, where FP8 is
  weight-only).
- ``mxfp8``: weights E4M3 with one E8M0 scale per 32 along K (``2^ceil(log2(amax / 448))``, so
  no element overflows), activations dynamic MXFP8 (``mxfp8-quantized``; block-scaled MMA).
- ``nvfp4``: weights E2M1 packed two per byte (the even element in the low nibble) with one E4M3
  scale per 16 along K and an FP32 tensor scale, activations the same with a static tensor scale
  from ``--act-amax`` (``tools/calibrate.py``) (``nvfp4-pack-quantized``; sm100 and sm120).
  Stored global scales are divisors, ``448 * 6 / amax``; linears vLLM fuses (``--fuse``) share
  theirs, the largest amax of the group.

Every stored scale is rounded to its stored dtype before the weight is divided by it, so the
stored weight is exact for the stored scale.

Groups: ``--group SCHEME SUFFIX [SUFFIX ...]`` (repeatable) takes the BF16 weights whose names end
with a suffix; ``--scheme S --suffixes ...`` is one group, ``--scheme S --like REF`` one group of
every weight REF stores with a ``weight_scale``. ``--keep`` suffixes stay BF16 and join
``ignore`` (keep a fused group whole: q/k/v, gate/up, qkv/z).

    python tools/quantize.py SRC DST --scheme fp8 --suffixes linear_attn.in_proj_qkv.weight \
        linear_attn.in_proj_z.weight linear_attn.out_proj.weight
    python tools/quantize.py qwen35-4b-bf16 qwen35-4b-int8 --scheme int8 --like qwen35-4b-fp8-gdn
    python tools/quantize.py qwen35-4b-bf16 qwen35-4b-mlp4 --act-amax amax.json \
        --group nvfp4 mlp.gate_proj.weight mlp.up_proj.weight mlp.down_proj.weight \
        --group fp8 self_attn.q_proj.weight self_attn.k_proj.weight self_attn.v_proj.weight \
        self_attn.o_proj.weight linear_attn.in_proj_qkv.weight linear_attn.in_proj_z.weight \
        linear_attn.out_proj.weight
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E4M3 = torch.float8_e4m3fn
E4M3_MAX = torch.finfo(E4M3).max
FP4_MAX = 6.0
# E2M1's magnitudes; a value's code is its index (the sign is bit 3)
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
SCHEMES = ("fp8", "int8", "mxfp8", "nvfp4")
DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
FUSE = ("q_proj,k_proj,v_proj", "gate_proj,up_proj", "in_proj_qkv,in_proj_z", "in_proj_b,in_proj_a")


def per_channel(w32: torch.Tensor, scheme: str, scale_dtype: torch.dtype) -> dict:
    qdtype, qmax = (E4M3, E4M3_MAX) if scheme == "fp8" else (torch.int8, 127.0)
    scale = (w32.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-12).to(scale_dtype)
    q = (w32 / scale.float()).clamp(-qmax, qmax)
    return {"weight": (q.round() if scheme == "int8" else q).to(qdtype), "weight_scale": scale}


def mx8(w32: torch.Tensor) -> dict:
    n, k = w32.shape
    blocks = w32.view(n, k // 32, 32)
    amax = blocks.abs().amax(-1).clamp(min=2.0**-126)
    exp = torch.ceil(torch.log2(amax / E4M3_MAX)).clamp(-127, 127)
    q = (blocks / torch.exp2(exp)[..., None]).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)
    return {"weight": q.view(n, k), "weight_scale": (exp + 127).to(torch.uint8)}


def e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 value (ties to the even code) and return the 4-bit codes."""
    grid = FP4_GRID.to(x.device)
    mag = x.abs().clamp(max=FP4_MAX)
    # The code is the number of midpoints below the value; on a midpoint the even code wins
    mids = (grid[1:] + grid[:-1]) / 2
    code = torch.bucketize(mag, mids, right=False)
    tie = (code < 7) & (mag == mids[code.clamp(max=6)]) & (code % 2 == 1)
    code = torch.where(tie, code + 1, code)
    return (code | ((x < 0) & (code > 0)).to(code.dtype) << 3).to(torch.uint8)


def nvfp4(w32: torch.Tensor, global_scale: torch.Tensor) -> dict:
    """``global_scale``: the stored divisor ``448 * 6 / amax`` (fp32)."""
    n, k = w32.shape
    blocks = w32.view(n, k // 16, 16)
    scale = (blocks.abs().amax(-1) / FP4_MAX * global_scale).clamp(max=E4M3_MAX).to(E4M3)
    step = scale.float() / global_scale
    codes = e2m1(torch.where(step[..., None] > 0, blocks / step[..., None], 0.0)).view(n, k)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return {
        "weight_packed": packed,
        "weight_scale": scale,
        "weight_global_scale": global_scale.reshape(1),
    }


def group_config(scheme: str) -> tuple[str, dict, dict]:
    """(format, weights args, input activations args) of a compressed-tensors config group."""
    base = {"symmetric": True, "actorder": None, "block_structure": None, "observer_kwargs": {}}
    match scheme:
        case "fp8" | "int8":
            kind = "float" if scheme == "fp8" else "int"
            w = {**base, "num_bits": 8, "type": kind, "strategy": "channel", "dynamic": False}
            a = {**base, "num_bits": 8, "type": kind, "strategy": "token", "dynamic": True}
            return f"{kind}-quantized", w, a
        case "mxfp8":
            mx = {"num_bits": 8, "type": "float", "strategy": "group", "group_size": 32}
            mx["scale_dtype"] = "torch.uint8"
            w = {**base, **mx, "dynamic": False}
            return "mxfp8-quantized", w, {**base, **mx, "dynamic": True}
        case "nvfp4":
            fp4 = {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16}
            fp4["scale_dtype"] = "torch.float8_e4m3fn"
            w = {**base, **fp4, "dynamic": False}
            return "nvfp4-pack-quantized", w, {**base, **fp4, "dynamic": "local"}
    raise ValueError(scheme)


def ref_names(ref: Path) -> set[str]:
    names = json.loads((ref / "model.safetensors.index.json").read_text())["weight_map"]
    return {k.removesuffix("_scale") for k in names if k.endswith(".weight_scale")}


def fused_with(name: str, fuse: list[list[str]]) -> list[str]:
    """``name`` (a linear) and the linears vLLM fuses it with."""
    for group in fuse:
        for part in group:
            if name.endswith("." + part):
                stem = name.removesuffix(part)
                return [stem + p for p in group]
    return [name]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--group", nargs="+", action="append", default=[], metavar="SCHEME SUFFIX")
    p.add_argument("--scheme", choices=SCHEMES)
    pick = p.add_mutually_exclusive_group()
    pick.add_argument("--suffixes", nargs="+")
    pick.add_argument("--like")
    p.add_argument("--keep", nargs="*", default=[], help="weight-name suffixes left in BF16")
    p.add_argument("--skip", nargs="*", default=[], help="weight-name prefixes left as they are")
    p.add_argument("--scale-dtype", choices=sorted(DTYPES), default="BF16", help="fp8 / int8")
    p.add_argument("--act-amax", default=None, help="tools/calibrate.py's output (nvfp4)")
    p.add_argument("--fuse", nargs="*", default=list(FUSE), help="linears vLLM fuses, a,b,c")
    args = p.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    # Groups: (scheme, a weight name -> whether it belongs)
    groups: list[tuple[str, callable]] = []
    for g in args.group:
        if g[0] not in SCHEMES or len(g) < 2:
            p.error(f"--group SCHEME SUFFIX ...: got {g}")
        groups.append((g[0], lambda k, s=tuple(g[1:]): k.endswith(s)))
    if args.scheme:
        if args.like:
            wanted = ref_names(Path(args.like))
            groups.append((args.scheme, lambda k: k in wanted))
        elif args.suffixes:
            groups.append((args.scheme, lambda k, s=tuple(args.suffixes): k.endswith(s)))
        else:
            p.error("--scheme needs --suffixes or --like")
    if not groups:
        p.error("no group: --group, or --scheme with --suffixes / --like")
    if any(s == "nvfp4" for s, _ in groups) and not args.act_amax:
        p.error("nvfp4 needs --act-amax")
    act_amax = json.loads(Path(args.act_amax).read_text())["amax"] if args.act_amax else {}
    fuse = [f.split(",") for f in args.fuse]

    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name != "config.json":
            shutil.copy(f, dst / f.name)

    def scheme_of(k: str) -> str | None:
        if any(k.endswith(x) for x in args.keep) or k.startswith(tuple(args.skip)):
            return None
        hits = [s for s, member in groups if member(k)]
        if len(set(hits)) > 1:
            raise ValueError(f"{k} is in groups of {sorted(set(hits))}")
        return hits[0] if hits else None

    # NVFP4 weight amax per fused group, read before any weight is written
    shards = sorted(src.glob("*.safetensors"))
    weight_amax: dict[str, float] = {}
    for shard in shards:
        with safe_open(shard, "pt") as s:
            for k in list(s.keys()):
                if scheme_of(k) == "nvfp4":
                    weight_amax[k.removesuffix(".weight")] = s.get_tensor(k).float().abs().amax()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def shared(name: str, amax: dict, what: str) -> torch.Tensor:
        parts = fused_with(name, fuse)
        found = [float(amax[x]) for x in parts if x in amax]
        if len(found) != len(parts) and len(found) != 1:
            raise ValueError(f"{what} amax of {name}'s fused group {parts}: found {len(found)}")
        return torch.tensor(E4M3_MAX * FP4_MAX / max(max(found), 1e-12), device=dev)

    weight_map, done, kept = {}, {}, set()
    sdt = DTYPES[args.scale_dtype]
    for shard in shards:
        out = {}
        with safe_open(shard, "pt") as s:
            meta = s.metadata()
            for k in list(s.keys()):
                t = s.get_tensor(k)
                scheme = scheme_of(k) if t.dtype == torch.bfloat16 else None
                if any(k.endswith(x) for x in args.keep):
                    kept.add(k.removesuffix(".weight"))
                if scheme is None:
                    out[k] = t
                    continue
                name = k.removesuffix(".weight")
                w32 = t.to(dev).float()
                match scheme:
                    case "fp8" | "int8":
                        q = per_channel(w32, scheme, sdt)
                    case "mxfp8":
                        q = mx8(w32)
                    case "nvfp4":
                        q = nvfp4(w32, shared(name, weight_amax, "weight"))
                        if name not in act_amax and not any(
                            x in act_amax for x in fused_with(name, fuse)
                        ):
                            raise ValueError(f"no calibrated activation amax for {name}")
                        q["input_global_scale"] = shared(name, act_amax, "input").reshape(1)
                for suffix, tensor in q.items():
                    out[f"{name}.{suffix}"] = tensor.cpu()
                done[name] = scheme
        save_file(out, dst / shard.name, metadata=meta)
        weight_map.update(dict.fromkeys(out, shard.name))

    index = src / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
        data["weight_map"] = weight_map
        (dst / index.name).write_text(json.dumps(data, indent=2))
    config = json.loads((src / "config.json").read_text())
    base = config.get("quantization_config") or {}
    if args.like:
        base = json.loads((Path(args.like) / "config.json").read_text())["quantization_config"]
    used = sorted(set(done.values()), key=SCHEMES.index)
    config_groups = {}
    for i, scheme in enumerate(used):
        fmt, w, a = group_config(scheme)
        # Targets by the name within a layer: vLLM's module paths differ from the checkpoint's
        tails = sorted(
            {re.sub(r"^.*?\.layers\.\d+\.", "", n) for n, s in done.items() if s == scheme}
        )
        config_groups[f"group_{i}"] = {
            "format": fmt,
            "targets": [r"re:.*\.layers\.\d+\.(" + "|".join(re.escape(t) for t in tails) + ")$"],
            "weights": w,
            "input_activations": a,
            "output_activations": None,
        }
    formats = {g["format"] for g in config_groups.values()}
    ignore = [n for n in base.get("ignore", []) if n not in done]
    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": formats.pop() if len(formats) == 1 else "mixed-precision",
        "config_groups": config_groups,
        "ignore": ignore + sorted(kept - set(ignore)),
        "quantization_status": "compressed",
        "kv_cache_scheme": None,
    }
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    counts = {s: sum(1 for v in done.values() if v == s) for s in used}
    print(f"quantized {counts}; {len(config['quantization_config']['ignore'])} left in ignore")
    return 0


if __name__ == "__main__":
    sys.exit(main())
