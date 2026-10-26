"""The 8-bit GDN states (``kohakuvllm.gdn.state8``: FP8 on sm89+, INT8) against vLLM's bf16 state
pool and an fp64 reference, over many decode steps.

From the same float32 starting states (each pool rounds them its own way), ``--steps`` decode
steps run on random inputs through vLLM's packed decode on a bf16 pool and through ours on each
8-bit pool; the fp64 recurrence from the float32 states is the ruler. At steps 1, 8, ... the
outputs and the states of every path are profiled against the ruler, side by side (no
threshold). Slots outside the batch must stay untouched. Then CUDA-graph-replayed GPU time per
step over batch sizes, per format and decode tile (the Triton kernel's ``TILE``).

    python bench/gdn/state8_check.py [--hv 32] [--h 16] [--steps 64] [--batches 1,32,128,256,384]
        [--tiles 8x1,16x1,32x1,...]
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
)

sys.path.insert(0, str(Path(__file__).parent))

from decode_check import DIM, errors, gpu_ms, graphed

from kohakuvllm.gdn import state8


def ruler_step(mixed, a, b, A_log, dt_bias, h, H, HV):
    """fp64 decode step on states ``h`` [B, HV, V, K]: outputs and the new states."""
    B = mixed.shape[0]
    x = mixed.double()
    q = x[:, : H * DIM].view(B, H, DIM)
    k = x[:, H * DIM : 2 * H * DIM].view(B, H, DIM)
    v = x[:, 2 * H * DIM :].view(B, HV, DIM)
    q = torch.nn.functional.normalize(q, dim=-1) * DIM**-0.5
    k = torch.nn.functional.normalize(k, dim=-1)
    rep = HV // H
    q, k = q.repeat_interleave(rep, 1), k.repeat_interleave(rep, 1)
    g = -A_log.double().exp() * torch.nn.functional.softplus(a.double() + dt_bias.double())
    beta = b.double().sigmoid()
    h = h * g.exp()[..., None, None]
    v = (v - (h * k[:, :, None, :]).sum(-1)) * beta[..., None]
    h = h + v[..., None] * k[:, :, None, :]
    return (h * q[:, :, None, :]).sum(-1), h


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hv", type=int, default=32)
    p.add_argument("--h", type=int, default=16)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--b", type=int, default=64, help="sequences in the accuracy run")
    p.add_argument("--batches", default="1,32,128,256,384")
    p.add_argument(
        "--tiles", default="8x1,16x1,32x1,16x2,32x2,64x2,32x4,64x4", help="decode tiles, ROWSxWARPS"
    )
    p.add_argument("--blocks", type=int, default=520)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    tiles = [tuple(int(x) for x in t.split("x")) for t in args.tiles.split(",")]
    dev = torch.device("cuda")
    fmts = ["int8"] + (["fp8"] if torch.cuda.get_device_capability(dev) >= (8, 9) else [])
    gen = torch.Generator(device=dev).manual_seed(0)
    H, HV, B = args.h, args.hv, args.b
    width = (2 * H + HV) * DIM
    A_log = torch.rand(HV, device=dev, generator=gen) * 2 - 1
    dt_bias = (torch.randn(HV, device=dev, generator=gen) * 0.5).bfloat16()

    # The same float32 starting states in every pool; slot 0 is vLLM's null slot
    start = torch.randn(args.blocks, HV, DIM, DIM, device=dev, generator=gen) * 0.05
    pool16 = start.bfloat16()
    pools = {}
    for fmt in fmts:
        shape = (args.blocks, HV, DIM, DIM + state8.SCALE_BYTES)
        pools[fmt] = state8.quantize(start, torch.empty(shape, dtype=torch.uint8, device=dev), fmt)
    before16 = pool16.clone()
    before = {fmt: pool.clone() for fmt, pool in pools.items()}
    slots = (torch.randperm(args.blocks - 1, generator=torch.Generator().manual_seed(1)) + 1)[:B]
    slots = slots.to(dev, torch.int32)
    h_ref = start[slots.long()].double()

    report = {"shape": {"H": H, "HV": HV, "K": DIM, "V": DIM}, "formats": fmts, "steps": {}}
    out16 = torch.empty(B, 1, HV, DIM, dtype=torch.bfloat16, device=dev)
    outs = {fmt: torch.empty_like(out16) for fmt in fmts}
    marks = {1, 8, 32, args.steps}
    for step in range(1, args.steps + 1):
        mixed = torch.randn(B, width, device=dev, generator=gen).bfloat16()
        a = torch.randn(B, HV, device=dev, generator=gen).bfloat16()
        b = torch.randn(B, HV, device=dev, generator=gen).bfloat16()
        o_ref, h_ref = ruler_step(mixed, a, b, A_log, dt_bias, h_ref, H, HV)
        common = {"a": a, "b": b, "A_log": A_log, "dt_bias": dt_bias, "scale": DIM**-0.5}
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed,
            initial_state=pool16,
            out=out16,
            ssm_state_indices=slots,
            use_qk_l2norm_in_kernel=True,
            **common,
        )
        for fmt in fmts:
            state8.packed_decode(
                mixed,
                initial_state=pools[fmt],
                out=outs[fmt],
                ssm_state_indices=slots,
                use_qk_l2norm_in_kernel=True,
                fmt=fmt,
                **common,
            )
        if step in marks:
            idx = slots.long()
            row = {"bf16_out": errors(out16[:, 0], o_ref), "bf16_state": errors(pool16[idx], h_ref)}
            for fmt in fmts:
                row[f"{fmt}_out"] = errors(outs[fmt][:, 0], o_ref)
                row[f"{fmt}_state"] = errors(state8.dequantize(pools[fmt][idx], DIM, fmt), h_ref)
            report["steps"][step] = row
            print(step, json.dumps(row), flush=True)

    rest = ~torch.isin(torch.arange(args.blocks, device=dev), slots.long())
    report["other_slots_untouched"] = {
        "bf16": bool(torch.equal(pool16[rest], before16[rest])),
        **{fmt: bool(torch.equal(pools[fmt][rest], before[fmt][rest])) for fmt in fmts},
    }

    for n in [int(x) for x in args.batches.split(",")]:
        mixed = torch.randn(n, width, device=dev, generator=gen).bfloat16()
        a = torch.randn(n, HV, device=dev, generator=gen).bfloat16()
        b = torch.randn(n, HV, device=dev, generator=gen).bfloat16()
        idx = (torch.randperm(args.blocks - 1, generator=torch.Generator().manual_seed(n)) + 1)[:n]
        idx = idx.to(dev, torch.int32)
        o16 = torch.empty(n, 1, HV, DIM, dtype=torch.bfloat16, device=dev)
        kw = {
            "a": a,
            "b": b,
            "A_log": A_log,
            "dt_bias": dt_bias,
            "scale": DIM**-0.5,
            "ssm_state_indices": idx,
            "use_qk_l2norm_in_kernel": True,
        }
        run16 = partial(
            fused_recurrent_gated_delta_rule_packed_decode,
            mixed_qkv=mixed,
            initial_state=pool16,
            out=o16,
            **kw,
        )
        row = {"bf16_ms_graph": gpu_ms(graphed(run16)), "bf16_ms_eager": gpu_ms(run16)}
        for fmt in fmts:
            for tile in tiles:
                run8 = partial(
                    state8.packed_decode,
                    mixed,
                    initial_state=pools[fmt],
                    out=torch.empty_like(o16),
                    fmt=fmt,
                    tile={"rows": tile[0], "warps": tile[1]},
                    **kw,
                )
                name = f"{fmt}_{tile[0]}x{tile[1]}"
                row[f"{name}_ms_graph"] = gpu_ms(graphed(run8))
                row[f"{name}_speedup_graph"] = round(
                    row["bf16_ms_graph"] / row[f"{name}_ms_graph"], 3
                )
        report[f"B{n}"] = row
        print(f"B{n}", json.dumps(row), flush=True)
    print(json.dumps(report["other_slots_untouched"]), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
