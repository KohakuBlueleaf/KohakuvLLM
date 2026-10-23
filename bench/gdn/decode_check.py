"""The GDN decode step (one token per sequence, recurrent state in vLLM's pool): vLLM's FLA
``fused_recurrent_gated_delta_rule_packed_decode`` against FlashInfer's
``gated_delta_rule_decode_pretranspose`` (bf16 state pool, gathered and written in place).

Both read the post-conv ``mixed_qkv`` ([q | k | v] per token) and the same state pool, laid out
as vLLM keeps it: ``[blocks, HV, V, K]`` bf16 inside a page that also holds the conv state (the
block stride is larger than one state). The fp64 reference runs the recurrence from the same
starting states; each path's output and updated states are compared against it, and the two
paths' errors are reported side by side (an error profile, no threshold). Slots not in the batch
must stay untouched. Then GPU time per call over batch sizes, eager and CUDA-graph replayed.

    python bench/gdn/decode_check.py [--hv 32] [--h 16] [--batches 1,8,32,64,128,192,256,384]
"""

import argparse
import json
import sys

import torch
import torch.nn.functional as F
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
)

DIM = 128


def reference(mixed, a, b, A_log, dt_bias, pool, slots, H, HV):
    """fp64 decode step: outputs [B, HV, V] and the updated states [B, HV, V, K]."""
    B = mixed.shape[0]
    x = mixed.double()
    q = x[:, : H * DIM].view(B, H, DIM)
    k = x[:, H * DIM : 2 * H * DIM].view(B, H, DIM)
    v = x[:, 2 * H * DIM :].view(B, HV, DIM)
    q = F.normalize(q, dim=-1) * DIM**-0.5
    k = F.normalize(k, dim=-1)
    rep = HV // H
    q, k = q.repeat_interleave(rep, 1), k.repeat_interleave(rep, 1)
    g = -A_log.double().exp() * F.softplus(a.double() + dt_bias.double())
    beta = b.double().sigmoid()
    h = pool[slots.long()].double() * g.exp()[..., None, None]
    v = (v - (h * k[:, :, None, :]).sum(-1)) * beta[..., None]
    h = h + v[..., None] * k[:, :, None, :]
    return (h * q[:, :, None, :]).sum(-1), h


class Step:
    """One decode step's inputs in persistent buffers, run by either path."""

    def __init__(self, B, H, HV, pool, A_log, dt_bias, gen):
        dev = pool.device
        width = (2 * H + HV) * DIM
        self.H, self.HV, self.pool, self.A_log, self.dt_bias = H, HV, pool, A_log, dt_bias
        self.mixed = torch.randn(B, width, device=dev, generator=gen).bfloat16()
        self.a = torch.randn(B, HV, device=dev, generator=gen).bfloat16()
        self.b = torch.randn(B, HV, device=dev, generator=gen).bfloat16()
        perm = torch.randperm(pool.shape[0] - 1, generator=torch.Generator().manual_seed(B)) + 1
        self.slots = perm[:B].to(dev, torch.int32)
        self.out = torch.zeros(B, HV, DIM, dtype=torch.bfloat16, device=dev)

    def fla(self):
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=self.mixed,
            a=self.a,
            b=self.b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=DIM**-0.5,
            initial_state=self.pool,
            out=self.out.unsqueeze(1),
            ssm_state_indices=self.slots,
            use_qk_l2norm_in_kernel=True,
        )

    def flashinfer(self):
        B, H, HV = self.mixed.shape[0], self.H, self.HV
        q = self.mixed[:, : H * DIM].view(B, 1, H, DIM)
        k = self.mixed[:, H * DIM : 2 * H * DIM].view(B, 1, H, DIM)
        v = self.mixed[:, 2 * H * DIM :].view(B, 1, HV, DIM)
        gated_delta_rule_decode_pretranspose(
            q,
            k,
            v,
            None,
            self.A_log,
            self.a.unsqueeze(1),
            self.dt_bias,
            self.b.unsqueeze(1),
            scale=DIM**-0.5,
            output=self.out.unsqueeze(1),
            use_qk_l2norm=True,
            initial_state=self.pool,
            initial_state_indices=self.slots,
            backend="flashinfer",
        )


def gpu_ms(fn, reps: int = 50) -> float:
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return round(a.elapsed_time(b) / reps, 4)


def graphed(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


def errors(got, ref) -> dict:
    d = (got.double() - ref).abs()
    mag = ref.abs()
    big = mag > mag.mean()
    rel = d[big] / mag[big]
    rel = rel[torch.randperm(rel.numel(), device=rel.device)[: 1 << 22]]
    return {
        "abs_max": float(d.max()),
        "abs_mean": float(d.mean()),
        "ref_abs_mean": float(mag.mean()),
        "rel_mean_where_ref_above_mean": float(rel.mean()),
        "rel_p99_where_ref_above_mean": float(rel.quantile(0.99)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hv", type=int, default=32)
    p.add_argument("--h", type=int, default=16)
    p.add_argument("--batches", default="1,8,32,64,128,192,256,384")
    p.add_argument("--blocks", type=int, default=520)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    H, HV = args.h, args.hv
    state = HV * DIM * DIM
    page = state + (2 * H + HV) * DIM * 3
    raw = torch.randn(args.blocks, page, device=dev, generator=gen).bfloat16() * 0.05
    pool0 = raw[:, :state].view(args.blocks, HV, DIM, DIM)
    A_log = torch.rand(HV, device=dev, generator=gen) * 2 - 1
    dt_bias = (torch.randn(HV, device=dev, generator=gen) * 0.5).bfloat16()
    report = {"shape": {"H": H, "HV": HV, "K": DIM, "V": DIM, "page_elems": page}}
    for B in [int(x) for x in args.batches.split(",")]:
        step = Step(B, H, HV, pool0, A_log, dt_bias, gen)
        ref_o, ref_h = reference(
            step.mixed, step.a, step.b, A_log, dt_bias, pool0, step.slots, H, HV
        )
        idx = step.slots.long()
        rest = ~torch.isin(torch.arange(args.blocks, device=dev), idx)
        row = {}
        for name in ("fla", "flashinfer"):
            raw_run = raw.clone()
            step.pool = raw_run[:, :state].view(args.blocks, HV, DIM, DIM)
            step.out.zero_()
            getattr(step, name)()
            torch.cuda.synchronize()
            o, h = step.out.clone(), step.pool[idx].clone()
            row[name] = {
                "out": errors(o, ref_o),
                "state": errors(h, ref_h),
                "other_slots_untouched": bool(
                    torch.equal(step.pool[rest], pool0[rest])
                    and torch.equal(raw_run[:, state:], raw[:, state:])
                ),
                "ms_eager": gpu_ms(getattr(step, name)),
                "ms_graph": gpu_ms(graphed(getattr(step, name))),
            }
        row["speedup_graph"] = round(row["fla"]["ms_graph"] / row["flashinfer"]["ms_graph"], 3)
        report[f"B{B}"] = row
        print(f"B{B}", json.dumps(row), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
