"""Our NVFP4 producers (``kohakuvllm.nvfp4``) against vLLM's, byte for byte.

- ``quant``: our quantization against ``scaled_fp4_quant`` on the same bf16 input.
- ``norm``: our add + RMSNorm + quantization (one and two outputs) against vLLM's ops run apart:
  ``fused_add_rms_norm``'s reference (fp32 math, the same as the compiled graph's) and then
  ``scaled_fp4_quant``.

Inputs are heavy-tailed (Student-t, 3 degrees of freedom, a few outlier channels), at the token
counts and widths of Qwen3.5 4B / 9B / 27B. Per case: the share of identical code, scale (padded
tile included) and residual bytes, and the decoded E2M1 differences of differing codes, so a tie
read one step apart shows apart from a real error. No threshold: the profile is for review.

    python bench/precision/nvfp4_ops.py [--out nvfp4_ops.json]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import vllm.ir.ops
from vllm import _custom_ops as vops

from kohakuvllm.kernels.backends.triton.nvfp4 import entry as triton_nvfp4
from kohakuvllm.nvfp4 import ops

GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
TOKENS = [1, 5, 64, 384, 1000, 8192]
WIDTHS = [2560, 4096, 5120, 9728]


def activation(m: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = torch.distributions.StudentT(3.0).sample((m, k)).cuda()
    cols = torch.randint(0, k, (4,), generator=g, device="cuda")
    t[:, cols] *= 30.0
    return t.to(torch.bfloat16)


def decode(q: torch.Tensor, k: int) -> torch.Tensor:
    """Packed codes [M, K/2] -> E2M1 values [M, K] (unscaled)."""
    lo, hi = q & 0xF, q >> 4
    codes = torch.stack([lo, hi], -1).reshape(q.shape[0], k).long()
    mag = GRID.to(q.device)[codes & 7]
    return torch.where(codes >= 8, -mag, mag)


def compare_codes(ours: torch.Tensor, ref: torch.Tensor, k: int) -> dict:
    same = (ours == ref).float().mean().item()
    a, b = decode(ours, k), decode(ref, k)
    diff = (a - b).abs()
    differ = diff > 0
    return {
        "code_bytes_same": round(same, 6),
        "values_differ": int(differ.sum()),
        "values": a.numel(),
        "max_value_diff_e2m1": float(diff.max()) if differ.any() else 0.0,
        "value_diffs": {str(v): int((diff == v).sum()) for v in diff[differ].unique().tolist()},
    }


def quant_case(m: int, k: int) -> dict:
    x = activation(m, k, m * 7 + k)
    gs = (2688.0 / x.float().abs().amax()).reshape(1)
    rq, rs = vops.scaled_fp4_quant(x, gs)
    row = {"op": "quant", "m": m, "k": k}
    for cvt in ("ptx", "arith"):
        triton_nvfp4.CVT["name"] = cvt
        q, s = ops.nvfp4_quant(x, gs)
        row[cvt] = {
            **compare_codes(q, rq, k),
            "scale_bytes_same": round(
                (s.view(torch.uint8) == rs.view(torch.uint8)).float().mean().item(), 6
            ),
        }
    return row


def norm_case(m: int, k: int, two: bool) -> dict:
    x = activation(m, k, m * 11 + k)
    r = activation(m, k, m * 13 + k)
    w = (torch.randn(k, device="cuda") * 0.1).float() + 1.0
    eps = 1e-6
    rms, res = vllm.ir.ops.fused_add_rms_norm.impls["native"].impl_fn(x, r, w, eps)
    gs = [(2688.0 / rms.float().abs().amax() * f).reshape(1) for f in (1.0, 0.5)]
    refs = [vops.scaled_fp4_quant(rms, g) for g in gs[: 2 if two else 1]]
    row = {"op": "add_rmsnorm_x2" if two else "add_rmsnorm", "m": m, "k": k}
    for cvt in ("ptx", "arith"):
        triton_nvfp4.CVT["name"] = cvt
        if two:
            q0, s0, q1, s1, ores = ops.add_rmsnorm_nvfp4x2(x, r, w, eps, gs[0], gs[1])
            outs = [(q0, s0), (q1, s1)]
        else:
            q0, s0, ores = ops.add_rmsnorm_nvfp4(x, r, w, eps, gs[0])
            outs = [(q0, s0)]
        row[cvt] = {
            "residual_bytes_same": round((ores == res).float().mean().item(), 6),
            "outputs": [
                {
                    **compare_codes(q, rq, k),
                    "scale_bytes_same": round(
                        (s.view(torch.uint8) == rs.view(torch.uint8)).float().mean().item(), 6
                    ),
                }
                for (q, s), (rq, rs) in zip(outs, refs)
            ],
        }
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rows = []
    for k in WIDTHS:
        for m in TOKENS:
            for row in (quant_case(m, k), norm_case(m, k, False), norm_case(m, k, True)):
                rows.append(row)
                print(json.dumps(row), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
