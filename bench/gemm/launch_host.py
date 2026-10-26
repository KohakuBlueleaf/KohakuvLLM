"""Host time per call of the NVFP4 producers when launched eagerly (vLLM runs the steps past its
largest captured CUDA graph eagerly): our Triton ops, through the custom-op dispatcher and
directly, against vLLM's ``scaled_fp4_quant``. Small problems, so the device never holds the host
back: the time is the launch path's.

    python bench/gemm/launch_host.py [--out launch_host.json]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from vllm import _custom_ops as vops

from kohakuvllm.nvfp4 import ops

CALLS = 2000


def host_us(call) -> float:
    for _ in range(50):
        call()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(CALLS):
        call()
    host = (time.perf_counter() - start) * 1e6 / CALLS
    torch.cuda.synchronize()
    return round(host, 2)


def shape(m: int, k: int) -> dict:
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x)
    w = torch.rand(k, device="cuda") + 0.5
    gs = torch.ones(1, device="cuda")
    q_ref, s_ref = vops.scaled_fp4_quant(x, gs)
    q, s = ops.nvfp4_quant(x, gs)
    same = torch.equal(q, q_ref) and torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))
    return {
        "bytes_equal_vllm": bool(same),
        "vllm_scaled_fp4_quant": host_us(lambda: vops.scaled_fp4_quant(x, gs)),
        "ours_quant_direct": host_us(lambda: ops.nvfp4_quant(x, gs)),
        "ours_quant_op": host_us(lambda: torch.ops.vllm.kohaku_nvfp4_quant(x, gs)),
        "ours_norm2_direct": host_us(lambda: ops.rmsnorm_nvfp4(x, r, w, 1e-6, gs, gs)),
        "ours_norm2_op": host_us(
            lambda: torch.ops.vllm.kohaku_rmsnorm_nvfp4(x, r, w, 1e-6, gs, gs)
        ),
        "torch_empty": host_us(lambda: torch.empty(m, k // 2, device="cuda")),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    report = {}
    for m, k in ((16, 2560), (384, 2560), (8192, 2560)):
        row = shape(m, k)
        report[f"m{m}_k{k}"] = row
        print(m, k, json.dumps(row), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
