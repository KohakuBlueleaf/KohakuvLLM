"""Measured matmul throughput of the card per dtype (TFLOPS / TOPS), square and engine-shaped
GEMMs, CUDA-event timed: what the checkpoint choice (BF16, INT8 W8A8, FP8) rests on.

    python bench/gemm/peak.py [--out peak.json]
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch

# (M, N, K): square, then Qwen3.5-4B's decode-step and prefill-step projections
SHAPES = [(8192, 8192, 8192), (384, 9728, 2560), (384, 2560, 9728), (8192, 9728, 2560)]
MATMUL = torch.backends.cuda.matmul


def timed(fn, iters: int = 50) -> float:
    for _ in range(5):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3


def operands(m, n, k, dtype, dev):
    if dtype == torch.int8:
        a = torch.randint(-128, 127, (m, k), device=dev, dtype=dtype)
        return a, torch.randint(-128, 127, (n, k), device=dev, dtype=dtype).t()
    if dtype == torch.float8_e4m3fn:
        a = torch.randn(m, k, device=dev).to(dtype)
        return a, torch.randn(n, k, device=dev).to(dtype).t()
    return torch.randn(m, k, device=dev, dtype=dtype), torch.randn(k, n, device=dev, dtype=dtype)


def scaled(a, b, one):
    return torch._scaled_mm(a, b, one, one, out_dtype=torch.bfloat16)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability(dev)
    one = torch.ones((), device=dev)
    # name: (operand dtype, the op, reduced-precision fp16 accumulation, tf32)
    cases = {
        "bf16": (torch.bfloat16, torch.matmul, False, False),
        "fp16": (torch.float16, torch.matmul, False, False),
        "fp16_reduced_acc": (torch.float16, torch.matmul, True, False),
        "tf32": (torch.float32, torch.matmul, False, True),
        "int8": (torch.int8, torch._int_mm, False, False),
    }
    if cap >= (8, 9):
        cases["fp8"] = (torch.float8_e4m3fn, partial(scaled, one=one), False, False)
    report = {"gpu": torch.cuda.get_device_name(dev), "capability": list(cap), "rows": []}
    for m, n, k in SHAPES:
        row = {"shape": [m, n, k]}
        for name, (dtype, op, reduced, tf32) in cases.items():
            MATMUL.allow_fp16_reduced_precision_reduction = reduced
            MATMUL.allow_tf32 = tf32
            a, b = operands(m, n, k, dtype, dev)
            row[name] = round(2 * m * n * k / timed(partial(op, a, b)) / 1e12, 1)
        report["rows"].append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
