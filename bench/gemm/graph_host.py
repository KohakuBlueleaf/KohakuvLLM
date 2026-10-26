"""Host cost of a CUDA graph launch per node type: each op is captured 128 times into one graph,
and the host time of ``graph.replay()`` (the launch call alone, no synchronization) is divided
by the node count. GPU time per node is reported next to it. Shapes are Qwen3.5-4B decode-step
projections (M tokens, K = 2560, N = 2560).

    python bench/gemm/graph_host.py [--m 384] [--out graph_host.json]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import flashinfer
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from flashinfer.autotuner import autotune
from vllm import _custom_ops as ops

NODES = 128


@triton.jit
def _copy(x, y, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + i, tl.load(x + i, mask=i < n), mask=i < n)


def measure(call) -> dict:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(NODES):
            call()
    host = []
    for _ in range(30):
        torch.cuda.synchronize()
        t = time.perf_counter()
        graph.replay()
        host.append(time.perf_counter() - t)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    host.sort()
    return {
        "host_us_per_node": round(host[len(host) // 2] / NODES * 1e6, 3),
        "gpu_us_per_node": round(start.elapsed_time(end) / 10 / NODES * 1e3, 3),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=384)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    m, k, n = args.m, 2560, 2560
    dev = torch.device("cuda")
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
    xg = (2688.0 / x.float().abs().amax()).reshape(1)
    wg = (2688.0 / w.float().abs().amax()).reshape(1)
    alpha = 1.0 / (xg * wg)
    x4, xs4 = ops.scaled_fp4_quant(x, xg)
    w4, ws4 = ops.scaled_fp4_quant(w, wg)
    x8, xs8 = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
    w8, ws8 = ops.scaled_fp8_quant(w, use_per_token_if_dynamic=True)
    buf = torch.empty(m * k, device=dev, dtype=torch.uint8)
    y = torch.empty_like(x)
    with autotune(True):
        flashinfer.mm_fp4(x4, w4.t(), xs4, ws4.t(), alpha, torch.bfloat16, backend="cutlass")
    torch.cuda.synchronize()
    swz = F.SwizzleType.SWIZZLE_32_4_4
    cases = {
        "memset": lambda: buf.zero_(),
        "triton_copy": lambda: _copy[(triton.cdiv(m * k, 1024),)](x, y, m * k, BLOCK=1024),
        "torch_add": lambda: torch.add(x, x, out=y),
        "fp4_quant_vllm": lambda: ops.scaled_fp4_quant(x, xg),
        "fp8_quant_vllm": lambda: ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True),
        "fp8_gemm_vllm_cutlass": lambda: ops.cutlass_scaled_mm(
            x8, w8.t(), xs8, ws8.reshape(1, -1), torch.bfloat16
        ),
        "fp4_gemm_flashinfer_cutlass": lambda: flashinfer.mm_fp4(
            x4, w4.t(), xs4, ws4.t(), alpha, torch.bfloat16, backend="cutlass"
        ),
        "fp4_gemm_vllm_cutlass": lambda: ops.cutlass_scaled_fp4_mm(
            x4, w4, xs4, ws4, alpha, torch.bfloat16
        ),
        "fp4_gemm_torch_cublaslt": lambda: F.scaled_mm(
            x4.view(torch.float4_e2m1fn_x2),
            w4.view(torch.float4_e2m1fn_x2).t(),
            [xs4.view(torch.float8_e4m3fn).flatten(), 1.0 / xg],
            [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            [ws4.view(torch.float8_e4m3fn).flatten(), 1.0 / wg],
            [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[swz, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[swz, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        ),
        "bf16_gemm": lambda: x @ w.t(),
    }
    report = {"m": m, "n": n, "k": k, "nodes": NODES, "cases": {}}
    for name, call in cases.items():
        try:
            report["cases"][name] = measure(call)
        except Exception as e:  # noqa: BLE001 - a failing backend is a result
            torch.cuda.synchronize()
            report["cases"][name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(name, report["cases"][name], flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
