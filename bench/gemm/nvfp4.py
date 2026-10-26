"""NVFP4 on the card: every W4A4 GEMM and activation-quantization kernel vLLM and FlashInfer ship,
against FP8 and BF16, on Qwen3.5-4B's projection shapes over the token counts a serving step
sees. Failures are recorded, never skipped.

Timing is GPU time as a served step sees it: each case runs inside one CUDA graph (no Python or
launch overhead), replayed; FlashInfer's autotuner runs before capture, as vLLM runs it at start.
A GEMM cycles through copies of its weight that together exceed twice the L2 cache (a model's
weights are never L2-resident from one layer to the next); a quantization reads one input, as it
reads the activation the kernel before it just wrote. GEMMs report TFLOPS (``2 M N K / t``);
quantization kernels report microseconds and GB/s over the bytes they must move (bf16 in, packed
E2M1 out, one E4M3 scale per 16 values, plus a fused kernel's extra operands).

    python bench/gemm/nvfp4.py [--out nvfp4.json]
"""

import argparse
import json
import sys
from pathlib import Path

import flashinfer
import torch
import torch.nn.functional as F
from flashinfer.autotuner import autotune
from vllm import _custom_ops as ops

# Qwen3.5-4B: hidden 2560, MLP 9728 (gate and up fused: 19456)
PROJ = {"gate_up": (19456, 2560), "down": (2560, 9728), "square": (2560, 2560)}
TOKENS = [16, 64, 384, 1024, 4096, 8192]
FP4_MAX, E4M3_MAX = 6.0, 448.0
L2_BYTES = 96 * 2**20


def graph_time(calls: list, replays: int = 20, least: int = 128) -> float:
    """Seconds per call: ``calls`` (cycled to at least ``least``, so a graph launch is spread
    over many kernels) captured once in one CUDA graph, the graph replayed."""
    calls = [calls[i % len(calls)] for i in range(max(least, len(calls)))]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for c in calls:
            c()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in calls:
            c()
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3 / (replays * len(calls))


def attempt(calls) -> tuple[float | None, str | None]:
    try:
        return graph_time(calls), None
    except Exception as e:  # noqa: BLE001 - every backend's failure is a result
        torch.cuda.synchronize()
        return None, f"{type(e).__name__}: {str(e)[:200]}"


def global_scale(x: torch.Tensor) -> torch.Tensor:
    """The tensor scale vLLM's NVFP4 linears take: ``448 * 6 / amax``, shape [1]."""
    return (E4M3_MAX * FP4_MAX / x.float().abs().amax()).reshape(1).to(torch.float32)


def gemm_cases(m, n, k, dev) -> dict:
    """name: one call per weight copy."""
    copies = max(2, -(-2 * L2_BYTES // (n * k // 2)))
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) for _ in range(copies)]
    xg = global_scale(x)
    wg = global_scale(ws[0])
    alpha = 1.0 / (xg * wg)
    x4, xs4 = ops.scaled_fp4_quant(x, xg)
    x8, xs8 = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
    w4s, w8s = [], []
    for w in ws:
        w4s.append(ops.scaled_fp4_quant(w, wg))
        w8 = ops.scaled_fp8_quant(w, use_per_token_if_dynamic=True)
        w8s.append((w8[0], w8[1].reshape(1, -1)))
    bf = ws
    swizzle = F.SwizzleType.SWIZZLE_32_4_4
    xa = x4.view(torch.float4_e2m1fn_x2)
    cases = {
        "bf16": [lambda w=w: x @ w.t() for w in bf],
        "fp8_cutlass": [
            lambda w=w, s=s: ops.cutlass_scaled_mm(x8, w.t(), xs8, s, torch.bfloat16)
            for w, s in w8s
        ],
        "fp8_torch_rowwise": [
            lambda w=w, s=s: torch._scaled_mm(x8, w.t(), xs8, s, out_dtype=torch.bfloat16)
            for w, s in w8s
        ],
        "fp4_vllm_cutlass": [
            lambda w=w, s=s: ops.cutlass_scaled_fp4_mm(x4, w, xs4, s, alpha, torch.bfloat16)
            for w, s in w4s
        ],
        "fp4_torch_cublaslt": [
            lambda w=w, s=s: F.scaled_mm(
                xa,
                w.view(torch.float4_e2m1fn_x2).t(),
                [xs4.view(torch.float8_e4m3fn).flatten(), 1.0 / xg],
                [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
                [s.view(torch.float8_e4m3fn).flatten(), 1.0 / wg],
                [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
                swizzle_a=[swizzle, F.SwizzleType.NO_SWIZZLE],
                swizzle_b=[swizzle, F.SwizzleType.NO_SWIZZLE],
                output_dtype=torch.bfloat16,
            )
            for w, s in w4s
        ],
    }
    for backend in ("cutlass", "cudnn", "b12x"):
        calls = [
            lambda w=w, s=s, b=backend: flashinfer.mm_fp4(
                x4, w.t(), xs4, s.t(), alpha, torch.bfloat16, backend=b
            )
            for w, s in w4s
        ]
        try:
            with autotune(True):
                calls[0]()
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001 - the timed attempt records the failure
            pass
        cases[f"fp4_flashinfer_{backend}"] = calls
    return cases


def quant_cases(m, k, dev) -> dict:
    """name: (calls, bytes moved per call)."""
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    res = torch.randn_like(x)
    weight = torch.randn(k, device=dev, dtype=torch.bfloat16)
    gate_up = torch.randn(m, 2 * k, device=dev, dtype=torch.bfloat16)
    xg = global_scale(x)
    xg_inv = 1.0 / xg
    fp4 = m * k // 2 + m * k // 16
    plain = m * k * 2 + fp4
    out = torch.empty(m, k // 2, dtype=torch.uint8, device=dev)
    sf = torch.empty(-(-m // 128) * 128, -(-k // 64) * 4, dtype=torch.uint8, device=dev)
    return {
        "fp8_vllm_per_token": (
            [lambda: ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)],
            m * k * 3 + m * 4,
        ),
        "fp4_vllm": ([lambda: ops.scaled_fp4_quant(x, xg)], plain),
        "fp4_flashinfer_cuda": ([lambda: flashinfer.fp4_quantize(x, xg)], plain),
        "fp4_flashinfer_cute_dsl": (
            [lambda: flashinfer.nvfp4_quantize(x, xg, backend="cute-dsl")],
            plain,
        ),
        "rmsnorm+fp4_flashinfer": (
            [lambda: flashinfer.rmsnorm_fp4quant(x, weight, global_scale=xg)],
            plain + k * 2,
        ),
        "add_rmsnorm+fp4_flashinfer": (
            [lambda: flashinfer.add_rmsnorm_fp4quant(x, res, weight, global_scale=xg)],
            plain + m * k * 4 + k * 2,
        ),
        "silu_mul+fp4_vllm": (
            [
                lambda: torch.ops._C.silu_and_mul_nvfp4_quant(
                    out, sf.view(torch.float8_e4m3fn), gate_up, xg_inv
                )
            ],
            m * 2 * k * 2 + fp4,
        ),
        "silu_mul+fp4_flashinfer": (
            [lambda: flashinfer.silu_and_mul_nvfp4_quantize(gate_up, xg)],
            m * 2 * k * 2 + fp4,
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    dev = torch.device("cuda")
    report = {
        "gpu": torch.cuda.get_device_name(dev),
        "capability": list(torch.cuda.get_device_capability(dev)),
        "gemm": [],
        "quant": [],
    }
    print(json.dumps({k: report[k] for k in ("gpu", "capability")}), flush=True)
    for proj, (n, k) in PROJ.items():
        for m in TOKENS:
            row = {"proj": proj, "shape": [m, n, k], "tflops": {}, "us": {}, "errors": {}}
            for name, calls in gemm_cases(m, n, k, dev).items():
                t, err = attempt(calls)
                if t is None:
                    row["errors"][name] = err
                else:
                    row["tflops"][name] = round(2 * m * n * k / t / 1e12, 1)
                    row["us"][name] = round(t * 1e6, 2)
            report["gemm"].append(row)
            print(json.dumps(row), flush=True)
    for k in (2560, 9728):
        for m in TOKENS:
            row = {"shape": [m, k], "us": {}, "gb_s": {}, "errors": {}}
            for name, (calls, nbytes) in quant_cases(m, k, dev).items():
                t, err = attempt(calls)
                if t is None:
                    row["errors"][name] = err
                    continue
                row["us"][name] = round(t * 1e6, 2)
                row["gb_s"][name] = round(nbytes / t / 1e9, 1)
            report["quant"].append(row)
            print(json.dumps(row), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
