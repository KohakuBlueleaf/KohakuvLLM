"""Speed of our NVFP4 producers (``kohakuvllm.nvfp4``) against the kernels a vLLM graph runs in
their place, and of the LM head in bf16 / FP8 / NVFP4, on the RTX 5090.

Producers: plain quantization (ours, every tile of ``TILES``, against ``scaled_fp4_quant``); add
+ RMSNorm + quantization, one and two outputs (ours against the compiled fp32 norm followed by
``scaled_fp4_quant``, which is what vLLM's graph runs for Qwen3.5, and FlashInfer's
``add_rmsnorm_fp4quant`` with a bf16 weight). Each case is one CUDA graph of 128 calls (no launch
overhead), the input re-read each call as the kernel before it wrote it; GB/s counts the bytes a
producer must move. Head: ``vocab x hidden`` GEMMs with their activation quantization.

    python bench/gemm/nvfp4_quant.py [--out nvfp4_quant.json]
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import flashinfer
import torch
import vllm.ir.ops
from vllm import _custom_ops as vops

from kohakuvllm.kernels.backends.triton.nvfp4 import entry as triton_nvfp4
from kohakuvllm.nvfp4 import head, ops

TOKENS = [16, 64, 384, 1024, 4096, 8192]
WIDTHS = [2560, 4096, 5120, 9728]
TILES = list(itertools.product((1, 2, 4), (256, 512, 1024, 2048, 4096, 8192), (2, 4, 8, 16)))
VOCAB = 248320
NODES = 128


def graph_us(call) -> float:
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
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(10):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (10 * NODES)


def best_tile(call) -> tuple[float, str, float]:
    """(fastest us, its tile, the default tile's us)."""
    triton_nvfp4.TILE["value"] = None
    default = graph_us(call)
    best = (default, "default")
    for t in TILES:
        triton_nvfp4.TILE["value"] = t
        try:
            best = min(best, (graph_us(call), "x".join(map(str, t))))
        except Exception:  # noqa: BLE001 - a tile that does not compile is skipped, not fatal
            torch.cuda.synchronize()
    triton_nvfp4.TILE["value"] = None
    return round(best[0], 2), best[1], round(default, 2)


def producers(m: int, k: int) -> dict:
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x)
    w = torch.rand(k, device="cuda") + 0.5
    wb = w.to(torch.bfloat16)
    gs = (2688.0 / x.float().abs().amax()).reshape(1)
    fp4 = m * k // 2 + m * k // 16
    norm = torch.compile(vllm.ir.ops.fused_add_rms_norm.impls["native"].impl_fn)
    row = {"m": m, "k": k, "us": {}, "tile": {}, "default_tile_us": {}}

    row["us"]["quant_vllm"] = round(graph_us(lambda: vops.scaled_fp4_quant(x, gs)), 2)
    t, tile, d = best_tile(lambda: ops.nvfp4_quant(x, gs))
    row["us"]["quant_ours"], row["tile"]["quant_ours"], row["default_tile_us"]["quant_ours"] = (
        t,
        tile,
        d,
    )

    def unfused(n):
        def call():
            y, _ = norm(x, r, w, 1e-6)
            for _ in range(n):
                vops.scaled_fp4_quant(y, gs)

        return call

    row["us"]["norm1_vllm_graph"] = round(graph_us(unfused(1)), 2)
    row["us"]["norm2_vllm_graph"] = round(graph_us(unfused(2)), 2)
    row["us"]["norm1_flashinfer_bf16w"] = round(
        graph_us(lambda: flashinfer.add_rmsnorm_fp4quant(x, r, wb, global_scale=gs)), 2
    )
    t, tile, d = best_tile(lambda: ops.add_rmsnorm_nvfp4(x, r, w, 1e-6, gs))
    row["us"]["norm1_ours"], row["tile"]["norm1_ours"], row["default_tile_us"]["norm1_ours"] = (
        t,
        tile,
        d,
    )
    t, tile, d = best_tile(lambda: ops.add_rmsnorm_nvfp4x2(x, r, w, 1e-6, gs, gs))
    row["us"]["norm2_ours"], row["tile"]["norm2_ours"], row["default_tile_us"]["norm2_ours"] = (
        t,
        tile,
        d,
    )

    moved = {
        "quant_ours": m * k * 2 + fp4,
        "norm1_ours": m * k * 2 * 2 * 2 + m * k * 2 + fp4 + k * 4,
        "norm2_ours": m * k * 2 * 2 * 2 + m * k * 2 + 2 * fp4 + k * 4,
    }
    row["gb_s"] = {n: round(b / (row["us"][n] * 1e3), 1) for n, b in moved.items()}
    return row


def heads(m: int, k: int) -> dict:
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    wt = torch.randn(VOCAB, k, device="cuda", dtype=torch.bfloat16) * 0.02
    row = {"m": m, "k": k, "vocab": VOCAB, "us": {}}
    row["us"]["bf16"] = round(graph_us(lambda: x @ wt.t()), 2)
    for mode in ("fp8", "nvfp4"):
        h = head.prepare(wt, mode)
        row["us"][mode] = round(graph_us(lambda h=h: head.project(h, x)), 2)
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    report = {"gpu": torch.cuda.get_device_name(), "producers": [], "head": []}
    for k in WIDTHS:
        for m in TOKENS:
            row = producers(m, k)
            report["producers"].append(row)
            print(json.dumps(row), flush=True)
    for k in (2560, 4096, 5120):
        for m in (1, 64, 384, 1024):
            row = heads(m, k)
            report["head"].append(row)
            print(json.dumps(row), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
