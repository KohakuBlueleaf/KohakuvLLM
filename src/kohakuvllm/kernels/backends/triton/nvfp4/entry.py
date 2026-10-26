"""The Triton NVFP4 producers (``_kernels``), byte for byte vLLM's ``scaled_fp4_quant``.

``CVT`` picks the E2M1 conversion (``ptx`` the hardware instruction, ``arith`` comparisons); the
tiles are ``TILE`` when set, else ``quant_tile`` / ``norm_tile``.
"""

import torch
import triton

from kohakuvllm.kernels import registry
from kohakuvllm.kernels.backends.triton.nvfp4 import _kernels

CVT = {"name": "ptx"}
# (BM rows per program, BK columns per chunk, warps); None: the tile functions pick
TILE: dict = {"value": None}
# The widest row the norm kernel holds in registers (one pass); wider rows take two passes
ONE_PASS_MAX = 8192


def quant_tile(m: int, k: int) -> tuple[int, int, int]:
    """The fastest tiles on the RTX 5090 (``bench/gemm/nvfp4_quant.py``, widths 2560-9728)."""
    if TILE["value"] is not None:
        return TILE["value"]
    return (2, 512, 4) if m <= 64 else (4, 512, 4)


def norm_tile(m: int, k: int) -> tuple[int, int, int]:
    if TILE["value"] is not None:
        return TILE["value"]
    bk = triton.next_power_of_2(k)
    if bk > ONE_PASS_MAX:
        return (2, 1024, 8)
    warps = 16 if bk >= 8192 or (bk >= 4096 and m >= 1024) else 8 if bk >= 4096 else 4
    return (1, bk, warps)


def _grid(m: int, bm: int) -> int:
    return triton.cdiv(-(-m // 128) * 128, bm)


def nvfp4_quant(x, gs, q, s) -> None:
    """``x`` [M, K] (unit column stride) -> ``q`` [M, K / 2] uint8, ``s`` the swizzled scales
    (uint8 view, rows rounded up to 128)."""
    m, k = x.shape
    bm, bk, warps = quant_tile(m, k)
    args = (x, q, s, gs, m, k, x.stride(0))
    meta = {"BM": bm, "BK": bk, "PTX": CVT["name"] == "ptx"}
    _kernels.quant[(_grid(m, bm), triton.cdiv(k, bk))](*args, **meta, num_warps=warps)


def rmsnorm_nvfp4(x, residual, weight, eps, gs0, gs1, res, q0, s0, q1, s1) -> None:
    """vLLM's ``fused_add_rms_norm`` (``rms_norm`` when ``residual`` is None; its sum into
    ``res``), the output quantized with ``gs0`` into ``q0`` / ``s0`` and, when ``gs1`` is given,
    with ``gs1`` into ``q1`` / ``s1`` (scales as uint8 views)."""
    m, k = x.shape
    bm, bk, warps = norm_tile(m, k)
    two = gs1 is not None
    args = (
        x,
        residual if residual is not None else x,
        weight,
        res,
        q0,
        s0,
        gs0,
        q1 if two else q0,
        s1 if two else s0,
        gs1 if two else gs0,
        m,
        k,
        x.stride(0),
        (residual if residual is not None else x).stride(0),
        eps,
    )
    meta = {
        "HAS_RES": residual is not None,
        "TWO": two,
        "ONE": bk >= k,
        "WBF16": weight.dtype != torch.float32,
        "BM": bm,
        "BK": bk,
        "PTX": CVT["name"] == "ptx",
    }
    _kernels.add_rmsnorm_quant[(_grid(m, bm),)](*args, **meta, num_warps=warps)


def supports(query: registry.Query) -> str | None:
    if query.arch not in ("sm100", "sm103", "sm120", "sm121") and CVT["name"] == "ptx":
        return "the PTX E2M1 conversion needs sm100a / sm120a (CVT arith runs elsewhere)"
    return None


registry.register("nvfp4_quant", "triton", nvfp4_quant, supports)
registry.register("rmsnorm_nvfp4", "triton", rmsnorm_nvfp4, supports)
