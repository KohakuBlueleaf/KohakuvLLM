"""The NVFP4 producers as vLLM custom ops (``torch.ops.vllm.kohaku_*``), so torch.compile graphs
and the fusion pass (``nvfp4.fusion``) can call them. Each returns the packed codes ``[M, K / 2]``
uint8 and the swizzled scales as vLLM's ``scaled_fp4_quant`` allocates them (int32, rows rounded
up to 128, scale columns to 4).

``CVT`` picks the E2M1 conversion (``ptx`` the hardware instruction, ``arith`` comparisons); the
tiles are ``TILE`` when set, else ``quant_tile`` / ``norm_tile``.
"""

import torch
import triton
from vllm._custom_ops import create_fp4_scale_tensor
from vllm.utils.torch_utils import direct_register_custom_op

from kohakuvllm.nvfp4 import kernels

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


def _alloc(m: int, k: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The codes and the scales (int32, four E4M3 bytes each: kernels take a uint8 view)."""
    q = torch.empty(m, k // 2, dtype=torch.uint8, device=device)
    return q, create_fp4_scale_tensor(m, k, device, True)


def _grid(m: int, bm: int) -> tuple[int]:
    return (triton.cdiv(-(-m // 128) * 128, bm),)


def nvfp4_quant(x: torch.Tensor, gs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    q, s = _alloc(m, k, x.device)
    if m:
        bm, bk, warps = quant_tile(m, k)
        grid = (_grid(m, bm)[0], triton.cdiv(k, bk))
        args = (x2, q, s.view(torch.uint8), gs, m, k, x2.stride(0))
        meta = {"BM": bm, "BK": bk, "PTX": CVT["name"] == "ptx"}
        kernels.quant[grid](*args, **meta, num_warps=warps)
    return q, s


def _nvfp4_quant_fake(x: torch.Tensor, gs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    k = x.shape[-1]
    return _alloc(x.numel() // k, k, x.device)


def rmsnorm_nvfp4(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    gs0: torch.Tensor,
    gs1: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """vLLM's ``fused_add_rms_norm(x, residual, weight, eps)`` (``rms_norm`` when ``residual`` is
    None), its output quantized with ``gs0`` and, when given, ``gs1``: ``(q0, s0, q1, s1,
    residual out)``, the unused ones empty."""
    k = x.shape[-1]
    x2 = x.reshape(-1, k).contiguous()
    r2 = residual.reshape(-1, k).contiguous() if residual is not None else x2
    m = x2.shape[0]
    res = torch.empty_like(x2) if residual is not None else x2.new_empty(0)
    q0, s0 = _alloc(m, k, x.device)
    q1, s1 = _alloc(m, k, x.device) if gs1 is not None else (q0.new_empty(0), s0.new_empty(0))
    if m:
        bm, bk, warps = norm_tile(m, k)
        args = (
            x2,
            r2,
            weight,
            res,
            q0,
            s0.view(torch.uint8),
            gs0,
            q1 if gs1 is not None else q0,
            (s1 if gs1 is not None else s0).view(torch.uint8),
            gs1 if gs1 is not None else gs0,
            m,
            k,
            x2.stride(0),
            r2.stride(0),
            eps,
        )
        meta = {
            "HAS_RES": residual is not None,
            "TWO": gs1 is not None,
            "ONE": bk >= k,
            "WBF16": weight.dtype != torch.float32,
            "BM": bm,
            "BK": bk,
            "PTX": CVT["name"] == "ptx",
        }
        kernels.add_rmsnorm_quant[_grid(m, bm)](*args, **meta, num_warps=warps)
    return q0, s0, q1, s1, res.view_as(x) if residual is not None else res


def _rmsnorm_nvfp4_fake(x, residual, weight, eps, gs0, gs1):
    k = x.shape[-1]
    m = x.numel() // k
    q0, s0 = _alloc(m, k, x.device)
    q1, s1 = _alloc(m, k, x.device) if gs1 is not None else (q0.new_empty(0), s0.new_empty(0))
    res = torch.empty_like(x) if residual is not None else x.new_empty(0)
    return q0, s0, q1, s1, res


def add_rmsnorm_nvfp4(x, residual, weight, eps: float, gs) -> tuple:
    q, s, _, _, res = rmsnorm_nvfp4(x, residual, weight, eps, gs, None)
    return q, s, res


def add_rmsnorm_nvfp4x2(x, residual, weight, eps: float, gs0, gs1) -> tuple:
    return rmsnorm_nvfp4(x, residual, weight, eps, gs0, gs1)


direct_register_custom_op("kohaku_nvfp4_quant", nvfp4_quant, [], _nvfp4_quant_fake)
direct_register_custom_op("kohaku_rmsnorm_nvfp4", rmsnorm_nvfp4, [], _rmsnorm_nvfp4_fake)
