"""The NVFP4 producers as vLLM custom ops (``torch.ops.vllm.kohaku_*``), so torch.compile graphs
and the fusion pass (``nvfp4.fusion``) can call them. Each returns the packed codes ``[M, K / 2]``
uint8 and the swizzled scales as vLLM's ``scaled_fp4_quant`` allocates them (int32, rows rounded
up to 128, scale columns to 4). The kernels are the registry's ``nvfp4_quant`` and
``rmsnorm_nvfp4`` (``kohakuvllm.kernels``), resolved once per device and dtype.
"""

import torch
from vllm._custom_ops import create_fp4_scale_tensor
from vllm.utils.torch_utils import direct_register_custom_op

from kohakuvllm.kernels import registry

_BOUND: dict = {}


def bound(op: str, x: torch.Tensor) -> registry.Bound:
    key = (op, x.device, x.dtype)
    if key not in _BOUND:
        _BOUND[key] = registry.resolve(op, x.device, x.dtype)
    return _BOUND[key]


def _alloc(m: int, k: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The codes and the scales (int32, four E4M3 bytes each: kernels take a uint8 view)."""
    q = torch.empty(m, k // 2, dtype=torch.uint8, device=device)
    return q, create_fp4_scale_tensor(m, k, device, True)


def nvfp4_quant(x: torch.Tensor, gs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    q, s = _alloc(m, k, x.device)
    if m:
        bound("nvfp4_quant", x2)(x2, gs, q, s.view(torch.uint8))
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
        two = gs1 is not None
        bound("rmsnorm_nvfp4", x2)(
            x2,
            r2 if residual is not None else None,
            weight,
            eps,
            gs0,
            gs1,
            res,
            q0,
            s0.view(torch.uint8),
            q1 if two else None,
            s1.view(torch.uint8) if two else None,
        )
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
