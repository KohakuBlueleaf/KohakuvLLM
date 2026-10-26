"""The GDN recurrent state in 8 bits (FP8 E4M3 or INT8): half the bytes of a bf16 state, read and
written by every decode step.

Layout: vLLM's state pool holds, per slot, ``[HV, V, K + 4]`` bytes; per value-head, its ``V * K``
values in 8 bits (row-major ``[V, K]``, so every row starts 16-byte aligned), then the ``V`` rows'
float32 scales (``amax / 448`` for FP8, ``amax / 127`` for INT8). ``install(fmt)`` makes vLLM
size the pool that way (the GDN state shape and dtype) and routes every use of the state through
this module:

- decodes (vLLM's packed decode and its sigmoid-gating update for single-token sequences): the
  registry's ``gdn_state8_decode`` (``kohakuvllm.kernels``) reads the rows, dequantizes, runs the
  gated delta rule step in float32, and writes the rows back requantized with fresh scales;
- prefills: vLLM gathers the prefill sequences' states and hands them to its chunked kernel;
  the gathered bytes are dequantized before it, and its final states quantized after it, so the
  pool takes them back as bytes.

FP8 needs sm89 and up (Triton's E4M3); INT8 runs on any card. Spec decoding is not handled.
``KOHAKU_GDN_STATE=fp8`` or ``int8`` turns it on (``kohakuvllm.plugin``).
"""

import torch
from vllm.model_executor.layers.mamba import mamba_utils
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

from kohakuvllm.kernels import registry
from kohakuvllm.kernels.backends.triton.gdn import entry as triton_gdn

FORMATS = {"fp8": (torch.float8_e4m3fn, 448.0), "int8": (torch.int8, 127.0)}
SCALE_BYTES = 4
_ORIGINAL = {}
_FORMAT = {"name": "fp8"}
_BOUND: dict = {}


def split(pool: torch.Tensor, k: int, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``pool`` [..., V, K + 4] uint8 -> its values [..., V, K] in ``fmt`` and scales [..., V]
    float32 (views; per head the values come first, then the scales)."""
    v = pool.shape[-2]
    flat = pool.flatten(-2)
    values = flat[..., : v * k].unflatten(-1, (v, k))
    return values.view(FORMATS[fmt][0]), flat[..., v * k :].view(torch.float32)


def quantize(state: torch.Tensor, out: torch.Tensor, fmt: str) -> torch.Tensor:
    """``state`` [..., V, K] float -> ``out`` [..., V, K + 4] uint8, per row."""
    dtype, top = FORMATS[fmt]
    k = state.shape[-1]
    values, scales = split(out, k, fmt)
    s = state.float().abs().amax(-1).clamp(min=1e-30) / top
    scales.copy_(s)
    x = state.float() / s[..., None]
    values.copy_(x.round() if dtype == torch.int8 else x)
    return out


def dequantize(pool: torch.Tensor, k: int, fmt: str, dtype=torch.float32) -> torch.Tensor:
    values, scales = split(pool, k, fmt)
    return (values.float() * scales[..., None]).to(dtype)


def decode(
    q, k, v, a, b, A_log, dt_bias, pool, slots, out, scale: float, l2norm: bool, fmt: str, tile=None
):
    """One decode step of ``N`` sequences on the 8-bit pool. ``q``, ``k`` [N, H * K], ``v``
    [N, HV * V] (row strides free), ``a``, ``b`` [N, HV], ``slots`` [N], ``out`` [N, HV * V];
    the registry's ``gdn_state8_decode`` (``tile``: the Triton kernel's ``{"rows", "warps"}``)."""
    key = (q.device, pool.dtype)
    if key not in _BOUND:
        _BOUND[key] = registry.resolve("gdn_state8_decode", q.device, q.dtype)
    _BOUND[key](q, k, v, a, b, A_log, dt_bias, pool, slots, out, scale, l2norm, fmt, tile=tile)


# ---- vLLM's entry points ------------------------------------------------------------------------


def packed_decode(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    out,
    ssm_state_indices,
    use_qk_l2norm_in_kernel=False,
    fmt=None,
    tile=None,
):
    """vLLM's ``fused_recurrent_gated_delta_rule_packed_decode`` on the 8-bit pool (``fmt``,
    ``tile``: the installed ones by default)."""
    if initial_state.dtype != torch.uint8:
        return _ORIGINAL["packed"](
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            out=out,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    HV, V = initial_state.shape[-3:-1]
    qk = (mixed_qkv.shape[1] - HV * V) // 2
    decode(
        mixed_qkv[:, :qk],
        mixed_qkv[:, qk : 2 * qk],
        mixed_qkv[:, 2 * qk :],
        a,
        b,
        A_log,
        dt_bias,
        initial_state,
        ssm_state_indices,
        out.view(out.shape[0], HV * V),
        scale,
        use_qk_l2norm_in_kernel,
        fmt or _FORMAT["name"],
        tile,
    )
    return out, initial_state


def sigmoid_update(A_log, a, b, dt_bias, q, k, v, **kw):
    """vLLM's ``fused_sigmoid_gating_delta_rule_update`` on the 8-bit pool, for sequences of one
    token each (non-spec decodes)."""
    state = kw.get("initial_state")
    if state is None or state.dtype != torch.uint8:
        return _ORIGINAL["sigmoid"](A_log, a, b, dt_bias, q, k, v, **kw)
    if kw.get("num_accepted_tokens") is not None:
        raise NotImplementedError("kohakuvllm 8-bit GDN state: spec decoding")
    T, H, K = q.shape[1], q.shape[2], q.shape[3]
    HV, V = v.shape[2], v.shape[3]
    out = q.new_empty(1, T, HV, V)
    scale = kw.get("scale") or K**-0.5
    decode(
        q.reshape(T, H * K),
        k.reshape(T, H * K),
        v.reshape(T, HV * V),
        a.reshape(T, HV),
        b.reshape(T, HV),
        A_log,
        dt_bias,
        state,
        kw["ssm_state_indices"],
        out.view(T, HV * V),
        scale,
        kw.get("use_qk_l2norm_in_kernel", False),
        _FORMAT["name"],
    )
    return out, state


class Chunk8(torch.nn.Module):
    """vLLM's chunked prefill with the gathered 8-bit states dequantized before it and its final
    states quantized after it (returned as the pool's bytes)."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, *args, initial_state=None, **kw):
        if initial_state is None or initial_state.dtype != torch.uint8:
            return self.inner(*args, initial_state=initial_state, **kw)
        k = initial_state.shape[-1] - SCALE_BYTES
        if k != kw["k"].shape[-1]:
            # vLLM's prefill warmup: a zero state of the pool's dtype, unpacked ([..., V, K])
            zeros = torch.zeros(initial_state.shape, dtype=torch.float32, device=kw["k"].device)
            return self.inner(*args, initial_state=zeros, **kw)
        fmt = _FORMAT["name"]
        out, final = self.inner(*args, initial_state=dequantize(initial_state, k, fmt), **kw)
        return out, quantize(final, torch.empty_like(initial_state), fmt)


def state_shape(*args, **kw):
    conv, temporal = _ORIGINAL["shape"](*args, **kw)
    return conv, (*temporal[:-1], temporal[-1] + SCALE_BYTES)


def state_dtype(cls, model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype="auto"):
    conv, _ = _ORIGINAL["dtype"](model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype)
    return conv, torch.uint8


def install(fmt: str, tile: dict | None = None) -> None:
    """Route vLLM's Qwen3.5 / Qwen3-Next GDN state through this module in ``fmt`` (``fp8`` or
    ``int8``; see the module docstring), the decode kernel at ``tile`` (``TILE`` keys)."""
    if fmt not in FORMATS:
        raise ValueError(f"8-bit GDN state: format {fmt!r} not in {sorted(FORMATS)}")
    _FORMAT["name"] = fmt
    triton_gdn.TILE.update(tile or {})
    calc_shape = mamba_utils.MambaStateShapeCalculator
    calc_dtype = mamba_utils.MambaStateDtypeCalculator
    _ORIGINAL.update(
        packed=gdn.fused_recurrent_gated_delta_rule_packed_decode,
        sigmoid=gdn.fused_sigmoid_gating_delta_rule_update,
        shape=calc_shape.gated_delta_net_state_shape,
        dtype=calc_dtype.gated_delta_net_state_dtype,
    )
    gdn.fused_recurrent_gated_delta_rule_packed_decode = packed_decode
    gdn.fused_sigmoid_gating_delta_rule_update = sigmoid_update
    calc_shape.gated_delta_net_state_shape = staticmethod(state_shape)
    calc_dtype.gated_delta_net_state_dtype = classmethod(state_dtype)
    init = gdn.QwenGatedDeltaNetAttention.__init__

    def wrapped_init(self, *args, **kw):
        init(self, *args, **kw)
        self.chunk_gated_delta_rule = Chunk8(self.chunk_gated_delta_rule)

    gdn.QwenGatedDeltaNetAttention.__init__ = wrapped_init
