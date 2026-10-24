"""The GDN recurrent state in 8 bits (FP8 E4M3 or INT8): half the bytes of a bf16 state, read and
written by every decode step.

Layout: vLLM's state pool holds, per slot, ``[HV, V, K + 4]`` bytes: each state row (``K``
values) in 8 bits, then that row's float32 scale (``amax / 448`` for FP8, ``amax / 127`` for
INT8). ``install(fmt)`` makes vLLM size the pool that way (the GDN state shape and dtype) and
routes every use of the state through this module:

- decodes (vLLM's packed decode and its sigmoid-gating update for single-token sequences): one
  Triton kernel reads a row tile, dequantizes, runs the gated delta rule step in float32, and
  writes the row back requantized with a fresh scale;
- prefills: vLLM gathers the prefill sequences' states and hands them to its chunked kernel;
  the gathered bytes are dequantized before it, and its final states quantized after it, so the
  pool takes them back as bytes.

FP8 needs sm89 and up (Triton's E4M3); INT8 runs on any card. Spec decoding is not handled.
``KOHAKU_GDN_STATE=fp8`` or ``int8`` turns it on (``kohakuvllm.plugin``).
"""

import torch
import triton
import triton.language as tl
from vllm.model_executor.layers.mamba import mamba_utils
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

FORMATS = {"fp8": (torch.float8_e4m3fn, 448.0), "int8": (torch.int8, 127.0)}
SCALE_BYTES = 4
_ORIGINAL = {}
_FORMAT = {"name": "fp8"}
# The decode kernel's state rows and warps per program (install's ``tile``); 8x1 and the larger
# tiles are level on a 5090 (1.9x the bf16 pool)
TILE = {"rows": 8, "warps": 1}


def split(pool: torch.Tensor, k: int, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``pool`` [..., V, K + 4] uint8 -> its values [..., V, K] in ``fmt`` and scales [..., V]
    float32 (views)."""
    return pool[..., :k].view(FORMATS[fmt][0]), pool[..., k:].view(torch.float32)[..., 0]


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


@triton.jit
def decode_kernel(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    o,
    pool,
    slots,
    scale,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    stride_o,
    stride_slot,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    L2NORM: tl.constexpr,
    INT8: tl.constexpr,
):
    """One token of one sequence, one value-head, ``BV`` state rows."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    p_o = o + i_n * stride_o + i_hv * V + o_v

    slot = tl.load(slots + i_n).to(tl.int64)
    if slot <= 0:
        tl.store(p_o, tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty))
        return

    ROW: tl.constexpr = K + 4
    p_row = pool + slot * stride_slot + (i_hv * V + o_v) * ROW
    raw = tl.load(p_row[:, None] + o_k[None, :])
    p_scale = (p_row + K).to(tl.pointer_type(tl.float32))
    if INT8:
        h = raw.to(tl.int8, bitcast=True).to(tl.float32)
    else:
        h = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    h *= tl.load(p_scale)[:, None]

    b_q = tl.load(q + i_n * stride_q + i_h * K + o_k).to(tl.float32)
    b_k = tl.load(k + i_n * stride_k + i_h * K + o_k).to(tl.float32)
    b_v = tl.load(v + i_n * stride_v + i_hv * V + o_v).to(tl.float32)
    if L2NORM:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    x = tl.load(a + i_n * stride_a + i_hv).to(tl.float32) + tl.load(dt_bias + i_hv).to(tl.float32)
    softplus = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    g = -tl.exp(tl.load(A_log + i_hv).to(tl.float32)) * softplus
    beta = tl.sigmoid(tl.load(b + i_n * stride_b + i_hv).to(tl.float32))

    h *= tl.exp(g)
    b_v -= tl.sum(h * b_k[None, :], 1)
    b_v *= beta
    h += b_v[:, None] * b_k[None, :]
    tl.store(p_o, tl.sum(h * b_q[None, :], 1).to(p_o.dtype.element_ty))

    amax = tl.maximum(tl.max(tl.abs(h), 1), 1e-30)
    if INT8:
        s = amax / 127.0
        x8 = h / s[:, None]
        q8 = tl.where(x8 >= 0, x8 + 0.5, x8 - 0.5).to(tl.int8)
    else:
        s = amax / 448.0
        q8 = (h / s[:, None]).to(tl.float8e4nv)
    tl.store(p_row[:, None] + o_k[None, :], q8.to(tl.uint8, bitcast=True))
    tl.store(p_scale, s)


def decode(
    q, k, v, a, b, A_log, dt_bias, pool, slots, out, scale: float, l2norm: bool, fmt: str, tile=None
):
    """One decode step of ``N`` sequences on the 8-bit pool. ``q``, ``k`` [N, H * K], ``v``
    [N, HV * V] (row strides free), ``a``, ``b`` [N, HV], ``slots`` [N], ``out`` [N, HV * V];
    ``tile``: ``{"rows", "warps"}``, ``TILE`` by default."""
    tile = tile or TILE
    n = q.shape[0]
    HV, V, row = pool.shape[-3:]
    K = row - SCALE_BYTES
    H = q.shape[1] // K
    BV = min(triton.next_power_of_2(V), tile["rows"])
    if triton.next_power_of_2(K) != K or V % BV:
        raise ValueError(
            f"8-bit GDN state: K {K} must be a power of two and V {V} a multiple of {BV}"
        )
    grid = (triton.cdiv(V, BV), n * HV)
    decode_kernel[grid](
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        out,
        pool,
        slots,
        scale,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        a.stride(0),
        b.stride(0),
        out.stride(0),
        pool.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BV=BV,
        L2NORM=l2norm,
        INT8=fmt == "int8",
        num_warps=tile["warps"],
    )


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
    TILE.update(tile or {})
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
