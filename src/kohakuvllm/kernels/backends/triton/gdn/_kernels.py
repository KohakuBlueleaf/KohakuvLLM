import triton
import triton.language as tl


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
    # sequences x value-heads on axis 0 (up to 2^31 - 1 programs; axis 1 stops at 65535)
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    p_o = o + i_n * stride_o + i_hv * V + o_v

    slot = tl.load(slots + i_n).to(tl.int64)
    if slot <= 0:
        tl.store(p_o, tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty))
        return

    p_head = pool + slot * stride_slot + i_hv * (V * (K + 4))
    p_row = p_head + o_v * K
    raw = tl.load(p_row[:, None] + o_k[None, :], eviction_policy="evict_first")
    p_scale = (p_head + V * K).to(tl.pointer_type(tl.float32)) + o_v
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
    tl.store(
        p_row[:, None] + o_k[None, :], q8.to(tl.uint8, bitcast=True), eviction_policy="evict_first"
    )
    tl.store(p_scale, s)
