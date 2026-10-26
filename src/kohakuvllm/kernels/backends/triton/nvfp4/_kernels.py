"""Triton NVFP4 producers for sm100 / sm120: a bf16 activation (or one a fused RMSNorm makes) to
packed E2M1 plus E4M3 block scales in the 128x4 swizzled layout the FP4 GEMMs read.

The arithmetic is vLLM's ``scaled_fp4_quant`` (``cvt_fp16_to_fp4``), step for step, so the bytes
match it: per 16 values ``sf = e4m3(gs * amax * rcp(6))`` and ``q = e2m1(v * rcp(sf * rcp(gs)))``,
``rcp`` the approximate reciprocal (``rcp.approx.ftz.f32``), both conversions round to nearest
even and saturate, a negative value that rounds to zero keeps its sign bit. ``PTX`` 1 converts
with the hardware ``cvt.rn.satfinite.e2m1x2.f32`` (sm100a / sm120a), 0 with comparisons against
the E2M1 rounding midpoints; the two give the same codes.

A program takes ``BM`` rows; the grid covers the rows rounded up to 128, and the rows past ``M``
write zero scales, so the padded scale tile needs no separate zeroing.
"""

import triton
import triton.language as tl


@triton.jit
def _rcp(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _e2m1(v):
    """4-bit E2M1 code of fp32 ``v``: round to nearest even, saturate at 6, sign bit kept."""
    a = tl.abs(v)
    code = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    return code | ((v < 0).to(tl.int32) << 3)


@triton.jit
def _e2m1x2_ptx(lo, hi):
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t; }",
        "=r,f,f",
        [lo, hi],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _quant_store(
    y, rows, k0, M, KB, KBP, Q, S, gs, BM: tl.constexpr, BK: tl.constexpr, PTX: tl.constexpr
):
    """Quantize ``y`` [BM, BK] fp32 (rows ``rows``, columns from ``k0``) with tensor scale ``gs``;
    store the packed codes (rows < M) and the swizzled scales (every row of the padded tile)."""
    nb: tl.constexpr = BK // 16
    yb = tl.reshape(y, (BM, nb, 16))
    amax = tl.max(tl.abs(yb), axis=2)
    gsb = tl.zeros((BM, nb), tl.float32) + gs
    sf = (gsb * (amax * _rcp(tl.full((BM, nb), 6.0, tl.float32)))).to(tl.float8e4nv)
    sff = sf.to(tl.float32)
    step = tl.where(sff != 0.0, _rcp(sff * _rcp(gsb)), 0.0)
    q = tl.reshape(yb * step[:, :, None], (BM, nb, 8, 2))
    lo, hi = tl.split(q)
    if PTX:  # noqa: SIM108 - a constexpr branch: Triton compiles only the taken side
        packed = _e2m1x2_ptx(lo, hi)
    else:
        packed = _e2m1(lo) | (_e2m1(hi) << 4)
    packed = tl.reshape(packed, (BM, BK // 2)).to(tl.uint8)
    pcols = k0 // 2 + tl.arange(0, BK // 2)
    kq = KB * 8
    tl.store(
        Q + rows[:, None] * kq + pcols[None, :],
        packed,
        mask=(rows < M)[:, None] & (pcols < kq)[None, :],
    )
    kb = k0 // 16 + tl.arange(0, nb)
    off = (
        ((rows // 128) * (KBP // 4))[:, None] * 512
        + ((kb // 4) * 512)[None, :]
        + ((rows % 32) * 16 + ((rows % 128) // 32) * 4)[:, None]
        + (kb % 4)[None, :]
    )
    tl.store(S + off, sf.to(tl.uint8, bitcast=True), mask=(kb < KB)[None, :])


@triton.jit
def _zero_scale_tail(rows, KB, KBP, S, BM: tl.constexpr):
    """Zero the scale columns between the last block and the next multiple of 4."""
    kb = KB + tl.arange(0, 4)
    off = (
        ((rows // 128) * (KBP // 4))[:, None] * 512
        + ((kb // 4) * 512)[None, :]
        + ((rows % 32) * 16 + ((rows % 128) // 32) * 4)[:, None]
        + (kb % 4)[None, :]
    )
    tl.store(S + off, tl.zeros((BM, 4), tl.uint8), mask=(kb < KBP)[None, :])


@triton.jit
def quant(X, Q, S, GS, M, K, sx, BM: tl.constexpr, BK: tl.constexpr, PTX: tl.constexpr):
    """Grid (row tiles, column chunks of ``BK``)."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    k0 = tl.program_id(1) * BK
    KB = K // 16
    KBP = (KB + 3) // 4 * 4
    gs = tl.load(GS)
    cols = k0 + tl.arange(0, BK)
    mask = (rows < M)[:, None] & (cols < K)[None, :]
    x = tl.load(X + rows[:, None] * sx + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    _quant_store(x, rows, k0, M, KB, KBP, Q, S, gs, BM, BK, PTX)
    if (KBP != KB) & (tl.program_id(1) == 0):
        _zero_scale_tail(rows, KB, KBP, S, BM)


@triton.jit
def _normed(h, rstd, w, DTYPE: tl.constexpr, WBF16: tl.constexpr):
    """vLLM's rounding of the normed row: a 32-bit weight (Gemma's ``1 + w``) multiplies the
    fp32 row, one rounding to ``DTYPE``; a 16-bit weight multiplies the row rounded first."""
    n = h * rstd[:, None]
    if WBF16:
        n = n.to(DTYPE).to(tl.float32)
    return (n * w[None, :]).to(DTYPE).to(tl.float32)


@triton.jit
def add_rmsnorm_quant(
    X,
    R,
    W,
    RES,
    Q0,
    S0,
    GS0,
    Q1,
    S1,
    GS1,
    M,
    K,
    sx,
    sr,
    eps,
    HAS_RES: tl.constexpr,
    TWO: tl.constexpr,
    ONE: tl.constexpr,
    WBF16: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    PTX: tl.constexpr,
):
    """``h = x + r`` (``RES`` takes it rounded to bf16), ``y = h * rsqrt(mean(h^2) + eps) * w``
    as vLLM's ``fused_add_rms_norm`` / ``rms_norm`` rounds it (``_normed``), then ``y`` quantized
    with tensor scale ``GS0`` (and, ``TWO``, again with ``GS1``). ``ONE``: the row in registers
    (``BK >= K``), read once; else two passes over ``BK`` chunks."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    in_m = rows < M
    KB = K // 16
    KBP = (KB + 3) // 4 * 4
    gs0 = tl.load(GS0)
    gs1 = gs0
    if TWO:
        gs1 = tl.load(GS1)
    if ONE:
        cols = tl.arange(0, BK)
        mask = in_m[:, None] & (cols < K)[None, :]
        h = tl.load(X + rows[:, None] * sx + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        if HAS_RES:
            h += tl.load(R + rows[:, None] * sr + cols[None, :], mask=mask, other=0.0).to(
                tl.float32
            )
            tl.store(RES + rows[:, None] * K + cols[None, :], h.to(RES.dtype.element_ty), mask=mask)
        rstd = tl.math.rsqrt(tl.sum(h * h, axis=1) / K + eps)
        w = tl.load(W + cols, mask=cols < K, other=0.0).to(tl.float32)
        y = _normed(h, rstd, w, X.dtype.element_ty, WBF16)
        _quant_store(y, rows, 0, M, KB, KBP, Q0, S0, gs0, BM, BK, PTX)
        if TWO:
            _quant_store(y, rows, 0, M, KB, KBP, Q1, S1, gs1, BM, BK, PTX)
    else:
        _two_pass(
            X,
            R,
            W,
            RES,
            Q0,
            S0,
            Q1,
            S1,
            gs0,
            gs1,
            rows,
            in_m,
            M,
            K,
            KB,
            KBP,
            sx,
            sr,
            eps,
            HAS_RES,
            TWO,
            WBF16,
            BM,
            BK,
            PTX,
        )
    if KBP != KB:
        _zero_scale_tail(rows, KB, KBP, S0, BM)
        if TWO:
            _zero_scale_tail(rows, KB, KBP, S1, BM)


@triton.jit
def _two_pass(
    X,
    R,
    W,
    RES,
    Q0,
    S0,
    Q1,
    S1,
    gs0,
    gs1,
    rows,
    in_m,
    M,
    K,
    KB,
    KBP,
    sx,
    sr,
    eps,
    HAS_RES: tl.constexpr,
    TWO: tl.constexpr,
    WBF16: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    PTX: tl.constexpr,
):
    acc = tl.zeros((BM,), tl.float32)
    for k0 in range(0, K, BK):
        cols = k0 + tl.arange(0, BK)
        mask = in_m[:, None] & (cols < K)[None, :]
        h = tl.load(X + rows[:, None] * sx + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        if HAS_RES:
            h += tl.load(R + rows[:, None] * sr + cols[None, :], mask=mask, other=0.0).to(
                tl.float32
            )
            tl.store(RES + rows[:, None] * K + cols[None, :], h.to(RES.dtype.element_ty), mask=mask)
        acc += tl.sum(h * h, axis=1)
    rstd = tl.math.rsqrt(acc / K + eps)
    for k0 in range(0, K, BK):
        cols = k0 + tl.arange(0, BK)
        mask = in_m[:, None] & (cols < K)[None, :]
        h = tl.load(X + rows[:, None] * sx + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        if HAS_RES:
            h += tl.load(R + rows[:, None] * sr + cols[None, :], mask=mask, other=0.0).to(
                tl.float32
            )
        w = tl.load(W + cols, mask=cols < K, other=0.0).to(tl.float32)
        y = _normed(h, rstd, w, X.dtype.element_ty, WBF16)
        _quant_store(y, rows, k0, M, KB, KBP, Q0, S0, gs0, BM, BK, PTX)
        if TWO:
            _quant_store(y, rows, k0, M, KB, KBP, Q1, S1, gs1, BM, BK, PTX)
