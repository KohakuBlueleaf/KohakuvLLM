"""Causal depthwise conv1d over a mixed batch, from device metadata only (CUDA-graph safe).

``varlen_conv``: every token of every sequence (a decode is a length-1 sequence) in one launch
over the padded token count. Token ``t`` of a sequence at position ``p`` reads its ``W - 1``
predecessors from the batch, or, for ``p < W - 1``, from the sequence's conv state slot (zeros
without an initial state). Tokens past the batch's real token count are written as zeros.

``update_state``: each sequence's slot then holds its last ``W - 1`` inputs (older ones from the
previous state when the sequence is shorter than ``W - 1``). It runs after ``varlen_conv``, which
reads the states it replaces.

The conv state is the ``(..., dim, W - 1)`` view vLLM's conv kernels take (newest input last);
any strides.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _varlen_conv_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    state_ptr,
    out_ptr,
    tok_seq_ptr,
    cu_ptr,
    slot_ptr,
    init_ptr,
    n_tok_ptr,
    T,
    DIM,
    stride_x_t,
    stride_o_t,
    stride_s_slot,
    stride_s_dim,
    stride_s_k,
    stride_w_d,
    W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    n_tok = tl.load(n_tok_ptr)
    in_t = t < T
    live = t < n_tok
    dm = d < DIM
    seq = tl.load(tok_seq_ptr + t, mask=live, other=0)
    start = tl.load(cu_ptr + seq, mask=live, other=0)
    slot = tl.load(slot_ptr + seq, mask=live, other=0).to(tl.int64)
    init = tl.load(init_ptr + seq, mask=live, other=0) != 0
    pos = t - start
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for j in tl.static_range(W):
        back = W - 1 - j
        src = pos - back
        from_x = src >= 0
        xm = (live & from_x)[:, None] & dm[None, :]
        xv = tl.load(x_ptr + (t - back)[:, None] * stride_x_t + d[None, :], mask=xm, other=0.0)
        k = src + (W - 1)
        sm = (live & (~from_x) & init)[:, None] & dm[None, :]
        sv = tl.load(
            state_ptr
            + slot[:, None] * stride_s_slot
            + d[None, :] * stride_s_dim
            + k[:, None] * stride_s_k,
            mask=sm,
            other=0.0,
        )
        wj = tl.load(w_ptr + d * stride_w_d + j, mask=dm, other=0.0).to(tl.float32)
        acc += (xv.to(tl.float32) + sv.to(tl.float32)) * wj[None, :]
    if HAS_BIAS:
        acc += tl.load(bias_ptr + d, mask=dm, other=0.0).to(tl.float32)[None, :]
    if SILU:
        acc = acc * tl.sigmoid(acc)
    acc = tl.where(live[:, None], acc, 0.0)
    tl.store(
        out_ptr + t[:, None] * stride_o_t + d[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=in_t[:, None] & dm[None, :],
    )


@triton.jit
def _update_state_kernel(
    x_ptr,
    state_ptr,
    cu_ptr,
    slot_ptr,
    init_ptr,
    DIM,
    stride_x_t,
    stride_s_slot,
    stride_s_dim,
    stride_s_k,
    W: tl.constexpr,
    KP: tl.constexpr,
    BD: tl.constexpr,
):
    s = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    dm = d < DIM
    start = tl.load(cu_ptr + s)
    length = tl.load(cu_ptr + s + 1) - start
    if length > 0:
        slot = tl.load(slot_ptr + s).to(tl.int64)
        init = tl.load(init_ptr + s) != 0
        base = state_ptr + slot * stride_s_slot + d * stride_s_dim
        kk = tl.arange(0, KP)
        new = tl.zeros((KP, BD), dtype=tl.float32)
        for k in tl.static_range(W - 1):
            src = length - (W - 1) + k
            vx = tl.load(x_ptr + (start + src) * stride_x_t + d, mask=dm & (src >= 0), other=0.0)
            vs = tl.load(base + (k + length) * stride_s_k, mask=dm & init & (src < 0), other=0.0)
            v = vx.to(tl.float32) + vs.to(tl.float32)
            new = tl.where((kk == k)[:, None], v[None, :], new)
        tl.store(
            base[None, :] + kk[:, None] * stride_s_k,
            new.to(state_ptr.dtype.element_ty),
            mask=(kk < W - 1)[:, None] & dm[None, :],
        )


def varlen_conv(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    tok_seq: torch.Tensor,
    cu: torch.Tensor,
    slot: torch.Tensor,
    init: torch.Tensor,
    n_tok: torch.Tensor,
    silu: bool = True,
) -> torch.Tensor:
    """``x`` [T, dim] (T padded), ``state`` [slots, dim, W - 1], ``weight`` [dim, W];
    ``tok_seq`` [T] sequence of each token, ``cu`` [S + 1], ``slot`` / ``init`` [S],
    ``n_tok`` [1] real tokens. Returns the conv output [T, dim] in ``x``'s dtype."""
    t, dim = x.shape
    out = torch.empty_like(x)
    bt, bd = 16, 256
    grid = (triton.cdiv(t, bt), triton.cdiv(dim, bd))
    _varlen_conv_kernel[grid](
        x,
        weight,
        bias if bias is not None else weight,
        state,
        out,
        tok_seq,
        cu,
        slot,
        init,
        n_tok,
        t,
        dim,
        x.stride(0),
        out.stride(0),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        weight.stride(0),
        W=weight.shape[1],
        HAS_BIAS=bias is not None,
        SILU=silu,
        BT=bt,
        BD=bd,
    )
    return out


def update_state(
    x: torch.Tensor,
    state: torch.Tensor,
    cu: torch.Tensor,
    slot: torch.Tensor,
    init: torch.Tensor,
    width: int,
) -> None:
    """Each of the ``len(slot)`` sequences' slots takes its last ``width - 1`` inputs of ``x``."""
    dim = x.shape[1]
    bd = 256
    grid = (slot.shape[0], triton.cdiv(dim, bd))
    _update_state_kernel[grid](
        x,
        state,
        cu,
        slot,
        init,
        dim,
        x.stride(0),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        W=width,
        KP=triton.next_power_of_2(width - 1),
        BD=bd,
    )
