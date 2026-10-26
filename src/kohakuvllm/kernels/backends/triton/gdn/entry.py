"""``gdn_state8_decode``: one program per (sequence, value-head, ``BV`` state rows) reads the 8-bit
rows, dequantizes, runs the gated delta rule step in float32 and writes the rows back requantized
(``kohakuvllm.gdn.state8`` holds the layout)."""

import triton

from kohakuvllm.kernels import registry
from kohakuvllm.kernels.backends.triton.gdn import _kernels
from kohakuvllm.kernels.errors import BackendUnavailable

SCALE_BYTES = 4
# State rows and warps per program; 8x1 and the larger tiles are level on a 5090 (1.9x the bf16
# pool)
TILE = {"rows": 8, "warps": 1}


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
        raise BackendUnavailable(
            f"8-bit GDN state: K {K} must be a power of two and V {V} a multiple of {BV}"
        )
    grid = (n * HV, triton.cdiv(V, BV))
    _kernels.decode_kernel[grid](
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


registry.register("gdn_state8_decode", "triton", decode)
