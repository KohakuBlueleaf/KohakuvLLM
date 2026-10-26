"""``gdn_state8_decode``'s reference in plain torch: one gated-delta-rule decode step of ``N``
sequences on the 8-bit GDN state pool (``kohakuvllm.gdn.state8``'s layout). Rows whose slot is
0 or below get a zero output and leave the pool alone. No host sync (CUDA-graph safe)."""

import torch
import torch.nn.functional as F

from kohakuvllm.kernels import registry

SCALE_BYTES = 4
FORMATS = {"fp8": (torch.float8_e4m3fn, 448.0), "int8": (torch.int8, 127.0)}


def _split(pool: torch.Tensor, k: int, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    v = pool.shape[-2]
    flat = pool.flatten(-2)
    values = flat[..., : v * k].unflatten(-1, (v, k))
    return values.view(FORMATS[fmt][0]), flat[..., v * k :].view(torch.float32)


def decode(q, k, v, a, b, A_log, dt_bias, pool, slots, out, scale, l2norm, fmt, tile=None):
    n = q.shape[0]
    HV, V, row = pool.shape[-3:]
    K = row - SCALE_BYTES
    H = q.shape[1] // K
    live = slots > 0
    idx = slots.long().clamp(min=0)
    rows = pool[idx]  # [N, HV, V, K + 4] bytes
    values, scales = _split(rows, K, fmt)
    h = values.float() * scales[..., None]
    bq = q.float().view(n, H, K)
    bk = k.float().view(n, H, K)
    if l2norm:
        bq = bq / torch.sqrt((bq * bq).sum(-1, keepdim=True) + 1e-6)
        bk = bk / torch.sqrt((bk * bk).sum(-1, keepdim=True) + 1e-6)
    bq = (bq * scale).repeat_interleave(HV // H, 1)
    bk = bk.repeat_interleave(HV // H, 1)
    bv = v.float().view(n, HV, V)
    x = a.float() + dt_bias.float()
    g = -torch.exp(A_log.float()) * torch.where(x <= 20.0, F.softplus(x), x)
    beta = torch.sigmoid(b.float())
    h = h * torch.exp(g)[..., None, None]
    bv = (bv - (h * bk[:, :, None, :]).sum(-1)) * beta[..., None]
    h = h + bv[..., None] * bk[:, :, None, :]
    o = (h * bq[:, :, None, :]).sum(-1)
    out.copy_(torch.where(live[:, None], o.reshape(n, HV * V), 0).to(out.dtype))

    dtype, top = FORMATS[fmt]
    s = h.abs().amax(-1).clamp(min=1e-30) / top
    x8 = h / s[..., None]
    x8 = torch.trunc(x8 + torch.where(x8 >= 0, 0.5, -0.5)) if dtype == torch.int8 else x8
    new = torch.empty_like(rows)
    new_values, new_scales = _split(new, K, fmt)
    new_values.copy_(x8)
    new_scales.copy_(s)
    pool.index_put_((idx,), torch.where(live[:, None, None, None], new, rows))


registry.register("gdn_state8_decode", "eager", decode)
