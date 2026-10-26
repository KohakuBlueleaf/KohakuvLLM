import triton
import triton.language as tl


@triton.jit
def mask_kernel(logits, stride, table, words, ids, V, BLOCK: tl.constexpr):
    row, blk = tl.program_id(0), tl.program_id(1)
    m = tl.load(ids + row)
    if m < 0:
        return
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    inside = offs < V
    w = tl.load(table + m.to(tl.int64) * words + offs // 32, mask=inside, other=0)
    keep = ((w >> (offs % 32).to(tl.int32)) & 1) != 0
    p = logits + row.to(tl.int64) * stride + offs
    x = tl.load(p, mask=inside)
    tl.store(p, tl.where(keep, x, float("-inf")), mask=inside)
