"""``grammar_mask`` in one launch: a program per (row, 2048 logits), rows with mask id -1 return
at once."""

import triton

from kohakuvllm.kernels import registry
from kohakuvllm.kernels.backends.triton.grammar import _kernels

BLOCK = 2048


def grammar_mask(logits, table, ids) -> None:
    B, V = logits.shape
    _kernels.mask_kernel[(B, triton.cdiv(V, BLOCK))](
        logits, logits.stride(0), table, table.shape[1], ids, V, BLOCK=BLOCK
    )


registry.register("grammar_mask", "triton", grammar_mask)
