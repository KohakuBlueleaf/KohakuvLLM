"""``grammar_mask(logits, table, ids)``: ``logits`` [B, V] in place, row ``r`` keeping the tokens
whose bit is set in mask ``table[ids[r]]`` ([M, words] int32, token ``t`` at bit ``t % 32`` of
word ``t // 32``); ``ids[r]`` -1 keeps every token. No host sync (CUDA-graph safe)."""

import torch

from kohakuvllm.kernels import registry


def grammar_mask(logits, table, ids) -> None:
    V = logits.shape[1]
    words = table[ids.long().clamp(min=0)]
    shifts = torch.arange(32, device=logits.device, dtype=torch.int32)
    bits = ((words[:, :, None] >> shifts) & 1).flatten(1)[:, :V]
    keep = (bits != 0) | (ids < 0)[:, None]
    logits.masked_fill_(~keep, float("-inf"))


registry.register("grammar_mask", "eager", grammar_mask)
