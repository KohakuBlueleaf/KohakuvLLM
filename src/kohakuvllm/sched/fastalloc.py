"""A shortcut through vLLM's ``KVCacheManager.allocate_slots`` for steps that need no block work.

Every scheduled request calls ``allocate_slots`` every step, and each call walks every KV cache
group in Python (skipped-block removal, block counts, allocation, prefix caching): about a fifth
of the engine core's time on a hybrid model at a thousand running requests. A running request
whose new tokens stay inside blocks it already holds, in every group, with its processed position
in the same block and no block newly full, gets nothing from that walk: no block allocated,
cached or freed. For those calls the shortcut returns the empty result directly; every other
call runs vLLM's own code.

``install("on")`` takes the shortcut; ``install("check")`` runs vLLM's code where the shortcut
would apply and counts calls in which it allocated, cached or freed anything (logged, 0 when the
shortcut is exact).
"""

import logging

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import RequestStatus

log = logging.getLogger("kohakuvllm")


def no_block_work(manager: KVCacheManager, request, num_new_tokens: int) -> bool:
    """Whether ``allocate_slots(request, num_new_tokens)`` with no other argument would leave
    every group's blocks of ``request`` unchanged."""
    if request.status != RequestStatus.RUNNING or num_new_tokens <= 0:
        return False
    computed = request.num_computed_tokens
    need = min(computed + num_new_tokens, manager.max_model_len)
    processed = max(0, computed - request.num_in_flight_tokens)
    to_cache = min(computed + num_new_tokens, request.num_tokens)
    rid = request.request_id
    for group in manager.coordinator.single_type_managers:
        size = group.block_size
        blocks = group.req_to_blocks.get(rid)
        last = (need - 1) // size
        # the slots exist, the state sits in the same block, no block becomes full
        if not blocks or len(blocks) < cdiv(need, size) or blocks[last].is_null:
            return False
        if processed // size != last:
            return False
        if to_cache // size > group.num_cached_block.get(rid, 0):
            return False
    return True


def _snapshot(manager: KVCacheManager, rid: str) -> tuple:
    return tuple(
        (len(g.req_to_blocks.get(rid, ())), g.num_cached_block.get(rid))
        for g in manager.coordinator.single_type_managers
    )


def install(mode: str) -> None:
    original = KVCacheManager.allocate_slots
    counts = {"calls": 0, "shortcut": 0, "changed": 0}

    def allocate_slots(self, request, num_new_tokens, *args, **kwargs):
        counts["calls"] += 1
        plain = not args and not any(kwargs.get(k) for k in kwargs if k != "has_scheduled_reqs")
        if plain and no_block_work(self, request, num_new_tokens):
            counts["shortcut"] += 1
            if mode == "on":
                return self.empty_kv_cache_blocks
            before = _snapshot(self, request.request_id)
            out = original(self, request, num_new_tokens, *args, **kwargs)
            after = _snapshot(self, request.request_id)
            if out is not self.empty_kv_cache_blocks or after != before:
                counts["changed"] += 1
            if counts["shortcut"] % 200_000 == 0:
                log.warning("kohakuvllm fastalloc check: %s", counts)
            return out
        return original(self, request, num_new_tokens, *args, **kwargs)

    KVCacheManager.allocate_slots = allocate_slots
    log.info("kohakuvllm: allocate_slots shortcut (%s)", mode)
