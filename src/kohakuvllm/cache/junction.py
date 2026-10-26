"""GDN states cached at the exact shared-prefix junction, without speculative decoding.

vLLM's Mamba "align" prefix caching stores a GDN state only at block boundaries (272 tokens for
Qwen3.5) and at the prompt tail. When requests share a long system prompt and then differ (an
image each), the state worth reusing sits at the end of that prompt, mid-block: vLLM's own
junction checkpoint (``enable_mamba_shared_prefix_checkpoint``) stores it there, but is gated to
EAGLE drafters. Without a drafter nothing is re-prefilled past the junction, so the gate only
keeps the feature off; ``install()`` lifts it. With ``--prefix-match-unit`` (e.g. 16), hits then
land on the junction itself instead of the block below it.

``KOHAKU_MAMBA_JUNCTION=1`` turns it on (``kohakuvllm.plugin``); needs prefix caching and
``--prefix-match-unit`` below the block size.
"""

import logging
import os

from vllm.v1.core import kv_cache_manager
from vllm.v1.core.single_type_kv_cache_manager import MambaManager

log = logging.getLogger("vllm.kohakuvllm")  # under vLLM's logger, so its handler prints it
_ORIGINAL = {}


def install() -> None:
    if "init" in _ORIGINAL:
        return
    cls = kv_cache_manager.KVCacheManager
    _ORIGINAL["init"] = init = cls.__init__

    def patched(self, *args, **kwargs):
        init(self, *args, **kwargs)
        coord = self.coordinator
        on = bool(
            coord.enable_partial_hash_hits
            and coord.num_reprefillable_tokens == 0
            and not coord.eagle_group_ids
        )
        if on:
            self.mamba_shared_prefix_checkpoint = True
            for manager in coord.single_type_managers:
                if isinstance(manager, MambaManager):
                    manager.shared_prefix_checkpoint = True
        log.info(
            "kohakuvllm: GDN junction checkpoint %s",
            "on" if on else "off (needs --prefix-match-unit)",
        )

    cls.__init__ = patched
    if os.environ.get("KOHAKU_MAMBA_JUNCTION_DEBUG") == "1":
        _ORIGINAL["lookup"] = lookup = cls.get_computed_blocks
        seen = [0]

        def logged(self, request):
            blocks, hit, junction = lookup(self, request)
            if seen[0] < 64:
                seen[0] += 1
                log.info(
                    "kohakuvllm junction: prompt %d hit %d junction %d (flag %s)",
                    request.num_prompt_tokens,
                    hit,
                    junction,
                    self.mamba_shared_prefix_checkpoint,
                )
            return blocks, hit, junction

        cls.get_computed_blocks = logged
