"""A scheduler that keeps prefill out of most steps.

Steps in between run the decoding requests only: vLLM's own prefill throttle (built for DP prefill
balancing) with its saturation escape turned off, so they are uniform decode batches and replay
full CUDA graphs, which a hybrid GDN model cannot do for a mixed batch. ``KOHAKU_PREFILL_EVERY``
(4): one step in that many admits prefill; 1 is vLLM's own scheduling.

It pays under a stream of short requests (GBC's replay on an RTX 5090: 9.2 -> 14.9 requests/s)
and costs under a saturated offline queue (17.8 -> 14.8), where prefill batches best freely.

    vllm serve ... --scheduler-cls kohakuvllm.sched.PhasedScheduler
"""

import os

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

EVERY = int(os.environ.get("KOHAKU_PREFILL_EVERY", "4"))


class PhasedScheduler(AsyncScheduler):
    _phase = 0

    def schedule(self, throttle_prefills: bool = False):
        self._phase += 1
        decode_only = EVERY > 1 and self._phase % EVERY != 0
        if decode_only:
            self.prefill_capacity_bound = False
        return super().schedule(throttle_prefills or decode_only)
