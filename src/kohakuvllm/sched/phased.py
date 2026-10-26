"""A scheduler that keeps prefill out of most steps.

Steps in between run the decoding requests only: vLLM's own prefill throttle (built for DP prefill
balancing) with its saturation escape turned off, so they are uniform decode batches and replay
full CUDA graphs, which a hybrid GDN model cannot do for a mixed batch. ``EVERY`` (feature
``sched=N``, 4 by default): one step in that many admits prefill; 1 is vLLM's own scheduling.

With a grammar that defers nothing in the scheduler (``kohakuvllm.grammar``) it pays under
image + JSON-schema requests (RTX 5090: 30.1 -> 35.0 requests/s; B300, Qwen3.5-9B NVFP4 at 16:
85.8 -> 92.7); with xgrammar, and under a saturated offline queue of uniform text prompts
(17.8 -> 14.8), it costs.

``install(every)`` makes it vLLM's scheduler whenever no ``--scheduler-cls`` is given and async
scheduling is on (vLLM's default); ``--scheduler-cls kohakuvllm.sched.PhasedScheduler`` also
works.
"""

import logging

from vllm.config.scheduler import SchedulerConfig
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

EVERY = {"n": 4}
log = logging.getLogger("vllm.kohakuvllm")


class PhasedScheduler(AsyncScheduler):
    _phase = 0

    def schedule(self, throttle_prefills: bool = False):
        self._phase += 1
        decode_only = EVERY["n"] > 1 and self._phase % EVERY["n"] != 0
        if decode_only:
            self.prefill_capacity_bound = False
        return super().schedule(throttle_prefills or decode_only)


def install(every: int) -> None:
    EVERY["n"] = every
    original = SchedulerConfig.get_scheduler_cls
    if getattr(original, "_kohaku", False):
        return

    def get_scheduler_cls(self):
        if self.scheduler_cls is None and self.async_scheduling:
            log.info("kohakuvllm: PhasedScheduler, prefill every %d steps", EVERY["n"])
            return PhasedScheduler
        return original(self)

    get_scheduler_cls._kohaku = True
    SchedulerConfig.get_scheduler_cls = get_scheduler_cls
