"""vLLM's general-plugin entry point: runs at import in every vLLM process.

It warns once when the installed vLLM is newer than the last tested one (TESTED.md). The parts
are opted into by configuration: ``--scheduler-cls kohakuvllm.sched.PhasedScheduler``, and
``KOHAKU_GDN=hybrid`` in the environment for the graph-captured mixed GDN step (``gdn.hybrid``).
"""

import logging
import os

import vllm
from packaging.version import Version

from kohakuvllm import TESTED_VLLM
from kohakuvllm.gdn import hybrid

log = logging.getLogger("kohakuvllm")


def register() -> None:
    if Version(vllm.__version__).release > Version(TESTED_VLLM).release:
        log.warning(
            "kohakuvllm was last tested on vLLM %s; running on %s", TESTED_VLLM, vllm.__version__
        )
    mode = os.environ.get("KOHAKU_GDN", "")
    if mode in ("hybrid", "check"):
        hybrid.install(os.environ.get("KOHAKU_GDN_CHECK", "gdn_check") if mode == "check" else None)
        log.info("kohakuvllm: hybrid GDN step installed (%s)", mode)
