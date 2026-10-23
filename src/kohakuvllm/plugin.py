"""vLLM's general-plugin entry point: runs at import in every vLLM process.

It warns once when the installed vLLM is newer than the last tested one (TESTED.md). The parts
are opted into by configuration: ``--scheduler-cls kohakuvllm.sched.PhasedScheduler``;
``KOHAKU_GDN=hybrid`` in the environment for the graph-captured mixed GDN step (``gdn.hybrid``);
``KOHAKU_GDN_STATE=fp8`` or ``int8`` for the GDN recurrent state in 8 bits (``gdn.state8``; not
with ``hybrid``), ``KOHAKU_GDN_STATE_TILE=ROWSxWARPS`` its decode kernel's tile (``state8.TILE``
by default).
"""

import logging
import os

import vllm
from packaging.version import Version

from kohakuvllm import TESTED_VLLM
from kohakuvllm.gdn import hybrid, state8

log = logging.getLogger("kohakuvllm")


def register() -> None:
    if Version(vllm.__version__).release > Version(TESTED_VLLM).release:
        log.warning(
            "kohakuvllm was last tested on vLLM %s; running on %s", TESTED_VLLM, vllm.__version__
        )
    mode = os.environ.get("KOHAKU_GDN", "")
    state = os.environ.get("KOHAKU_GDN_STATE", "")
    if state:
        if mode:
            raise ValueError("KOHAKU_GDN_STATE and KOHAKU_GDN=hybrid / check cannot combine")
        tile = os.environ.get("KOHAKU_GDN_STATE_TILE", "")
        rows, _, warps = tile.partition("x")
        state8.install(state, {"rows": int(rows), "warps": int(warps)} if tile else None)
        log.info("kohakuvllm: GDN recurrent state in %s", state)
    if mode in ("hybrid", "check"):
        hybrid.install(os.environ.get("KOHAKU_GDN_CHECK", "gdn_check") if mode == "check" else None)
        log.info("kohakuvllm: hybrid GDN step installed (%s)", mode)
