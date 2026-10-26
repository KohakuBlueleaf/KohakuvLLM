"""vLLM's general-plugin entry point: runs at import in every vLLM process and installs the
features ``KOHAKU_VLLM`` names (``kohakuvllm.features``). With it unset, vLLM runs as shipped.

It warns once when the installed vLLM is newer than the last tested one (TESTED.md).
"""

import logging

import vllm
from packaging.version import Version

from kohakuvllm import TESTED_VLLM, features, nvfp4
from kohakuvllm.cache import junction
from kohakuvllm.gdn import state8
from kohakuvllm.grammar import logits as grammar
from kohakuvllm.sched import fastalloc, phased

log = logging.getLogger("vllm.kohakuvllm")


def register() -> None:
    if Version(vllm.__version__).release > Version(TESTED_VLLM).release:
        log.warning(
            "kohakuvllm was last tested on vLLM %s; running on %s", TESTED_VLLM, vllm.__version__
        )
    chosen = features.from_env()
    if not chosen:
        return
    if "sched" in chosen:
        phased.install(int(chosen["sched"]))
    if "alloc" in chosen:
        fastalloc.install(chosen["alloc"])
    if "state" in chosen:
        state8.install(chosen["state"])
    if "nvfp4" in chosen or "lmhead" in chosen:
        spec = ",".join(chosen.get("nvfp4", "").split("+"))
        if "lmhead" in chosen:
            spec += f",head={chosen['lmhead']}"
        nvfp4.install(spec)
    if "junction" in chosen:
        junction.install()
    if "grammar_ws" in chosen:
        grammar.WS["bytes"] = int(chosen["grammar_ws"])
    log.info("kohakuvllm: %s", ",".join(f"{k}={v}" for k, v in chosen.items()))
