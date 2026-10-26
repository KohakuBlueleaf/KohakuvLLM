"""The plugin's one switch: ``KOHAKU_VLLM``, a comma list of features (``name`` or ``name=value``).

- ``sched[=N]`` (4): ``PhasedScheduler`` in place of vLLM's scheduler; one step in ``N`` admits
  prefill, the others are decode-only and replay full CUDA graphs.
- ``alloc[=check]``: ``allocate_slots`` skips its block walk for requests that need no block
  work (``check``: runs vLLM's code too and counts any difference).
- ``state=int8|fp8`` (int8): the GDN recurrent state in 8 bits.
- ``nvfp4[=norm+quant]`` (both): our NVFP4 producers in place of vLLM's in the compiled graph
  (acts on NVFP4 checkpoints only).
- ``lmhead=fp8|nvfp4``: the LM head in FP8 or NVFP4 (not in the default set).
- ``junction``: GDN states cached at the shared-prefix junction (needs ``--prefix-match-unit``;
  not in the default set).
- ``grammar_ws=N`` (8): whitespace bytes the JSON grammar allows per gap.

``KOHAKU_VLLM=1`` (or ``on``) is the default set: ``sched,alloc,state,nvfp4``. Items apply in
order: ``KOHAKU_VLLM=1,sched=16`` is the default set with ``sched`` at 16, and a name prefixed
``-`` drops it (``1,-alloc``).
Unset or empty: vLLM runs as shipped (the JSON grammar still serves requests that ask for it).
"""

import logging
import os

log = logging.getLogger("vllm.kohakuvllm")

DEFAULTS = {"sched": "4", "alloc": "on", "state": "int8", "nvfp4": "norm+quant"}
OPTIONAL = {"lmhead": None, "junction": "on", "grammar_ws": "8"}
# The environment variables before KOHAKU_VLLM, still read (with a warning) as their features
LEGACY = {
    "KOHAKU_PREFILL_EVERY": "sched",
    "KOHAKU_FAST_ALLOC": "alloc",
    "KOHAKU_GDN_STATE": "state",
    "KOHAKU_MAMBA_JUNCTION": "junction",
    "KOHAKU_GRAMMAR_WS": "grammar_ws",
}


def parse(spec: str) -> dict[str, str]:
    """``KOHAKU_VLLM``'s text -> {feature: value}."""
    out: dict[str, str] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, eq, value = item.partition("=")
        if name in ("1", "on", "all") and not eq:
            for key, default in DEFAULTS.items():
                out.setdefault(key, default)
        elif name.startswith("-") and not eq:
            out.pop(name[1:], None)
        elif name in DEFAULTS or name in OPTIONAL:
            default = DEFAULTS.get(name) or OPTIONAL.get(name)
            out[name] = value if eq else default
            if out[name] is None:
                raise ValueError(f"KOHAKU_VLLM: {name} needs a value ({name}=...)")
        else:
            raise ValueError(f"KOHAKU_VLLM: unknown feature {item!r}")
    return out


def from_env(env=os.environ) -> dict[str, str]:
    chosen = parse(env.get("KOHAKU_VLLM", ""))
    for var, name in LEGACY.items():
        value = env.get(var, "")
        if not value or name in chosen:
            continue
        if name == "junction" and value != "1":
            continue
        if name == "alloc" and value not in ("on", "check"):
            continue
        log.warning("kohakuvllm: %s is deprecated, use KOHAKU_VLLM=%s=%s", var, name, value)
        chosen[name] = "on" if name == "junction" else value
    nvfp4 = env.get("KOHAKU_NVFP4", "")
    if nvfp4 and "nvfp4" not in chosen and "lmhead" not in chosen:
        log.warning("kohakuvllm: KOHAKU_NVFP4 is deprecated, use KOHAKU_VLLM=nvfp4=...,lmhead=...")
        parts = [p for p in nvfp4.split(",") if p in ("norm", "quant")]
        if parts:
            chosen["nvfp4"] = "+".join(parts)
        heads = [p.partition("=")[2] for p in nvfp4.split(",") if p.startswith("head=")]
        if heads:
            chosen["lmhead"] = heads[0]
    return chosen
