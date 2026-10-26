"""NVFP4 serving on sm100 / sm120: our activation producers (``kernels``, ``ops``), the
torch.compile pass that puts them in place of vLLM's (``fusion``), and the LM head in FP8 or NVFP4
(``head``).

``KOHAKU_NVFP4`` (``kohakuvllm.plugin``) is a comma list: ``norm`` (RMSNorm fused with the
quantizations it feeds), ``quant`` (our kernel for the other quantizations), ``head=fp8`` or
``head=nvfp4``, ``cvt=ptx`` or ``cvt=arith`` (the E2M1 conversion, ``ptx`` by default),
``tile=BMxBKxW`` (the producers' tile), ``head_graph=0`` (the head launched op by op; replayed
from CUDA graphs by default).
"""

from kohakuvllm.nvfp4 import fusion, head, ops


def install(spec: str) -> dict:
    parts, done = set(), {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        key, _, value = item.partition("=")
        if key in ("norm", "quant"):
            parts.add(key)
        elif key == "head":
            head.install(value)
            done["head"] = value
        elif key == "cvt":
            ops.CVT["name"] = value
            done["cvt"] = value
        elif key == "head_graph":
            head.GRAPH["on"] = value != "0"
            done["head_graph"] = value
        elif key == "tile":
            ops.TILE["value"] = tuple(int(v) for v in value.split("x"))
            done["tile"] = value
        else:
            raise ValueError(f"KOHAKU_NVFP4: unknown part {item!r}")
    if parts:
        fusion.install(parts)
        done["fusion"] = sorted(parts)
    return done
