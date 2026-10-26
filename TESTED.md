# Tested versions

KohakuvLLM follows upstream vLLM. This file names the last versions every part was verified on;
a newer vLLM may work, and a part that breaks on it is a bug to report. `kohakuvllm.TESTED_VLLM`
holds the vLLM version of the newest row, and the plugin warns when the installed vLLM is newer.

| Part | vLLM | torch | GPU | Date |
|---|---|---|---|---|
| `kohakuvllm.sched.PhasedScheduler` | 0.32.0 (PyPI wheel) | 2.13.0+cu130 | RTX 5090 (sm120), RTX 3090 (sm86) | 2026-10-24 |
| `kohakuvllm.gdn.state8` | 0.32.0 (PyPI wheel) | 2.13.0+cu130 | RTX 5090 (sm120: FP8 and INT8 in the engine), RTX 3090 (sm86: INT8 in the engine) | 2026-10-24 |
| `kohakuvllm.grammar` (Model Runner V2), `kohakuvllm-grammar` 0.0.1 | 0.32.0 (PyPI wheel) | 2.13.0+cu130 | RTX 5090 (sm120), RTX 3090 (sm86); in-process and behind `vllm serve` | 2026-10-24 |
| `kohakuvllm.plugin` | 0.32.0 (PyPI wheel) | 2.13.0+cu130 | RTX 5090 (sm120), RTX 3090 (sm86) | 2026-10-24 |
| `kohakuvllm.plugin` + `features` (`KOHAKU_VLLM`), `PhasedScheduler` installed without `--scheduler-cls`, grammar as a `vllm.logits_processors` plugin | 0.32.0 (PyPI wheel) | 2.13.0+cu130 | B300 (sm103) | 2026-10-26 |
| `kohakuvllm.sched.fastalloc` (`alloc=check`: 0 of 600k shortcut calls changed a block) | 0.32.0 (PyPI wheel) | 2.13.0+cu130 | B300 (sm103) | 2026-10-26 |
| `kohakuvllm.kernels` (`bench/kernels/check.py`: every Triton op against its eager reference) | 0.32.0 (PyPI wheel) | 2.13.0+cu130 | B300 (sm103) | 2026-10-26 |
| `tools/quantize.py`, `tools/smooth.py` | 0.32.0 (PyPI wheel) | 2.13.0+cu130 | RTX 5090 (FP8), RTX 3090 (INT8) | 2026-10-24 |
| `bench/` | 0.32.0 (PyPI wheel) | 2.13.0+cu130 | RTX 5090 (sm120), RTX 3090 (sm86) | 2026-10-24 |
