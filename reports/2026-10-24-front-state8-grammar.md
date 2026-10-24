# Frontend, 8-bit GDN state and grammar (2026-10-24)

vLLM 0.32.0 (Model Runner V2), torch 2.13 cu130; the V100 runs vLLM 0.32.0 built from source for
sm70 on torch 2.13 cu126. Qwen3.5-4B: the FP8 checkpoint with FP8 GDN projections on the 5090,
SmoothQuant INT8 (`sq65all`) on the 3090. The machine is KohakuGBC's
`configs/machine/qwen35_card.py` with the `mid` + `sam3` controls, 480 images per card, the VLM
in-process with the PhasedScheduler (prefill every 16 steps), KV cache fixed at 12 GiB (5090) /
6 GiB (3090) so every cell has the same cache. Cells compared against each other ran in the same
window on different cards of one box.

## Frontend (`kohakuvllm.front`)

Against a float32 run of the reference tower (the ruler), on 16 images (18704 patches):

| Tower | mean relative error (ruler above its mean) | p99 |
|---|---|---|
| reference model, bf16 | 3.35% | 30.6% |
| ours, bf16 | 3.38% | 31.6% |
| ours, FP8 linears | 13.3% | 90.5% |

The engine with each input path, 64 requests, against the engine fed the ruler's embeddings:

| Path | prompt ids identical | replies identical | agreed prefix | logprob \|diff\| mean / p99 |
|---|---|---|---|---|
| vLLM vs vLLM in reversed order (the floor) | 64 | 2 | 0.33 | 0.073 / 1.21 |
| vLLM's own input path | 64 | 1 | 0.25 | 0.208 / 2.34 |
| ours, bf16 tower | 64 | 2 | 0.28 | 0.225 / 2.32 |
| ours, FP8 tower | 64 | 1 | 0.23 | 0.249 / 2.68 |

Our bf16 path is as far from the ruler as vLLM's own. In the machine (5090, four cells in one
window): ours 1.21 img/s (bf16 and FP8 towers alike) against 1.44-1.46 for vLLM's input path.
The host ran at load 94 (four preparing processes per machine beside SAM 3 and EmbeddingGemma),
and our tower runs eagerly where vLLM's encoder is compiled and graph-captured. The frontend
stays a knob, off by default.

## 8-bit GDN state (`kohakuvllm.gdn.state8`)

Kernel, 64 decode steps against an fp64 ruler, mean relative error per state row:

| Pool | state | output |
|---|---|---|
| bf16 (vLLM's) | 0.17% | 0.17% |
| INT8 | 0.64% | 0.36% |
| FP8 | 2.7% | 1.25% |

No growth over the steps; slots outside the batch untouched. Decode step time, CUDA-graph
replayed, 8x1 tile, relative to the bf16 pool:

| Card | batch 128 | batch 256-384 |
|---|---|---|
| RTX 5090 | INT8 2.4x, FP8 2.4x | 1.9x both |
| RTX 2080 Ti | INT8 1.6x | 1.8-1.9x |
| V100 | INT8 1.3x | 1.5x |

Engine precision (5090, 431 recorded conversations, against the bf16 state):

| State | teacher-forced KL | top-1 | greedy replies identical | valid JSON |
|---|---|---|---|---|
| float32 (the checkpoint's own) | 0.0045 | 98.14% | | |
| INT8 | 0.0043 | 98.06% | 45.9% | 345 / 431 |
| FP8 | 0.0043 | 98.04% | 39.2% | 340 / 431 |
| bf16, batch 32 instead of 64 (the floor) | | | 44.3% | 349 / 431 |

INT8 sits at the floor; FP8 a little below it. In the machine:

| Card | bf16 state | INT8 state | KV tokens in the same cache |
|---|---|---|---|
| RTX 5090 (clean window) | 1.560 | 1.591 | 320k -> 355k |
| RTX 3090 | see the 3090 table | | 160k -> 177k |

The 5090 is power-bound (96-98% busy at 545-560 W of 575), so the larger cache does not turn
into throughput there. Engine alone, 1000 in / 100 out: 2080 Ti 3.35 -> 3.52 req/s (INT8); V100
2.65 -> 2.58 and 1.91 -> 1.86 on two cards (prefill dominates there and pays the quantize).

## Grammar (`kohakuvllm.grammar`)

Masks, every step over the whole vocabulary of generated documents and random walks on GBC's
seven schemas, compared with xgrammar's; every disagreement and a sample of the agreements
decided by `bench/grammar/reference.py`: ours 0 wrong of 841k judged on the first schema run;
xgrammar forbids escapes and allows raw control bytes inside `maxLength` strings, and rejected
valid documents. CPU: 0.14 ms per new mask, 0.011 ms per step from the memo; xgrammar 0.61 ms per
fill. In the engine: 231 of 231 replies (32 sampled at temperature 1.0 + 1 greedy per schema)
are complete valid documents.

The machine, 5090, four cells in one window:

| Structured replies | img/s | GPU busy |
|---|---|---|
| KohakuvLLM grammar | 1.589 | 96.8% |
| prompt + one retry | 1.539 | 98.6% |
| vLLM structured outputs, guidance | 1.426 | 85.4% |
| vLLM structured outputs, xgrammar | 1.226 | 74.8% |

vLLM's structured outputs build the bitmask in the scheduler, which has to wait for each
request's last token: with async scheduling about 250 requests stay "Deferred" and the card
idles. Ours reads the last token on the worker before the step and costs no throughput.

## The best configuration, four 5090s

KohakuvLLM grammar + INT8 GDN state, one machine per card at once: 1.592 / 1.617 / 1.599 / 1.650
img/s, 6.46 img/s for the box, 1920 of 1920 jobs, 94-95% busy at 532-555 W.
