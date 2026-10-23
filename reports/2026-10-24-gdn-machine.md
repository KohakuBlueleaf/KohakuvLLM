# GDN path and the GBC machine on RTX 5090 / 3090 (2026-10-24)

vLLM 0.32.0, torch 2.13 cu130. Qwen3.5-4B W8A8 (FP8 with the GDN projections in FP8 on the
5090; SmoothQuant INT8 on the 3090), bf16 GDN state. The machine is KohakuGBC's
`configs/machine/qwen35_card.py` with the `mid` + `sam3` controls over 480 images per card: the
VLM in-process, GBC's SAM 3 (fp8 row) and EmbeddingGemma on the same card.

## GDN

| Piece | Result |
|---|---|
| `gdn.conv` varlen conv + state update | states bitwise equal to vLLM's `causal_conv1d_fn`, both conv-state layouts; 0.06-0.08 ms vs 0.4-1.1 ms |
| hybrid mixed step (FLA decode + FlashInfer prefill) | within the fp32 reference's error of vLLM's split path; capture/replay bitwise; 0.18-0.58 ms vs 0.43-0.93 ms per layer |
| hybrid in the engine | generations and prompt logprobs within the stock eager-vs-graph noise floor (KL 0.0058) |
| hybrid in the machine | no gain (1.51-1.53 img/s vs 1.50-1.55): mixed steps above the graph sizes run piecewise either way, and the card is power-bound |
| decode kernel, FLA vs FlashInfer bf16-state pool | identical error against fp64; both 0.317 ms at 256 sequences = 1.62 TB/s, 90% of the card's bandwidth |

The decode step is bandwidth-bound on the state (bf16, 1 MB per sequence read and written); a
smaller state (fp8) is the remaining lever, at most about 4% of the machine's GPU time.
`KOHAKU_GDN=hybrid` stays a knob, off by default.

## Machine, 5090 (img/s per card)

| Config | img/s |
|---|---|
| stock vLLM, share 0.55 | 0.95 |
| PhasedScheduler, prefill every 16 steps | 1.36 |
| + share 0.75, 384 lanes | 1.50-1.55 (repeats) |
| prefill every 4 / 8 / 32 | 1.52 / 1.54 / 1.47 |
| SAM 3 batch 16, 512 lanes, 16k step tokens | 1.54 / 1.53 / 1.51 |
| attention FLASHINFER / TRITON_ATTN (bf16 KV) | 1.55 / 1.54 |
| prefix caching off | 1.24 |
| FP8 KV, prefix caching on | 0.77 (about 8 running) |
| FP8 KV, prefix caching off | 1.28 |

At share 0.75 the card is 96% busy (Nsight) and at its power cap (558 W of 575): vLLM's engine
61% of kernel time (FP8 GEMM 39%, GDN decode 14%, attention 15%), SAM 3 + EmbeddingGemma 39%.
FP8 KV doubles vLLM's block size to 544 tokens (an attention page must cover a Mamba page);
with prefix caching vLLM's Mamba `align` mode then cuts prefill chunks at block boundaries ahead
of the image-encoder limits, and few requests can be scheduled.

## Machine, 3090 (img/s per card, INT8 SmoothQuant, SAM 3 bf16)

| Config | img/s |
|---|---|
| RTN INT8, share 0.55 | 0.29 |
| SmoothQuant INT8, share 0.70-0.72, 128 lanes, prefill every 16 | 0.39-0.44 |
| share 0.73 + SAM 3 batch 4 | 0.41 |
| down projections in BF16 (FP8-level accuracy) | 0.36 |
| FP8 KV | 0.31 |

The 3090 runs KV-capacity bound (cache 91-99% used).
