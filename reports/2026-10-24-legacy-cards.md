# Qwen3.5 on vLLM 0.32: RTX 2080 Ti (sm75) and V100 (sm70) (2026-10-24)

What stock vLLM lacks on these cards, what runs, and which kernels a TileLang / CUDA port has to
replace, from the probes (`bench/legacy/probe.sh`), the matmul peaks (`bench/gemm/peak.py`) and
Nsight captures of the engine alone (`bench/profile/inproc_nsys.sh`; 1000 in / 100 out, 128
sequences, fp16).

## What vLLM lacks

| | RTX 2080 Ti (sm75) | V100 (sm70) |
|---|---|---|
| vLLM 0.32 wheels | run (the cu129 / cu130 wheels carry sm75) | no sm70 kernels in any wheel; torch cu129 / cu130 drop sm70. Built from source with `TORCH_CUDA_ARCH_LIST=7.0` on torch 2.13 cu126 |
| bfloat16 | engine init fails; bf16 matmul is emulated (7 TFLOPS vs 52 fp16) | refused ("Bfloat16 is only supported on compute capability >= 8.0") |
| float16 | runs (`--dtype float16`, `mamba_ssm_cache_dtype float16`) | runs |
| INT8 W8A8 (compressed-tensors) | runs, cutlass sm75 INT8 GEMM | refused: the scheme needs capability 7.5 |
| FP8 checkpoints | weight-only, fp16, eager | not tried (no FP8 path below sm80 in vLLM) |
| FlashAttention 2 | none | none |
| attention backend chosen | TRITON_ATTN | TRITON_ATTN |
| CUDA graphs | run | run on the 32 GB cards; the 16 GB card runs eager only (graph capture does not fit beside 8.6 GiB of fp16 weights) |

Measured, engine alone: 2080 Ti, the INT8 checkpoint, fp16 graphs 5.07 req/s at 32 sequences,
3.83 at 128 under Nsight; V100 32 GB, bf16 checkpoint run in fp16, 5.69 req/s at 32 sequences
(graphs) / 4.85 (eager), 2.65 at 128; V100 16 GB eager, 2.02.

## Matmul peaks

| | fp16 | bf16 | tf32 | INT8 |
|---|---|---|---|---|
| RTX 2080 Ti | 50-53 TFLOPS | 7 (emulated) | 12 | 120-139 TOPS |
| RTX 3090 (fp32 accumulate unlocked) | 119-127 | 123-130 | 63-66 | 214-250 |

## Where the time goes

| Kernel | V100 | 2080 Ti |
|---|---|---|
| GDN chunked prefill: `chunk_gated_delta_rule_fwd_kernel_h` (FLA, Triton) | 46.3% | 15.3% |
| GDN chunked prefill: `chunk_fwd_kernel_o` | 5.7% | 12.5% |
| other GDN prefill pieces (`recompute_w_u`, `chunk_scaled_dot_kkt`, inverse) | 2.2% | 3.9% |
| attention: `kernel_unified_attention` (Triton) | 14.0% | 34.3% |
| linear layers: cutlass / cuBLAS GEMM | 27% (fp16 s884) | 23% (INT8) + 1.4% activation quantization |
| GDN decode (`fused_recurrent_gated_delta_rule_packed_decode`) | 0.9% | 2.7% |

Both cards are 99.6-99.7% busy: the engine keeps them fed; the kernels are slow.

## Kernels to replace

1. **GDN chunked prefill** (`chunk_gated_delta_rule_fwd_kernel_h` first, then `chunk_fwd_kernel_o`):
   54% of the V100's time and 32% of the 2080 Ti's (likely Triton's weak or absent tensor-core lowering for these tiles below sm80; the capture
   shows the time, not the cause). A TileLang or CUDA kernel on `mma.sync` m8n8k4 (sm70) /
   m16n8k8 (sm75) fp16 with fp32 accumulation.
2. **Attention for prefill and decode**: no FlashAttention below sm80; vLLM falls back to its
   Triton unified attention, 34% of the 2080 Ti's time and 14% of the V100's. A paged
   FlashAttention-style kernel for sm75 / sm70 (fp16 `mma.sync`).
3. **INT8 W8A8 linears on sm70**: vLLM's scheme stops at 7.5; V100 has no INT8 tensor cores, so
   W8A16 (INT8 weights dequantized in the GEMM's main loop, fp16 tensor cores) is the path that
   halves weight bytes there.
4. The GDN decode kernel and the 8-bit state (`gdn.state8`) already run on both (INT8 state
   1.3-1.9x the bf16 pool's decode step); no replacement needed.

bf16 has no route on either card: fp16 everywhere, with the GDN state in fp16 or INT8.
