# Qwen3.5 on vLLM 0.32: RTX 2080 Ti (sm75) and V100 (sm70) (2026-10-24)

What stock vLLM lacks on these cards, from the probes (`bench/legacy/probe.sh`) and Nsight captures
of the engine alone. No performance number is reported: these cards have not been measured with
kernels built for them.

## What vLLM lacks

| | RTX 2080 Ti (sm75) | V100 (sm70) |
|---|---|---|
| vLLM 0.32 wheels | run (the cu129 / cu130 wheels carry sm75) | no sm70 kernels in any wheel; torch cu129 / cu130 drop sm70. Built from source with `TORCH_CUDA_ARCH_LIST=7.0` on torch 2.13 cu126 |
| bfloat16 | engine init fails; bf16 matmul is emulated | refused ("Bfloat16 is only supported on compute capability >= 8.0") |
| float16 | runs (`--dtype float16`, `mamba_ssm_cache_dtype float16`) | runs |
| INT8 W8A8 (compressed-tensors) | runs, cutlass sm75 INT8 GEMM | refused: the scheme needs capability 7.5 |
| FP8 checkpoints | weight-only, fp16, eager | no FP8 path below sm80 in vLLM |
| FlashAttention 2 | none | none |
| attention backend chosen | TRITON_ATTN | TRITON_ATTN |
| vision encoder CUDA graphs | | capture fails (its SDPA path copies `cu_seqlens` to the host) |

## Kernels without an sm70 / sm75 implementation

The Nsight captures name the kernels the engine runs there; each needs one written for these
cards (TileLang, fp16 MMA with fp32 accumulation):

1. GDN chunked prefill: FLA's Triton `chunk_gated_delta_rule_fwd_kernel_h`, `chunk_fwd_kernel_o`,
   `recompute_w_u_fwd_kernel`, `chunk_scaled_dot_kkt_fwd_kernel`, `merge_16x16_to_64x64_inverse_kernel`.
2. Attention, prefill and decode: vLLM's Triton `kernel_unified_attention`.
3. Linear layers: cuBLAS / cutlass fp16 GEMM; W8A16 on sm70, where the INT8 scheme is refused.
4. The vision encoder's attention, capturable in CUDA graphs.
5. The small pieces: causal conv and its update, the post-conv prep, RMSNorm, SiLU-and-mul, the
   GDN decode step.
