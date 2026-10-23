# KohakuvLLM

Our own frontend and kernels for running small hybrid VLMs (Qwen3.5: Gated DeltaNet + full
attention, with a vision tower) at full GPU load, with [vLLM](https://github.com/vllm-project/vllm)'s
engine as the backend. It is a drop-in plugin package: install it beside a stock vLLM wheel;
nothing is forked, patched on disk or recompiled. It follows upstream vLLM;
[TESTED.md](TESTED.md) names the last versions each part was verified on, and the plugin warns
when the installed vLLM is newer.

```bash
pip install kohakuvllm            # beside vLLM; or pip install -e . from a clone
vllm serve Qwen/... --scheduler-cls kohakuvllm.sched.PhasedScheduler
```

The same parts apply to vLLM's engine driven in-process (`AsyncLLM`, as KohakuGBC's in-process
VLM entry does) and behind `vllm serve`: every part is opted into by an engine argument or an
environment variable, and with none set vLLM runs as shipped.

## What it changes in vLLM

Every change, where it hooks in, and how it is turned on. Nothing else in vLLM is touched.

| Part | Hooks into vLLM | On by | What it does |
|---|---|---|---|
| `kohakuvllm.plugin` | the `vllm.general_plugins` entry point (runs at import in every vLLM process) | installing the package | warns once when vLLM is newer than `TESTED_VLLM`; installs `gdn.hybrid` when `KOHAKU_GDN` asks for it; nothing else |
| `kohakuvllm.sched.PhasedScheduler` | a subclass of `vllm.v1.core.sched.async_scheduler.AsyncScheduler`, loaded by vLLM's `scheduler_cls` engine argument | `--scheduler-cls kohakuvllm.sched.PhasedScheduler` (`scheduler_cls=` in-process) | keeps prefill out of most steps: between prefill steps it calls vLLM's own prefill throttle with the saturation escape off, so those steps are uniform decode batches that replay full CUDA graphs. `KOHAKU_PREFILL_EVERY` (4): one step in that many admits prefill; 1 is vLLM's own scheduling |
| `kohakuvllm.gdn.hybrid` | replaces `QwenGatedDeltaNetAttention._forward_core` and `GDNAttentionBackend.get_builder_cls` (a subclass of vLLM's GDN metadata builder), and takes `vllm::qwen_gdn_attention_core` out of `CompilationConfig`'s splitting ops | `KOHAKU_GDN=hybrid`; `KOHAKU_GDN=check` runs it beside vLLM's core and logs the difference per layer and step (`KOHAKU_GDN_CHECK`: the log path prefix) | the GDN core of mixed prefill/decode steps as one fixed launch sequence over persistent padded buffers, so piecewise CUDA graphs capture it: `gdn.conv`'s varlen causal conv and state update, vLLM's `fused_post_conv_prep`, FLA's recurrent kernel for decodes and FlashInfer's chunked kernel for prefills, both on the state pool by slot. Decode-only steps keep vLLM's path; speculative decoding keeps vLLM's path |
| `kohakuvllm.gdn.conv` | none (kernels called by `gdn.hybrid`) | with `gdn.hybrid` | the causal conv over a packed varlen batch and each sequence's conv-state update, in Triton, for both of vLLM's conv-state layouts |

Measured (RTX 5090, vLLM 0.32.0; [reports](reports/)):

| Part | Result |
|---|---|
| `PhasedScheduler` | GBC's flagship machine on one card: 0.95 -> 1.36 img/s at vLLM share 0.55, 1.50-1.55 at 0.75 with 384 lanes (prefill every 16 steps; 4 / 8 / 32 within 3%). Under a saturated offline queue of uniform prompts it costs (17.8 -> 14.8 requests/s): prefill batches best there |
| `gdn.conv` | states bitwise equal to vLLM's `causal_conv1d_fn`; 0.06-0.08 ms vs 0.4-1.1 ms per call |
| `gdn.hybrid` | within the fp32 reference's error of vLLM's split path, capture/replay bitwise, 0.18-0.58 ms vs 0.43-0.93 ms per layer; no gain in the machine (the card is power-bound and the large mixed steps run piecewise either way), so it stays off by default |

## Tools and benchmarks

| Path | What it is |
|---|---|
| `tools/quantize.py` | W8A8 checkpoints from a BF16 one: FP8 or INT8 per-channel weights (round to nearest), dynamic per-token activations, over the linears named by suffix or by a reference checkpoint (`--like`), so the GDN projections can go to 8 bits too; vLLM loads them as compressed-tensors |
| `tools/smooth.py` | SmoothQuant folded into Qwen3.5's zero-centred `(1 + w)` RMSNorms before INT8 quantization |
| `tools/gptq_w8a8.py` | GPTQ W8A8 through llm-compressor |
| `bench/inproc/run.py` | the engine alone, in-process, pre-tokenized prompts: the backend's ceiling and its GPU load, any engine argument and scheduler |
| `bench/profile/` | Nsight Systems capture of that run (`inproc_nsys.sh`); GPU busy, idle gaps and time per kernel class (`nsys_summary.py`) |
| `bench/gdn/` | each GDN piece against vLLM's and an fp64 reference: conv (`conv_check.py`), the mixed step (`mixed_check.py`), FlashInfer capture (`fi_multi_capture.py`), the decode kernels (`decode_check.py`) |
| `bench/precision/` | generations and prompt logprobs of two engine configurations compared (KL, argmax agreement) |
| `bench/legacy/probe.sh` | stock vLLM on older cards (2080 Ti sm75, V100 sm70) per checkpoint, dtype and graph mode |
| `reports/` | measurements, dated |

## Why

On an RTX 5090, Qwen3.5-4B FP8 behind `vllm serve` with GBC's captioning requests (one image,
~1000 prompt tokens, ~100 reply tokens, JSON schema) leaves the GPU 35-55% busy: vLLM's engine
thread spends its time running the GDN layers eagerly on every mixed prefill/decode step and
filling grammar bitmasks, and the GPU waits for it. Driven in-process with no server, the same
engine keeps the card 87% busy. Both paths stay: HTTP to a hosted server, and the engine
in-process for local runs; KohakuvLLM's parts apply to both.

## License

Apache-2.0.
