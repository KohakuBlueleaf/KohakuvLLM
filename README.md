# KohakuvLLM

Our own frontend and kernels for running small hybrid VLMs (Qwen3.5: Gated DeltaNet + full
attention, with a vision tower) at full GPU load, with [vLLM](https://github.com/vllm-project/vllm)'s
engine as the backend. It is a drop-in plugin package: install it beside a stock vLLM wheel;
nothing is forked, patched on disk or recompiled. It follows upstream vLLM;
[TESTED.md](TESTED.md) names the last versions each part was verified on, and the plugin warns
when the installed vLLM is newer.

```bash
pip install kohakuvllm            # beside vLLM; or pip install -e . from a clone
pip install ./grammar             # the Rust grammar matcher (maturin, a Rust toolchain)
vllm serve Qwen/... --scheduler-cls kohakuvllm.sched.PhasedScheduler
```

The same parts apply to vLLM's engine driven in-process (`AsyncLLM`, as KohakuGBC's in-process
VLM entry does) and behind `vllm serve`: every part is opted into by an engine argument or an
environment variable, and with none set vLLM runs as shipped.

## What it changes in vLLM

Every change, where it hooks in, and how it is turned on. Nothing else in vLLM is touched.

| Part | Hooks into vLLM | On by | What it does |
|---|---|---|---|
| `kohakuvllm.plugin` | the `vllm.general_plugins` entry point (runs at import in every vLLM process) | installing the package | warns once when vLLM is newer than `TESTED_VLLM`; installs `gdn.hybrid` when `KOHAKU_GDN` asks for it and `gdn.state8` when `KOHAKU_GDN_STATE` does; nothing else |
| `kohakuvllm.sched.PhasedScheduler` | a subclass of `vllm.v1.core.sched.async_scheduler.AsyncScheduler`, loaded by vLLM's `scheduler_cls` engine argument | `--scheduler-cls kohakuvllm.sched.PhasedScheduler` (`scheduler_cls=` in-process) | keeps prefill out of most steps: between prefill steps it calls vLLM's own prefill throttle with the saturation escape off, so those steps are uniform decode batches that replay full CUDA graphs. `KOHAKU_PREFILL_EVERY` (4): one step in that many admits prefill; 1 is vLLM's own scheduling |
| `kohakuvllm.gdn.hybrid` | replaces `QwenGatedDeltaNetAttention._forward_core` and `GDNAttentionBackend.get_builder_cls` (a subclass of vLLM's GDN metadata builder), and takes `vllm::qwen_gdn_attention_core` out of `CompilationConfig`'s splitting ops | `KOHAKU_GDN=hybrid`; `KOHAKU_GDN=check` runs it beside vLLM's core and logs the difference per layer and step (`KOHAKU_GDN_CHECK`: the log path prefix) | the GDN core of mixed prefill/decode steps as one fixed launch sequence over persistent padded buffers, so piecewise CUDA graphs capture it: `gdn.conv`'s varlen causal conv and state update, vLLM's `fused_post_conv_prep`, FLA's recurrent kernel for decodes and FlashInfer's chunked kernel for prefills, both on the state pool by slot. Decode-only steps keep vLLM's path; speculative decoding keeps vLLM's path |
| `kohakuvllm.gdn.conv` | none (kernels called by `gdn.hybrid`) | with `gdn.hybrid` | the causal conv over a packed varlen batch and each sequence's conv-state update, in Triton, for both of vLLM's conv-state layouts |
| `kohakuvllm.gdn.state8` | `MambaStateShapeCalculator.gated_delta_net_state_shape` / `MambaStateDtypeCalculator.gated_delta_net_state_dtype` (the pool becomes `uint8 [HV, V, K + 4]`), the GDN module's `fused_recurrent_gated_delta_rule_packed_decode` and `fused_sigmoid_gating_delta_rule_update`, and `QwenGatedDeltaNetAttention.__init__` (its chunked prefill wrapped) | `KOHAKU_GDN_STATE=int8` or `fp8` (FP8: sm89+); `KOHAKU_GDN_STATE_TILE=ROWSxWARPS` | the GDN recurrent state in 8 bits with a float32 scale per row: half the state bytes. Decodes run one Triton kernel that dequantizes, steps and requantizes each row; prefills dequantize the gathered states before vLLM's chunked kernel and quantize its final states. Not with `gdn.hybrid` or speculative decoding |
| `kohakuvllm.front` | vLLM's `enable_mm_embeds` input path: requests go in as token ids plus `image_embeds` / `image_grid_thw` | KohakuGBC's in-process entry (`frontend="kohakuvllm"`), or `Frontend(llm, model)` around any `AsyncLLM` | the input side outside vLLM's engine: chat template and image processor in a pool of worker processes (`prep`), and the model's own vision tower (bf16, or FP8 linears on sm89+) batched on its own CUDA stream in the caller's process (`vision`); the engine then only decodes |
| `kohakuvllm.grammar` (with the `kohakuvllm-grammar` package, `grammar/`) | a vLLM v1 logits processor, loaded by vLLM's `logits_processors` engine argument; per request through `SamplingParams.extra_args` (`vllm_xargs` over HTTP) | `--logits-processors kohakuvllm.grammar:JsonSchemaLogits` and `extra_args={"kohaku_json_schema": SCHEMA}`; `KOHAKU_GRAMMAR_WS` (whitespace per gap, 0) | JSON-schema replies: a Rust byte-level matcher (object, array, string with lengths, number, boolean, null, enum) whose counters clamp to the longest token, so each distinct clamped state's mask is built once (a token-trie walk; string bodies from precomputed masks) and copied once to a GPU table. A step costs one Rust call for the batch and one mask id per row; one Triton kernel masks the logits |

Measured (vLLM 0.32.0; RTX 5090 unless named; [reports](reports/)). GBC's flagship machine, one
per card on four 5090s with `PhasedScheduler`, `grammar` and the INT8 `gdn.state8`: 6.46 img/s
for the box (1.59-1.65 per card), 17% over stock vLLM on the same machine, in-process and behind
`vllm serve` alike.

| Part | Result |
|---|---|
| `PhasedScheduler` | GBC's flagship machine on one card: 0.95 -> 1.36 img/s at vLLM share 0.55, 1.50-1.55 at 0.75 with 384 lanes (prefill every 16 steps; 4 / 8 / 32 within 3%). Under a saturated offline queue of uniform prompts it costs (17.8 -> 14.8 requests/s): prefill batches best there |
| `gdn.conv` | states bitwise equal to vLLM's `causal_conv1d_fn`; 0.06-0.08 ms vs 0.4-1.1 ms per call |
| `gdn.hybrid` | within the fp32 reference's error of vLLM's split path, capture/replay bitwise, 0.18-0.58 ms vs 0.43-0.93 ms per layer; no gain in the machine (the card is power-bound and the large mixed steps run piecewise either way), so it stays off by default |
| `gdn.state8` (kernel, 64 decode steps vs an fp64 ruler) | state error per row (mean relative where the ruler is above its mean): bf16 pool 0.17%, INT8 0.64%, FP8 2.7%; outputs 0.17% / 0.36% / 1.25%; no growth over steps; other slots untouched. Decode step at batch 256-384: 1.9x the bf16 pool on a 5090 (INT8 and FP8 alike), 1.6-1.9x on a 2080 Ti (INT8, 8x1 tile). Engine: INT8 teacher-forced KL and greedy agreement at the batching floor, FP8 slightly below. Machine: the same cache holds 11% more KV tokens; KV-bound RTX 3090s +9.4% (+6 to +14% per card, every card), the power-bound 5090 +2% |
| `front` | prompt ids identical to vLLM's input path; bf16 tower as close to a float32 tower as the reference model's; in the machine 1.21 img/s vs 1.44-1.46 (host-bound preparing, eager tower), so it stays a knob |
| `grammar` (vs xgrammar 0.1.x on GBC's schemas, every step over the whole vocabulary, disagreements decided by an independent judge, `bench/grammar/`) | ours 0 wrong decisions; xgrammar forbids escapes and allows raw control bytes inside `maxLength` strings. 0.14 ms per new mask, 0.011 ms per step from the memo vs 0.61 ms per xgrammar fill. In the engine 231 / 231 replies valid. Machine: 1.589 img/s vs prompt + retry 1.539, vLLM structured outputs with guidance 1.426 and xgrammar 1.226 (the scheduler-side bitmask defers requests under async scheduling) |

## Tools and benchmarks

| Path | What it is |
|---|---|
| `tools/quantize.py` | W8A8 checkpoints from a BF16 one: FP8 or INT8 per-channel weights (round to nearest), dynamic per-token activations, over the linears named by suffix or by a reference checkpoint (`--like`), so the GDN projections can go to 8 bits too; vLLM loads them as compressed-tensors |
| `tools/smooth.py` | SmoothQuant folded into Qwen3.5's zero-centred `(1 + w)` RMSNorms before INT8 quantization |
| `tools/gptq_w8a8.py` | GPTQ W8A8 through llm-compressor |
| `bench/inproc/run.py` | the engine alone, in-process, pre-tokenized prompts: the backend's ceiling and its GPU load, any engine argument and scheduler |
| `bench/profile/` | Nsight Systems capture of that run (`inproc_nsys.sh`); GPU busy, idle gaps and time per kernel class (`nsys_summary.py`) |
| `bench/gdn/` | each GDN piece against vLLM's and an fp64 reference: conv (`conv_check.py`), the mixed step (`mixed_check.py`), FlashInfer capture (`fi_multi_capture.py`), the decode kernels (`decode_check.py`), the 8-bit states and their decode tiles (`state8_check.py`) |
| `bench/front/check.py` | the frontend against vLLM's input path: vision towers vs a float32 ruler, and the engine's prompt ids, greedy replies and teacher-forced logprobs per path |
| `bench/grammar/` | `check.py`: our masks vs xgrammar's at every step of generated documents and random walks, with `reference.py`, a separate recursive-descent judge, deciding each disagreement |
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
