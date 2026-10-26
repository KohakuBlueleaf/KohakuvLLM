"""KohakuvLLM's kernels: an op registry with per-arch backend priority by card family, so a
kernel written for one card serves every card that can run it and the others fall back.

Each op has one eager reference (``backends/eager``, plain torch or vLLM's own op) and per-arch
implementations (``backends/triton`` on sm80+, ``backends/gluon`` on the sm10x family) registered
with ``registry.register``; a caller resolves an op once (``registry.resolve``) to a ``Bound``
that falls back along the arch's backend priority when a backend cannot build or launch.

Ops (each with Triton and eager implementations):

- ``nvfp4_quant``: a bf16 activation to packed E2M1 and swizzled E4M3 scales, byte for byte
  vLLM's ``scaled_fp4_quant``;
- ``rmsnorm_nvfp4``: (fused add) RMSNorm, its output quantized to NVFP4 once or twice;
- ``grammar_mask``: logits rows set to -inf outside each row's grammar mask;
- ``gdn_state8_decode``: one gated-delta-rule decode step on the 8-bit GDN state.
"""
