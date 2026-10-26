"""KohakuvLLM: a vLLM plugin that makes ``vllm serve`` faster on small hybrid VLMs (Qwen3.5:
Gated DeltaNet + full attention, with a vision tower). No fork, no patch on disk, no rebuild;
``KOHAKU_VLLM`` (``kohakuvllm.features``) turns its parts on."""

__version__ = "0.1.0"
TESTED_VLLM = "0.32.0"
