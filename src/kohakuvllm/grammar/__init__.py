"""JSON-schema replies with token masks from KohakuvLLM's Rust matcher (``grammar/``, the
``kohakuvllm-grammar`` package): ``logits.JsonSchemaLogits``, a vLLM logits processor."""

from kohakuvllm.grammar.logits import JsonSchemaLogits

__all__ = ["JsonSchemaLogits"]
