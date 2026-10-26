"""The LM head's GEMM in FP8 or NVFP4: checkpoints keep the head in bf16, and on a small model
reading its ``vocab x hidden`` weight every step is a large share of decode (Qwen3.5-4B: 1.27 GB
a step, 7% of GPU time).

- ``fp8``: the weight E4M3 per output row, the hidden states per token (dynamic), cuBLAS's
  row-wise scaled GEMM.
- ``nvfp4``: the weight NVFP4 with its own tensor scale, the hidden states NVFP4 with a tensor
  scale taken from their amax each call (no calibration), FlashInfer's CUTLASS FP4 GEMM.

The quantized copy is made at the head's first call (vLLM's profiling run, before the KV cache is
sized); an untied head then frees its bf16 weight. ``install(mode)`` routes ``LogitsProcessor``'s
head through it, adds the head GEMM to vLLM's FlashInfer autotune pass (which never calls it) and
after that pass captures the head's CUDA graphs (``Graphed``): vLLM runs the head eagerly, and
its dozen launches a step held the 4B's decode back (GPU busy 83% against 96%).
"""

import torch
import vllm.utils.flashinfer as fi_utils
from flashinfer.gemm import mm_fp4
from vllm import _custom_ops as vops
from vllm.model_executor.layers import logits_processor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.warmup import kernel_warmup

from kohakuvllm.nvfp4 import ops

E2M1_E4M3 = 6.0 * 448.0
MODE = {"name": None}
# on: the head replayed from CUDA graphs (``Graphed``), else launched op by op every step
GRAPH = {"on": True}


def amax(t: torch.Tensor) -> torch.Tensor:
    """``max |t|`` as fp32 ``[1]`` without a full-size temporary (a 27B head is 2.5 GB)."""
    lo, hi = torch.aminmax(t)
    return torch.maximum(-lo, hi).float().clamp(min=1e-12).reshape(1)


def prepare(weight: torch.Tensor, mode: str) -> tuple:
    if mode == "fp8":
        w8, ws = vops.scaled_fp8_quant(weight, use_per_token_if_dynamic=True)
        return ("fp8", w8, ws.reshape(1, -1).contiguous())
    gs = E2M1_E4M3 / amax(weight)
    w4, ws4 = ops.nvfp4_quant(weight, gs)
    return ("nvfp4", w4, ws4, gs)


def head_of(lm_head) -> tuple:
    """``lm_head``'s quantized copy, made on first use (an untied head's bf16 weight freed)."""
    head = getattr(lm_head, "_kohaku_head", None)
    if head is None or head[0] != MODE["name"]:
        head = prepare(lm_head.weight.data, MODE["name"])
        lm_head._kohaku_head = head
        lm_head._kohaku_hidden = lm_head.weight.shape[1]
        if type(lm_head) is ParallelLMHead and not getattr(lm_head, "_kohaku_tied", False):
            lm_head.weight.data = lm_head.weight.data.new_empty(0, lm_head.weight.shape[1])
    return head


def project(head: tuple, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``x @ head.T``, into ``out`` when given (``[rows, vocab]``, contiguous)."""
    x2 = x.reshape(-1, x.shape[-1])
    if head[0] == "fp8":
        _, w8, ws = head
        xq, xs = vops.scaled_fp8_quant(x2, use_per_token_if_dynamic=True)
        if out is None:
            out = torch._scaled_mm(xq, w8.t(), xs, ws, out_dtype=x.dtype)
        else:
            torch.ops.aten._scaled_mm.out(xq, w8.t(), xs, ws, out_dtype=x.dtype, out=out)
    else:
        _, w4, ws4, wgs = head
        gs = E2M1_E4M3 / amax(x2)
        x4, xs4 = ops.nvfp4_quant(x2, gs)
        out = mm_fp4(
            x4,
            w4.t(),
            xs4.view(torch.uint8),
            ws4.view(torch.uint8).t(),
            1.0 / (gs * wgs),
            x.dtype,
            out=out,
            block_size=16,
            backend="cutlass",
        )
    return out.view(*x.shape[:-1], out.shape[-1])


class Graphed:
    """The head replayed from one CUDA graph per row bucket (powers of two, then ``rows``): its
    launches (the amax, the quantization, the GEMM) cost one replay on the host. Rows past the
    call's are zeroed, so the per-call amax sees the call's rows only; every bucket writes the
    one ``[rows, vocab]`` output, read before the next step's head runs on the same stream."""

    def __init__(self, head: tuple, hidden: int, vocab: int, rows: int, device):
        self.head = head
        self.buckets = [1 << i for i in range(rows.bit_length()) if 1 << i < rows] + [rows]
        self.inp = torch.zeros(rows, hidden, dtype=torch.bfloat16, device=device)
        self.out = torch.empty(rows, vocab, dtype=torch.bfloat16, device=device)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        for b in reversed(self.buckets):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                project(head, self.inp[:b], self.out[:b])
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                project(head, self.inp[:b], self.out[:b])
            self.graphs[b] = graph

    def __call__(self, x: torch.Tensor) -> torch.Tensor | None:
        m = x.shape[0]
        b = next((b for b in self.buckets if b >= m), None)
        if b is None:
            return None
        self.inp[:m].copy_(x)
        if b > m:
            self.inp[m:b].zero_()
        self.graphs[b].replay()
        return self.out[:m]


def tune(runner) -> None:
    """Profile the NVFP4 head GEMM's tactics for every token bucket up to ``max_num_seqs``
    (inside vLLM's autotune context)."""
    rows = runner.max_num_reqs
    for module in runner.get_model().modules():
        head = getattr(module, "_kohaku_head", None)
        if head is None or head[0] != "nvfp4":
            continue
        x = torch.randn(rows, module._kohaku_hidden, device=head[1].device, dtype=torch.bfloat16)
        buckets = fi_utils.flashinfer_get_hybrid_num_tokens_buckets(rows)
        with fi_utils.autotune(tuning_buckets=buckets, round_up=True):
            project(head, x)


def capture(runner) -> None:
    """Every quantized head's ``Graphed`` (``GRAPH``), for up to ``max_num_seqs`` rows."""
    if not GRAPH["on"]:
        return
    for module in runner.get_model().modules():
        head = getattr(module, "_kohaku_head", None)
        if head is not None:
            vocab = head[1].shape[0]
            with torch.inference_mode():
                module._kohaku_graphed = Graphed(
                    head, module._kohaku_hidden, vocab, runner.max_num_reqs, head[1].device
                )


def install(mode: str) -> None:
    MODE["name"] = mode
    original = logits_processor.LogitsProcessor._apply_head
    if getattr(original, "_kohaku", False):
        return

    def apply_head(self, lm_head, hidden_states, embedding_bias):
        if MODE["name"] is None or lm_head.weight.dtype not in (torch.bfloat16, torch.float16):
            return original(self, lm_head, hidden_states, embedding_bias)
        graphed = getattr(lm_head, "_kohaku_graphed", None)
        out = graphed(hidden_states) if graphed is not None else None
        if out is None:
            out = project(head_of(lm_head), hidden_states)
        return out + embedding_bias if embedding_bias is not None else out

    apply_head._kohaku = True
    logits_processor.LogitsProcessor._apply_head = apply_head

    # A tied head shares the embedding's Parameter: marked here, its weight is never freed
    tie = ParallelLMHead.tie_weights

    def tie_weights(self, embed_tokens):
        self._kohaku_tied = True
        return tie(self, embed_tokens)

    ParallelLMHead.tie_weights = tie_weights

    dummy_runs = kernel_warmup._run_flashinfer_autotune_dummy_runs

    def dummy_runs_and_head(runner, **kwargs) -> None:
        dummy_runs(runner, **kwargs)
        tune(runner)

    kernel_warmup._run_flashinfer_autotune_dummy_runs = dummy_runs_and_head

    # The graphs are captured once the autotune context has closed (its tactics then fixed)
    autotune = kernel_warmup.flashinfer_autotune

    def autotune_and_capture(runner, **kwargs) -> None:
        autotune(runner, **kwargs)
        capture(runner)

    kernel_warmup.flashinfer_autotune = autotune_and_capture
