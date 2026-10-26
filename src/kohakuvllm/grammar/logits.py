"""JSON-schema replies through a vLLM logits processor (Model Runner V2's interface), masks from
``kohakuvllm_grammar``.

A request opts in with ``SamplingParams(extra_args={"kohaku_json_schema": SCHEMA_TEXT})`` (over
HTTP: ``vllm_xargs``). Every step, one Rust call feeds each grammar request its new tokens and
returns the id of the mask its next token must satisfy; masks are memoised per (schema, clamped
state), so a mask is built once and copied to a GPU table once, and a step uploads only one id
per row. One kernel (the registry's ``grammar_mask``) then sets the masked logits to -inf.

The new tokens come from the runner's request state on the device: before each forward
(``apply_staged_writes``), a side stream ordered after the previous step's sampling copies every
slot's length and last committed token to pinned host memory, so ``apply`` waits for that copy
only, never for the forward in flight.

The package registers it as a vLLM logits-processor plugin (``vllm.logits_processors``), so
``vllm serve`` loads it with no flag; a request without the key never touches it. Given also by
``--logits-processors kohakuvllm.grammar:JsonSchemaLogits``, the second instance stays inert.

``WS`` (feature ``grammar_ws``): whitespace bytes allowed in each gap between JSON tokens (8;
0: compact JSON, where Qwen3.5 runs some replies to max_tokens). Spec decoding is not handled.
"""

import json
import logging
import time

import numpy as np
import torch
from kohakuvllm_grammar import Store
from transformers import AutoTokenizer
from vllm.v1.worker.gpu.sample.logits_processor import LogitsContext, LogitsProcessor

from kohakuvllm.grammar.tokens import eos_ids, token_bytes
from kohakuvllm.kernels import registry

KEY = "kohaku_json_schema"
STATS_EVERY_S = 60.0
log = logging.getLogger("vllm.kohakuvllm")
_BOUND: dict = {}
# Whitespace bytes allowed per gap (feature grammar_ws)
WS = {"bytes": 8}


def apply_masks(logits: torch.Tensor, table: torch.Tensor, ids: torch.Tensor) -> None:
    """``logits`` [B, V] in place: row ``r`` keeps the tokens of mask ``ids[r]`` (-1: all); the
    registry's ``grammar_mask``, resolved once per device and dtype."""
    key = (logits.device, logits.dtype)
    if key not in _BOUND:
        _BOUND[key] = registry.resolve("grammar_mask", logits.device, logits.dtype)
    _BOUND[key](logits, table, ids)


class JsonSchemaLogits(LogitsProcessor):
    _live = False  # one instance per process serves; another (loaded twice) stays inert

    def __init__(self, vllm_config, req_states):
        self.handles: dict[int, int] = {}  # request slot -> matcher handle
        self.inert = JsonSchemaLogits._live
        if self.inert:
            return
        JsonSchemaLogits._live = True
        mc = vllm_config.model_config
        tok = AutoTokenizer.from_pretrained(mc.tokenizer, trust_remote_code=mc.trust_remote_code)
        vocab = req_states.vocab_size
        ws = WS["bytes"]
        self.store = Store(token_bytes(tok, vocab), eos_ids(tok, mc.model), vocab, ws)
        self.rs = req_states
        self.device = req_states.device
        n = req_states.max_num_reqs
        self.grammars: dict[str, int] = {}
        self.slot_handle = np.full(n, -1, dtype=np.int64)  # the same, -1 for none
        self.seen = np.zeros(n, dtype=np.int64)  # output tokens fed per slot
        self.total = torch.empty(n, dtype=torch.int32, pin_memory=True)
        self.last = torch.empty(n, dtype=torch.int32, pin_memory=True)
        self.total_np, self.last_np = self.total.numpy(), self.last.numpy()
        self.side = torch.cuda.Stream(self.device)
        self.ready, self.copied = torch.cuda.Event(), torch.cuda.Event()
        # The per-row mask ids, staged in a ring so an in-flight copy is never overwritten
        self.staged = [torch.empty(n, dtype=torch.int32, pin_memory=True) for _ in range(4)]
        self.turn = 0
        self.table = torch.empty((256, self.store.words), dtype=torch.int32, device=self.device)
        self.uploaded = 0
        self.stats_at = time.monotonic()

    @classmethod
    def validate_params(cls, params) -> None:
        schema = (params.extra_args or {}).get(KEY)
        if schema is not None:
            if not isinstance(schema, str):
                raise ValueError(f"{KEY} must be a JSON schema as text")
            json.loads(schema)

    def add_request(self, req_idx: int, sampling_params) -> bool:
        if self.inert:
            return False
        old = self.handles.pop(req_idx, None)
        self.slot_handle[req_idx] = -1
        if old is not None:
            self.store.release(old)
        schema = (sampling_params.extra_args or {}).get(KEY)
        if schema is None:
            return False
        g = self.grammars.get(schema)
        if g is None:
            g = self.grammars[schema] = self.store.compile(schema)
        self.handles[req_idx] = self.slot_handle[req_idx] = self.store.matcher(g)
        self.seen[req_idx] = 0
        return True

    def apply_staged_writes(self) -> None:
        if not self.handles:
            return
        self.ready.record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.side):
            self.side.wait_event(self.ready)
            total = self.rs.total_len.gpu
            at = (total.long() - 1).clamp(min=0)[:, None]
            last = self.rs.all_token_ids.gpu.gather(1, at)[:, 0]
            self.total.copy_(total, non_blocking=True)
            self.last.copy_(last, non_blocking=True)
            self.copied.record(self.side)

    def _resumed(self, slot: int, done: int, out_len: int) -> list[int]:
        """Several tokens since the last step (a resumed request), read from the token rows."""
        start = int(self.rs.prompt_len.np[slot])
        return self.rs.all_token_ids.gpu[slot, start + done : start + out_len].tolist()

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        if not self.handles:
            return logits
        slots = ctx.idx_mapping_np
        if logits.shape[0] != len(slots):
            raise NotImplementedError("kohakuvllm grammar: speculative decoding")
        self.copied.synchronize()
        # One numpy pass over the batch (a Python loop over ~1000 rows cost milliseconds a step)
        hs = self.slot_handle[slots]
        rows = np.flatnonzero(hs >= 0)
        if not rows.size:
            return logits
        sl = slots[rows]
        out_len = self.total_np[sl].astype(np.int64) - self.rs.prompt_len.np[sl]
        done = self.seen[sl]
        if (out_len < done).any():
            raise RuntimeError("kohakuvllm grammar: a slot lost output tokens")
        self.seen[sl] = out_len
        step = (out_len - done).tolist()
        last = self.last_np[sl].tolist()
        news = [
            [t]
            if k == 1
            else []
            if k == 0
            else self._resumed(int(sl[i]), int(done[i]), int(out_len[i]))
            for i, (k, t) in enumerate(zip(step, last))
        ]
        handles = hs[rows].tolist()
        ids = self.store.step(handles, news)
        self._upload()
        self.turn = (self.turn + 1) % len(self.staged)
        dense = self.staged[self.turn][: logits.shape[0]]
        view = dense.numpy()
        view.fill(-1)
        view[rows] = ids
        apply_masks(logits, self.table, dense.to(self.device, non_blocking=True))
        if time.monotonic() - self.stats_at > STATS_EVERY_S:
            self.stats_at = time.monotonic()
            masks, mask_s, steps, step_s, tokens, rejected = self.store.stats()
            log.info(
                "kohakuvllm grammar: %d masks (%.2f s), %d steps (%.2f s), %d tokens, %d rejected",
                masks,
                mask_s,
                steps,
                step_s,
                tokens,
                rejected,
            )
        return logits

    def _upload(self) -> None:
        raw = self.store.take_new()
        if not raw:
            return
        # Pinned and asynchronous: a pageable copy waits for the forward in flight (the caching
        # host allocator keeps the source alive until the copy has run)
        new = torch.frombuffer(bytearray(raw), dtype=torch.int32).view(-1, self.store.words)
        new = new.pin_memory()
        end = self.uploaded + new.shape[0]
        if end > self.table.shape[0]:
            grown = torch.empty(
                (max(end, 2 * self.table.shape[0]), self.store.words),
                dtype=torch.int32,
                device=self.device,
            )
            grown[: self.uploaded] = self.table[: self.uploaded]
            self.table = grown
        self.table[self.uploaded : end].copy_(new, non_blocking=True)
        self.uploaded = end
