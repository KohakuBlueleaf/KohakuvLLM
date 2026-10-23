"""A mixed decode/prefill GDN step that CUDA graphs capture: vLLM's Qwen3.5 / Qwen3-Next GDN core
then runs inside the piecewise graphs instead of eagerly between them.

Per step, the metadata builder writes the batch into persistent padded buffers (one set per GDN
KV-cache group, since each group has its own state slots): sequence boundaries over all, decode
and prefill sequences, state slots, initial-state flags, the real and decode token counts. The
layer's core then runs a fixed launch sequence over the padded token count, reading only those
buffers:

1. the causal conv over every token (``conv.varlen_conv``), then each sequence's conv state
   (``conv.update_state``);
2. split, L2 norm and gating (vLLM's ``fused_post_conv_prep``);
3. the recurrent state of prefills without an initial state zeroed in place;
4. decodes through FLA's recurrent kernel and prefills through FlashInfer's chunked kernel, both
   reading and writing the state pool by slot (padding: zero-length sequences on the null slot 0,
   which the recurrent kernel skips and the chunked kernel never writes);
5. the two outputs joined by a token mask.

Decode-only steps run eagerly or as FULL graphs keep vLLM's own path; PIECEWISE graphs always
hold the hybrid step (a piecewise graph serves every batch of its token count, mixed or not).
Speculative decoding is not handled. ``install`` swaps the core and the builder and takes the
core out of the splitting ops.
"""

import json
import logging
import os

import torch
from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_chunk
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from vllm.v1.attention.backends import gdn_attn

from kohakuvllm.gdn.conv import update_state, varlen_conv

log = logging.getLogger("kohakuvllm")
CORE_OP = "vllm::qwen_gdn_attention_core"
HOST_RING = 8
BUFFERS: dict[str, "Buffers"] = {}
BUILT: list["Buffers"] = []
CHECK_LOG = None
DEBUG = int(os.environ.get("KOHAKU_GDN_DEBUG", "0"))
_ORIGINAL_CORE = QwenGatedDeltaNetAttention._forward_core
_ORIGINAL_BUILDER = gdn_attn.GDNAttentionBackend.get_builder_cls


class Buffers:
    """One GDN KV-cache group's step metadata, at fixed addresses, for ``seqs`` sequences and
    ``tokens`` tokens at most. int32 block: ``cu_all`` / ``cu_dec`` / ``cu_pre`` [seqs + 1] each,
    ``n_tok``, ``n_dec``, ``live`` [seqs] (the sequence has tokens this step); slots: ``slot_all``
    / ``slot_dec`` / ``slot_pre`` / ``init`` [seqs] int32, ``zero_slot`` [seqs] int64."""

    def __init__(self, seqs: int, tokens: int, device: torch.device):
        self.seqs, self.tokens = seqs, tokens
        self.builders, self.fills = 0, 0
        n = 3 * (seqs + 1) + 2 + seqs
        self.packed = torch.zeros(n, dtype=torch.int32, device=device)
        self.host = [torch.zeros(n, dtype=torch.int32).pin_memory() for _ in range(HOST_RING)]
        self.turn = 0
        s1 = seqs + 1
        self.cu_all = self.packed[0:s1]
        self.cu_dec = self.packed[s1 : 2 * s1]
        self.cu_pre = self.packed[2 * s1 : 3 * s1]
        self.n_tok = self.packed[3 * s1 : 3 * s1 + 1]
        self.n_dec = self.packed[3 * s1 + 1 : 3 * s1 + 2]
        self.live = self.packed[3 * s1 + 2 :]
        self.slot_all = torch.zeros(seqs, dtype=torch.int32, device=device)
        self.slot_dec = torch.zeros_like(self.slot_all)
        self.slot_pre = torch.zeros_like(self.slot_all)
        self.init = torch.zeros_like(self.slot_all)
        self.zero_slot = torch.zeros(seqs, dtype=torch.int64, device=device)
        self.arange = torch.arange(tokens, dtype=torch.int32, device=device)

    def fill(self, qsl_cpu: torch.Tensor, n_dec: int, slots: torch.Tensor, init) -> None:
        """``qsl_cpu`` [n + 1] the step's query start locations (decodes first), ``slots`` [n]
        their state slots (device), ``init`` [n] has-initial-state (device bool) or None when
        every sequence has one."""
        S = self.seqs
        self.fills += 1
        n = qsl_cpu.shape[0] - 1
        total = int(qsl_cpu[n])
        host = self.host[self.turn]
        self.turn = (self.turn + 1) % HOST_RING
        s1 = S + 1
        host[: n + 1] = qsl_cpu
        host[n + 1 : s1] = total
        host[s1 : s1 + n_dec + 1] = torch.arange(n_dec + 1, dtype=torch.int32)
        host[s1 + n_dec + 1 : 2 * s1] = n_dec
        n_pre = n - n_dec
        host[2 * s1 : 2 * s1 + n_pre + 1] = qsl_cpu[n_dec:]
        host[2 * s1 + n_pre + 1 : 3 * s1] = total
        host[3 * s1] = total
        host[3 * s1 + 1] = n_dec
        live = host[3 * s1 + 2 :]
        live.zero_()
        live[:n] = (qsl_cpu[1:] > qsl_cpu[:-1]).to(torch.int32)
        self.packed.copy_(host, non_blocking=True)

        self.slot_all.zero_()
        self.slot_all[:n].copy_(slots[:n])
        self.slot_dec.zero_()
        self.slot_dec[:n_dec].copy_(slots[:n_dec])
        self.slot_pre.zero_()
        self.slot_pre[:n_pre].copy_(slots[n_dec:n])
        if init is None:
            self.init.fill_(1)
        else:
            self.init.zero_()
            self.init[:n].copy_(init[:n])
        # Only sequences with tokens this step start from a zero state: the requests vLLM pads a
        # graph batch with have none, and their slot entries can be live requests' slots
        self.zero_slot.copy_(self.slot_all)
        self.zero_slot.masked_fill_(self.init.bool() | (self.live == 0), 0)
        if DEBUG and self.fills <= DEBUG:
            s = slots[:n].long()
            log.warning(
                "kohakuvllm GDN fill %d: reqs %d decodes %d tokens %d live %d slots [%d, %d] "
                "init %s qsl %s",
                self.fills,
                n,
                n_dec,
                total,
                int(live.sum()),
                int(s.min()) if n else -1,
                int(s.max()) if n else -1,
                "all" if init is None else int(init[:n].sum()),
                qsl_cpu[: min(n + 1, 12)].tolist(),
            )


def hybrid_core(self, mixed_qkv, b, a, core_attn_out) -> None:
    """The fixed launch sequence of the module docstring over this layer group's buffers."""
    # Inside the compiled graph the inputs can arrive as strided views; the kernels below take
    # rows with a contiguous inner dimension
    mixed_qkv, a, b = mixed_qkv.contiguous(), a.contiguous(), b.contiguous()
    T = mixed_qkv.shape[0]
    buf = BUFFERS[self.prefix]
    kv = self.kv_cache
    conv_state = kv[0] if is_conv_state_dim_first() else kv[0].transpose(-1, -2)
    ssm_state = kv[1]
    weight = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))

    tok_seq = torch.searchsorted(buf.cu_all[1:], buf.arange[:T], right=True).to(torch.int32)
    conv_out = varlen_conv(
        mixed_qkv,
        conv_state,
        weight,
        self.conv1d.bias,
        tok_seq,
        buf.cu_all,
        buf.slot_all,
        buf.init,
        buf.n_tok,
        silu=self.activation in ("silu", "swish", True),
    )
    update_state(mixed_qkv, conv_state, buf.cu_all, buf.slot_all, buf.init, weight.shape[1])

    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv_out,
        a=a,
        b=b,
        A_log=self.A_log,
        dt_bias=self.dt_bias,
        num_k_heads=self.num_k_heads // self.tp_size,
        head_k_dim=self.head_k_dim,
        head_v_dim=self.head_v_dim,
        apply_l2norm=True,
        output_g_exp=False,
    )
    ssm_state.index_fill_(0, buf.zero_slot, 0)
    scale = self.head_k_dim**-0.5
    o_dec, _ = fused_recurrent_gated_delta_rule(
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        scale=scale,
        initial_state=ssm_state,
        inplace_final_state=True,
        cu_seqlens=buf.cu_dec,
        ssm_state_indices=buf.slot_dec,
    )
    o_pre = torch.empty_like(v)
    fi_chunk(
        q=q,
        k=k,
        v=v,
        g=g.exp(),
        beta=beta,
        scale=scale,
        initial_state=ssm_state,
        output_final_state=True,
        cu_seqlens=buf.cu_pre,
        output=o_pre,
        output_state=ssm_state,
        state_indices=buf.slot_pre,
        use_cp=False,
        backend="flashinfer",
    )
    is_dec = (buf.arange[:T] < buf.n_dec).view(T, 1, 1)
    torch.where(is_dec, o_dec.squeeze(0), o_pre, out=core_attn_out[:T])


def stat(err: torch.Tensor, ref: torch.Tensor) -> list[float]:
    e, r = err.abs().float(), ref.abs().float()
    return [float(e.max()), float(e.mean()), float(r.mean())]


def check_core(self, mixed_qkv, b, a, core_attn_out, md) -> None:
    """vLLM's core, then the hybrid one from the same states on the same inputs: one JSON line of
    [max, mean |err|, mean |ref|] for the output, conv states and recurrent states of the step's
    slots; vLLM's results are kept, so the run stays on vLLM's trajectory."""
    n_tok = md.num_actual_tokens
    idx = md.non_spec_state_indices_tensor.long()
    idx = idx[idx > 0].unique()
    kv = self.kv_cache
    conv0, ssm0 = kv[0][idx].clone(), kv[1][idx].clone()
    out_ref = core_attn_out.clone()
    _ORIGINAL_CORE(self, mixed_qkv, b, a, out_ref)
    conv1, ssm1 = kv[0][idx].clone(), kv[1][idx].clone()
    kv[0][idx], kv[1][idx] = conv0, ssm0
    hybrid_core(self, mixed_qkv, b, a, core_attn_out)
    row = {
        "layer": self.prefix,
        "decodes": md.num_decodes,
        "prefills": md.num_prefills,
        "tokens": n_tok,
        "out": stat(core_attn_out[:n_tok] - out_ref[:n_tok], out_ref[:n_tok]),
        "conv": stat(kv[0][idx] - conv1, conv1),
        "ssm": stat(kv[1][idx].float() - ssm1.float(), ssm1),
    }
    CHECK_LOG.write(json.dumps(row) + "\n")
    CHECK_LOG.flush()
    kv[0][idx], kv[1][idx] = conv1, ssm1
    core_attn_out.copy_(out_ref)


def forward_core(self, mixed_qkv, b, a, core_attn_out):
    ctx = get_forward_context()
    md = ctx.attn_metadata
    md = md.get(self.prefix) if isinstance(md, dict) else None
    # vLLM's own core: before the KV cache and the builders exist (memory profiling), and for
    # speculative steps; for decode-only steps run eagerly or captured as FULL graphs (vLLM's
    # persistent decode buffers). A PIECEWISE graph is replayed for any batch of its size, so
    # it always captures the hybrid step, whatever its dummy capture batch is.
    piecewise = ctx.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
    decode_only = md is not None and md.num_prefills == 0 and md.num_decodes > 0
    if (
        self.prefix not in BUFFERS
        or (md is not None and md.spec_sequence_masks is not None)
        or (decode_only and not piecewise)
    ):
        return _ORIGINAL_CORE(self, mixed_qkv, b, a, core_attn_out)
    if CHECK_LOG is not None and md is not None:
        return check_core(self, mixed_qkv, b, a, core_attn_out, md)
    hybrid_core(self, mixed_qkv, b, a, core_attn_out)


class HybridBuilder(gdn_attn.GDNAttentionMetadataBuilder):
    """vLLM's GDN builder, plus this group's ``Buffers`` filled on every step."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        sched = vllm_config.scheduler_config
        # vLLM may build a group's builder again after the graphs are captured; the graphs hold
        # the first buffers' addresses, so a later builder fills those same buffers
        self.kohaku = BUFFERS.get(layer_names[0])
        if self.kohaku is None:
            self.kohaku = Buffers(sched.max_num_seqs, sched.max_num_batched_tokens, device)
            BUILT.append(self.kohaku)
        for name in layer_names:
            BUFFERS[name] = self.kohaku
        self.kohaku.builders += 1
        log.warning(
            "kohakuvllm GDN buffers: group of %d layers (%s...), builder %d, %d buffer sets",
            len(layer_names),
            layer_names[0],
            self.kohaku.builders,
            len(BUILT),
        )

    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        md = super().build(common_prefix_len, common_attn_metadata, *args, **kwargs)
        if md.spec_sequence_masks is None:
            m = common_attn_metadata
            n = m.num_reqs
            init = md.has_initial_state if md.num_prefills > 0 else None
            self.kohaku.fill(
                m.query_start_loc_cpu[: n + 1].to(torch.int32),
                md.num_decodes,
                md.non_spec_state_indices_tensor,
                init,
            )
        return md


def install(check_path: str | None = None) -> None:
    """Swap in the hybrid core and builder; the core leaves the default splitting ops. With
    ``check_path`` (eager engines only), every mixed step runs both cores (``check_core``) and
    logs their differences there, one line per layer and step, per process."""
    global CHECK_LOG
    if check_path:
        CHECK_LOG = open(f"{check_path}.{os.getpid()}.jsonl", "a", encoding="utf-8")  # noqa: SIM115
    QwenGatedDeltaNetAttention._forward_core = forward_core
    gdn_attn.GDNAttentionBackend.get_builder_cls = staticmethod(lambda: HybridBuilder)
    if CORE_OP in CompilationConfig._attention_ops:
        CompilationConfig._attention_ops = [
            op for op in CompilationConfig._attention_ops if op != CORE_OP
        ]
