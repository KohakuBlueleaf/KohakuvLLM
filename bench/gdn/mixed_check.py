"""One launch for a whole mixed GDN step: every sequence (decodes as length-1 sequences, prefill
chunks) through FlashInfer's SM120 chunked delta-rule kernel, reading and writing the state pool
in place by slot. On one card, over several step layouts written into the same padded buffers:

- agreement with an fp32 token-by-token recurrence (torch), beside the split production path
  (FLA recurrent kernel for decodes, FlashInfer chunked kernel over gathered fp32 states for
  prefills): output and final-state error profiles (|err| mean / p99 / max against the
  reference's own magnitude);
- CUDA-graph capture of the unified call at a padded sequence count, replayed after other layouts
  are written into the same buffers: bitwise against the eager unified call;
- GPU time per call: unified eager, unified graph replay, split path.

    python bench/gdn/mixed_check.py [--state bfloat16] [--tokens 2048] [--seqs 384]
"""

import argparse
import json
import sys

import torch
import torch.nn.functional as F
from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_chunk
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)

HK, HV, D = 16, 32, 128
LAYOUTS = {
    "mixed_120d_2p": (120, [900, 600]),
    "decode_250": (250, []),
    "prefill_1x2048": (0, [2048]),
    "mixed_40d_5p": (40, [300] * 5),
    "mixed_300d_1p": (300, [1500]),
}


def make_inputs(lens: list[int], dev) -> dict:
    t = sum(lens)
    g = torch.Generator(device=dev).manual_seed(len(lens) * 7919 + t)
    q = F.normalize(torch.randn(t, HK, D, device=dev, generator=g), dim=-1).bfloat16()
    k = F.normalize(torch.randn(t, HK, D, device=dev, generator=g), dim=-1).bfloat16()
    v = torch.randn(t, HV, D, device=dev, generator=g).bfloat16()
    alpha = torch.empty(t, HV, device=dev).uniform_(0.85, 1.0, generator=g)
    beta = torch.rand(t, HV, device=dev, generator=g)
    return {"q": q, "k": k, "v": v, "alpha": alpha, "beta": beta}


def reference(x: dict, lens: list[int], slots: list[int], pool: torch.Tensor, scale: float):
    """fp32 recurrence: S = a S; S += b (v - S k) k^T; o = scale S q (S is [V, K] per head)."""
    pool = pool.float().clone()
    out = torch.empty(x["v"].shape, dtype=torch.float32, device=x["v"].device)
    rep = HV // HK
    q = x["q"].float().repeat_interleave(rep, dim=1)
    k = x["k"].float().repeat_interleave(rep, dim=1)
    v, a, b = x["v"].float(), x["alpha"], x["beta"]
    start = 0
    for n, slot in zip(lens, slots):
        s = pool[slot]
        for t in range(start, start + n):
            s = s * a[t, :, None, None]
            sk = torch.einsum("hvk,hk->hv", s, k[t])
            s = s + torch.einsum("hv,hk->hvk", b[t, :, None] * (v[t] - sk), k[t])
            out[t] = scale * torch.einsum("hvk,hk->hv", s, q[t])
        pool[slot] = s
        start += n
    return out, pool


def profile(err: torch.Tensor, ref: torch.Tensor) -> dict:
    e = err.abs().flatten().float()
    r = ref.abs().flatten().float()
    return {
        "err_mean": float(e.mean()),
        "err_p99": float(e[:: -(-e.numel() // 2**23)].quantile(0.99)),
        "err_max": float(e.max()),
        "ref_mean": float(r.mean()),
        "rel_mean": float(e.mean() / r.mean().clamp_min(1e-30)),
    }


class Unified:
    """Padded persistent buffers for one T bucket and one sequence cap; ``load`` writes a layout
    in, ``call`` runs the single FlashInfer launch over them."""

    def __init__(self, tokens: int, seqs: int, pool: torch.Tensor, scratch: int):
        dev = pool.device
        self.pool, self.scratch, self.tokens, self.seqs = pool, scratch, tokens, seqs
        self.q = torch.zeros(tokens, HK, D, dtype=torch.bfloat16, device=dev)
        self.k = torch.zeros_like(self.q)
        self.v = torch.zeros(tokens, HV, D, dtype=torch.bfloat16, device=dev)
        self.alpha = torch.ones(tokens, HV, device=dev)
        self.beta = torch.zeros(tokens, HV, device=dev)
        self.out = torch.zeros(tokens, HV, D, dtype=torch.bfloat16, device=dev)
        self.cu = torch.zeros(seqs + 1, dtype=torch.int32, device=dev)
        self.slots = torch.full((seqs,), scratch, dtype=torch.int32, device=dev)

    def load(self, x: dict, lens: list[int], slots: list[int]) -> None:
        t, n = sum(lens), len(lens)
        for name in ("q", "k", "v", "alpha", "beta"):
            buf = getattr(self, name)
            buf.zero_() if name != "alpha" else buf.fill_(1.0)
            buf[:t].copy_(x[name])
        cu = [0]
        for m in lens:
            cu.append(cu[-1] + m)
        cu += [t] * (self.seqs - n)
        self.cu.copy_(torch.tensor(cu, dtype=torch.int32))
        self.slots.copy_(torch.tensor(slots + [self.scratch] * (self.seqs - n), dtype=torch.int32))

    def call(self, scale: float) -> None:
        fi_chunk(
            q=self.q,
            k=self.k,
            v=self.v,
            g=self.alpha,
            beta=self.beta,
            scale=scale,
            initial_state=self.pool,
            output_final_state=True,
            cu_seqlens=self.cu,
            output=self.out,
            output_state=self.pool,
            state_indices=self.slots,
            use_cp=False,
            backend="flashinfer",
        )


class Hybrid(Unified):
    """The same padded buffers, decodes through the FLA recurrent kernel and prefills through
    FlashInfer's chunked kernel, both over the full buffers with their own padded ``cu_seqlens``
    and slots (padding: zero-length sequences on slot 0, which the recurrent kernel skips and the
    chunked kernel never writes); the outputs are joined by a token mask."""

    def __init__(self, tokens: int, seqs: int, pool: torch.Tensor, scratch: int):
        super().__init__(tokens, seqs, pool, scratch)
        dev = pool.device
        self.log_alpha = torch.zeros(tokens, HV, device=dev)
        self.cu_dec = torch.zeros(seqs + 1, dtype=torch.int32, device=dev)
        self.slots_dec = torch.zeros(seqs, dtype=torch.int32, device=dev)
        self.cu_pre = torch.zeros(seqs + 1, dtype=torch.int32, device=dev)
        self.slots_pre = torch.zeros(seqs, dtype=torch.int32, device=dev)
        self.is_dec = torch.zeros(tokens, 1, 1, dtype=torch.bool, device=dev)
        self.out_pre = torch.zeros_like(self.out)

    def load_split(self, x: dict, lens: list[int], slots: list[int], n_dec: int) -> None:
        self.load(x, lens, slots)
        self.log_alpha.copy_(self.alpha.log())
        t, n_pre = sum(lens), len(lens) - n_dec
        cu_dec = list(range(n_dec + 1)) + [n_dec] * (self.seqs - n_dec)
        cu_pre = [n_dec]
        for m in lens[n_dec:]:
            cu_pre.append(cu_pre[-1] + m)
        cu_pre += [t] * (self.seqs - n_pre)
        self.cu_dec.copy_(torch.tensor(cu_dec, dtype=torch.int32))
        self.cu_pre.copy_(torch.tensor(cu_pre, dtype=torch.int32))
        self.slots_dec.copy_(torch.tensor(slots[:n_dec] + [0] * (self.seqs - n_dec)))
        self.slots_pre.copy_(torch.tensor(slots[n_dec:] + [0] * (self.seqs - n_pre)))
        self.is_dec.copy_((torch.arange(self.tokens) < n_dec).view(-1, 1, 1))

    def call(self, scale: float) -> None:
        o_dec, _ = fused_recurrent_gated_delta_rule(
            q=self.q.unsqueeze(0),
            k=self.k.unsqueeze(0),
            v=self.v.unsqueeze(0),
            g=self.log_alpha.unsqueeze(0),
            beta=self.beta.unsqueeze(0),
            scale=scale,
            initial_state=self.pool,
            inplace_final_state=True,
            cu_seqlens=self.cu_dec,
            ssm_state_indices=self.slots_dec,
        )
        fi_chunk(
            q=self.q,
            k=self.k,
            v=self.v,
            g=self.alpha,
            beta=self.beta,
            scale=scale,
            initial_state=self.pool,
            output_final_state=True,
            cu_seqlens=self.cu_pre,
            output=self.out_pre,
            output_state=self.pool,
            state_indices=self.slots_pre,
            use_cp=False,
            backend="flashinfer",
        )
        torch.where(self.is_dec, o_dec.squeeze(0), self.out_pre, out=self.out)


def split(x: dict, lens: list[int], slots: list[int], pool: torch.Tensor, scale: float, n_dec):
    """vLLM's production split: decodes through the FLA recurrent kernel in place, prefills
    through FlashInfer over gathered fp32 states scattered back."""
    dev = pool.device
    out = torch.empty_like(x["v"])
    if n_dec:
        o, _ = fused_recurrent_gated_delta_rule(
            q=x["q"][:n_dec].unsqueeze(0),
            k=x["k"][:n_dec].unsqueeze(0),
            v=x["v"][:n_dec].unsqueeze(0),
            g=x["alpha"][:n_dec].log().unsqueeze(0),
            beta=x["beta"][:n_dec].to(torch.bfloat16).unsqueeze(0),
            scale=scale,
            initial_state=pool,
            inplace_final_state=True,
            cu_seqlens=torch.arange(n_dec + 1, dtype=torch.int32, device=dev),
            ssm_state_indices=torch.tensor(slots[:n_dec], dtype=torch.int32, device=dev),
        )
        out[:n_dec] = o.squeeze(0)
    if len(lens) > n_dec:
        pl = lens[n_dec:]
        cu = torch.tensor([0] + list(torch.tensor(pl).cumsum(0)), dtype=torch.int64, device=dev)
        idx = torch.tensor(slots[n_dec:], dtype=torch.long, device=dev)
        o, st = fi_chunk(
            q=x["q"][n_dec:].contiguous(),
            k=x["k"][n_dec:].contiguous(),
            v=x["v"][n_dec:].contiguous(),
            g=x["alpha"][n_dec:].contiguous(),
            beta=x["beta"][n_dec:].contiguous(),
            scale=scale,
            initial_state=pool[idx].float(),
            output_final_state=True,
            cu_seqlens=cu,
            use_cp=False,
            backend="flashinfer",
        )
        out[n_dec:] = o
        pool[idx] = st.to(pool.dtype)
    return out


def gpu_ms(fn, reps: int = 30) -> float:
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--seqs", type=int, default=384)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    dev = torch.device("cuda")
    sdt = getattr(torch, args.state)
    scale = D**-0.5
    n_pool = args.seqs + 64
    scratch = n_pool - 1
    g = torch.Generator(device=dev).manual_seed(0)
    pool0 = (0.05 * torch.randn(n_pool, HV, D, D, device=dev, generator=g)).to(sdt)
    modes = {
        "unified": Unified(args.tokens, args.seqs, pool0.clone(), scratch),
        "hybrid": Hybrid(args.tokens, args.seqs, pool0.clone(), scratch),
    }
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    # Slot 0 is the null slot (the recurrent kernel skips it); the last slot is the scratch slot
    perm = (torch.randperm(n_pool - 2, generator=torch.Generator().manual_seed(1)) + 1).tolist()
    report = {"state": args.state, "tokens": args.tokens, "seqs": args.seqs, "layouts": {}}

    for li, (name, (n_dec, prefills)) in enumerate(LAYOUTS.items()):
        lens = [1] * n_dec + prefills
        if sum(lens) > args.tokens or len(lens) > args.seqs:
            continue
        slots = perm[li * 7 : li * 7 + len(lens)]
        x = make_inputs(lens, dev)
        ref_o, ref_pool = reference(x, lens, slots, pool0, scale)
        idx = torch.tensor(slots, device=dev)
        rest = ~torch.isin(torch.arange(n_pool, device=dev), idx)
        t = sum(lens)
        row = {"decodes": n_dec, "prefills": prefills}

        sp_pool = pool0.clone()
        s_o = split(x, lens, slots, sp_pool, scale, n_dec)
        row["split_out"] = profile(s_o.float() - ref_o, ref_o)
        row["split_state"] = profile(sp_pool[idx].float() - ref_pool[idx], ref_pool[idx])
        row["ms_split"] = round(
            gpu_ms(
                lambda x=x, lens=lens, slots=slots, pool=sp_pool, n=n_dec: split(
                    x, lens, slots, pool, scale, n
                )
            ),
            4,
        )

        for mode, m in modes.items():

            def load(m=m, x=x, lens=lens, slots=slots, n=n_dec):
                m.pool.copy_(pool0)
                m.load_split(x, lens, slots, n) if isinstance(m, Hybrid) else m.load(x, lens, slots)

            load()
            m.call(scale)
            torch.cuda.synchronize()
            o, pool = m.out[:t].clone(), m.pool.clone()
            row[f"{mode}_out"] = profile(o.float() - ref_o, ref_o)
            row[f"{mode}_state"] = profile(pool[idx].float() - ref_pool[idx], ref_pool[idx])
            row[f"{mode}_other_slots_equal"] = bool(torch.equal(pool[rest], pool0[rest]))
            if mode not in graphs:
                m.call(scale)
                torch.cuda.synchronize()
                graphs[mode] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[mode]):
                    m.call(scale)
                row[f"{mode}_graph_captured_here"] = True
            load()
            graphs[mode].replay()
            torch.cuda.synchronize()
            row[f"{mode}_graph_out_bitwise"] = bool(torch.equal(m.out[:t], o))
            row[f"{mode}_graph_state_bitwise"] = bool(torch.equal(m.pool, pool))
            row[f"ms_{mode}_eager"] = round(gpu_ms(lambda m=m: m.call(scale)), 4)
            row[f"ms_{mode}_graph"] = round(gpu_ms(graphs[mode].replay), 4)
        report["layouts"][name] = row
        print(name, json.dumps(row), flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
