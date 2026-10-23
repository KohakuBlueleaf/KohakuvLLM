"""``kohakuvllm.gdn.conv`` against vLLM's ``causal_conv1d_fn`` (the conv of vLLM's mixed GDN
step): conv output and updated conv states over decode/prefill layouts with and without initial
states, both conv-state layouts (dim-first, and width-first seen through a transpose), padded
token and sequence counts; CUDA-graph capture replayed after other layouts are written into the
same buffers (bitwise against eager); GPU time per call of both.

    python bench/gdn/conv_check.py [--dim 8192] [--tokens 4096] [--seqs 384]
"""

import argparse
import json
import sys

import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

from kohakuvllm.gdn.conv import update_state, varlen_conv

W = 4
LAYOUTS = {
    "mixed_120d_2p": (120, [900, 600]),
    "decode_250": (250, []),
    "prefill_short": (0, [1, 2, 3, 5]),
    "mixed_40d_5p": (40, [300] * 5),
    "mixed_300d_1p": (300, [1500]),
}


class Bench:
    """Padded persistent buffers and the conv-state store for one layout of the state."""

    def __init__(self, store0, dim_first, weight, bias, tokens, seqs):
        dev = store0.device
        self.store0, self.dim_first, self.weight, self.bias = store0, dim_first, weight, bias
        self.store = store0.clone()
        self.S = seqs
        self.x = torch.zeros(tokens, weight.shape[0], dtype=torch.bfloat16, device=dev)
        self.cu = torch.zeros(seqs + 1, dtype=torch.int32, device=dev)
        self.slot = torch.zeros(seqs, dtype=torch.int32, device=dev)
        self.init = torch.zeros(seqs, dtype=torch.int32, device=dev)
        self.n_tok = torch.zeros(1, dtype=torch.int32, device=dev)
        self.ar = torch.arange(tokens, dtype=torch.int32, device=dev)

    def view(self, store):
        return store if self.dim_first else store.transpose(-1, -2)

    def load(self, x, cul, slots, inits):
        t, S = x.shape[0], self.S
        self.store.copy_(self.store0)
        self.x.zero_()
        self.x[:t].copy_(x)
        self.cu.copy_(torch.tensor(cul + [t] * (S + 1 - len(cul)), dtype=torch.int32))
        self.slot.copy_(torch.tensor(slots + [0] * (S - len(slots)), dtype=torch.int32))
        self.init.copy_(torch.tensor(inits + [0] * (S - len(inits)), dtype=torch.int32))
        self.n_tok.fill_(t)

    def step(self):
        tok_seq = torch.searchsorted(self.cu[1:], self.ar, right=True).to(torch.int32)
        st = self.view(self.store)
        out = varlen_conv(
            self.x, st, self.weight, self.bias, tok_seq, self.cu, self.slot, self.init, self.n_tok
        )
        update_state(self.x, st, self.cu, self.slot, self.init, W)
        return out

    def vllm(self, store, x, cul, slots, inits):
        dev = x.device
        return causal_conv1d_fn(
            x.transpose(0, 1),
            self.weight,
            self.bias,
            conv_states=self.view(store),
            query_start_loc=torch.tensor(cul, dtype=torch.int32, device=dev),
            cache_indices=torch.tensor(slots, dtype=torch.int32, device=dev),
            has_initial_state=torch.tensor(inits, dtype=torch.bool, device=dev),
            activation="silu",
        ).transpose(0, 1)


def gpu_ms(fn, reps: int = 20) -> float:
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return round(a.elapsed_time(b) / reps, 4)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=8192)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--seqs", type=int, default=384)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    n_pool, dim = args.seqs + 32, args.dim
    weight = torch.randn(dim, W, device=dev, generator=g).bfloat16() * 0.5
    bias = torch.randn(dim, device=dev, generator=g).bfloat16() * 0.1
    perm = (torch.randperm(n_pool - 2, generator=torch.Generator().manual_seed(3)) + 1).tolist()
    report = {}
    for dim_first in (True, False):
        shape = (n_pool, dim, W - 1) if dim_first else (n_pool, W - 1, dim)
        store0 = torch.randn(*shape, device=dev, generator=g).bfloat16()
        b = Bench(store0, dim_first, weight, bias, args.tokens, args.seqs)
        graph, gout = None, None
        for li, (name, (n_dec, prefills)) in enumerate(LAYOUTS.items()):
            lens = [1] * n_dec + prefills
            t = sum(lens)
            slots = perm[li * 5 : li * 5 + len(lens)]
            inits = [1] * n_dec + [int(i % 2 == 0) for i in range(len(prefills))]
            x = torch.randn(t, dim, device=dev, generator=g).bfloat16()
            cul = [0]
            for m in lens:
                cul.append(cul[-1] + m)
            ref_store = store0.clone()
            ref = b.vllm(ref_store, x, cul, slots, inits)

            b.load(x, cul, slots, inits)
            out = b.step()
            torch.cuda.synchronize()
            o_eager, st_eager = out[:t].clone(), b.store.clone()
            idx = torch.tensor(slots, device=dev)
            rest = ~torch.isin(torch.arange(n_pool, device=dev), idx)
            d_out = (o_eager.float() - ref.float()).abs()
            d_st = (st_eager[idx].float() - ref_store[idx].float()).abs()
            row = {
                "out_err_max": float(d_out.max()),
                "out_err_mean": float(d_out.mean()),
                "out_ref_mean": float(ref.float().abs().mean()),
                "state_err_max": float(d_st.max()),
                "state_bitwise": bool(torch.equal(st_eager[idx], ref_store[idx])),
                "other_slots_equal": bool(torch.equal(st_eager[rest], store0[rest])),
                "pad_rows_zero": bool((out[t:] == 0).all()),
            }
            if graph is None:
                b.load(x, cul, slots, inits)
                b.step()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                b.load(x, cul, slots, inits)
                with torch.cuda.graph(graph):
                    gout = b.step()
            b.load(x, cul, slots, inits)
            graph.replay()
            torch.cuda.synchronize()
            row["graph_out_bitwise"] = bool(torch.equal(gout[:t], o_eager))
            row["graph_state_bitwise"] = bool(torch.equal(b.store, st_eager))
            row["ms_mine_graph"] = gpu_ms(graph.replay)
            row["ms_vllm"] = gpu_ms(
                lambda rs=ref_store, x=x, c=cul, s=slots, i=inits, b=b: b.vllm(rs, x, c, s, i)
            )
            key = f"{'dim_first' if dim_first else 'width_first'}/{name}"
            report[key] = row
            print(key, json.dumps(row), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
