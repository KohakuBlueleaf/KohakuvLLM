"""FlashInfer's chunked GDN kernel, several calls captured in one CUDA graph (as vLLM's piecewise
graphs hold one per GDN layer): each call on its own buffers and token count; after replay,
every call's output bitwise against its eager run.

    python bench/gdn/fi_multi_capture.py [--calls 3]
"""

import argparse
import sys

import torch
import torch.nn.functional as F
from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_chunk

HK, HV, D = 16, 32, 128


class Call:
    def __init__(self, tokens: int, seed: int):
        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(seed)
        self.q = F.normalize(torch.randn(tokens, HK, D, device=dev, generator=g), dim=-1).bfloat16()
        self.k = F.normalize(torch.randn(tokens, HK, D, device=dev, generator=g), dim=-1).bfloat16()
        self.v = torch.randn(tokens, HV, D, device=dev, generator=g).bfloat16()
        self.alpha = torch.empty(tokens, HV, device=dev).uniform_(0.85, 1.0, generator=g)
        self.beta = torch.rand(tokens, HV, device=dev, generator=g)
        self.pool0 = (0.05 * torch.randn(8, HV, D, D, device=dev, generator=g)).bfloat16()
        self.pool = self.pool0.clone()
        self.out = torch.zeros(tokens, HV, D, dtype=torch.bfloat16, device=dev)
        half = tokens // 2
        self.cu = torch.tensor([0, half, tokens], dtype=torch.int32, device=dev)
        self.slots = torch.tensor([1, 2], dtype=torch.int32, device=dev)

    def run(self):
        fi_chunk(
            q=self.q,
            k=self.k,
            v=self.v,
            g=self.alpha,
            beta=self.beta,
            initial_state=self.pool,
            output_final_state=True,
            cu_seqlens=self.cu,
            output=self.out,
            output_state=self.pool,
            state_indices=self.slots,
            use_cp=False,
            backend="flashinfer",
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calls", type=int, default=3)
    args = p.parse_args()
    calls = [Call(256 * (i + 2), i) for i in range(args.calls)]
    eager = []
    for c in calls:
        c.pool.copy_(c.pool0)
        c.run()
        torch.cuda.synchronize()
        eager.append((c.out.clone(), c.pool.clone()))
    graph = torch.cuda.CUDAGraph()
    for c in calls:
        c.pool.copy_(c.pool0)
    with torch.cuda.graph(graph):
        for c in calls:
            c.run()
    for c in calls:
        c.pool.copy_(c.pool0)
        c.out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    for i, (c, (o, st)) in enumerate(zip(calls, eager)):
        print(
            f"call {i} tokens {c.q.shape[0]}: out bitwise {torch.equal(c.out, o)} "
            f"(max err {(c.out.float() - o.float()).abs().max():.3g}), "
            f"state bitwise {torch.equal(c.pool, st)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
