"""The error profile of models against a reference, from ``prompt_logprobs.py`` files.

Per reply position: the actual token's log-probability difference, whether the top-1 token
agrees, and the KL divergence of the reference's top-k distribution against the model's over the
union of both top-k sets (a token outside a model's top k takes that model's k-th log-probability,
both renormalised). Prints the distributions (mean, p50, p90, p99, max) side by side, and the
positions with the largest difference together with the reference probability there, so a large
difference on an improbable token reads as what it is. No threshold: the profile is for review.

    python bench/precision/compare.py ref.pt a.pt b.pt [--worst 10]
"""

import argparse
import math
import sys

import torch


def kl_topk(ref_ids, ref_lp, ids, lp) -> float:
    union = sorted((set(ref_ids.tolist()) | set(ids.tolist())) - {-1})
    rmap = dict(zip(ref_ids.tolist(), ref_lp.tolist()))
    tmap = dict(zip(ids.tolist(), lp.tolist()))
    rfloor = ref_lp[ref_lp.isfinite()].min().item()
    tfloor = lp[lp.isfinite()].min().item()
    r = torch.tensor([rmap.get(t, rfloor) for t in union], dtype=torch.float64)
    t = torch.tensor([tmap.get(x, tfloor) for x in union], dtype=torch.float64)
    r, t = r.log_softmax(0), t.log_softmax(0)
    return float((r.exp() * (r - t)).sum())


def stats(x: torch.Tensor) -> str:
    q = torch.quantile(x, torch.tensor([0.5, 0.9, 0.99], dtype=x.dtype))
    return f"mean {x.mean():.4g}  p50 {q[0]:.4g}  p90 {q[1]:.4g}  p99 {q[2]:.4g}  max {x.max():.4g}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ref")
    p.add_argument("models", nargs="+")
    p.add_argument("--worst", type=int, default=10)
    args = p.parse_args()
    ref = torch.load(args.ref)
    print(f"reference {ref['model']}: {len(ref['actual'])} positions")
    for path in args.models:
        m = torch.load(path)
        assert len(m["actual"]) == len(ref["actual"]), "different positions"
        diff = (m["actual"] - ref["actual"]).abs()
        top1 = (m["top_ids"][:, 0] == ref["top_ids"][:, 0]).double()
        kl = torch.tensor(
            [
                kl_topk(ref["top_ids"][i], ref["top_lp"][i], m["top_ids"][i], m["top_lp"][i])
                for i in range(len(diff))
            ],
            dtype=torch.float64,
        )
        print(f"\n== {m['model']}")
        print(f"  |d logprob| actual token: {stats(diff)}")
        print(f"  KL(ref || model) top-k:   {stats(kl)}")
        print(f"  top-1 agreement: {100 * top1.mean():.2f}%")
        print("  largest |d logprob| (ref p, model p):")
        for i in diff.argsort(descending=True)[: args.worst].tolist():
            print(
                f"    pos {i:6d}  d {diff[i]:.3f}  ref p {math.exp(ref['actual'][i]):.4f}  "
                f"model p {math.exp(m['actual'][i]):.4f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
