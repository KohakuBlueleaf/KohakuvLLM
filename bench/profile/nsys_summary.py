"""An nsys capture, summarised from its SQLite export (``nsys export --type sqlite``).

Over the steady window (the kernel span without its first and last ``--trim`` share):

- GPU busy: the union of kernel and memcpy time over the window; the idle gaps between kernels;
- GPU time per kernel class (FP8 GEMM, BF16 GEMM, attention, GDN, norm/elementwise, sampling,
  copies, other) and the top kernels by name;
- with NVTX ranges in the capture, GPU time per range: each kernel goes to the innermost range
  open on its launching thread at launch, layer indices folded to ``N``;
- with kernels from several processes (vLLM's engine core beside a host's own models), GPU time
  and busy share per process, and each one's top kernels;
- the CPU's CUDA API time (launches, syncs).

    python bench/profile/nsys_summary.py REPORT.sqlite [--trim 0.1] [--out summary.json]
"""

import argparse
import bisect
import json
import re
import sqlite3
import sys
from collections import defaultdict

CLASSES = [
    ("fp8_gemm", r"sm120.*Gemm|Sm120.*Gemm|scaled_mm|fp8.*gemm|gemm.*fp8|e4m3"),
    ("bf16_gemm", r"s16816gemm|gemm.*bf16|bf16.*gemm|nvjet|cublas|sm80_xmma_gemm|gemv"),
    ("attention", r"flash_fwd|flash::|fmha|attention|attn"),
    ("gdn", r"delta_rule|gated_delta|gdn|causal_conv1d|post_conv|rearrange_mixed|mamba"),
    ("sampling", r"sample|topk|top_k|top_p|gumbel|argmax|softmax|logits|bitmask"),
    ("copy", r"[Cc]opy|memcpy|Cat|index_|scatter|gather|fill|zeros"),
    ("norm_elementwise", r"triton_|norm|rms|elementwise|silu|gelu|rotary|rope|act_and_mul"),
]


def classify(name: str) -> str:
    for cls, pattern in CLASSES:
        if re.search(pattern, name):
            return cls
    return "other"


def union(intervals) -> int:
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s if cur_e is not None else 0)


def module_of(text: str) -> str:
    text = text.split(" ")[0].split("(")[0]
    return re.sub(r"\.\d+\.", ".N.", text)


def nvtx_modules(cur, kernels) -> dict:
    """GPU ns per module: kernel -> launch (correlationId) -> innermost NVTX range on its thread."""
    ranges = defaultdict(list)
    for s, e, text, tid in cur.execute(
        "select start, end, coalesce(text, ''), globalTid from NVTX_EVENTS "
        "where end is not null and end > start"
    ):
        ranges[tid].append((s, e, text))
    if not ranges:
        return {}
    launches = {
        corr: (tid, s)
        for corr, tid, s in cur.execute(
            "select correlationId, globalTid, start from CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    }
    index = {}
    for tid, rs in ranges.items():
        rs.sort()
        index[tid] = ([r[0] for r in rs], rs)
    out = defaultdict(int)
    for s, e, _, corr, _ in kernels:
        tid, t = launches.get(corr, (None, None))
        if tid not in index:
            out["(no range)"] += e - s
            continue
        starts, rs = index[tid]
        i = bisect.bisect_right(starts, t) - 1
        best = None
        # Nested ranges start later than their parents: the latest-starting range that still
        # contains the launch is the innermost one
        while i >= 0:
            rs_s, rs_e, text = rs[i]
            if rs_e >= t:
                best = text
                break
            if t - rs_s > 5_000_000_000:
                break
            i -= 1
        out[module_of(best) if best else "(no range)"] += e - s
    return dict(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sqlite")
    p.add_argument("--trim", type=float, default=0.1)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    db = sqlite3.connect(args.sqlite)
    cur = db.cursor()
    tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
    names = dict(cur.execute("select id, value from StringIds"))
    allk = cur.execute(
        "select start, end, demangledName, correlationId, globalPid "
        "from CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchall()
    t0, t1 = min(k[0] for k in allk), max(k[1] for k in allk)
    lo, hi = t0 + args.trim * (t1 - t0), t1 - args.trim * (t1 - t0)
    ker = [k for k in allk if k[0] >= lo and k[1] <= hi]
    mem = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        mem = [
            (s, e)
            for s, e in cur.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY")
            if s >= lo and e <= hi
        ]
    window = hi - lo
    busy_k = union([(s, e) for s, e, *_ in ker])
    starts = sorted((s, e) for s, e, *_ in ker)
    gaps, end = [], starts[0][1]
    for s, e in starts[1:]:
        if s > end:
            gaps.append(s - end)
        end = max(end, e)
    big = [g for g in gaps if g > 100_000]
    summary = {
        "window_s": round(window / 1e9, 3),
        "kernels": len(ker),
        "kernels_per_s": round(len(ker) / (window / 1e9)),
        "gpu_busy_pct": round(100 * busy_k / window, 1),
        "gpu_busy_with_memcpy_pct": round(
            100 * union([(s, e) for s, e, *_ in ker] + mem) / window, 1
        ),
        "gaps_over_100us": len(big),
        "gaps_over_100us_pct": round(100 * sum(big) / window, 1),
    }
    by_class, by_name = defaultdict(int), defaultdict(lambda: [0, 0])
    for s, e, n, *_ in ker:
        name = names.get(n, str(n))
        by_class[classify(name)] += e - s
        by_name[name][0] += e - s
        by_name[name][1] += 1
    total = sum(by_class.values())
    summary["class_pct"] = {
        c: round(100 * t / total, 1) for c, t in sorted(by_class.items(), key=lambda x: -x[1])
    }
    summary["top_kernels"] = [
        {"pct": round(100 * t / total, 1), "ms": round(t / 1e6, 1), "count": c, "name": n[:160]}
        for n, (t, c) in sorted(by_name.items(), key=lambda x: -x[1][0])[: args.top]
    ]
    pids = sorted({k[4] for k in ker})
    if len(pids) > 1:
        label = {}
        if "PROCESSES" in tables:
            label = {
                g: f"{name} ({pid})"
                for g, pid, name in cur.execute("select globalPid, pid, name from PROCESSES")
            }
        summary["processes"] = {}
        for g in pids:
            mine = [k for k in ker if k[4] == g]
            names_ns = defaultdict(int)
            for s, e, n, *_ in mine:
                names_ns[names.get(n, str(n))] += e - s
            t = sum(names_ns.values())
            summary["processes"][label.get(g, str(g))] = {
                "kernel_pct": round(100 * t / total, 1),
                "busy_pct": round(100 * union([(s, e) for s, e, *_ in mine]) / window, 1),
                "top": [
                    {"pct": round(100 * v / t, 1), "name": n[:120]}
                    for n, v in sorted(names_ns.items(), key=lambda x: -x[1])[:12]
                ],
            }
    if "NVTX_EVENTS" in tables:
        mods = nvtx_modules(cur, ker)
        if mods:
            mt = sum(mods.values())
            summary["module_pct"] = {
                m: round(100 * t / mt, 1)
                for m, t in sorted(mods.items(), key=lambda x: -x[1])[: args.top * 2]
            }
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        api = defaultdict(lambda: [0, 0])
        for s, e, n in cur.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
            if s >= lo and e <= hi:
                api[names.get(n, str(n))][0] += e - s
                api[names.get(n, str(n))][1] += 1
        summary["cuda_api_ms"] = {
            n: [round(t / 1e6, 1), c]
            for n, (t, c) in sorted(api.items(), key=lambda x: -x[1][0])[:10]
        }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
