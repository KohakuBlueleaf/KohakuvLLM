"""Attribute GPU idle time and non-GEMM kernel time in an Nsight Systems sqlite export.

For every gap between busy GPU intervals (kernels, memcpy, memset) it records the step
it falls in (vLLM's `execute_context_C(...)_generation_G(...)` NVTX ranges: C > 0 is a
step with prefill, C == 0 decode only), the operations on both sides, the API that
launched the next operation, and how late the host issued that launch. Host lateness
is the time from the gap start to the start of the launching runtime call; for a late
launch, the runtime calls the launching thread made inside the gap are summed by name.

Usage: gaps.py PROFILE.sqlite OUT.json [--window-start-s S]
"""

import argparse
import bisect
import collections
import json
import re
import sqlite3

STEP = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
BUCKETS_US = [0, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 1e12]


def load(c, strings):
    ops = []
    for s, e, name, corr, graph in c.execute(
        "select start, end, shortName, correlationId, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        ops.append((s, e, strings.get(name, "?"), corr, graph is not None))
    for s, e, corr, graph, kind in c.execute(
        "select start, end, correlationId, graphNodeId, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"
    ):
        ops.append((s, e, f"memcpy[{kind}]", corr, graph is not None))
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        for s, e, corr, graph in c.execute(
            "select start, end, correlationId, graphNodeId from CUPTI_ACTIVITY_KIND_MEMSET"
        ):
            ops.append((s, e, "memset", corr, graph is not None))
    ops.sort()
    return ops


def steps(c, strings):
    rows = []
    for s, e, text in c.execute(
        "select start, end, coalesce(text, textId) from NVTX_EVENTS where end is not null"
    ):
        text = strings.get(text, text) if isinstance(text, int) else text
        m = STEP.match(text or "")
        if m:
            rows.append((s, e, int(m.group(2)), int(m.group(4))))
    rows.sort()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("out")
    ap.add_argument("--window-start-s", type=float, default=None)
    args = ap.parse_args()

    c = sqlite3.connect(args.db)
    strings = dict(c.execute("select id, value from StringIds"))
    ops = load(c, strings)
    step_rows = steps(c, strings)

    runtime = {}
    by_tid = collections.defaultdict(list)
    for s, e, name, tid, corr in c.execute(
        "select start, end, nameId, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        name = strings.get(name, "?")
        runtime[corr] = (s, e, name, tid)
        by_tid[tid].append((s, e, name))
    for rows in by_tid.values():
        rows.sort()
    tid_starts = {tid: [r[0] for r in rows] for tid, rows in by_tid.items()}

    # The benchmark window starts at the first decode step (graph capture and warm-up
    # come before it) unless given explicitly.
    t0 = step_rows[0][0] if step_rows else ops[0][0]
    if args.window_start_s is not None:
        t0 = ops[0][0] + int(args.window_start_s * 1e9)
    ops = [o for o in ops if o[0] >= t0]
    t1 = max(o[1] for o in ops)
    wall = t1 - t0

    step_starts = [r[0] for r in step_rows]

    def step_of(t):
        i = bisect.bisect_right(step_starts, t) - 1
        if i >= 0 and step_rows[i][1] >= t:
            ctx = step_rows[i][2]
            return "prefill" if ctx > 0 else "decode"
        return "between_steps"

    busy = 0
    gaps = []
    end = ops[0][1]
    last = ops[0]
    for o in ops[1:]:
        if o[0] > end:
            gaps.append((end, o[0], last, o))
        if o[1] > end:
            end = o[1]
            last = o
    for o in ops:
        busy += o[1] - o[0]
    merged_busy = wall - sum(g[1] - g[0] for g in gaps)

    hist = collections.Counter()
    hist_t = collections.Counter()
    by_step = collections.Counter()
    by_api = collections.Counter()
    by_pair = collections.Counter()
    pair_n = collections.Counter()
    host_late = 0
    host_late_calls = collections.Counter()
    gpu_queue = 0
    for gs, ge, prev, nxt in gaps:
        d = ge - gs
        us = d / 1e3
        b = next(i for i in range(len(BUCKETS_US) - 1) if us < BUCKETS_US[i + 1])
        label = f"{BUCKETS_US[b]:g}-{BUCKETS_US[b + 1]:g}us"
        hist[label] += 1
        hist_t[label] += d
        by_step[step_of(gs)] += d
        rt = runtime.get(nxt[3])
        api = rt[2] if rt else "unknown"
        by_api[(api, "graph" if nxt[4] else "eager")] += d
        key = (prev[2][:70], nxt[2][:70], api)
        by_pair[key] += d
        pair_n[key] += 1
        if rt and rt[0] > gs:
            late = min(rt[0], ge) - gs
            host_late += late
            rows = by_tid[rt[3]]
            i = bisect.bisect_left(tid_starts[rt[3]], gs)
            while i < len(rows) and rows[i][0] < rt[0]:
                s, e, name = rows[i]
                host_late_calls[name] += min(e, rt[0]) - max(s, gs)
                i += 1
        else:
            gpu_queue += d

    kern = collections.Counter()
    kern_n = collections.Counter()
    for o in ops:
        kern[o[2]] += o[1] - o[0]
        kern_n[o[2]] += 1

    n_steps = collections.Counter()
    step_wall = collections.Counter()
    for s, e, ctx, _ in step_rows:
        if s >= t0:
            k = "prefill" if ctx > 0 else "decode"
            n_steps[k] += 1
            step_wall[k] += e - s

    def sec(x):
        return round(x / 1e9, 3)

    report = {
        "wall_s": sec(wall),
        "gpu_busy_s": sec(merged_busy),
        "gpu_busy_pct": round(100 * merged_busy / wall, 2),
        "idle_s": sec(wall - merged_busy),
        "gaps": len(gaps),
        "gap_hist": {k: {"n": hist[k], "s": sec(hist_t[k])} for k in hist},
        "idle_by_step_kind_s": {k: sec(v) for k, v in by_step.items()},
        "steps": {k: {"n": n_steps[k], "wall_s": sec(step_wall[k])} for k in n_steps},
        "idle_by_next_launch_s": {f"{a} ({g})": sec(v) for (a, g), v in by_api.most_common(20)},
        "idle_host_late_s": sec(host_late),
        "idle_launch_already_issued_s": sec(gpu_queue),
        "host_calls_inside_late_gaps_s": {k: sec(v) for k, v in host_late_calls.most_common(25)},
        "idle_by_kernel_pair": [
            {"prev": p, "next": n, "api": a, "n": pair_n[(p, n, a)], "s": sec(v)}
            for (p, n, a), v in by_pair.most_common(40)
        ],
        "kernel_time": [
            {"name": k[:120], "n": kern_n[k], "s": sec(v), "pct_wall": round(100 * v / wall, 2)}
            for k, v in kern.most_common(60)
        ],
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in list(report)[:12]}, indent=1))


if __name__ == "__main__":
    main()
