"""Greedy generations against a reference's, from ``generate.py`` files.

Per pair of files: the share of generations identical token for token, the agreed prefix (tokens
before the first difference, as a share of the reference's length; mean, p10, p50), and the share
of replies that parse as JSON in each. A first run of the same configuration against itself
measures the floor that batching nondeterminism alone gives.

    python bench/precision/compare_gen.py ref.jsonl a.jsonl b.jsonl
"""

import argparse
import json
import statistics
import sys


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ref")
    p.add_argument("others", nargs="+")
    args = p.parse_args()
    ref = load(args.ref)
    js_ref = sum(is_json(r["text"]) for r in ref)
    print(f"reference {args.ref}: {len(ref)} generations, json {js_ref}")
    for path in args.others:
        rows = load(path)
        assert len(rows) == len(ref)
        same = sum(r["ids"] == o["ids"] for r, o in zip(ref, rows))
        share = sorted(
            prefix(r["ids"], o["ids"]) / max(1, len(r["ids"])) for r, o in zip(ref, rows)
        )
        js = sum(is_json(o["text"]) for o in rows)
        print(
            f"{path}: identical {100 * same / len(ref):.1f}%  agreed prefix mean "
            f"{100 * statistics.mean(share):.1f}% p10 {100 * share[len(share) // 10]:.1f}% "
            f"p50 {100 * share[len(share) // 2]:.1f}%  json {js}/{len(rows)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
