"""KohakuvLLM's grammar masks (``kohakuvllm_grammar``) against xgrammar's and an independent
judge (``reference.py``), token by token.

For each schema, the JSON grammar with at most ``--ws`` whitespace bytes per gap (0: compact;
xgrammar with ``any_whitespace=False`` and separators ``,`` ``:``, or ``max_whitespace_cnt``) is
walked along two kinds of paths: documents generated to the schema (strings at and near their
limits, escapes, multi-byte characters, arrays at their item limits, gaps of 0 to ``--ws``
whitespace bytes), tokenized; and random walks that pick each next token uniformly from our mask. At every step
both masks are compared over the whole vocabulary. The judge decides the tokens the two masks
disagree on (all of them, or a random ``--diff-cap`` of them), end-of-sequence, and (every
``--judge-every`` steps) a random sample of the vocabulary, so each side's errors are counted
against the JSON and schema rules themselves; examples are kept (no threshold). Then CPU time per
mask: ours (memoised; the first build of each counted apart) and xgrammar's fill.

    python bench/grammar/check.py --model DIR --schemas DIR [--ws 0] [--docs 40] [--walks 40]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import xgrammar as xgr
from kohakuvllm_grammar import Store
from reference import judge
from transformers import AutoConfig, AutoTokenizer

from kohakuvllm.grammar.tokens import eos_ids, token_bytes

ALPHABET = list("abc xyz,.:;{}[]-_'") + ['"', "\\", "\n", "\t", "é", "中", "😀", "/"]
KEEP = 12


def text_of(rng: random.Random, lo: int, hi: int) -> str:
    n = rng.choice([lo, hi, rng.randint(lo, hi), max(lo, hi - 1)])
    return "".join(rng.choice(ALPHABET) for _ in range(n))


def instance(schema: dict, rng: random.Random):
    """A value valid under ``schema`` (the supported subset), limits hit often."""
    if "enum" in schema:
        return rng.choice(schema["enum"])
    match schema.get("type"):
        case "object":
            req = set(schema.get("required", []))
            return {
                k: instance(v, rng)
                for k, v in schema.get("properties", {}).items()
                if k in req or rng.random() < 0.5
            }
        case "array":
            lo, hi = schema.get("minItems", 0), schema.get("maxItems", 6)
            n = rng.choice([lo, hi, rng.randint(lo, hi)])
            return [instance(schema["items"], rng) for _ in range(n)]
        case "string":
            return text_of(rng, schema.get("minLength", 0), min(schema.get("maxLength", 40), 900))
        case "integer":
            return rng.randint(-1000, 10**6)
        case "number":
            return rng.choice([0, -1.5, 3.25e-3, 12345.0])
        case "boolean":
            return rng.random() < 0.5
        case "null":
            return None
    raise ValueError(f"unsupported {schema}")


def spaced(v, rng: random.Random, ws: int) -> str:
    """``v`` as JSON with 0 to ``ws`` whitespace bytes in each gap the grammar allows."""

    def gap() -> str:
        return "".join(rng.choice(" \n\t\r") for _ in range(rng.randint(0, ws)))

    def join(elems: list[str]) -> str:
        return "".join(("," + gap() if k else "") + e + gap() for k, e in enumerate(elems))

    if isinstance(v, dict):
        items = [
            json.dumps(k, ensure_ascii=False) + gap() + ":" + gap() + spaced(x, rng, ws)
            for k, x in v.items()
        ]
        return "{" + gap() + join(items) + "}"
    if isinstance(v, list):
        return "[" + gap() + join([spaced(x, rng, ws) for x in v]) + "]"
    return json.dumps(v, ensure_ascii=False)


def unpack(words: np.ndarray, vocab: int) -> np.ndarray:
    return np.unpackbits(words.view(np.uint8), bitorder="little")[:vocab].astype(bool)


class Checker:
    def __init__(self, store, pieces, vocab, eos, ws, bitmask, rng, judge_every, sample, diff_cap):
        self.store, self.pieces, self.vocab, self.eos, self.ws = store, pieces, vocab, eos, ws
        self.bitmask, self.rng = bitmask, rng
        self.judge_every, self.sample, self.diff_cap = judge_every, sample, diff_cap
        self.ours_s = self.xgr_s = 0.0

    def allowed(self, schema: dict, prefix: bytes, t: int) -> bool:
        if t in self.eos:
            return judge(schema, prefix, self.ws) == "complete"
        piece = self.pieces[t]
        return piece is not None and judge(schema, prefix + piece, self.ws) != "bad"

    def path(self, schema, gid, compiled, rows, tokens, steps) -> None:
        """One path, both masks compared and judged before each token: ``tokens`` given, or
        (None) a random walk over our mask."""
        h = self.store.matcher(gid)
        xm = xgr.GrammarMatcher(compiled)
        prefix: list[int] = []
        data = b""
        for i in range(steps):
            t0 = time.perf_counter()
            mid = self.store.step([h], [prefix[-1:]])[0]
            self.ours_s += time.perf_counter() - t0
            ours = unpack(np.frombuffer(self.store.mask(mid), dtype=np.uint32), self.vocab)
            t0 = time.perf_counter()
            xm.fill_next_token_bitmask(self.bitmask)
            self.xgr_s += time.perf_counter() - t0
            theirs = unpack(self.bitmask[0].numpy().view(np.uint32), self.vocab)
            rows["steps"] += 1
            differ = np.flatnonzero(ours != theirs).tolist()
            rows["agree"] += not differ
            if len(differ) > self.diff_cap:
                differ = self.rng.sample(differ, self.diff_cap)
            judged = set(differ) | set(self.eos)
            if rows["steps"] % self.judge_every == 0:
                judged |= set(self.rng.sample(range(self.vocab), self.sample))
                rows["sampled_steps"] += 1
            for t in judged:
                truth = self.allowed(schema, data, t)
                rows["judged"] += 1
                for side, mask in (("ours", ours), ("xgrammar", theirs)):
                    if mask[t] != truth:
                        rows[f"{side}_wrong"] += 1
                        if len(rows[f"{side}_examples"]) < KEEP:
                            rows[f"{side}_examples"].append(
                                {
                                    "prefix": data[-80:].decode("utf-8", "replace"),
                                    "token": [t, repr(self.pieces[t])],
                                    "mask_says": bool(mask[t]),
                                    "judge_says": truth,
                                }
                            )
            if tokens is None:
                choices = np.flatnonzero(ours).tolist()
                if not choices:
                    break
                nxt = int(self.rng.choice(choices))
            elif i == len(tokens):
                break
            else:
                nxt = tokens[i]
            if nxt in self.eos:
                break
            if not xm.accept_token(nxt):
                rows["xgr_rejected_path"] += 1
                break
            prefix.append(nxt)
            data += self.pieces[nxt]
        self.store.release(h)
        rows["paths"] += 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--schemas", required=True)
    p.add_argument("--ws", type=int, default=0, help="whitespace bytes allowed per gap")
    p.add_argument("--docs", type=int, default=40)
    p.add_argument("--walks", type=int, default=40)
    p.add_argument("--walk-steps", type=int, default=600)
    p.add_argument("--judge-every", type=int, default=25)
    p.add_argument("--sample", type=int, default=2000)
    p.add_argument("--diff-cap", type=int, default=200, help="disagreeing tokens judged per step")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rng = random.Random(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    vocab = AutoConfig.from_pretrained(args.model).get_text_config().vocab_size
    pieces = token_bytes(tok, vocab)
    eos = eos_ids(tok, args.model)
    t0 = time.perf_counter()
    store = Store(pieces, eos, vocab, args.ws)
    store_s = time.perf_counter() - t0
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab, stop_token_ids=eos)
    compiler = xgr.GrammarCompiler(info)
    bitmask = xgr.allocate_token_bitmask(1, vocab)
    check = Checker(
        store,
        pieces,
        vocab,
        eos,
        args.ws,
        bitmask,
        rng,
        args.judge_every,
        args.sample,
        args.diff_cap,
    )
    if args.ws:
        spacing = {"any_whitespace": True, "max_whitespace_cnt": args.ws}
    else:
        spacing = {"any_whitespace": False, "separators": (",", ":")}
    report = {"vocab": vocab, "eos": eos, "ws": args.ws, "store_build_s": store_s, "schemas": {}}
    for path in sorted(Path(args.schemas).glob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        gid = store.compile(json.dumps(schema))
        compiled = compiler.compile_json_schema(json.dumps(schema), **spacing)
        rows = {
            "paths": 0,
            "steps": 0,
            "agree": 0,
            "sampled_steps": 0,
            "judged": 0,
            "ours_wrong": 0,
            "xgrammar_wrong": 0,
            "xgr_rejected_path": 0,
            "ours_examples": [],
            "xgrammar_examples": [],
        }
        for _ in range(args.docs):
            doc = spaced(instance(schema, rng), rng, args.ws)
            ids = tok(doc, add_special_tokens=False)["input_ids"] + [eos[0]]
            check.path(schema, gid, compiled, rows, ids, len(ids) + 1)
        for _ in range(args.walks):
            check.path(schema, gid, compiled, rows, None, args.walk_steps)
        report["schemas"][path.name] = rows
        print(path.name, json.dumps({k: v for k, v in rows.items() if "examples" not in k}))
        for side in ("ours", "xgrammar"):
            for e in rows[f"{side}_examples"][:3]:
                print(f"  {side}:", json.dumps(e, ensure_ascii=False)[:300])

    masks, mask_s, steps, _, _, rejected = store.stats()
    report["time"] = {
        "ours_step_total_s": check.ours_s,
        "ours_masks_built": masks,
        "ours_build_s": mask_s,
        "ours_build_ms_each": 1e3 * mask_s / max(1, masks),
        "ours_ms_per_step": 1e3 * check.ours_s / max(1, steps),
        "ours_ms_per_step_excl_builds": 1e3 * (check.ours_s - mask_s) / max(1, steps),
        "xgrammar_ms_per_fill": 1e3 * check.xgr_s / max(1, steps),
        "steps": steps,
        "rejected": rejected,
    }
    print(json.dumps(report["time"]))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
