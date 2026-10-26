"""KohakuvLLM's grammar logits processor inside vLLM's engine: replies for every schema, sampled at
temperature 1.0 (wide walks through the grammar) and greedy, each judged by ``reference.py``
(complete documents only), with the replies that end unfinished (``max_tokens``) counted apart.

    python bench/grammar/engine_check.py --model DIR --schemas DIR [--n 32] [--out r.json]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from reference import judge
from vllm import LLM, SamplingParams

PROMPT = (
    "Describe an imaginary photo of a busy street market. Answer only with JSON that follows "
    "this schema:\n{schema}"
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--schemas", required=True)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--util", type=float, default=0.6)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    llm = LLM(
        args.model,
        gpu_memory_utilization=args.util,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 4, "video": 0},
        logits_processors=["kohakuvllm.grammar:JsonSchemaLogits"],
    )
    ws = int(os.environ.get("KOHAKU_GRAMMAR_WS", "8"))  # the engine's whitespace allowance
    report = {"ws": ws, "schemas": {}}
    for path in sorted(Path(args.schemas).glob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(schema)
        messages = [{"role": "user", "content": PROMPT.format(schema=text)}]
        rows = {}
        for name, temp in (("t1", 1.0), ("greedy", 0.0)):
            sp = SamplingParams(
                temperature=temp,
                max_tokens=args.max_tokens,
                n=args.n if temp else 1,
                seed=0,
                extra_args={"kohaku_json_schema": text},
            )
            t0 = time.perf_counter()
            out = llm.chat(
                messages, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False}
            )
            dt = time.perf_counter() - t0
            verdicts = {"complete": 0, "prefix": 0, "bad": 0}
            bad = []
            for o in out[0].outputs:
                v = judge(schema, o.text.encode(), ws)
                if v == "prefix" and o.finish_reason != "length":
                    v = "bad"
                verdicts[v] += 1
                if v == "bad" and len(bad) < 3:
                    bad.append(o.text[-300:])
            rows[name] = {"replies": len(out[0].outputs), **verdicts, "s": dt, "bad_tails": bad}
        report["schemas"][path.name] = rows
        print(
            path.name,
            json.dumps({k: {**v, "bad_tails": len(v["bad_tails"])} for k, v in rows.items()}),
        )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
