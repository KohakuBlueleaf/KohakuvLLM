"""Greedy generations of a model over recorded conversations, for comparing decode paths.

Each JSONL row is ``{"messages": [...]}`` (a ``reply`` field is ignored); the chat template
(thinking off) renders the messages and the model decodes greedily up to ``--max-tokens``.
``--engine`` passes extra ``vllm.LLM`` arguments as JSON (``{"mamba_ssm_cache_dtype":
"bfloat16"}``). ``compare_gen.py`` reads two such files.

    python bench/precision/generate.py --model DIR --data rows.jsonl --out a.jsonl
"""

import argparse
import json
import sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--util", type=float, default=0.55)
    p.add_argument("--engine", default="{}")
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    prompts = [
        TokensPrompt(
            prompt_token_ids=tok(
                tok.apply_chat_template(
                    row["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
                add_special_tokens=False,
            )["input_ids"]
        )
        for row in rows
    ]
    llm = LLM(
        args.model,
        gpu_memory_utilization=args.util,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1, "video": 0},
        **json.loads(args.engine),
    )
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    with open(args.out, "w", encoding="utf-8") as f:
        for o in outs:
            c = o.outputs[0]
            f.write(json.dumps({"ids": list(c.token_ids), "text": c.text}) + "\n")
    print(f"{len(outs)} generations -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
