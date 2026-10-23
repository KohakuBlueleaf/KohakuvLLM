"""Teacher-forced log-probabilities of a model over recorded conversations, for comparing models.

Each JSONL row is ``{"messages": [...], "reply": str}``; the chat template (thinking off) renders
the messages, the reply is appended, and vLLM scores every position (``prompt_logprobs``). For the
reply positions the script keeps the actual token's log-probability and the top ``--k`` tokens
with theirs; ``compare.py`` reads two such files.

    python bench/precision/prompt_logprobs.py --model DIR --data rows.jsonl --out ref.pt
"""

import argparse
import json
import sys

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--util", type=float, default=0.55)
    p.add_argument("--eager", action="store_true", help="no CUDA graphs, no torch.compile")
    p.add_argument("--engine", default="{}", help="more vLLM engine arguments, JSON")
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    prompts, spans = [], []
    for row in rows:
        head = tok.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids_head = tok(head, add_special_tokens=False)["input_ids"]
        ids_reply = tok(row["reply"], add_special_tokens=False)["input_ids"]
        prompts.append(TokensPrompt(prompt_token_ids=ids_head + ids_reply))
        spans.append((len(ids_head), len(ids_head) + len(ids_reply)))
    llm = LLM(
        args.model,
        gpu_memory_utilization=args.util,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1, "video": 0},
        max_logprobs=args.k,
        enforce_eager=args.eager,
        **json.loads(args.engine),
    )
    sp = SamplingParams(max_tokens=1, prompt_logprobs=args.k, temperature=0.0)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    actual, top_ids, top_lp = [], [], []
    for out, prompt, (lo, hi) in zip(outs, prompts, spans):
        ids = prompt["prompt_token_ids"]
        for pos in range(lo, hi):
            d = out.prompt_logprobs[pos]
            actual.append(d[ids[pos]].logprob)
            best = sorted(d.items(), key=lambda kv: -kv[1].logprob)[: args.k]
            best += [(-1, None)] * (args.k - len(best))
            top_ids.append([t for t, _ in best])
            top_lp.append([lp.logprob if lp else float("-inf") for _, lp in best])
    torch.save(
        {
            "model": args.model,
            "actual": torch.tensor(actual, dtype=torch.float64),
            "top_ids": torch.tensor(top_ids),
            "top_lp": torch.tensor(top_lp, dtype=torch.float64),
            "spans": spans,
        },
        args.out,
    )
    print(f"{len(actual)} reply positions over {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
