"""W8A8 with GPTQ weights (llm-compressor): weights INT8 or FP8 per output channel, solved layer by
layer against calibration activations; activations dynamic per token. The linears are those a
reference compressed-tensors checkpoint quantizes (its ``ignore`` list is reused as is).

SmoothQuant is not applied: Qwen3.5's RMSNorms scale by ``1 + weight``, which llm-compressor's
smoothing (a plain multiply into the norm weight) does not fold correctly.

Calibration: ``--samples`` chat conversations from ``--dataset`` (HF datasets, ``messages``
column), up to ``--seq-len`` tokens each. Runs in its own environment (llm-compressor pins its
own transformers).

    python tools/gptq_w8a8.py qwen35-4b-bf16 qwen35-4b-int8-gptq --like qwen35-4b-fp8-gdn \
        --scheme int8
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoModelForImageTextToText, AutoTokenizer

SCHEMES = {"int8": "W8A8", "fp8": "FP8_DYNAMIC"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--like", required=True)
    p.add_argument("--scheme", choices=sorted(SCHEMES), default="int8")
    p.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--split", default="train_sft")
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--dampening", type=float, default=0.01)
    args = p.parse_args()
    ignore = json.loads((Path(args.like) / "config.json").read_text())["quantization_config"][
        "ignore"
    ]
    tok = AutoTokenizer.from_pretrained(args.src)
    model = AutoModelForImageTextToText.from_pretrained(args.src, dtype="auto")
    rows = load_dataset(args.dataset, split=f"{args.split}[:{args.samples}]").shuffle(seed=0)

    def encode(row):
        text = tok.apply_chat_template(row["messages"], tokenize=False, enable_thinking=False)
        return tok(text, max_length=args.seq_len, truncation=True, add_special_tokens=False)

    data = rows.map(encode, remove_columns=rows.column_names)
    recipe = GPTQModifier(
        targets="Linear",
        scheme=SCHEMES[args.scheme],
        ignore=ignore,
        dampening_frac=args.dampening,
    )
    oneshot(
        model=model,
        processor=tok,
        dataset=data,
        recipe=recipe,
        max_seq_length=args.seq_len,
        num_calibration_samples=args.samples,
        sequential_targets=["Qwen3_5DecoderLayer"],
    )
    model.save_pretrained(args.dst, save_compressed=True)
    tok.save_pretrained(args.dst)
    print(f"saved {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
