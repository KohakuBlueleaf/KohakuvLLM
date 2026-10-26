"""The largest absolute input of every linear of a model over recorded conversations: the
activation calibration ``tools/quantize.py`` turns into NVFP4's static input global scales.

Each JSONL row is ``{"messages": [...], "reply": str}`` in OpenAI chat form (pictures as
``image_url`` data URLs); the reply is appended as the assistant turn, so the prefill and the
reply's positions both count. The BF16 model runs through transformers (thinking off) with a
forward pre-hook on every ``nn.Linear`` under ``--prefix``.

    python tools/calibrate.py --model qwen35-4b-bf16 --data rows.jsonl --out amax.json
"""

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def picture(url: str) -> Image.Image:
    data = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def to_hf(messages: list[dict], reply: str) -> list[dict]:
    """OpenAI chat messages (+ the reply) in the processor's form: parts as typed lists."""
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        else:
            parts = []
            for c in content:
                match c["type"]:
                    case "text":
                        parts.append({"type": "text", "text": c["text"]})
                    case "image_url":
                        parts.append({"type": "image", "image": picture(c["image_url"]["url"])})
                    case other:
                        raise ValueError(f"message part {other!r}")
        out.append({"role": m["role"], "content": parts})
    out.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prefix", default="model.language_model.", help="linears whose name starts so")
    p.add_argument("--limit", type=int, default=0, help="rows (0: all)")
    args = p.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.to("cuda").eval()

    # One running maximum per linear, kept on the device until the end
    amax: dict[str, torch.Tensor] = {}

    def hook(name: str):
        def pre(_module, inputs):
            m = inputs[0].detach().abs().amax().float()
            amax[name] = torch.maximum(amax[name], m) if name in amax else m

        return pre

    linears = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and n.startswith(args.prefix)
    ]
    for name, module in linears:
        module.register_forward_pre_hook(hook(name))

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[: args.limit]
    tokens, start = 0, time.perf_counter()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            inputs = processor.apply_chat_template(
                to_hf(row["messages"], row["reply"]),
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to("cuda")
            model(**inputs, use_cache=False)
            tokens += inputs["input_ids"].shape[1]
            if (i + 1) % 50 == 0:
                print(f"{i + 1} / {len(rows)} rows, {tokens} tokens", flush=True)

    missing = [n for n, _ in linears if n not in amax]
    report = {
        "model": args.model,
        "rows": len(rows),
        "tokens": tokens,
        "seconds": round(time.perf_counter() - start, 1),
        "missing": missing,
        "amax": {n: v.item() for n, v in sorted(amax.items())},
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{len(amax)} linears over {len(rows)} rows ({tokens} tokens); {len(missing)} never ran")
    return 0


if __name__ == "__main__":
    sys.exit(main())
