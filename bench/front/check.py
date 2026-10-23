"""Our frontend (``kohakuvllm.front``) against vLLM's own input path, on real images.

1. Vision tower: ours (bf16, and fp8 on sm89+) and the reference model's own tower in bf16, each
   against the reference tower in float32 (the ruler), over the same pixel patches: an error
   profile per path, side by side (no threshold).
2. Engine: the same requests through ``AsyncLLM`` with vLLM's input path (text prompt and
   images; again in reversed order, the floor batching alone gives) and through our frontend
   (token ids and image embeddings) with the tower in float32 (the ruler), bf16 and fp8 (sm89+);
   each path against the ruler: prompt token ids exactly, greedy replies token for token (share
   identical, agreed prefix), the teacher-forced log-probabilities of the text after the image.

    python bench/front/check.py MODEL IMAGES --ref BF16_MODEL [--n 64] [--out check.json]

``MODEL``: the checkpoint served; ``BF16_MODEL``: a BF16 checkpoint of the same model (its tower
is the reference). Requests: per image, a system prompt, a label, the image and a question.
"""

import argparse
import asyncio
import io
import json
import sys
import uuid
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

from kohakuvllm.front import engine as front
from kohakuvllm.front.prep import prepare
from kohakuvllm.front.vision import load_tower, to_fp8

SYSTEM = "You are a careful image describer. Answer in one short paragraph."
QUERY = "Describe the main objects in the image and how they are arranged."


def requests(images: Path, n: int, side: int) -> list[tuple[list[dict], list[bytes]]]:
    out = []
    for path in sorted(
        p for p in images.iterdir() if p.suffix.lower() in (".jpg", ".png", ".webp")
    )[:n]:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((side, side))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=90)
        messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Image {path.stem}:"},
                    {"type": "image"},
                    {"type": "text", "text": QUERY},
                ],
            },
        ]
        out.append((messages, [buf.getvalue()]))
    return out


def profile(got: torch.Tensor, ruler: torch.Tensor) -> dict:
    d = (got.double() - ruler.double()).abs()
    mag = ruler.double().abs()
    big = mag > mag.mean()
    rel = d[big] / mag[big]
    return {
        "abs_max": float(d.max()),
        "abs_mean": float(d.mean()),
        "ruler_abs_mean": float(mag.mean()),
        "rel_mean_where_ruler_above_mean": float(rel.mean()),
        "rel_p99_where_ruler_above_mean": float(rel.quantile(0.99)),
    }


@torch.inference_mode()
def tower_check(model: str, ref: str, reqs: list, device: torch.device) -> dict:
    processor = AutoProcessor.from_pretrained(model)
    prepared = [prepare(m, i, processor) for m, i in reqs[:16]]
    pixels = torch.cat([torch.from_numpy(p.pixels) for p in prepared]).to(device)
    grid = torch.cat([torch.from_numpy(p.grid) for p in prepared]).to(device)

    full = AutoModelForImageTextToText.from_pretrained(ref, torch_dtype=torch.float32)
    tower32 = next(m for n, m in full.named_modules() if n.endswith("visual")).to(device)
    ruler = tower32(pixels, grid_thw=grid).pooler_output.float()
    ref16 = tower32.to(torch.bfloat16)(pixels.bfloat16(), grid_thw=grid).pooler_output
    del full, tower32
    torch.cuda.empty_cache()

    ours = load_tower(model, device, torch.bfloat16)
    out = {
        "patches": int(pixels.shape[0]),
        "tokens": int(ruler.shape[0]),
        "reference_bf16": profile(ref16, ruler),
        "ours_bf16": profile(ours(pixels.bfloat16(), grid_thw=grid).pooler_output, ruler),
    }
    if torch.cuda.get_device_capability(device) >= (8, 9):
        to_fp8(ours.blocks)
        out["ours_fp8"] = profile(ours(pixels.bfloat16(), grid_thw=grid).pooler_output, ruler)
    return out


async def run_engine(args: dict, reqs: list, frontend: dict | None, order: list[int]) -> list:
    llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**args))
    fe = front.Frontend(llm, args["model"], **frontend) if frontend is not None else None
    processor = AutoProcessor.from_pretrained(args["model"])
    greedy = SamplingParams(temperature=0.0, max_tokens=96)
    scored = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    async def one(i: int, params: SamplingParams):
        messages, images = reqs[i]
        rid = uuid.uuid4().hex
        if fe is not None:
            return i, await fe.generate(messages, images, params, rid)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        pics = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        prompt = {"prompt": text, "multi_modal_data": {"image": pics}}
        final = None
        async for o in llm.generate(prompt, params, rid):
            final = o
        return i, final

    try:
        done = dict(await asyncio.gather(*(one(i, greedy) for i in order)))
        logp = dict(await asyncio.gather(*(one(i, scored) for i in order)))
    finally:
        if fe is not None:
            fe.close()
        llm.shutdown()
    out = []
    for i in range(len(reqs)):
        ids = list(done[i].prompt_token_ids)
        lp = [None if d is None else d[t].logprob for t, d in zip(ids, logp[i].prompt_logprobs)]
        out.append((ids, list(done[i].outputs[0].token_ids), lp))
    return out


def agreement(a: list, b: list, pad: int) -> dict:
    """Prompt ids, greedy replies, and the teacher-forced log-probability of every prompt token
    after each request's last image token (the text the image conditions)."""
    same_prompt = sum(x[0] == y[0] for x, y in zip(a, b))
    same_reply = sum(x[1] == y[1] for x, y in zip(a, b))
    prefix, diffs = [], []
    for (ids, ra, la), (_, rb, lb) in zip(a, b):
        k = next((j for j, (u, v) in enumerate(zip(ra, rb)) if u != v), min(len(ra), len(rb)))
        prefix.append(k / max(1, len(ra)))
        start = max(j for j, t in enumerate(ids) if t == pad) + 1
        pairs = zip(la[start:], lb[start:])
        diffs += [abs(x - y) for x, y in pairs if x is not None and y is not None]
    diffs = torch.tensor(diffs, dtype=torch.float64)
    first = next((j for j, (x, y) in enumerate(zip(a, b)) if x[0] != y[0]), None)
    return {
        "requests": len(a),
        "prompt_ids_identical": same_prompt,
        "first_prompt_mismatch": (
            None
            if first is None
            else {"request": first, "lens": [len(a[first][0]), len(b[first][0])]}
        ),
        "replies_identical": same_reply,
        "agreed_prefix_mean": round(sum(prefix) / len(prefix), 4),
        "logprob_abs_diff": {
            "tokens": int(diffs.numel()),
            "mean": float(diffs.mean()) if diffs.numel() else None,
            "p99": float(diffs.quantile(0.99)) if diffs.numel() else None,
            "max": float(diffs.max()) if diffs.numel() else None,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("images")
    p.add_argument("--ref", required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--side", type=int, default=768)
    p.add_argument("--util", type=float, default=0.6)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    reqs = requests(Path(args.images), args.n, args.side)
    report = {"tower": tower_check(args.model, args.ref, reqs, torch.device("cuda:0"))}
    print(json.dumps(report["tower"]), flush=True)
    torch.cuda.empty_cache()

    engine = {
        "model": args.model,
        "gpu_memory_utilization": args.util,
        "max_model_len": 8192,
        "max_num_seqs": 64,
        "mamba_ssm_cache_dtype": "bfloat16",
        "limit_mm_per_prompt": {"image": 4, "video": 0},
    }
    order = list(range(len(reqs)))
    ours_args = front.engine_args(engine)
    runs = {
        "stock": asyncio.run(run_engine(engine, reqs, None, order)),
        "stock_reversed": asyncio.run(run_engine(engine, reqs, None, order[::-1])),
        "ruler": asyncio.run(
            run_engine(ours_args, reqs, {"workers": 4, "vision_dtype": "float32"}, order)
        ),
        "ours_bf16": asyncio.run(run_engine(ours_args, reqs, {"workers": 4}, order)),
    }
    if "ours_fp8" in report["tower"]:
        fp8 = {"workers": 4, "vision_fp8": True}
        runs["ours_fp8"] = asyncio.run(run_engine(ours_args, reqs, fp8, order))
    tok = AutoProcessor.from_pretrained(args.model)
    pad = tok.tokenizer.convert_tokens_to_ids(tok.image_token)

    # Every path against the float32 tower's embeddings; the floor is vLLM against itself
    report["engine"] = {
        "floor_stock_vs_stock_reversed": agreement(runs["stock"], runs["stock_reversed"], pad),
        **{
            f"{name}_vs_ruler": agreement(runs["ruler"], runs[name], pad)
            for name in runs
            if name not in ("ruler", "stock_reversed")
        },
    }
    print(json.dumps(report["engine"]), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
