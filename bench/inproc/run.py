"""The engine alone: vLLM driven in-process (``vllm.LLM``) with pre-tokenized prompts, no API server
and no tokenizing during the run. Random prompt ids of ``--input`` tokens and ``--output`` tokens
each (ignore_eos), all submitted at once; the engine's own scheduler admits them.

Prints one JSON row (requests/s, tokens/s, the card's utilisation and power sampled every 0.2 s)
and writes it to ``--out``. ``--profile`` brackets the run with vLLM's CUDA profiler calls, so an
``nsys profile --capture-range=cudaProfilerApi`` around this script captures the run only.

    python bench/inproc/run.py --model MODEL --gpu 0 --input 1000 --output 100 --n 3000
"""

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def sampler(gpu: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=utilization.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits",
            "-lms",
            "200",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )


def card_stats(lines: list[str]) -> dict:
    rows = [[float(x) for x in line.split(",")] for line in lines if line.strip()]
    util = [r[0] for r in rows]
    steady = util[len(util) // 10 : -len(util) // 10] or util
    return {
        "util_mean": round(statistics.mean(steady), 1),
        "util_p10": sorted(steady)[len(steady) // 10],
        "util_stdev": round(statistics.pstdev(steady), 1),
        "power_w": round(statistics.mean(r[1] for r in rows)),
        "mem_peak_mib": max(r[2] for r in rows),
        "samples": len(rows),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", type=int, required=True, help="the card nvidia-smi samples")
    p.add_argument("--input", type=int, default=1000)
    p.add_argument("--output", type=int, default=100)
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--util", type=float, default=0.55)
    p.add_argument("--seqs", type=int, default=256)
    p.add_argument("--batched", type=int, default=8192)
    p.add_argument("--scheduler-cls", default=None)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--engine", default="{}", help="extra vllm.LLM arguments, as JSON")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    extra = json.loads(args.engine)
    if args.scheduler_cls:
        extra["scheduler_cls"] = args.scheduler_cls
    if args.profile:
        extra["profiler_config"] = {"profiler": "cuda"}
    llm = LLM(
        args.model,
        gpu_memory_utilization=args.util,
        max_model_len=8192,
        max_num_seqs=args.seqs,
        max_num_batched_tokens=args.batched,
        limit_mm_per_prompt={"image": 1, "video": 0},
        **extra,
    )
    rng = random.Random(0)
    prompts = [
        TokensPrompt(prompt_token_ids=[rng.randrange(1000, 150_000) for _ in range(args.input)])
        for _ in range(args.n)
    ]
    sp = SamplingParams(max_tokens=args.output, ignore_eos=True, temperature=0.1)
    llm.generate(prompts[:64], sp, use_tqdm=False)

    smi = sampler(args.gpu)
    if args.profile:
        llm.start_profile()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    if args.profile:
        llm.stop_profile()
    smi.terminate()
    gen = sum(len(o.outputs[0].token_ids) for o in outs)
    row = {
        "input": args.input,
        "output": args.output,
        "n": len(outs),
        "scheduler": args.scheduler_cls or "vllm",
        "engine": json.loads(args.engine),
        "wall_s": round(wall, 2),
        "req_s": round(len(outs) / wall, 2),
        "out_tok_s": round(gen / wall),
        "in_tok_s": round(args.input * len(outs) / wall),
        **card_stats(smi.stdout.read().splitlines()),
    }
    print(json.dumps(row), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
