"""``tools/quantize.py``'s weight encodings against vLLM's own quantizers, byte for byte: NVFP4
(packed codes, E4M3 block scales) against ``scaled_fp4_quant`` (CUDA, unswizzled scales) and
MXFP8 (values, E8M0 scales) against ``mxfp8_e4m3_quantize``. Real weights of a checkpoint's
first layers and random tensors over several scales. Prints the share of bytes that differ.

    python bench/precision/quant_layout.py --model qwen35-4b-bf16
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "quantize.py"
spec = importlib.util.spec_from_file_location("quantize", TOOLS)
quantize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantize)


def tensors(model: str) -> dict[str, torch.Tensor]:
    out = {}
    root = Path(model)
    names = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    for k, shard in names.items():
        layer = ".layers.0." in k or ".layers.3." in k
        if layer and k.endswith("_proj.weight") and "language_model" in k:
            with safe_open(root / shard, "pt") as s:
                out[k] = s.get_tensor(k)
    gen = torch.Generator().manual_seed(0)
    for scale in (1e-3, 1.0, 40.0):
        out[f"randn*{scale}"] = (torch.randn(512, 2048, generator=gen) * scale).bfloat16()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    args = p.parse_args()
    for name, w in tensors(args.model).items():
        w = w.cuda()
        w32 = w.float()
        g = torch.tensor(quantize.E4M3_MAX * quantize.FP4_MAX / w32.abs().amax().item()).cuda()
        ours = quantize.nvfp4(w32, g)
        ref_q, ref_s = ops.scaled_fp4_quant(w, g, is_sf_swizzled_layout=False)
        ref_s = ref_s.view(torch.float8_e4m3fn)[: w.shape[0], : w.shape[1] // 16]
        q_diff = (ours["weight_packed"] != ref_q).float().mean().item()
        s_diff = (ours["weight_scale"].view(torch.uint8) != ref_s.view(torch.uint8)).float()
        mx = quantize.mx8(w32)
        mq, ms = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
        mq_diff = (mx["weight"].view(torch.uint8) != mq.view(torch.uint8)).float().mean().item()
        ms_diff = (mx["weight_scale"] != ms.view(torch.uint8)).float().mean().item()
        print(
            json.dumps(
                {
                    "tensor": name,
                    "shape": list(w.shape),
                    "nvfp4_codes_differ": q_diff,
                    "nvfp4_scales_differ": s_diff.mean().item(),
                    "mxfp8_values_differ": mq_diff,
                    "mxfp8_scales_differ": ms_diff,
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
