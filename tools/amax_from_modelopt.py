"""Activation amax per linear from a ModelOpt checkpoint's static input scales, in
``tools/calibrate.py``'s format, so ``tools/quantize.py`` can requantize the BF16 model with the
calibration NVIDIA ran: ``amax = input_scale * 448`` for an FP8 linear, ``* 448 * 6`` for NVFP4
(ModelOpt stores ``amax / bound``).

    python tools/amax_from_modelopt.py qwen38-27b-nvfp4 amax.json
"""

import argparse
import json
import sys
from pathlib import Path

from safetensors import safe_open

BOUND = {"FP8": 448.0, "NVFP4": 448.0 * 6.0}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src")
    p.add_argument("out")
    args = p.parse_args()
    src = Path(args.src)
    layers = json.loads((src / "hf_quant_config.json").read_text())["quantization"]
    layers = layers["quantized_layers"]
    amax, algos = {}, {}
    for shard in sorted(src.glob("*.safetensors")):
        with safe_open(shard, "pt") as s:
            for k in list(s.keys()):
                if not k.endswith(".input_scale"):
                    continue
                name = k.removesuffix(".input_scale")
                algo = layers[name]["quant_algo"]
                amax[name] = float(s.get_tensor(k).float().item()) * BOUND[algo]
                algos[algo] = algos.get(algo, 0) + 1
    Path(args.out).write_text(json.dumps({"source": str(src), "amax": amax}, indent=1))
    values = sorted(amax.values())
    print(
        f"{len(amax)} linears {algos}; amax min {values[0]:.3g} median "
        f"{values[len(values) // 2]:.3g} max {values[-1]:.3g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
