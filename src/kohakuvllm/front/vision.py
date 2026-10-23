"""The checkpoint's vision tower outside vLLM's engine.

A thread with its own CUDA stream takes the waiting requests' pixel patches (up to
``max_patches`` per call), runs the tower once over them and answers each request with its
images' merged embeddings on the CPU ([tokens, hidden], the model dtype): what vLLM's engine
takes as ``image_embeds``. The tower is the transformers module of the checkpoint (any
Qwen2.5-VL / Qwen3-VL / Qwen3.5 model), its weights read alone from the checkpoint. ``fp8``
runs the blocks' linears as FP8 E4M3 GEMMs: weights per output channel, activations per token
(sm89 and up).
"""

import json
import queue
import threading
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForImageTextToText

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max


class Fp8Linear(torch.nn.Module):
    """``y = x W^T + b`` with ``W`` in FP8 per output channel and ``x`` quantized per token."""

    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        w = linear.weight.detach().float()
        scale = w.abs().amax(1, keepdim=True).clamp(min=1e-12) / FP8_MAX
        self.weight = (w / scale).to(FP8)
        self.w_scale = scale.t().contiguous()
        self.bias = None if linear.bias is None else linear.bias.detach()
        self.out_dtype = linear.weight.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        s = x.abs().amax(1, keepdim=True).float().clamp(min=1e-12) / FP8_MAX
        q = (x.float() / s).to(FP8)
        y = torch._scaled_mm(
            q,
            self.weight.t(),
            scale_a=s,
            scale_b=self.w_scale,
            bias=self.bias,
            out_dtype=self.out_dtype,
        )
        return y.reshape(*shape[:-1], y.shape[-1])


def to_fp8(blocks: torch.nn.Module) -> int:
    """Every ``nn.Linear`` under ``blocks`` as an ``Fp8Linear``; returns how many."""
    n = 0
    for name, child in list(blocks.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(blocks, name, Fp8Linear(child))
            n += 1
        else:
            n += to_fp8(child)
    return n


def load_tower(model: str, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """The checkpoint's vision tower with its own weights, on ``device``."""
    config = AutoConfig.from_pretrained(model)
    # The model's tower class, found on a weightless copy of the whole model, then built alone
    with torch.device("meta"):
        full = AutoModelForImageTextToText.from_config(config, torch_dtype=dtype)
    meta = next(m for n, m in full.named_modules() if n.endswith("visual"))
    with torch.device(device):
        tower = type(meta)._from_config(meta.config, torch_dtype=dtype)

    root = Path(model)
    index = root / "model.safetensors.index.json"
    files = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.exists()
        else ["model.safetensors"]
    )
    params = dict(tower.named_parameters()) | dict(tower.named_buffers())
    loaded = set()
    for file in files:
        with safe_open(root / file, framework="pt", device=str(device)) as f:
            for key in f.keys():  # noqa: SIM118 - safe_open is not a mapping
                if "visual." not in key:
                    continue
                local = key.split("visual.", 1)[1]
                if local in params:
                    params[local].data.copy_(f.get_tensor(key))
                    loaded.add(local)
    missing = sorted(k for k in dict(tower.named_parameters()) if k not in loaded)
    if missing:
        raise ValueError(f"{model}: vision weights missing from the checkpoint: {missing[:5]}")
    return tower.eval()


class VisionEncoder:
    """``submit(pixels, grid)`` -> a future of the request's merged image embeddings."""

    def __init__(
        self,
        model: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        fp8: bool = False,
        max_patches: int = 32768,
    ):
        self.device = torch.device(device)
        self.dtype, self.max_patches = dtype, max_patches
        self.tower = load_tower(model, self.device, dtype)
        self.fp8_linears = to_fp8(self.tower.blocks) if fp8 else 0
        self.merge = self.tower.spatial_merge_size**2
        self.stream = torch.cuda.Stream(self.device)
        self.queue: queue.Queue = queue.Queue()
        self.counts = {"calls": 0, "images": 0, "patches": 0}
        self.thread = threading.Thread(target=self._loop, name="kohakuvllm-vision", daemon=True)
        self.thread.start()

    def submit(self, pixels: np.ndarray, grid: np.ndarray) -> Future:
        future: Future = Future()
        self.queue.put((pixels, grid, future))
        return future

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()

    def _take(self) -> list | None:
        first = self.queue.get()
        if first is None:
            return None
        batch, patches = [first], first[0].shape[0]
        while patches < self.max_patches:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self.queue.put(None)
                break
            batch.append(item)
            patches += item[0].shape[0]
        return batch

    @torch.inference_mode()
    def _loop(self) -> None:
        torch.cuda.set_device(self.device)
        while (batch := self._take()) is not None:
            try:
                with torch.cuda.stream(self.stream):
                    pixels = torch.from_numpy(np.concatenate([b[0] for b in batch]))
                    grid = torch.from_numpy(np.concatenate([b[1] for b in batch]))
                    x = pixels.pin_memory().to(self.device, self.dtype, non_blocking=True)
                    out = self.tower(x, grid_thw=grid.to(self.device)).pooler_output
                    host = out.to("cpu", non_blocking=True)
                self.stream.synchronize()
            except Exception as exc:  # noqa: BLE001 - every request of the call gets the error
                for *_, future in batch:
                    future.set_exception(exc)
                continue
            self.counts["calls"] += 1
            self.counts["images"] += int(grid.shape[0])
            self.counts["patches"] += int(pixels.shape[0])
            start = 0
            for _, g, future in batch:
                n = int((g.prod(1) // self.merge).sum())
                future.set_result(host[start : start + n])
                start += n
