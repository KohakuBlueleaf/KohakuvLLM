"""Chat requests into model inputs in worker processes, off the caller's GIL.

Each worker holds the checkpoint's HF processor and turns ``(messages, images)`` into the
prompt token ids (chat template, thinking off) and the pixel patches with their grids, exactly
as the processor computes them for vLLM's own path. Each image keeps one placeholder token in
the ids: vLLM expands it to the image's token count from the grid itself.
"""

import io
import multiprocessing as mp
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image
from transformers import AutoProcessor

_PROCESSOR = None


@dataclass
class Prepared:
    """One request's model inputs: ``ids`` the prompt token ids (one placeholder per image);
    ``pixels`` [patches, C * T * P * P] float32 and ``grid`` [n_images, 3] (t, h, w in
    patches) over every image."""

    ids: list[int]
    pixels: np.ndarray | None
    grid: np.ndarray | None


def _init(model: str) -> None:
    global _PROCESSOR
    _PROCESSOR = AutoProcessor.from_pretrained(model)


def prepare(messages: list[dict], images: list[bytes], processor=None) -> Prepared:
    """``messages``: chat messages whose content holds ``{"type": "image"}`` items, one per
    entry of ``images`` (encoded image files), in order."""
    p = processor or _PROCESSOR
    text = p.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    pictures = []
    for data in images:
        with Image.open(io.BytesIO(data)) as img:
            pictures.append(img.convert("RGB"))
    out = p(text=[text], images=pictures or None, return_tensors="np")
    ids = out["input_ids"][0].tolist()
    if not pictures:
        return Prepared(ids, None, None)
    pad = p.tokenizer.convert_tokens_to_ids(p.image_token)
    ids = [t for i, t in enumerate(ids) if t != pad or i == 0 or ids[i - 1] != pad]
    return Prepared(ids, out["pixel_values"], out["image_grid_thw"])


class Preparer:
    """``workers`` spawned processes preparing requests; ``submit`` returns a future."""

    def __init__(self, model: str, workers: int = 4):
        self.pool = ProcessPoolExecutor(
            workers, mp_context=mp.get_context("spawn"), initializer=_init, initargs=(model,)
        )

    def submit(self, messages: list[dict], images: list[bytes]) -> Future:
        return self.pool.submit(prepare, messages, images)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
