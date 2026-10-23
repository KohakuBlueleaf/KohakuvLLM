"""vLLM's ``AsyncLLM`` behind our frontend: requests are prepared in worker processes
(``prep.Preparer``) and their images encoded by our vision tower (``vision.VisionEncoder``), then
handed to the engine as prompt token ids with ``image_embeds``; vLLM tokenizes nothing and runs
no image processor or encoder. The engine needs ``enable_mm_embeds=True`` (``engine_args``
adds it).
"""

import asyncio

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

from kohakuvllm.front.prep import Preparer
from kohakuvllm.front.vision import VisionEncoder


def engine_args(args: dict) -> dict:
    """``args`` with what this frontend needs from the engine."""
    return {**args, "enable_mm_embeds": True}


class Frontend:
    """``generate(messages, images, params, request_id)`` -> vLLM's final ``RequestOutput``.

    ``workers``: preparing processes; ``vision_device``, ``vision_dtype``, ``vision_fp8``,
    ``max_patches``: the vision tower's card, dtype (``bfloat16``; ``float32`` as a reference),
    FP8 linears, and patches per call."""

    def __init__(
        self,
        llm: AsyncLLM,
        model: str,
        workers: int = 4,
        vision_device: str = "cuda:0",
        vision_dtype: str = "bfloat16",
        vision_fp8: bool = False,
        max_patches: int = 32768,
    ):
        self.llm = llm
        self.prep = Preparer(model, workers)
        dtype = getattr(torch, vision_dtype)
        self.vision = VisionEncoder(model, vision_device, dtype, vision_fp8, max_patches)

    async def inputs(self, messages: list[dict], images: list[bytes]) -> dict:
        """The engine prompt of one request."""
        prepared = await asyncio.wrap_future(self.prep.submit(messages, images))
        prompt = {"prompt_token_ids": prepared.ids}
        if prepared.pixels is not None:
            embeds = await asyncio.wrap_future(self.vision.submit(prepared.pixels, prepared.grid))
            prompt["multi_modal_data"] = {
                "image": {
                    "image_embeds": embeds,
                    "image_grid_thw": torch.from_numpy(prepared.grid),
                }
            }
        return prompt

    async def generate(
        self, messages: list[dict], images: list[bytes], params: SamplingParams, request_id: str
    ):
        prompt = await self.inputs(messages, images)
        final = None
        async for out in self.llm.generate(prompt, params, request_id):
            final = out
        return final

    def close(self) -> None:
        self.prep.close()
        self.vision.close()
