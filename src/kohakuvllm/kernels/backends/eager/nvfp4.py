"""The NVFP4 producers' reference: vLLM's native ``rms_norm`` / ``fused_add_rms_norm`` (the
compiled graph's fp32 math) and its ``scaled_fp4_quant``, written into the given outputs."""

import torch

from kohakuvllm.kernels import registry


def nvfp4_quant(x, gs, q, s) -> None:
    from vllm import _custom_ops as vllm_ops  # noqa: PLC0415 (needs vLLM's compiled ops)

    qq, ss = vllm_ops.scaled_fp4_quant(x, gs)
    q.copy_(qq)
    s.view(-1).copy_(ss.view(torch.uint8).reshape(-1))


def rmsnorm_nvfp4(x, residual, weight, eps, gs0, gs1, res, q0, s0, q1, s1) -> None:
    from vllm.ir import ops as ir  # noqa: PLC0415

    # the native (fp32) norms: what the compiled graph runs, whose rounding the fused kernel keeps
    if residual is not None:
        h, r = ir.fused_add_rms_norm.impls["native"].impl_fn(x, residual, weight, eps)
        res.copy_(r)
    else:
        h = ir.rms_norm.impls["native"].impl_fn(x, weight, eps)
    nvfp4_quant(h, gs0, q0, s0)
    if gs1 is not None:
        nvfp4_quant(h, gs1, q1, s1)


registry.register("nvfp4_quant", "eager", nvfp4_quant)
registry.register("rmsnorm_nvfp4", "eager", rmsnorm_nvfp4)
