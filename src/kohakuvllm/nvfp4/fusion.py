"""A torch.compile pass that hands NVFP4 activation quantization to our producers (``nvfp4.ops``),
by rewriting vLLM's post-grad graph directly:

- ``norm``: an RMSNorm (``fused_add_rms_norm``, or ``rms_norm`` without a residual) whose output
  feeds only NVFP4 quantizations (through views), one or two of them (a GDN layer quantizes the
  normed input once per projection, each with its own tensor scale), becomes one kernel: the
  normed activation never leaves registers (vLLM's own fusion takes a bf16 weight and one
  quantization; Qwen3.5 has an fp32 ``1 + w`` and quantizes twice in GDN layers).
- ``quant``: every remaining NVFP4 quantization runs our kernel.

``install()`` appends the pass to vLLM's post-grad pass manager (before IR lowering) and folds
its source into ``VllmConfig.compute_hash``: vLLM keys its compile caches (the AOT-compiled model
included) on that hash, and would otherwise load a graph compiled without the pass.
"""

import hashlib
import inspect
import logging
import operator
import sys

import torch
from torch import fx
from torch._guards import detect_fake_mode
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from vllm.compilation.passes import pass_manager
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig

from kohakuvllm.nvfp4 import kernels, ops

# Under vLLM's logger, so the lines reach vLLM's handler in every process
log = logging.getLogger("vllm.kohakuvllm")
aten = torch.ops.aten
QUANT = torch.ops._C.scaled_fp4_quant.out
NORMS = {
    torch.ops.vllm_ir.fused_add_rms_norm.default: True,
    torch.ops.vllm_ir.rms_norm.default: False,
}
VIEWS = (aten.view.default, aten.reshape.default, aten._unsafe_view.default)
# Which rewrites run: "norm" (norm + quantization) and "quant" (the remaining quantizations)
PARTS = {"norm", "quant"}


def is_quant(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is auto_functionalized
        and node.args[0] is QUANT
        and node.kwargs.get("is_sf_swizzled_layout") is True
    )


def getitems(node: fx.Node) -> dict[int, fx.Node]:
    return {u.args[1]: u for u in node.users if u.target is operator.getitem}


def fed_quants(node: fx.Node) -> tuple[list, list] | None:
    """The quantizations ``node`` feeds (through views keeping its shape) and those views; None
    when anything else reads it."""
    shape = node.meta["val"].shape
    quants, views, stack = [], [], [node]
    while stack:
        cur = stack.pop()
        for u in cur.users:
            if u.op == "call_function" and u.target in VIEWS and u.meta["val"].shape == shape:
                views.append(u)
                stack.append(u)
            elif is_quant(u) and u.kwargs["input"] is cur:
                quants.append(u)
            else:
                return None
    return quants, views


def call(graph: fx.Graph, op, args: tuple, fake, outputs: int) -> list[fx.Node]:
    """Insert ``op(*args)`` and its ``outputs`` getitems, their meta values from ``fake``."""
    node = graph.call_function(op, args)
    vals = [a.meta["val"] if isinstance(a, fx.Node) else a for a in args]
    with detect_fake_mode([v for v in vals if isinstance(v, torch.Tensor)]):
        out = fake(*vals)
    node.meta["val"] = out
    items = []
    for i in range(outputs):
        item = graph.call_function(operator.getitem, (node, i))
        item.meta["val"] = out[i]
        items.append(item)
    return items


def take_quant(graph: fx.Graph, quant: fx.Node, q: fx.Node, s: fx.Node) -> None:
    """Route ``quant``'s codes and scales to ``q`` / ``s`` and drop it."""
    for i, item in getitems(quant).items():
        if i in (1, 2):
            item.replace_all_uses_with(q if i == 1 else s)
        graph.erase_node(item)
    graph.erase_node(quant)


def fuse_norms(graph: fx.Graph) -> int:
    order = {n: i for i, n in enumerate(graph.nodes)}
    done = 0
    for norm in list(graph.nodes):
        if norm.op != "call_function" or norm.target not in NORMS:
            continue
        has_res = NORMS[norm.target]
        x, *rest = norm.args
        residual = rest.pop(0) if has_res else None
        weight, eps = rest[0], rest[1]
        if (len(rest) > 2 and rest[2] is not None) or norm.kwargs.get("variance_size") is not None:
            continue
        if weight is None or x.meta["val"].dtype not in (torch.bfloat16, torch.float16):
            continue
        items = getitems(norm) if has_res else {0: norm}
        if 0 not in items:
            continue
        found = fed_quants(items[0])
        if found is None or not 1 <= len(found[0]) <= 2:
            continue
        quants, views = found
        first = min(quants, key=order.__getitem__)
        scales = [q.kwargs["input_scale"] for q in quants]
        late = [u for u in (items[1].users if 1 in items else ()) if order[u] < order[first]]
        if late or any(order.get(s, -1) > order[first] for s in scales):
            continue
        with graph.inserting_before(first):
            q0, s0, q1, s1, res = call(
                graph,
                torch.ops.vllm.kohaku_rmsnorm_nvfp4.default,
                (x, residual, weight, eps, scales[0], scales[1] if len(scales) > 1 else None),
                ops._rmsnorm_nvfp4_fake,
                5,
            )
        for quant, (q, s) in zip(quants, ((q0, s0), (q1, s1))):
            take_quant(graph, quant, q, s)
        for view in reversed(views):
            graph.erase_node(view)
        if has_res:
            items[1].replace_all_uses_with(res)
            graph.erase_node(items[1])
            graph.erase_node(items[0])
        graph.erase_node(norm)
        done += 1
    return done


def replace_quants(graph: fx.Graph) -> int:
    done = 0
    for quant in list(graph.nodes):
        if not is_quant(quant):
            continue
        with graph.inserting_before(quant):
            q, s = call(
                graph,
                torch.ops.vllm.kohaku_nvfp4_quant.default,
                (quant.kwargs["input"], quant.kwargs["input_scale"]),
                ops._nvfp4_quant_fake,
                2,
            )
        take_quant(graph, quant, q, s)
        done += 1
    return done


class Nvfp4FusionPass(VllmInductorPass):
    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        norms = fuse_norms(graph) if "norm" in PARTS else 0
        quants = replace_quants(graph) if "quant" in PARTS else 0
        graph.eliminate_dead_code()
        log.info("kohakuvllm nvfp4: %d norm+quant fused, %d quantizations ours", norms, quants)

    def uuid(self) -> str:
        return self.hash_source(self, kernels, ops, f"parts={sorted(PARTS)}")


def source_hash() -> str:
    text = "".join(inspect.getsource(m) for m in (kernels, ops, sys.modules[__name__]))
    return hashlib.sha256((text + repr(sorted(PARTS))).encode()).hexdigest()


def install(parts: set[str]) -> None:
    PARTS.clear()
    PARTS.update(parts)
    configure = pass_manager.PostGradPassManager.configure
    if getattr(configure, "_kohaku", False):
        return

    def configured(self, config: VllmConfig) -> None:
        configure(self, config)
        self.add(Nvfp4FusionPass(config))

    configured._kohaku = True
    pass_manager.PostGradPassManager.configure = configured

    compute = VllmConfig.compute_hash

    def compute_hash(self, include_version: bool = True) -> str:
        base = compute(self, include_version)
        return hashlib.sha256((base + source_hash()).encode()).hexdigest()[: len(base)]

    VllmConfig.compute_hash = compute_hash
