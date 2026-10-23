"""SmoothQuant for a BF16 Qwen3.5 checkpoint: migrate activation outliers into the weights of the
linears that consume them, writing a mathematically equivalent BF16 checkpoint (same tensor names,
shapes and config) that quantizes to W8A8 with less activation error.

For a group of linears sharing one input X, the per-input-channel factor is
``s_j = max|X_j|^alpha / max|W_j|^(1 - alpha)`` (``max|W_j|`` over every weight of the group,
column j). The op producing X is divided by s and the group's columns are multiplied by it.
Groups of the text model, ``model.language_model.layers.N``:

- ``attn_in``: ``input_layernorm`` -> ``self_attn.q_proj/k_proj/v_proj`` (full-attention layers).
- ``gdn_in``: ``input_layernorm`` -> ``linear_attn.in_proj_qkv/in_proj_z/in_proj_b/in_proj_a``
  (GDN layers; b and a are scaled too since their input changes, whatever their precision).
- ``mlp_in``: ``post_attention_layernorm`` -> ``mlp.gate_proj/up_proj``.
- ``down``: rows of ``mlp.up_proj`` -> ``mlp.down_proj`` (``act(gate) * up`` is linear in each
  row of up).
- ``o_proj``: rows of ``self_attn.v_proj`` -> ``self_attn.o_proj``. The o_proj input is
  ``(softmax(qk) v) * sigmoid(gate)`` per channel, linear in v, and every q head of a kv head reads
  the same v rows, so s is per (kv head, dim), max over that kv head's q heads.
- ``gdn_out``: ``linear_attn.norm.weight`` -> ``linear_attn.out_proj``. The gated norm is
  ``weight * rmsnorm(x) * silu(z)`` with one ``[head_v_dim]`` weight shared by every head, so s
  is per dim, max over heads.

The decoder RMSNorm gain is ``1 + weight``, stored as ``bf16(g / s - 1)``; consumers take the
factor the stored weight realises, ``g / (1 + bf16(w'))``, absorbing the coarse bf16 rounding
near w' = -1 (a gain rounding to 0 is held at +-2^-8). The gated norm (plain ``weight``) folds
the same way without the offset. Inner groups (down, o_proj, gdn_out) fold first so the outer
groups' ``max|W_j|`` sees the weights that get quantized.

alpha is ``--alpha``, ``--group-alpha GROUP=A`` per group, or, for the groups in ``--search``,
chosen per layer from ``--grid`` (or no smoothing, when that is lower) by the W8A8 error it
gives on stored calibration tokens: the
mean over tokens of ``|Y_q - Y|^2 / |Y|^2`` summed over the group's quantized consumers, with
per-token symmetric INT8 activations and per-row INT8 weights (bf16 scale) as ``quantize.py``
writes them (gdn_in measures in_proj_qkv/z; b/a stay BF16).

Calibration runs forward pre-hooks on the HF model (bf16, one unpadded sample per forward) over
``--samples`` ultrachat_200k ``train_sft`` chats (chat template, ``enable_thinking=False``,
``--seq-len`` tokens). It records each tapped input's per-channel absmax and, for the search,
``--search-tokens`` random token rows spread over the samples plus each sample's largest-norm
row. ``--stats`` caches both, so a sweep calibrates once.

    python tools/smooth.py qwen35-4b-bf16 qwen35-4b-sq --alpha 0.5 --stats sq_stats.pt
    python tools/smooth.py qwen35-4b-bf16 qwen35-4b-sqs --stats sq_stats.pt \
        --groups attn_in gdn_in mlp_in down o_proj gdn_out --search down mlp_in
"""

import argparse
import math
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

PREFIX = "model.language_model.layers"
GROUPS = ("attn_in", "gdn_in", "mlp_in", "down", "o_proj", "gdn_out")
DEFAULT_GROUPS = ("attn_in", "gdn_in", "mlp_in", "down")
# group -> (module whose input is recorded, layer type it exists in or None for every layer)
TAPS = {
    "attn_in": ("self_attn.q_proj", "full_attention"),
    "gdn_in": ("linear_attn.in_proj_qkv", "linear_attention"),
    "mlp_in": ("mlp.gate_proj", None),
    "down": ("mlp.down_proj", None),
    "o_proj": ("self_attn.o_proj", "full_attention"),
    "gdn_out": ("linear_attn.out_proj", "linear_attention"),
}
MIN_GAIN = 2.0**-8
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Group:
    name: str
    tap: str  # linear whose input X is smoothed
    producer: str  # tensor that takes 1/s
    offset: float | None  # gain offset of a norm producer; None: the producer's rows take 1/s
    consumers: tuple[str, ...]  # linears whose columns take s
    quantized: tuple[str, ...]  # consumers the search measures
    to_s: Callable  # column-space vector -> s space
    to_cols: Callable  # s space -> consumer columns
    to_rows: Callable  # s space -> producer rows


def calibrate(src: Path, samples: int, seq_len: int, seed: int, n_tokens: int) -> dict:
    """Per-input-channel absmax and sampled token rows of every tapped linear's input."""
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16).cuda().eval()
    text_cfg = model.config.get_text_config()
    absmax: dict[str, torch.Tensor] = {}
    rows: dict[str, list[torch.Tensor]] = {}
    pick = {"idx": None}
    gen = torch.Generator().manual_seed(seed)

    def tap(name: str):
        def hook(_mod, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1])
            a = x.abs().amax(0).float()
            absmax[name] = torch.maximum(absmax[name], a) if name in absmax else a
            idx = torch.cat([pick["idx"], x.float().norm(dim=-1).argmax()[None].cpu()])
            rows.setdefault(name, []).append(x[idx.cuda()].cpu())

        return hook

    handles = []
    for i, layer in enumerate(model.model.language_model.layers):
        for sub, kind in TAPS.values():
            if kind in (None, text_cfg.layer_types[i]):
                handles.append(
                    layer.get_submodule(sub).register_forward_pre_hook(tap(f"{PREFIX}.{i}.{sub}"))
                )

    data = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=seed)
    per = math.ceil(n_tokens / samples)
    tokens = 0
    with torch.no_grad():
        for n in range(samples):
            text = tok.apply_chat_template(
                data[n]["messages"], tokenize=False, enable_thinking=False
            )
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :seq_len]
            pick["idx"] = torch.randperm(ids.shape[1], generator=gen)[:per]
            model.model(input_ids=ids.cuda(), use_cache=False)
            tokens += ids.shape[1]
            if (n + 1) % 64 == 0:
                print(f"calibrated {n + 1}/{samples} samples, {tokens} tokens", flush=True)
    for h in handles:
        h.remove()
    print(f"calibration: {samples} samples, {tokens} tokens, {len(absmax)} taps", flush=True)
    return {
        "absmax": {k: v.cpu() for k, v in absmax.items()},
        "tokens": {k: torch.cat(v) for k, v in rows.items()},
    }


def factor(ax: torch.Tensor, aw: torch.Tensor, alpha: float) -> torch.Tensor:
    return ax.clamp_min(1e-5).pow(alpha) / aw.clamp_min(1e-5).pow(1 - alpha)


def col_absmax(*ws: torch.Tensor) -> torch.Tensor:
    return torch.stack([w.float().abs().amax(0) for w in ws]).amax(0)


def fake_int8(x: torch.Tensor, bf16_scale: bool) -> torch.Tensor:
    """Symmetric per-row INT8 round trip in fp32."""
    scale = (x.abs().amax(dim=-1, keepdim=True) / 127).clamp_min(1e-12)
    if bf16_scale:
        scale = scale.bfloat16().float()
    return (x / scale).round().clamp(-127, 127) * scale


def w8a8_error(x: torch.Tensor, ws: list[torch.Tensor], s: torch.Tensor) -> float:
    """Mean per-token relative squared output error of W8A8 with columns scaled by s, summed."""
    xq = fake_int8((x / s).bfloat16().float(), False)
    err = 0.0
    for w in ws:
        y = x @ w.T
        yq = xq @ fake_int8((w * s).bfloat16().float(), True).T
        err += ((yq - y).pow(2).sum(-1) / y.pow(2).sum(-1).clamp_min(1e-12)).mean().item()
    return err


def fold_gain(w: torch.Tensor, s: torch.Tensor, offset: float):
    """Divide the gain ``offset + w`` by s; returns the stored weight and the factor it realises."""
    g = w.float() + offset
    w_new = (g / s - offset).to(w.dtype)
    g_new = w_new.float() + offset
    dead = (g_new == 0) | (torch.sign(g_new) != torch.sign(g))
    if dead.any():
        hold = (torch.sign(g) * MIN_GAIN - offset).to(w.dtype)
        w_new = torch.where(dead, hold, w_new)
        g_new = w_new.float() + offset
    zero = g == 0
    w_new = torch.where(zero, w, w_new)
    s_eff = torch.where(zero, torch.ones_like(g), g / torch.where(zero, 1.0, g_new))
    return w_new, s_eff


def layer_groups(kind: str, cfg) -> list[Group]:
    """The layer's groups in fold order: inner (row/gated-norm producers) before outer (norms)."""

    def same(v: torch.Tensor) -> torch.Tensor:
        return v

    heads, kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    nv, vd = cfg.linear_num_value_heads, cfg.linear_value_head_dim
    d = ("mlp.down_proj",)
    down = Group("down", "mlp.down_proj", "mlp.up_proj", None, d, d, same, same, same)
    mlp = ("mlp.gate_proj", "mlp.up_proj")
    mlp_in = Group(
        "mlp_in", "mlp.gate_proj", "post_attention_layernorm", 1.0, mlp, mlp, same, same, same
    )
    if kind == "full_attention":
        o = ("self_attn.o_proj",)
        o_proj = Group(
            "o_proj",
            "self_attn.o_proj",
            "self_attn.v_proj",
            None,
            o,
            o,
            lambda v: v.view(kv, heads // kv, hd).amax(1),
            lambda s: s.repeat_interleave(heads // kv, dim=0).flatten(),
            lambda s: s.flatten(),
        )
        qkv = tuple(f"self_attn.{x}_proj" for x in "qkv")
        inp = Group(
            "attn_in", "self_attn.q_proj", "input_layernorm", 1.0, qkv, qkv, same, same, same
        )
        return [down, o_proj, mlp_in, inp]
    o = ("linear_attn.out_proj",)
    gdn_out = Group(
        "gdn_out",
        "linear_attn.out_proj",
        "linear_attn.norm",
        0.0,
        o,
        o,
        lambda v: v.view(nv, vd).amax(0),
        lambda s: s.repeat(nv),
        same,
    )
    gdn = tuple(f"linear_attn.in_proj_{x}" for x in ("qkv", "z", "b", "a"))
    inp = Group(
        "gdn_in", "linear_attn.in_proj_qkv", "input_layernorm", 1.0, gdn, gdn[:2], same, same, same
    )
    return [down, gdn_out, mlp_in, inp]


def smooth_layer(t: dict, stats: dict, i: int, kind: str, cfg, args) -> list[str]:
    """Fold the chosen groups of layer i into the tensors of ``t`` in place; returns a report."""
    p = f"{PREFIX}.{i}"
    report = []
    for g in layer_groups(kind, cfg):
        if g.name not in args.groups:
            continue
        ws = [t[f"{p}.{c}.weight"] for c in g.consumers]
        ax = g.to_s(stats["absmax"][f"{p}.{g.tap}"].float())
        aw = g.to_s(col_absmax(*ws))
        alpha, line = args.alpha[g.name], ""
        if g.name in args.search:
            x = stats["tokens"][f"{p}.{g.tap}"].to(DEV).float()
            wq = [t[f"{p}.{c}.weight"].to(DEV).float() for c in g.quantized]
            errs = [w8a8_error(x, wq, g.to_cols(factor(ax, aw, a).to(DEV))) for a in args.grid]
            alpha = args.grid[min(range(len(errs)), key=errs.__getitem__)]
            base = w8a8_error(x, wq, torch.ones(x.shape[1], device=DEV))
            line = f" searched: err {min(errs):.3e} (unsmoothed {base:.3e}); grid " + " ".join(
                f"{a:g}:{e:.2e}" for a, e in zip(args.grid, errs)
            )
            del x, wq
            if base <= min(errs):
                report.append(f"{p} {g.name}: left unsmoothed;{line}")
                continue
        s = factor(ax, aw, alpha)
        prod = f"{p}.{g.producer}.weight"
        if g.offset is None:
            t[prod] = (t[prod].float() / g.to_rows(s)[:, None]).to(t[prod].dtype)
            s_eff = s
        else:
            t[prod], s_eff = fold_gain(t[prod], s, g.offset)
        cols = g.to_cols(s_eff)
        for c, w in zip(g.consumers, ws):
            t[f"{p}.{c}.weight"] = (w.float() * cols[None, :]).to(w.dtype)
        report.append(
            f"{p} {g.name}: alpha {alpha:g}, s in [{s.min():.3g}, {s.max():.3g}] median "
            f"{s.median():.3g}, realised/target max {(s_eff / s - 1).abs().max():.2e}" + line
        )
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument(
        "--group-alpha", nargs="*", default=[], metavar="GROUP=ALPHA", help="per-group override"
    )
    p.add_argument("--groups", nargs="+", choices=GROUPS, default=list(DEFAULT_GROUPS))
    p.add_argument("--search", nargs="*", choices=GROUPS, default=[])
    p.add_argument("--grid", nargs="+", type=float, default=[i / 20 for i in range(21)])
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--search-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stats", help="calibration cache: loaded if present, else written")
    args = p.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    alpha = dict.fromkeys(GROUPS, args.alpha)
    for kv in args.group_alpha:
        g, v = kv.split("=")
        if g not in GROUPS:
            p.error(f"unknown group {g!r}")
        alpha[g] = float(v)
    args.alpha = alpha
    cfg = AutoConfig.from_pretrained(src).get_text_config()

    if args.stats and Path(args.stats).exists():
        stats = torch.load(args.stats)
        print(f"loaded {len(stats['absmax'])} taps from {args.stats}", flush=True)
    else:
        stats = calibrate(src, args.samples, args.seq_len, args.seed, args.search_tokens)
        if args.stats:
            torch.save(stats, args.stats)
        torch.cuda.empty_cache()

    shards = sorted(src.glob("*.safetensors"))
    tensors, meta, owner = {}, {}, {}
    for shard in shards:
        with safe_open(shard, "pt") as f:
            meta[shard.name] = f.metadata()
            for k in f.keys():  # noqa: SIM118
                tensors[k] = f.get_tensor(k)
                owner[k] = shard.name
    before = {k: (v.shape, v.dtype) for k, v in tensors.items()}
    for i, kind in enumerate(cfg.layer_types[: cfg.num_hidden_layers]):
        for line in smooth_layer(tensors, stats, i, kind, cfg, args):
            print(line, flush=True)
    assert before == {k: (v.shape, v.dtype) for k, v in tensors.items()}

    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy(f, dst / f.name)
    for shard in shards:
        part = {k: v.contiguous() for k, v in tensors.items() if owner[k] == shard.name}
        save_file(part, dst / shard.name, metadata=meta[shard.name])
    print(f"smoothed {args.groups} alpha {alpha} search {args.search} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
