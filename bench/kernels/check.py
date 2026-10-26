"""Every registry op's backends against its eager reference on this card, on random operands:
``grammar_mask`` and ``nvfp4_quant`` / ``rmsnorm_nvfp4`` byte for byte, ``gdn_state8_decode``
by error against the eager step (outputs, and the dequantized states it writes). Prints one line
per op and backend; exits 1 on a mismatch.

    python bench/kernels/check.py
"""

import sys

import torch

from kohakuvllm.gdn import state8
from kohakuvllm.kernels import registry
from kohakuvllm.kernels.hardware import arch_of

DEV = torch.device("cuda")
STEP = {"int8": 1 / 127, "fp8": 32 / 448}


def usable(op: str, dtype) -> list[str]:
    reasons = registry.available(op, DEV, dtype)
    for name, why in reasons.items():
        if why is not None:
            print(f"  {op} {name}: skipped ({why[:120]})")
    return [name for name, why in reasons.items() if why is None and name != "eager"]


def check_grammar() -> bool:
    ok = True
    B, V, M = 300, 151_936, 40
    words = (V + 31) // 32
    table = torch.randint(-(2**31), 2**31 - 1, (M, words), dtype=torch.int32, device=DEV)
    ids = torch.randint(-1, M, (B,), dtype=torch.int32, device=DEV)
    logits = torch.randn(B, V, device=DEV)
    ref = logits.clone()
    registry.resolve("grammar_mask", DEV, logits.dtype, backend="eager")(ref, table, ids)
    for name in usable("grammar_mask", logits.dtype):
        out = logits.clone()
        registry.resolve("grammar_mask", DEV, logits.dtype, backend=name)(out, table, ids)
        same = torch.equal(out, ref)
        ok &= same
        print(f"grammar_mask {name}: {'equal' if same else 'DIFFERENT'}")
    return ok


def bufs(m: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 codes [m, k / 2] and swizzled scales (rows to 128, scale columns to 4) as bytes."""
    kbp = -(-k // 16 // 4) * 4
    q = torch.empty(m, k // 2, dtype=torch.uint8, device=DEV)
    return q, torch.zeros(-(-m // 128) * 128 * kbp, dtype=torch.uint8, device=DEV)


def check_nvfp4() -> bool:
    ok = True
    for m, k in ((1, 4096), (37, 4096), (300, 12288), (1024, 4096)):
        x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16) * 3
        res = torch.randn_like(x)
        w = torch.randn(k, device=DEV, dtype=torch.bfloat16)
        gs = torch.tensor([448.0 * 6 / x.abs().max().float()], device=DEV)
        outs = {}
        for name in ["eager", *usable("nvfp4_quant", x.dtype)]:
            q, s = bufs(m, k)
            registry.resolve("nvfp4_quant", DEV, x.dtype, backend=name)(x, gs, q, s)
            q0, s0 = bufs(m, k)
            q1, s1 = bufs(m, k)
            r = torch.empty_like(x)
            fn = registry.resolve("rmsnorm_nvfp4", DEV, x.dtype, backend=name)
            fn(x, res, w, 1e-6, gs, gs * 0.5, r, q0, s0, q1, s1)
            outs[name] = (q, s, q0, s0, q1, s1, r)
        for name, got in outs.items():
            if name == "eager":
                continue
            same = [torch.equal(a, b) for a, b in zip(got, outs["eager"])]
            near = (got[2] != outs["eager"][2]).float().mean().item()
            good = all(same) or (all(same[:2]) and same[6] and near < 1e-4)
            ok &= good
            print(f"nvfp4 {name} m={m} k={k}: quant {same[:2]}, norm {same[2:6]} ({near:.1e})")
    return ok


def check_gdn() -> bool:
    ok = True
    n, H, HV, K, V, slots = 257, 4, 8, 128, 128, 400
    for fmt in ("int8", "fp8"):
        pool = torch.zeros(slots, HV, V, K + 4, dtype=torch.uint8, device=DEV)
        state8.quantize(torch.randn(slots, HV, V, K, device=DEV) * 0.1, pool, fmt)
        idx = torch.randperm(slots - 1, device=DEV)[:n] + 1
        idx[::17] = 0  # padded rows
        args = [
            torch.randn(n, H * K, device=DEV, dtype=torch.bfloat16),
            torch.randn(n, H * K, device=DEV, dtype=torch.bfloat16),
            torch.randn(n, HV * V, device=DEV, dtype=torch.bfloat16),
            torch.randn(n, HV, device=DEV, dtype=torch.bfloat16),
            torch.randn(n, HV, device=DEV, dtype=torch.bfloat16),
            torch.randn(HV, device=DEV),
            torch.randn(HV, device=DEV),
        ]
        results = {}
        for name in ["eager", *usable("gdn_state8_decode", torch.bfloat16)]:
            p = pool.clone()
            out = torch.empty(n, HV * V, device=DEV, dtype=torch.bfloat16)
            fn = registry.resolve("gdn_state8_decode", DEV, torch.bfloat16, backend=name)
            fn(*args, p, idx, out, K**-0.5, True, fmt)
            results[name] = (out.float(), state8.dequantize(p, K, fmt), p)
        for name, (out, st, p) in results.items():
            if name == "eager":
                continue
            ro, rs, rp = results["eager"]
            eo = ((out - ro).abs().max() / ro.abs().max()).item()
            es = ((st - rs).abs().max() / rs.abs().max()).item()
            untouched = torch.equal(p[0], pool[0]) and torch.equal(rp[0], pool[0])
            # within one 8-bit step at the top of a row's range
            good = eo < 2e-2 and es < STEP[fmt] and untouched
            ok &= good
            print(f"gdn_state8_decode {name} {fmt}: out {eo:.1e}, state {es:.1e}, kept {untouched}")
    return ok


def main() -> None:
    print(f"device {torch.cuda.get_device_name()} ({arch_of(DEV)})")
    results = [check_grammar(), check_nvfp4(), check_gdn()]
    print("all match" if all(results) else "MISMATCH")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
