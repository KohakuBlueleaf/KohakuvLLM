"""Each token's bytes, from a Hugging Face tokenizer (byte-level BPE or SentencePiece)."""

import re

from transformers import GenerationConfig

_BYTE_PIECE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def byte_decoder() -> dict[str, int]:
    """GPT-2's byte-level alphabet: printable stand-in character -> byte."""
    keep = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1)]
    keep += range(ord("®"), ord("ÿ") + 1)
    chars, n = list(keep), 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(keep, chars)}


def token_bytes(tokenizer, vocab: int) -> list[bytes | None]:
    """``vocab`` entries: each id's bytes; None for special tokens and ids past the tokenizer."""
    pieces = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    special = set(tokenizer.all_special_ids) | set(tokenizer.added_tokens_decoder)
    byte_level = any(p is not None and "Ġ" in p for p in pieces)
    decoder = byte_decoder() if byte_level else None
    out: list[bytes | None] = [None] * vocab
    for i, p in enumerate(pieces[:vocab]):
        if p is None or i in special:
            continue
        if decoder is not None:
            if all(c in decoder for c in p):
                out[i] = bytes(decoder[c] for c in p)
        elif m := _BYTE_PIECE.fullmatch(p):
            out[i] = bytes([int(m.group(1), 16)])
        else:
            out[i] = p.replace("▁", " ").encode()
    return out


def eos_ids(tokenizer, model: str) -> list[int]:
    """The tokenizer's end-of-sequence id and the generation config's."""
    ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    try:
        eos = GenerationConfig.from_pretrained(model).eos_token_id
    except OSError:
        eos = None
    ids |= set(eos if isinstance(eos, list) else [] if eos is None else [eos])
    return sorted(ids)
