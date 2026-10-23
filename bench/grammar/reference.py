"""A plain recursive-descent judge of compact JSON against a schema, written apart from the Rust
matcher: ``judge(schema, data)`` is ``"complete"`` (a whole valid document), ``"prefix"`` (the
start of one) or ``"bad"``. Strings are JSON strings (escapes, no raw control bytes, UTF-8),
lengths counted in characters; numbers are JSON numbers (``integer``: no fraction, no exponent;
at most 24 digits, as the matcher bounds them)."""

import json


class Prefix(Exception):
    """The data ended while everything so far was valid."""


class Bad(Exception):
    pass


MAX_DIGITS = 24


class Judge:
    def __init__(self, data: bytes):
        self.d = data

    def peek(self, i: int) -> int:
        if i >= len(self.d):
            raise Prefix
        return self.d[i]

    def lit(self, i: int, text: bytes) -> int:
        for k, b in enumerate(text):
            if self.peek(i + k) != b:
                raise Bad
        return i + len(text)

    def value(self, s: dict, i: int) -> int:
        if "enum" in s or "const" in s:
            return self.literals(s.get("enum", [s.get("const")]), i)
        t = s.get("type")
        if t == "object":
            return self.obj(s, i)
        if t == "array":
            return self.arr(s, i)
        if t == "string":
            return self.string(i, s.get("minLength", 0), s.get("maxLength"))
        if t in ("integer", "number"):
            return self.number(i, t == "integer")
        if t == "boolean":
            return self.literals([True, False], i)
        if t == "null":
            return self.literals([None], i)
        raise ValueError(t)

    def literals(self, values: list, i: int) -> int:
        texts = [_dump(v) for v in values]
        alive = texts
        k = 0
        while True:
            done = [t for t in alive if len(t) == k]
            longer = [t for t in alive if len(t) > k]
            if i + k >= len(self.d):
                if longer or done:
                    raise Prefix
                raise Bad
            b = self.d[i + k]
            nxt = [t for t in longer if t[k] == b]
            if not nxt:
                if done:
                    return i + k
                raise Bad
            alive = nxt
            k += 1

    def obj(self, s: dict, i: int) -> int:
        props = list(s.get("properties", {}).items())
        req = set(s.get("required", []))
        i = self.lit(i, b"{")
        nxt = 0
        first = True
        while True:
            rest = props[nxt:]
            if self.peek(i) == ord("}"):
                if any(k in req for k, _ in rest):
                    raise Bad
                return i + 1
            if not first:
                i = self.lit(i, b",")
            first = False
            cands = []
            for j, (k, _) in enumerate(rest):
                cands.append(nxt + j)
                if k in req:
                    break
            if not cands:
                raise Bad
            keys = {_dump(props[c][0]): c for c in cands}
            end = self.literals([props[c][0] for c in cands], i)
            c = keys[self.d[i:end]]
            i = self.lit(end, b":")
            i = self.value(props[c][1], i)
            nxt = c + 1

    def arr(self, s: dict, i: int) -> int:
        lo, hi = s.get("minItems", 0), s.get("maxItems")
        i = self.lit(i, b"[")
        n = 0
        while True:
            if self.peek(i) == ord("]"):
                if n < lo:
                    raise Bad
                return i + 1
            if n:
                i = self.lit(i, b",")
            if hi is not None and n >= hi:
                raise Bad
            i = self.value(s["items"], i)
            n += 1

    def string(self, i: int, lo: int, hi) -> int:
        i = self.lit(i, b'"')
        n = 0
        while True:
            b = self.peek(i)
            if b == ord('"'):
                if n < lo:
                    raise Bad
                return i + 1
            if b < 0x20:
                raise Bad
            if b == ord("\\"):
                e = self.peek(i + 1)
                if e in b'"\\/bfnrt':
                    size = 2
                elif e == ord("u"):
                    for k in range(4):
                        if self.peek(i + 2 + k) not in b"0123456789abcdefABCDEF":
                            raise Bad
                    size = 6
                else:
                    raise Bad
            elif b < 0x80:
                size = 1
            elif 0xC2 <= b <= 0xDF:
                size = 2
            elif 0xE0 <= b <= 0xEF:
                size = 3
            elif 0xF0 <= b <= 0xF4:
                size = 4
            else:
                raise Bad
            if b >= 0x80:
                for k in range(1, size):
                    if self.peek(i + k) & 0xC0 != 0x80:
                        raise Bad
            n += 1
            if hi is not None and n > hi:
                raise Bad
            i += size

    def number(self, i: int, integer: bool) -> int:
        digits = 0
        j = i
        if self.peek(j) == ord("-"):
            j += 1
        b = self.peek(j)
        if b == ord("0"):
            j, digits = j + 1, 1
        elif ord("1") <= b <= ord("9"):
            while j < len(self.d) and self.d[j] in b"0123456789":
                j, digits = j + 1, digits + 1
        else:
            raise Bad
        if not integer:
            if j < len(self.d) and self.d[j] == ord("."):
                j += 1
                if self.peek(j) not in b"0123456789":
                    raise Bad
                while j < len(self.d) and self.d[j] in b"0123456789":
                    j, digits = j + 1, digits + 1
            if j < len(self.d) and self.d[j] in b"eE":
                j += 1
                if self.peek(j) in b"+-":
                    j += 1
                if self.peek(j) not in b"0123456789":
                    raise Bad
                while j < len(self.d) and self.d[j] in b"0123456789":
                    j, digits = j + 1, digits + 1
        if digits > MAX_DIGITS:
            raise Bad
        if j >= len(self.d):
            raise Prefix
        return j


def _dump(v) -> bytes:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":")).encode()


def judge(schema: dict, data: bytes) -> str:
    try:
        end = Judge(data).value(schema, 0)
    except Prefix:
        return "prefix"
    except Bad:
        return "bad"
    return "complete" if end == len(data) else "bad"
