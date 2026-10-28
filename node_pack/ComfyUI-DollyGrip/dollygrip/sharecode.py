"""Share codes: a camera move as a short copy-pasteable string.

A code is "dg1-" + base64url(zlib(json)). It carries the canonical DSL and
strength (and optionally duration), which is everything needed to rebuild
the exact trajectory.
"""

from __future__ import annotations

import base64
import json
import zlib

from .trajectory import format_dsl, parse

PREFIX = "dg1-"


def encode(dsl: str, strength: float = 1.0, duration_s: float | None = None) -> str:
    payload = {"d": format_dsl(parse(dsl)), "s": round(float(strength), 3)}
    if duration_s is not None:
        payload["t"] = round(float(duration_s), 3)
    raw = zlib.compress(json.dumps(payload, separators=(",", ":")).encode(), 9)
    return PREFIX + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode(code: str) -> dict:
    code = code.strip()
    if not code.startswith(PREFIX):
        raise ValueError("not a DollyGrip share code")
    body = code[len(PREFIX):]
    raw = zlib.decompress(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    payload = json.loads(raw)
    parse(payload["d"])  # validate
    return {"dsl": payload["d"], "strength": payload.get("s", 1.0), "duration_s": payload.get("t")}
