"""Keep secrets out of anything that gets written to a log or shown on a page.

Two layers: the exact values of the keys this project loads from .env (so a key that
leaks in any format is still caught), and patterns for the shapes secrets take in
URLs and headers (so one we have never loaded is still caught).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

MASK = "[redacted]"
SECRET_ENV = (
    "OPEN_ROUTER_API_KEY",
    "OPENROUTER_API_KEY",
    "PIXELVAULT_API_KEY",
    "META_ACCESS_TOKEN",
    "LANGSMITH_API_KEY",
    "HF_API_KEY",
)
MIN_SECRET_LENGTH = 8  # a shorter value would mask ordinary words

_PATTERNS = (
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), rf"\1{MASK}"),
    (re.compile(r"(?i)\b(access_token|api[_-]?key|token)=([^&\s\"']+)"), rf"\1={MASK}"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), MASK),
)


def redact(text: str, environ: Mapping[str, str] | None = None) -> str:
    """`text` with every known secret value and secret-shaped token masked."""
    environ = os.environ if environ is None else environ
    for name in SECRET_ENV:
        value = environ.get(name)
        if value and len(value) >= MIN_SECRET_LENGTH:
            text = text.replace(value, MASK)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_deep(value: Any, environ: Mapping[str, str] | None = None) -> Any:
    """`redact` applied to every string inside dicts, lists and tuples."""
    if isinstance(value, str):
        return redact(value, environ)
    if isinstance(value, dict):
        return {key: redact_deep(item, environ) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_deep(item, environ) for item in value]
    return value
