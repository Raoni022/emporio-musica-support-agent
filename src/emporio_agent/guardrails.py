"""Output guardrail: every R$ value in an answer must come from the tools.

The prompt can be talked out of its rules; this check cannot (OWASP LLM01).
It is deterministic and runs before the customer reads the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TOLERANCE = 0.05  # cents of rounding

_BRL_RE = re.compile(r"R\$\s*(\d[\d.,]*\d|\d)")


def parse_brl(text: str) -> float | None:
    """'1.967,18' -> 1967.18, '2.199' -> 2199, '196,72' -> 196.72, '1967.18' -> 1967.18."""
    text = text.strip(".,")
    if "," in text:
        integer, _, decimals = text.rpartition(",")
        integer = integer.replace(".", "")
        if not integer.isdigit() or not decimals.isdigit() or len(decimals) > 2:
            return None
        return float(f"{integer}.{decimals}")
    if "." in text:
        head, *groups = text.split(".")
        if groups and all(len(g) == 3 for g in groups):  # thousands separators
            return float(head + "".join(groups))
        if len(groups) == 1 and len(groups[0]) <= 2:  # decimal point
            return float(text)
        return None
    return float(text)


def extract_amounts(text: str) -> list[float]:
    """Every R$ value written in a text."""
    values = (parse_brl(m) for m in _BRL_RE.findall(text or ""))
    return [v for v in values if v is not None]


def collect_numbers(value, found: set[float] | None = None) -> set[float]:
    """All numbers in a JSON-like structure, plus R$ values written inside its strings
    (e.g. the policy text: "parcela mínima de R$ 80,00")."""
    found = set() if found is None else found
    if isinstance(value, bool):
        return found
    if isinstance(value, (int, float)):
        found.add(float(value))
    elif isinstance(value, str):
        found.update(extract_amounts(value))
    elif isinstance(value, dict):
        for item in value.values():
            collect_numbers(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            collect_numbers(item, found)
    return found


def format_brl(value: float) -> str:
    text = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {text}"


@dataclass
class GuardrailResult:
    ok: bool
    unverified: list[float] = field(default_factory=list)


def validate_response(answer: str, sources: list) -> GuardrailResult:
    """Check that every R$ value in the answer appears in the sources (tool results,
    tool arguments and the customer's own messages), within a few cents."""
    allowed = collect_numbers(sources)
    unverified = [
        amount for amount in extract_amounts(answer)
        if not any(abs(amount - known) <= TOLERANCE for known in allowed)
    ]
    return GuardrailResult(ok=not unverified, unverified=unverified)
