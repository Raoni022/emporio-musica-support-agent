"""One JSON line per turn in logs/traces.jsonl, with e-mails and phones masked.

The trace shows which tools were called, with which arguments, how long they
took and what failed, plus tokens and an estimated cost. It is the base for
scripts/eval.py. Writing a trace never breaks the conversation.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from emporio_agent.agent import TurnResult
from emporio_agent.policies import OFFICIAL_EMAIL, OFFICIAL_PHONE

# USD per 1M tokens (input, output), from the providers' public price lists.
# An estimate: unknown models (free tiers, local models) get no cost.
MODEL_PRICES = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
}

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Brazilian phones: optional +55, DDD with or without parentheses, 8 or 9 digits.
# Lookarounds keep tracking codes (BRCX9876543BR), prices and dates untouched.
_PHONE_RE = re.compile(
    r"(?<![\w.,/])(?:\+?55[\s-]?)?\(?(\d{2})\)?[\s-]?(9?\d{4})[\s-]?(\d{4})(?![\w.,/])"
)
_STORE_CONTACTS = {OFFICIAL_EMAIL, OFFICIAL_PHONE}


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[:1]}***@{domain}"


def mask_phone(ddd: str, first: str, last: str) -> str:
    return f"({ddd}) {first[0]}{'*' * (len(first) - 1)}-{last}"


def mask_text(text: str) -> str:
    """Mask customer e-mails and phones; the store's own contact is kept."""
    text = _EMAIL_RE.sub(lambda m: m[0] if m[0] in _STORE_CONTACTS else mask_email(m[0]), text)
    return _PHONE_RE.sub(
        lambda m: m[0] if m[0] in _STORE_CONTACTS else mask_phone(m[1], m[2], m[3]), text
    )


def mask(value):
    """Mask PII in any JSON-like structure."""
    if isinstance(value, str):
        return mask_text(value)
    if isinstance(value, dict):
        return {k: mask(v) for k, v in value.items()}
    if isinstance(value, list):
        return [mask(v) for v in value]
    return value


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    prices = MODEL_PRICES.get(model)
    if prices is None:
        return None
    return round((prompt_tokens * prices[0] + completion_tokens * prices[1]) / 1_000_000, 6)


def build_trace(
    turn: TurnResult, user_message: str, session_id: str, model: str, timestamp: datetime
) -> dict:
    tool_calls = []
    policy_versions = []
    for call in turn.tool_calls:
        tool_calls.append({
            "name": call.name,
            "arguments": call.arguments,
            "latency_ms": call.latency_ms,
            "error": call.error,
        })
        if call.name == "get_policy" and "section" in call.result:
            policy_versions.append({"section": call.result["section"],
                                    "version": call.result["version"]})
    handoff_unanswered = any(
        c.name == "register_handoff" and isinstance(c.arguments, dict)
        and c.arguments.get("reason") == "unanswered"
        for c in turn.tool_calls
    )
    return mask({
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "session_id": session_id,
        "model": model,
        "user_message": user_message,
        "tool_calls": tool_calls,
        "policy_versions": policy_versions,
        "iterations": turn.iterations,
        "prompt_tokens": turn.prompt_tokens,
        "completion_tokens": turn.completion_tokens,
        "estimated_cost_usd": estimate_cost(model, turn.prompt_tokens, turn.completion_tokens),
        "latency_ms": turn.latency_ms,
        "error_type": turn.error_type,
        "resolved": turn.error_type is None and not handoff_unanswered,
        "final_response": turn.answer,
    })


class Tracer:
    def __init__(self, path: Path, session_id: str, model: str):
        self.path = Path(path)
        self.session_id = session_id
        self.model = model

    def record(self, turn: TurnResult, user_message: str, timestamp: datetime) -> dict | None:
        """Append the turn's trace. Returns it, or None if it could not be written."""
        trace = build_trace(turn, user_message, self.session_id, self.model, timestamp)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
        except OSError:
            return None
        return trace
