"""Settings from environment variables (.env).

OPENAI_API_KEY   required (any non-empty value for local servers such as Ollama)
OPENAI_BASE_URL  optional; any OpenAI-compatible provider (Gemini, Groq, Ollama...)
MODEL            default gpt-4.1-mini
REFERENCE_DATE   optional fixed "now" (YYYY-MM-DD or YYYY-MM-DDTHH:MM; 10:00 if no time)
STRICT_TOOLS     default true; set false for providers that reject strict tool schemas
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

from dotenv import load_dotenv

from emporio_agent.tools import STORE_TZ

ROOT = Path(__file__).resolve().parents[2]
ASSISTANT_NAME = "Tom"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_REFERENCE_TIME = time(10, 0)


class ConfigError(RuntimeError):
    """Raised at startup when the configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str | None
    model: str
    reference_now: datetime | None
    strict_tools: bool = True
    assistant_name: str = ASSISTANT_NAME
    handoffs_path: Path = ROOT / "handoffs.jsonl"
    logs_dir: Path = ROOT / "logs"

    def now(self) -> datetime:
        """Current time in the store's timezone, or the fixed reference time."""
        return self.reference_now or datetime.now(STORE_TZ)


def parse_reference_date(value: str | None) -> datetime | None:
    if not value or not value.strip():
        return None
    value = value.strip()
    try:
        if "T" in value:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M")
        else:
            parsed = datetime.combine(datetime.strptime(value, "%Y-%m-%d").date(),
                                      DEFAULT_REFERENCE_TIME)
    except ValueError as exc:
        raise ConfigError(
            f"REFERENCE_DATE must be YYYY-MM-DD or YYYY-MM-DDTHH:MM, got {value!r}."
        ) from exc
    return parsed.replace(tzinfo=STORE_TZ)


def _bool(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def load_settings(require_api_key: bool = True, env_file: Path = ROOT / ".env") -> Settings:
    # override=True: the project's .env wins over variables already set in the
    # system (e.g. an old OPENAI_API_KEY in the user's environment).
    load_dotenv(env_file, override=True)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise ConfigError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in "
            "(for a local server such as Ollama, any non-empty value works)."
        )
    return Settings(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
        model=os.getenv("MODEL", "").strip() or DEFAULT_MODEL,
        reference_now=parse_reference_date(os.getenv("REFERENCE_DATE")),
        strict_tools=_bool(os.getenv("STRICT_TOOLS"), True),
    )
