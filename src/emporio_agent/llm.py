"""Thin wrapper over the LLM provider. The only module that imports openai.

Any OpenAI-compatible endpoint works through OPENAI_BASE_URL. Timeouts and
retries with exponential backoff come from the SDK itself (no extra library).
FakeLLM replays scripted responses so the agent loop can be tested offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import openai

from emporio_agent.config import Settings

TIMEOUT_SECONDS = 30
MAX_RETRIES = 2
TEMPERATURE = 0.2


class LLMError(RuntimeError):
    """The provider failed after the SDK's own retries."""

    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, as returned by the model


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_message(self) -> dict:
        """The assistant message to append to the conversation history."""
        message: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return message


class OpenAIClient:
    def __init__(self, settings: Settings):
        self.model = settings.model
        self.strict_tools = settings.strict_tools
        self._client = openai.OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
        )

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        if not self.strict_tools:
            tools = [_without_strict(t) for t in tools]
        kwargs = {"tools": tools} if tools else {}
        try:
            response = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=TEMPERATURE, **kwargs
            )
        except openai.APITimeoutError as exc:
            raise LLMError("timeout", str(exc)) from exc
        except openai.APIConnectionError as exc:
            raise LLMError("connection", str(exc)) from exc
        except openai.RateLimitError as exc:
            raise LLMError("rate_limit", str(exc)) from exc
        except openai.AuthenticationError as exc:
            raise LLMError("authentication", str(exc)) from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"http_{exc.status_code}", str(exc)) from exc

        message = response.choices[0].message
        usage = response.usage
        return LLMResponse(
            content=message.content,
            tool_calls=[
                ToolCall(c.id, c.function.name, c.function.arguments or "{}")
                for c in message.tool_calls or []
            ],
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


def _without_strict(tool: dict) -> dict:
    function = {k: v for k, v in tool["function"].items() if k != "strict"}
    return {**tool, "function": function}


class FakeLLM:
    """Replays scripted responses and records what it was sent.

    Each script item is an LLMResponse, an exception to raise, or a callable
    taking the messages and returning one of those.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls.append(json.loads(json.dumps(messages, default=str)))
        if not self.script:
            raise AssertionError("FakeLLM script exhausted")
        item = self.script.pop(0)
        if callable(item) and not isinstance(item, LLMResponse):
            item = item(messages)
        if isinstance(item, Exception):
            raise item
        return item


def tool_call(name: str, call_id: str = "call_1", **arguments) -> ToolCall:
    """Convenience for tests and scripted examples."""
    return ToolCall(call_id, name, json.dumps(arguments))
