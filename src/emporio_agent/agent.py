"""The agent loop: message -> LLM -> tools -> LLM -> answer (ReAct style, no framework).

All tool calls in one model response run before the next LLM call. Every
failure ends in a defined answer: the customer always gets a reply and the
agent never fills gaps with guesses.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from emporio_agent import tools
from emporio_agent.config import Settings
from emporio_agent.data import StoreData
from emporio_agent.llm import LLMError, ToolCall
from emporio_agent.policies import PolicyManual
from emporio_agent.prompts import (
    FALLBACK_MESSAGE,
    LOOP_LIMIT_MESSAGE,
    TOO_LONG_MESSAGE,
    build_system_prompt,
)
from emporio_agent.tool_schemas import build_tool_schemas, validate_arguments

MAX_ITERATIONS = 5  # LLM calls per customer message
TIME_BUDGET_SECONDS = 60
MAX_MESSAGE_CHARS = 2000
MAX_HISTORY_TURNS = 10  # a turn = one customer message and everything after it


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict | str
    result: dict
    latency_ms: int

    @property
    def error(self) -> str | None:
        return self.result.get("error") if isinstance(self.result, dict) else None


@dataclass
class TurnResult:
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error_type: str | None = None  # set when a fixed message replaced the LLM answer


class Agent:
    def __init__(
        self,
        llm,
        store: StoreData,
        manual: PolicyManual,
        settings: Settings,
        on_tool: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.llm = llm
        self.store = store
        self.manual = manual
        self.settings = settings
        self.on_tool = on_tool
        self.clock = clock
        self.tool_schemas = build_tool_schemas(store)
        self._schemas_by_name = {t["function"]["name"]: t for t in self.tool_schemas}
        self.history: list[dict] = []
        self.tools_used: list[str] = []

    def reset(self) -> None:
        self.history.clear()
        self.tools_used.clear()

    # --- Tools ---------------------------------------------------------------

    def _tool_functions(self, now: datetime) -> dict[str, Callable[[dict], dict]]:
        store, manual, today = self.store, self.manual, now.date()
        return {
            "search_products": lambda a: tools.search_products(
                store,
                query=a.get("query"),
                category=a.get("category"),
                min_price=a.get("min_price"),
                max_price=a.get("max_price"),
                string_type=a.get("string_type"),
                electronics=a.get("electronics"),
                include_unavailable=bool(a.get("include_unavailable")),
            ),
            "get_product_details": lambda a: tools.get_product_details(store, a["product"]),
            "list_categories": lambda a: tools.list_categories(store),
            "list_active_promotions": lambda a: tools.list_active_promotions(store),
            "calculate_installments": lambda a: tools.calculate_installments(a["amount"]),
            "get_order_status": lambda a: tools.get_order_status(
                store, a["order_id"], a["contact"], today
            ),
            "check_return_eligibility": lambda a: tools.check_return_eligibility(
                store, a["order_id"], a["contact"], a["reason"], today
            ),
            "get_shipping_options": lambda a: tools.get_shipping_options(
                store, a["city"], a["order_value"], a.get("product_ids")
            ),
            "get_store_status": lambda a: tools.get_store_status(now),
            "get_policy": lambda a: tools.get_policy(manual, a["topic"]),
            "register_handoff": lambda a: tools.register_handoff(
                store,
                self.settings.handoffs_path,
                reason=a["reason"],
                summary=a["summary"],
                now=now,
                contact=a.get("contact"),
                order_id=a.get("order_id"),
                suggested_next_action=a.get("suggested_next_action"),
                tools_consulted=list(self.tools_used),
            ),
        }

    def _execute(self, call: ToolCall, functions: dict) -> ToolCallRecord:
        start = self.clock()
        arguments: dict | str = call.arguments
        if call.name not in functions:
            result = {"error": f"Unknown tool: {call.name}.", "retryable": False}
        else:
            try:
                arguments = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                result = {"error": "Arguments are not valid JSON.", "retryable": False}
            else:
                problem = validate_arguments(self._schemas_by_name[call.name], arguments)
                if problem:
                    result = {"error": f"Invalid arguments: {problem}.", "retryable": False}
                else:
                    if self.on_tool:
                        self.on_tool(call.name)
                    try:
                        result = functions[call.name](arguments)
                    except Exception as exc:  # a failing tool must not end the turn
                        result = {"error": f"Tool failed: {type(exc).__name__}: {exc}",
                                  "retryable": True}
                    if call.name not in self.tools_used:
                        self.tools_used.append(call.name)
        latency_ms = int((self.clock() - start) * 1000)
        return ToolCallRecord(call.name, arguments, result, latency_ms)

    # --- History -------------------------------------------------------------

    def _trim_history(self, keep_turns: int) -> None:
        """Drop whole turns from the start, so a tool call is never separated from its
        result (the API rejects orphan tool messages)."""
        user_indexes = [i for i, m in enumerate(self.history) if m["role"] == "user"]
        if len(user_indexes) > keep_turns:
            cut = user_indexes[len(user_indexes) - keep_turns] if keep_turns else len(self.history)
            del self.history[:cut]

    # --- Turn ----------------------------------------------------------------

    def respond(self, message: str) -> TurnResult:
        if len(message) > MAX_MESSAGE_CHARS:
            return TurnResult(TOO_LONG_MESSAGE.format(limit=MAX_MESSAGE_CHARS),
                              error_type="message_too_long")

        start = self.clock()
        now = self.settings.now()
        functions = self._tool_functions(now)
        self._trim_history(MAX_HISTORY_TURNS - 1)
        turn_start = len(self.history)
        self.history.append({"role": "user", "content": message})
        result = TurnResult(answer="")

        for _ in range(MAX_ITERATIONS):
            if self.clock() - start > TIME_BUDGET_SECONDS:
                result.error_type = "time_budget"
                break
            messages = [
                {"role": "system",
                 "content": build_system_prompt(now, self.settings.assistant_name)},
                *self.history,
            ]
            try:
                response = self.llm.complete(messages, self.tool_schemas)
            except Exception as exc:
                # Keep the customer's message, drop this turn's partial tool exchange.
                del self.history[turn_start + 1:]
                self.history.append({"role": "assistant", "content": FALLBACK_MESSAGE})
                result.answer = FALLBACK_MESSAGE
                result.error_type = exc.error_type if isinstance(exc, LLMError) else "unexpected_error"
                return result

            result.iterations += 1
            result.prompt_tokens += response.prompt_tokens
            result.completion_tokens += response.completion_tokens
            self.history.append(response.as_message())

            if not response.tool_calls:
                if response.content and response.content.strip():
                    result.answer = response.content.strip()
                    return result
                self.history[-1] = {"role": "assistant", "content": FALLBACK_MESSAGE}
                result.answer = FALLBACK_MESSAGE
                result.error_type = "empty_response"
                return result

            for call in response.tool_calls:
                record = self._execute(call, functions)
                result.tool_calls.append(record)
                self.history.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(record.result, ensure_ascii=False, default=str),
                })
        else:
            result.error_type = "max_iterations"

        self.history.append({"role": "assistant", "content": LOOP_LIMIT_MESSAGE})
        result.answer = LOOP_LIMIT_MESSAGE
        return result
