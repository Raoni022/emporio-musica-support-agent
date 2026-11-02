import json
from datetime import datetime

import pytest

from emporio_agent.agent import ToolCallRecord, TurnResult
from emporio_agent.prompts import FALLBACK_MESSAGE
from emporio_agent.tools import STORE_TZ
from emporio_agent.tracing import Tracer, build_trace, estimate_cost, mask, mask_text

NOW = datetime(2026, 3, 25, 10, 0, tzinfo=STORE_TZ)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("meu e-mail é lucas.mendes@jmail.com", "meu e-mail é l***@jmail.com"),
        ("(67) 99812-3456", "(67) 9****-3456"),
        ("67998123456", "(67) 9****-3456"),
        ("+55 67 99812-3456", "(67) 9****-3456"),
        ("67 3341-4445", "(67) 3***-4445"),  # landline, not the store's
    ],
)
def test_customer_contacts_are_masked(text, expected):
    assert mask_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "R$ 2.199,00 em 12x de R$ 183,25",
        "pedido 8732 de 2026-03-25T10:00",
        "rastreio BRCX9876543BR",
        "CNPJ 12.345.678/0001-90",
        "WhatsApp (67) 3341-4444 ou contato@emporiodamusica.com.br",  # the store's own contact
    ],
)
def test_non_personal_values_are_kept(text):
    assert mask_text(text) == text


def test_mask_walks_nested_structures():
    masked = mask({"args": {"contact": "isabela.moreira@jmail.com"}, "list": ["67998123456", 3]})
    assert masked == {"args": {"contact": "i***@jmail.com"}, "list": ["(67) 9****-3456", 3]}


def test_fallback_message_survives_masking_intact():
    assert mask_text(FALLBACK_MESSAGE) == FALLBACK_MESSAGE


def test_cost_estimate():
    assert estimate_cost("gpt-4.1-mini", 1_000_000, 1_000_000) == 2.0
    assert estimate_cost("gpt-4.1-mini", 1500, 200) == 0.00092
    assert estimate_cost("gemini-2.5-flash", 1500, 200) is None


def turn(**overrides):
    defaults = dict(
        answer="Achei seu pedido, Isabela!",
        tool_calls=[
            ToolCallRecord("get_order_status",
                           {"order_id": "3", "contact": "isabela.moreira@jmail.com"},
                           {"verified": True}, 12),
            ToolCallRecord("get_policy", {"topic": "returns_and_exchanges"},
                           {"section": 4, "version": "Versão 2.1 — Junho de 2025", "text": "..."}, 1),
        ],
        iterations=2, prompt_tokens=1500, completion_tokens=200, latency_ms=2300,
    )
    return TurnResult(**{**defaults, **overrides})


def test_trace_fields():
    trace = build_trace(turn(), "pedido 3, isabela.moreira@jmail.com", "s1", "gpt-4.1-mini", NOW)
    assert trace["user_message"] == "pedido 3, i***@jmail.com"
    assert trace["tool_calls"][0] == {
        "name": "get_order_status",
        "arguments": {"order_id": "3", "contact": "i***@jmail.com"},
        "latency_ms": 12,
        "error": None,
    }
    assert trace["policy_versions"] == [{"section": 4, "version": "Versão 2.1 — Junho de 2025"}]
    assert trace["estimated_cost_usd"] == 0.00092
    assert trace["resolved"] is True
    assert trace["timestamp"] == "2026-03-25T10:00:00-04:00"


def test_tool_results_are_not_traced():
    trace = build_trace(turn(), "x", "s1", "gpt-4.1-mini", NOW)
    assert "result" not in trace["tool_calls"][0]


def test_failed_turn_is_unresolved():
    trace = build_trace(turn(error_type="timeout", tool_calls=[]), "x", "s1", "m", NOW)
    assert trace["resolved"] is False
    assert trace["error_type"] == "timeout"


def test_unanswered_handoff_is_unresolved():
    handoff = ToolCallRecord("register_handoff", {"reason": "unanswered", "summary": "x"},
                             {"registered": True}, 3)
    assert build_trace(turn(tool_calls=[handoff]), "x", "s1", "m", NOW)["resolved"] is False


def test_tracer_appends_json_lines(tmp_path):
    tracer = Tracer(tmp_path / "logs" / "traces.jsonl", "s1", "gpt-4.1-mini")
    tracer.record(turn(), "primeira", NOW)
    tracer.record(turn(), "segunda", NOW)
    lines = (tmp_path / "logs" / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["user_message"] for line in lines] == ["primeira", "segunda"]
    assert "isabela.moreira" not in "".join(lines)


def test_trace_write_failure_does_not_raise(tmp_path):
    assert Tracer(tmp_path, "s1", "m").record(turn(), "x", NOW) is None  # path is a directory
