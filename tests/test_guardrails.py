from datetime import datetime

import pytest

from emporio_agent.agent import Agent
from emporio_agent.config import Settings
from emporio_agent.data import load_store_data
from emporio_agent.guardrails import extract_amounts, format_brl, parse_brl, validate_response
from emporio_agent.llm import FakeLLM, LLMResponse, tool_call
from emporio_agent.policies import load_policy_manual
from emporio_agent.prompts import GUARDRAIL_SAFE_MESSAGE
from emporio_agent.tools import STORE_TZ
from emporio_agent.tracing import build_trace

NOW = datetime(2026, 3, 25, 10, 0, tzinfo=STORE_TZ)


# --- Parsing -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text, value",
    [("1.967,18", 1967.18), ("2.199", 2199.0), ("196,72", 196.72), ("1967.18", 1967.18),
     ("1000", 1000.0), ("12.000,00", 12000.0), ("35", 35.0)],
)
def test_parse_brl(text, value):
    assert parse_brl(text) == value


def test_extract_amounts_from_an_answer():
    answer = "De R$ 2.399,00 por R$ 1.967,18, ou 10x de R$196,72. Frete R$ 35."
    assert extract_amounts(answer) == [2399.0, 1967.18, 196.72, 35.0]


def test_numbers_without_currency_are_ignored():
    assert extract_amounts("Pedido 8732, 12x, 18% de desconto, CEP 79202-333") == []


def test_format_brl():
    assert format_brl(1967.18) == "R$ 1.967,18"


# --- Validation --------------------------------------------------------------

CRAFTER = {"product": {"price": 2399.0, "final_price": 1967.18, "pix_price": None}}


def test_values_from_tools_pass():
    assert validate_response("De R$ 2.399,00 por R$ 1.967,18.", [CRAFTER]).ok


def test_value_no_tool_returned_is_blocked():
    # 5% on top of the promotion: exactly what the rules forbid.
    result = validate_response("No PIX fica R$ 1.868,82.", [CRAFTER])
    assert not result.ok
    assert result.unverified == [1868.82]


def test_different_formats_of_the_same_value_match():
    assert validate_response("Sai por R$ 1.967,18", [{"amount": 1967.18}]).ok
    assert validate_response("Sai por R$ 1967.18", [{"amount": 1967.18}]).ok


def test_rounding_cents_are_tolerated():
    assert validate_response("10x de R$ 196,72", [{"installment_value": 196.718}]).ok


def test_values_written_inside_tool_text_count():
    policy = {"text": "Parcelamento de 4x a 6x: sem juros, parcela mínima de R$ 80,00."}
    assert validate_response("A parcela mínima é R$ 80,00.", [policy]).ok


def test_answer_without_values_passes():
    assert validate_response("A loja abre às 9h.", []).ok


# --- In the agent loop -------------------------------------------------------


@pytest.fixture(scope="module")
def store():
    return load_store_data()


@pytest.fixture(scope="module")
def manual():
    return load_policy_manual()


@pytest.fixture
def make_agent(store, manual, tmp_path):
    def make(script):
        settings = Settings(api_key="test", base_url=None, model="fake", reference_now=NOW,
                            handoffs_path=tmp_path / "handoffs.jsonl")
        return Agent(FakeLLM(script), store, manual, settings)
    return make


def details(product="Crafter HT-100"):
    return LLMResponse(None, [tool_call("get_product_details", product=product)])


def answer(text):
    return LLMResponse(text)


def test_grounded_answer_passes_without_retry(make_agent):
    agent = make_agent([details(), answer("De R$ 2.399,00 por R$ 1.967,18 (18% off).")])
    result = agent.respond("Quanto custa o Crafter?")
    assert result.guardrail_triggered is False
    assert len(agent.llm.calls) == 2


def test_hallucinated_price_gets_one_corrected_retry(make_agent):
    agent = make_agent([
        details(),
        answer("No PIX fica R$ 1.868,82."),
        answer("No PIX fica R$ 1.967,18, o mesmo da promoção."),
    ])
    result = agent.respond("Quanto fica o Crafter no PIX?")
    assert result.answer == "No PIX fica R$ 1.967,18, o mesmo da promoção."
    assert result.guardrail_triggered is True
    assert result.unverified_values == [1868.82]
    warning = agent.llm.calls[2][-1]
    assert warning["role"] == "user" and "R$ 1.868,82" in warning["content"]
    # The rejected answer and the warning are not kept in the history.
    contents = [m.get("content") or "" for m in agent.history]
    assert not any("1.868,82" in c for c in contents)
    assert contents[-1] == "No PIX fica R$ 1.967,18, o mesmo da promoção."


def test_repeated_hallucination_gives_a_safe_answer(make_agent):
    agent = make_agent([details(), answer("R$ 1.868,82"), answer("Fica R$ 1.868,82 mesmo.")])
    result = agent.respond("Quanto fica o Crafter no PIX?")
    assert result.answer == GUARDRAIL_SAFE_MESSAGE
    assert result.error_type == "guardrail_blocked"
    assert "R$" not in GUARDRAIL_SAFE_MESSAGE
    assert agent.history[-1] == {"role": "assistant", "content": GUARDRAIL_SAFE_MESSAGE}


def test_the_warning_itself_does_not_approve_the_rejected_value(make_agent):
    # The warning quotes R$ 1.868,82; repeating it must still be blocked.
    agent = make_agent([details(), answer("R$ 1.868,82"), answer("R$ 1.868,82")])
    assert agent.respond("PIX?").error_type == "guardrail_blocked"


def test_values_from_earlier_turns_and_the_customer_are_allowed(make_agent):
    agent = make_agent([
        details(),
        answer("Está R$ 1.967,18."),
        answer("Sim, os R$ 1.967,18 que te passei. Abaixo de R$ 2.000 que você citou!"),
    ])
    agent.respond("Quanto custa o Crafter?")
    result = agent.respond("Então fica abaixo de R$ 2.000?")
    assert result.guardrail_triggered is False


def test_trace_records_the_guardrail(make_agent):
    agent = make_agent([details(), answer("R$ 1.868,82"), answer("R$ 1.967,18")])
    turn = agent.respond("PIX?")
    trace = build_trace(turn, "PIX?", "s1", "fake", NOW)
    assert trace["guardrail_triggered"] is True
    assert trace["unverified_values"] == [1868.82]
