from datetime import datetime

import pytest

from emporio_agent.config import Settings
from emporio_agent.data import load_store_data
from emporio_agent.llm import FakeLLM, LLMResponse, tool_call
from emporio_agent.policies import load_policy_manual
from emporio_agent.tool_schemas import build_tool_schemas
from scripts.eval import check_case, load_cases, render, run_case, summarize

CASE = {
    "id": "x",
    "messages": ["Quanto custa o Takamine GD20?"],
    "expected_tools": ["get_product_details"],
    "must_contain": ["2.199", ["PIX", "à vista"]],
    "must_not_contain": ["1.935"],
}


def test_passing_case():
    assert check_case(CASE, ["Sai por R$ 2.199,00, ou R$ 2.089,05 no pix."],
                      ["get_product_details"]) == []


def test_missing_tool_text_and_forbidden_text_are_reported():
    failures = check_case(CASE, ["Com 12% sai por R$ 1.935,12."], [])
    assert "tool not called: get_product_details" in failures
    assert "missing: 2.199" in failures
    assert "must not contain: 1.935" in failures


def test_comparison_ignores_case_and_accents():
    case = {"must_contain": ["terça", "Violões"], "must_not_contain": []}
    assert check_case(case, ["Abrimos TERCA. Temos violoes."], []) == []


def test_forbidden_text_is_checked_in_every_answer():
    case = {"must_contain": [], "must_not_contain": ["Isabela"]}
    assert check_case(case, ["Olá, Isabela!", "Tudo certo."], []) != []


# --- Dataset sanity ----------------------------------------------------------


@pytest.fixture(scope="module")
def cases():
    return load_cases()


def test_dataset_size_and_unique_ids(cases):
    # The spec planned 15-20; regression cases from real CLI sessions took it past 20.
    assert 15 <= len(cases) <= 30
    assert len({c["id"] for c in cases}) == len(cases)


def test_brief_questions_are_in_the_dataset_verbatim(cases):
    first_messages = {c["messages"][0] for c in cases}
    for question in [
        "Quais opções de violões disponíveis custando até R$1000?",
        "Qual o endereço da loja?",
        "Quanto custa o Takamine GD20?",
        "Me arrependi da minha compra, posso devolver meu pedido?",
    ]:
        assert question in first_messages


def test_every_tool_is_expected_at_least_once(cases):
    store = load_store_data()
    tool_names = {s["function"]["name"] for s in build_tool_schemas(store)}
    expected = {t for c in cases for t in c["expected_tools"]}
    assert tool_names <= expected


def test_case_fields(cases):
    for case in cases:
        assert set(case) <= {"id", "messages", "expected_tools", "must_contain",
                             "must_not_contain", "reference_date", "note"}
        assert case["messages"]


# --- Runner with the fake LLM ------------------------------------------------


def test_run_case_and_report(tmp_path):
    store, manual = load_store_data(), load_policy_manual()
    settings = Settings("k", None, "gpt-4.1-mini", None, handoffs_path=tmp_path / "h.jsonl")
    llm = FakeLLM([
        LLMResponse(None, [tool_call("get_product_details", product="Takamine GD20")], 100, 10),
        LLMResponse("Sai por R$ 2.199,00, ou R$ 2.089,05 no PIX.", [], 100, 10),
    ])
    result = run_case(CASE, llm, store, manual, settings)
    assert result.passed, result.failures
    metrics = summarize([result], "gpt-4.1-mini")
    assert metrics["pass_rate"] == 1.0 and metrics["messages"] == 1
    report = render([result], metrics, "gpt-4.1-mini", datetime(2026, 11, 2, 15, 0))
    assert "1/1 cases passed (100%)" in report
