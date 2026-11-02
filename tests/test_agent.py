import json
from datetime import datetime

import pytest

from emporio_agent.agent import MAX_HISTORY_TURNS, MAX_ITERATIONS, MAX_MESSAGE_CHARS, Agent
from emporio_agent.config import ConfigError, Settings, load_settings, parse_reference_date
from emporio_agent.data import load_store_data
from emporio_agent.llm import FakeLLM, LLMError, LLMResponse, tool_call
from emporio_agent.policies import OFFICIAL_PHONE, load_policy_manual
from emporio_agent.prompts import FALLBACK_MESSAGE, LOOP_LIMIT_MESSAGE
from emporio_agent.tool_schemas import build_tool_schemas, validate_arguments
from emporio_agent.tools import STORE_TZ

NOW = datetime(2026, 3, 25, 10, 0, tzinfo=STORE_TZ)


@pytest.fixture(scope="module")
def store():
    return load_store_data()


@pytest.fixture(scope="module")
def manual():
    return load_policy_manual()


@pytest.fixture
def make_agent(store, manual, tmp_path):
    def make(script, **kwargs):
        settings = Settings(api_key="test", base_url=None, model="fake", reference_now=NOW,
                            handoffs_path=tmp_path / "handoffs.jsonl")
        return Agent(FakeLLM(script), store, manual, settings, **kwargs)
    return make


def answer(text):
    return LLMResponse(content=text, prompt_tokens=100, completion_tokens=20)


def calls(*tool_calls):
    return LLMResponse(content=None, tool_calls=list(tool_calls), prompt_tokens=100,
                       completion_tokens=10)


def tool_messages(messages):
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


def assert_valid_history(history):
    """Every tool message answers a tool call from the assistant message before it."""
    pending: set[str] = set()
    for message in history:
        if message["role"] == "tool":
            assert message["tool_call_id"] in pending, "orphan tool message"
            pending.discard(message["tool_call_id"])
        else:
            assert not pending, "tool call without result"
            pending = {c["id"] for c in message.get("tool_calls", [])}
    assert not pending


# --- Loop --------------------------------------------------------------------


def test_plain_answer_without_tools(make_agent):
    agent = make_agent([answer("Oi! Sou o Tom 🎵")])
    result = agent.respond("oi")
    assert result.answer == "Oi! Sou o Tom 🎵"
    assert result.iterations == 1
    assert result.error_type is None


def test_tool_result_goes_back_to_the_model(make_agent):
    agent = make_agent([
        calls(tool_call("get_product_details", product="Takamine GD20")),
        answer("O Takamine GD20 sai por R$ 2.199,00."),
    ])
    result = agent.respond("Quanto custa o Takamine GD20?")
    assert result.answer.startswith("O Takamine")
    [record] = result.tool_calls
    assert record.name == "get_product_details"
    assert record.result["product"]["price"] == 2199.0
    second_call = agent.llm.calls[1]
    assert tool_messages(second_call)[0]["product"]["pix_price"] == 2089.05
    assert result.prompt_tokens == 200 and result.completion_tokens == 30
    assert_valid_history(agent.history)


def test_parallel_tool_calls_run_before_the_next_llm_call(make_agent):
    agent = make_agent([
        calls(tool_call("get_product_details", "c1", product="Crafter HT-100"),
              tool_call("get_policy", "c2", topic="payment")),
        answer("ok"),
    ])
    result = agent.respond("Quanto fica o Crafter no PIX?")
    assert [r.name for r in result.tool_calls] == ["get_product_details", "get_policy"]
    assert len(agent.llm.calls) == 2
    results = tool_messages(agent.llm.calls[1])
    assert results[0]["product"]["pix_price"] is None
    assert results[1]["section"] == 3
    assert_valid_history(agent.history)


def test_system_prompt_has_persona_and_reference_time(make_agent):
    agent = make_agent([answer("oi")])
    agent.respond("oi")
    system = agent.llm.calls[0][0]
    assert system["role"] == "system"
    assert "Você é Tom" in system["content"]
    assert "quarta-feira, 25/03/2026 10:00" in system["content"]


def test_tool_status_callback(make_agent):
    seen = []
    agent = make_agent([calls(tool_call("get_store_status")), answer("Abertos!")],
                       on_tool=seen.append)
    agent.respond("Vocês estão abertos?")
    assert seen == ["get_store_status"]


def test_handoff_receives_tools_consulted_from_the_loop(make_agent, tmp_path):
    agent = make_agent([
        calls(tool_call("get_order_status", order_id="8", contact="anacarol.ferreira@coldmail.com")),
        calls(tool_call("register_handoff", reason="late_order", summary="Pedido 8 atrasado.",
                        contact="anacarol.ferreira@coldmail.com", order_id="8",
                        suggested_next_action="Contatar a transportadora.")),
        answer("Encaminhei para a equipe."),
    ])
    agent.respond("Meu pedido 8 não chegou, e-mail anacarol.ferreira@coldmail.com")
    record = json.loads((tmp_path / "handoffs.jsonl").read_text(encoding="utf-8"))
    assert record["tools_consulted"] == ["get_order_status"]
    assert record["identity_verified"] is True


# --- Failures (section 7c) ---------------------------------------------------


def test_loop_stops_after_max_iterations(make_agent):
    agent = make_agent([calls(tool_call("list_categories", f"c{i}")) for i in range(10)])
    result = agent.respond("loop")
    assert result.answer == LOOP_LIMIT_MESSAGE
    assert result.error_type == "max_iterations"
    assert len(agent.llm.calls) == MAX_ITERATIONS
    assert_valid_history(agent.history)


def test_time_budget_stops_the_turn(make_agent):
    ticks = iter(range(0, 1000, 40))  # every clock read advances 40 s
    agent = make_agent([calls(tool_call("list_categories", f"c{i}")) for i in range(5)],
                       clock=lambda: next(ticks))
    result = agent.respond("lento")
    assert result.error_type == "time_budget"
    assert result.answer == LOOP_LIMIT_MESSAGE
    assert_valid_history(agent.history)


@pytest.mark.parametrize("error", [LLMError("timeout", "timed out"), LLMError("rate_limit", "429"),
                                   RuntimeError("boom")])
def test_llm_failure_returns_fixed_fallback(make_agent, error):
    agent = make_agent([calls(tool_call("list_categories")), error])
    result = agent.respond("oi")
    assert result.answer == FALLBACK_MESSAGE
    assert OFFICIAL_PHONE in result.answer
    assert result.error_type in ("timeout", "rate_limit", "unexpected_error")
    # The partial tool exchange is dropped; the customer's message stays.
    assert [m["role"] for m in agent.history] == ["user", "assistant"]


def test_conversation_continues_after_a_failure(make_agent):
    agent = make_agent([LLMError("timeout", "x"), answer("Voltei!")])
    agent.respond("oi")
    assert agent.respond("oi de novo").answer == "Voltei!"


def test_failing_tool_becomes_structured_error(make_agent, monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("disk unavailable")
    monkeypatch.setattr("emporio_agent.tools.list_categories", explode)
    agent = make_agent([calls(tool_call("list_categories")), answer("Não consegui verificar agora.")])
    result = agent.respond("quais categorias?")
    assert result.tool_calls[0].result == {"error": "Tool failed: OSError: disk unavailable",
                                           "retryable": True}
    assert result.answer == "Não consegui verificar agora."


@pytest.mark.parametrize(
    "bad_call, message",
    [
        (tool_call("delete_everything"), "Unknown tool"),
        (tool_call("get_policy", topic="refunds"), "Invalid arguments"),
        (tool_call("calculate_installments", amount="mil"), "Invalid arguments"),
    ],
)
def test_bad_tool_calls_are_reported_to_the_model(make_agent, bad_call, message):
    agent = make_agent([calls(bad_call), answer("ok")])
    result = agent.respond("x")
    assert message in result.tool_calls[0].error


def test_invalid_json_arguments(make_agent):
    from emporio_agent.llm import ToolCall
    agent = make_agent([calls(ToolCall("c1", "get_policy", "{not json")), answer("ok")])
    assert "not valid JSON" in agent.respond("x").tool_calls[0].error


def test_empty_model_answer_becomes_fallback(make_agent):
    agent = make_agent([answer("   ")])
    result = agent.respond("oi")
    assert result.answer == FALLBACK_MESSAGE
    assert result.error_type == "empty_response"


def test_too_long_message_is_not_sent_to_the_llm(make_agent):
    agent = make_agent([])
    result = agent.respond("a" * (MAX_MESSAGE_CHARS + 1))
    assert result.error_type == "message_too_long"
    assert agent.llm.calls == []
    assert agent.history == []


# --- History -----------------------------------------------------------------


def test_history_is_kept_within_the_session(make_agent):
    agent = make_agent([answer("Qual o e-mail?"), answer("ok")])
    agent.respond("Quero ver o pedido 3")
    agent.respond("isabela.moreira@jmail.com")
    contents = [m["content"] for m in agent.llm.calls[1] if m["role"] == "user"]
    assert contents == ["Quero ver o pedido 3", "isabela.moreira@jmail.com"]


def test_history_is_trimmed_on_turn_boundaries(make_agent):
    script = []
    for i in range(15):
        script += [calls(tool_call("list_categories", f"c{i}")), answer(f"resposta {i}")]
    agent = make_agent(script)
    for i in range(15):
        agent.respond(f"mensagem {i}")
    users = [m for m in agent.history if m["role"] == "user"]
    assert len(users) == MAX_HISTORY_TURNS
    assert agent.history[0] == {"role": "user", "content": "mensagem 5"}
    assert_valid_history(agent.history)


def test_reset_clears_the_session(make_agent):
    agent = make_agent([calls(tool_call("list_categories")), answer("ok")])
    agent.respond("x")
    agent.reset()
    assert agent.history == [] and agent.tools_used == []


# --- Schemas -----------------------------------------------------------------


def test_every_tool_has_a_strict_schema(store):
    schemas = build_tool_schemas(store)
    assert len(schemas) == 11
    for schema in schemas:
        function = schema["function"]
        params = function["parameters"]
        assert function["strict"] is True
        assert params["additionalProperties"] is False
        assert set(params["required"]) == set(params["properties"])
        assert function["description"]


def test_category_enum_comes_from_the_data(store):
    schema = next(s for s in build_tool_schemas(store) if s["function"]["name"] == "search_products")
    enum = schema["function"]["parameters"]["properties"]["category"]["enum"]
    assert "Violões" in enum and None in enum


@pytest.mark.parametrize(
    "arguments, ok",
    [
        ({"amount": 100}, True),
        ({"amount": 99.9}, True),
        ({"amount": True}, False),  # bool is not a number
        ({"amount": "100"}, False),
        ({"product": 95}, False),  # product is a string
        ({"product": "Crafter HT-100", "amount": None}, True),
        ({"amount": 100, "extra": 1}, False),
    ],
)
def test_validate_arguments(store, arguments, ok):
    schema = next(s for s in build_tool_schemas(store)
                  if s["function"]["name"] == "calculate_installments")
    assert (validate_arguments(schema, arguments) is None) is ok


def test_missing_nullable_argument_is_accepted(store):
    schema = next(s for s in build_tool_schemas(store) if s["function"]["name"] == "search_products")
    assert validate_arguments(schema, {"category": "Violões", "include_unavailable": False}) is None


# --- Config ------------------------------------------------------------------


def test_reference_date_without_time_defaults_to_10am():
    assert parse_reference_date("2026-03-25") == datetime(2026, 3, 25, 10, 0, tzinfo=STORE_TZ)


def test_reference_date_with_time():
    assert parse_reference_date("2026-03-22T10:30").hour == 10


@pytest.mark.parametrize("value", ["25/03/2026", "2026-13-01", "tomorrow"])
def test_invalid_reference_date(value):
    with pytest.raises(ConfigError):
        parse_reference_date(value)


def test_missing_api_key_fails_clearly(monkeypatch):
    monkeypatch.setattr("emporio_agent.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        load_settings()


def test_env_file_wins_over_system_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "old-key-from-the-system")
    monkeypatch.setenv("MODEL", "old-model")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=key-from-dotenv\nMODEL=gpt-4.1-mini\n", encoding="utf-8")
    settings = load_settings(env_file=env_file)
    assert settings.api_key == "key-from-dotenv"
    assert settings.model == "gpt-4.1-mini"


@pytest.mark.parametrize(
    "message, language",
    [
        ("Hi, do you sell ukuleles under 300 reais?", "en"),
        ("How much is the Takamine GD20?", "en"),
        ("Vocês têm ukulele até 300 reais?", "pt"),
        ("Quanto custa o Takamine GD20?", "pt"),
        ("oi", "pt"),
        ("Pedido 7, e-mail leticia.rocha@jmail.com", "pt"),
    ],
)
def test_detect_language(message, language):
    from emporio_agent.prompts import detect_language
    assert detect_language(message) == language


def test_english_turn_gets_a_language_line_in_the_system_prompt(make_agent):
    agent = make_agent([answer("Hello!"), answer("We have four.")])
    agent.respond("Vocês abrem hoje?")
    agent.respond("Hi, do you sell ukuleles under 300 reais?")
    assert "IDIOMA DESTE TURNO" not in agent.llm.calls[0][0]["content"]
    assert "Responda em inglês" in agent.llm.calls[1][0]["content"]
