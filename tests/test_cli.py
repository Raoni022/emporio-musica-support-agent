import json
from datetime import datetime

import pytest

import cli
from emporio_agent.agent import Agent
from emporio_agent.config import ConfigError, Settings
from emporio_agent.data import load_store_data
from emporio_agent.llm import FakeLLM, LLMResponse, tool_call
from emporio_agent.policies import load_policy_manual
from emporio_agent.tools import STORE_TZ
from emporio_agent.tracing import Tracer

NOW = datetime(2026, 3, 25, 10, 0, tzinfo=STORE_TZ)


@pytest.fixture(scope="module")
def store():
    return load_store_data()


@pytest.fixture(scope="module")
def manual():
    return load_policy_manual()


@pytest.fixture
def run(store, manual, tmp_path):
    """Run the chat loop with scripted input; return (output text, agent, trace path)."""
    def _run(inputs, script, debug=False):
        settings = Settings(api_key="test", base_url=None, model="gpt-4.1-mini",
                            reference_now=NOW, handoffs_path=tmp_path / "handoffs.jsonl")
        output: list[str] = []
        agent = Agent(FakeLLM(script), store, manual, settings,
                      on_tool=lambda t: output.append(cli.tool_status(t, "Tom")))
        trace_path = tmp_path / "traces.jsonl"
        tracer = Tracer(trace_path, "s1", settings.model)
        pending = list(inputs)

        def read(prompt):
            if not pending:
                raise EOFError
            return pending.pop(0)

        cli.chat(agent, tracer, debug=debug, read=read, write=output.append)
        return "\n".join(output), agent, trace_path
    return _run


def answer(text):
    return LLMResponse(content=text, prompt_tokens=10, completion_tokens=5)


def test_conversation_and_status_line(run):
    output, _, _ = run(
        ["Vocês estão abertos?", "/quit"],
        [LLMResponse(None, [tool_call("get_store_status")]), answer("Estamos abertos até 18h!")],
    )
    assert "🎵 Tom está consultando o horário da loja..." in output
    assert "Tom: Estamos abertos até 18h!" in output
    assert "Valeu pela conversa" in output


def test_history_persists_within_the_session(run):
    _, agent, _ = run(["Quero ver o pedido 3", "isabela.moreira@jmail.com"],
                      [answer("Qual o e-mail?"), answer("ok")])
    assert [m["content"] for m in agent.history if m["role"] == "user"] == [
        "Quero ver o pedido 3", "isabela.moreira@jmail.com",
    ]


def test_reset_clears_history(run):
    output, agent, _ = run(["oi", "/reset", "oi de novo"], [answer("Olá!"), answer("Olá de novo!")])
    assert "(conversa reiniciada)" in output
    assert [m["content"] for m in agent.history if m["role"] == "user"] == ["oi de novo"]


def test_empty_lines_are_ignored(run):
    _, agent, _ = run(["", "   ", "oi"], [answer("Olá!")])
    assert len(agent.llm.calls) == 1


def test_every_turn_is_traced_and_masked(run):
    _, _, trace_path = run(["meu e-mail é isabela.moreira@jmail.com"], [answer("ok")])
    [trace] = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert trace["user_message"] == "meu e-mail é i***@jmail.com"


def test_debug_shows_tool_calls_and_tokens(run):
    output, _, _ = run(
        ["Quanto custa o Takamine GD20?"],
        [LLMResponse(None, [tool_call("get_product_details", product="Takamine GD20")]),
         answer("R$ 2.199,00")],
        debug=True,
    )
    assert 'get_product_details({"product": "Takamine GD20"})' in output
    assert "2089.05" in output  # the tool result
    assert "2 LLM call(s)" in output
    assert "US$" in output


def test_end_of_input_exits_cleanly(run):
    output, _, _ = run([], [])
    assert "Valeu pela conversa" in output


def test_startup_error_is_friendly(monkeypatch, capsys):
    def fail():
        raise ConfigError("OPENAI_API_KEY is not set.")
    monkeypatch.setattr(cli, "load_settings", fail)
    assert cli.main([]) == 1
    assert "Erro ao iniciar: OPENAI_API_KEY is not set." in capsys.readouterr().err


def test_invalid_reference_date_flag_is_friendly(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_settings", lambda: Settings("k", None, "m", None))
    assert cli.main(["--reference-date", "amanhã"]) == 1
    assert "REFERENCE_DATE" in capsys.readouterr().err


def test_status_line_is_shown_once_per_tool_per_turn():
    lines = []
    status = cli.StatusPrinter("Tom", write=lines.append)
    for _ in range(8):
        status("get_product_details")
    status("get_policy")
    status.reset()
    status("get_product_details")
    assert lines == [
        "🎵 Tom está consultando o produto...",
        "🎵 Tom está consultando as políticas da loja...",
        "🎵 Tom está consultando o produto...",
    ]
