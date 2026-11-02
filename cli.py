"""Chat with Tom in the terminal.

Usage: python cli.py [--debug] [--reference-date YYYY-MM-DD[THH:MM]]

Commands: /reset starts a new session (history cleared), /quit exits.
History is kept within the session only, on purpose (see README).
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emporio_agent.agent import Agent, TurnResult  # noqa: E402
from emporio_agent.config import ConfigError, load_settings, parse_reference_date  # noqa: E402
from emporio_agent.data import DataLoadError, load_store_data  # noqa: E402
from emporio_agent.llm import OpenAIClient  # noqa: E402
from emporio_agent.policies import PolicyLoadError, load_policy_manual  # noqa: E402
from emporio_agent.prompts import TOOL_STATUS_MESSAGES  # noqa: E402
from emporio_agent.tracing import Tracer, estimate_cost  # noqa: E402

QUIT_COMMANDS = {"/quit", "/sair", "/exit"}
RESET_COMMANDS = {"/reset"}
DEBUG_RESULT_CHARS = 600


def tool_status(name: str, assistant_name: str) -> str:
    return f"🎵 {assistant_name} está {TOOL_STATUS_MESSAGES.get(name, 'consultando o sistema')}..."


def debug_report(turn: TurnResult, model: str) -> str:
    lines = [f"  ┌ debug: {turn.iterations} LLM call(s), {turn.prompt_tokens}+"
             f"{turn.completion_tokens} tokens, {turn.latency_ms} ms"]
    cost = estimate_cost(model, turn.prompt_tokens, turn.completion_tokens)
    if cost is not None:
        lines[0] += f", ~US$ {cost:.5f}"
    if turn.error_type:
        lines.append(f"  │ error_type: {turn.error_type}")
    for call in turn.tool_calls:
        args = json.dumps(call.arguments, ensure_ascii=False)
        result = json.dumps(call.result, ensure_ascii=False, default=str)
        if len(result) > DEBUG_RESULT_CHARS:
            result = result[:DEBUG_RESULT_CHARS] + "…"
        lines.append(f"  │ {call.name}({args}) [{call.latency_ms} ms]")
        lines.append(f"  │   → {result}")
    lines.append("  └")
    return "\n".join(lines)


def chat(
    agent: Agent,
    tracer: Tracer,
    debug: bool = False,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> None:
    name = agent.settings.assistant_name
    while True:
        try:
            message = read("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            write("")
            break
        if not message:
            continue
        if message.lower() in QUIT_COMMANDS:
            break
        if message.lower() in RESET_COMMANDS:
            agent.reset()
            tracer.session_id = uuid.uuid4().hex[:8]
            write("(conversa reiniciada)\n")
            continue
        turn = agent.respond(message)
        tracer.record(turn, message, agent.settings.now())
        if debug:
            write(debug_report(turn, agent.settings.model))
        write(f"{name}: {turn.answer}\n")
    write(f"{name}: Valeu pela conversa! Qualquer coisa, é só chamar. 🎶")


def _utf8_console() -> None:
    # Emojis and accents on Windows consoles and pipes.
    for stream in (sys.stdin, sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with Tom, the Empório da Música assistant.")
    parser.add_argument("--debug", action="store_true", help="show tool calls, tokens and latency")
    parser.add_argument("--reference-date", help="override REFERENCE_DATE for this session")
    args = parser.parse_args(argv)
    _utf8_console()

    try:
        settings = load_settings()
        if args.reference_date:
            settings = replace(settings, reference_now=parse_reference_date(args.reference_date))
        store = load_store_data()
        manual = load_policy_manual()
    except (ConfigError, DataLoadError, PolicyLoadError) as exc:
        print(f"Erro ao iniciar: {exc}", file=sys.stderr)
        return 1

    agent = Agent(OpenAIClient(settings), store, manual, settings,
                  on_tool=lambda tool: print(tool_status(tool, settings.assistant_name)))
    tracer = Tracer(settings.logs_dir / "traces.jsonl", uuid.uuid4().hex[:8], settings.model)

    now = settings.now()
    reference = "fixa" if settings.reference_now else "atual"
    print(f"🎵 Empório da Música — {settings.assistant_name}, assistente virtual "
          f"(modelo {settings.model})")
    print(f"   Data de referência ({reference}): {now:%d/%m/%Y %H:%M}. "
          "Comandos: /reset recomeça, /quit sai.\n")
    chat(agent, tracer, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
