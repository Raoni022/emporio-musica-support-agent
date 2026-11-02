"""Read the store policy manual (PDF) and split it into its 10 numbered sections.

The agent never gets the whole manual: get_policy(topic) returns one section,
so the decision to consult a policy is explicit and visible in the trace.
Loading fails loudly if the 10 expected headings or the version are not
found, instead of running with a policy missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from emporio_agent.data import DATA_DIR

MANUAL_PATH = DATA_DIR / "políticas_da_loja.pdf"

# Section number -> (topic used by get_policy, heading as written in the manual).
# Matching the exact headings makes the split deterministic; if the manual is
# restructured, startup fails and this table must be updated.
SECTIONS = {
    1: ("about_and_contact", "Sobre a Empório da Música"),
    2: ("opening_hours", "Horário de Funcionamento"),
    3: ("payment", "Formas de Pagamento"),
    4: ("returns_and_exchanges", "Política de Trocas e Devoluções"),
    5: ("shipping", "Política de Frete e Entregas"),
    6: ("promotions", "Promoções e Descontos"),
    7: ("customer_service", "Atendimento via WhatsApp"),
    8: ("warranty", "Garantia"),
    9: ("privacy", "Privacidade e Proteção de Dados"),
    10: ("final_provisions", "Disposições Finais"),
}
TOPICS = {topic: number for number, (topic, _) in SECTIONS.items()}

# Official contact, from the manual's cover and the company data table (1.2).
# The manual also lists another WhatsApp number in section 7 and an accented
# e-mail in 1.2; see docs/data_quality.md.
OFFICIAL_PHONE = "(67) 3341-4444"
OFFICIAL_EMAIL = "contato@emporiodamusica.com.br"

# Reconciliation notes returned with a section. The manual text itself is never
# edited; these say which value to use where it contradicts itself. Portuguese,
# because they are read by the LLM, like the prompts.
SECTION_NOTES = {
    1: [
        f"O e-mail oficial é {OFFICIAL_EMAIL} (capa do manual). A grafia com acento "
        "na tabela de dados da empresa é um erro de digitação."
    ],
    3: [
        "A tabela cita parcela mínima de R$ 100,00 para o crédito em geral, mas as "
        "regras detalhadas da seção 3.1 prevalecem. Use calculate_installments para "
        "as parcelas; não calcule."
    ],
    7: [
        f"O WhatsApp oficial é {OFFICIAL_PHONE} (capa e seção 1.2). O número citado "
        "nesta seção diverge do restante do manual; não o informe ao cliente.",
        "O manual não lista lojas parceiras. Não sugira nomes de outras lojas.",
    ],
}

_PAGE_HEADER = "Empório da Música Manual de Políticas e Procedimentos"
_VERSION_RE = re.compile(r"Versão (\d+(?:\.\d+)*) — (\w+ de \d{4})")


class PolicyLoadError(RuntimeError):
    """Raised when the manual cannot be read or split into its sections."""


@dataclass(frozen=True)
class PolicySection:
    number: int
    topic: str
    title: str
    text: str


@dataclass(frozen=True)
class PolicyManual:
    version: str
    cover: str
    sections: dict[int, PolicySection]

    @property
    def full_text(self) -> str:
        return "\n\n".join([self.cover, *(s.text for s in self.sections.values())])

    def by_topic(self, topic: str) -> PolicySection:
        return self.sections[TOPICS[topic]]


def _flatten(text: str) -> str:
    """pypdf returns one word per line here; rebuild running text."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"(\d) - (\d)", r"\1-\2", text)  # "3341 - 4444" -> "3341-4444"


def _clean_page(text: str) -> str:
    text = _flatten(text)
    if text.startswith(_PAGE_HEADER):
        text = text[len(_PAGE_HEADER):]
    return re.sub(r"\s*Página \d+$", "", text).strip()


def _format_section(text: str) -> str:
    """Restore some structure for readability: subsections, bullets, list items."""
    text = re.sub(r" (?=\d+\.\d+ [A-ZÁÉÍÓÚÂÊÔÃÕÇ])", "\n\n", text)
    text = re.sub(r" (?=• )", "\n", text)
    text = re.sub(r" (?=\d\. [A-ZÁÉÍÓÚÂÊÔÃÕÇ]\w*:)", "\n", text)
    return text.strip()


def parse_manual(pages: list[str]) -> PolicyManual:
    """Split the extracted page texts into cover + 10 sections."""
    if not pages:
        raise PolicyLoadError("The policy manual has no pages.")
    cover = _clean_page(pages[0])
    body = " ".join(_clean_page(p) for p in pages[1:])

    version_match = _VERSION_RE.search(cover)
    if not version_match:
        raise PolicyLoadError("Manual version ('Versão X — Mês de AAAA') not found on the cover.")
    version = f"Versão {version_match.group(1)} — {version_match.group(2)}"

    # Find each heading in order, so list items like "1. Saudação" inside
    # section 7 can never be mistaken for a section start.
    starts: list[int] = []
    position = 0
    missing = []
    for number, (_, title) in SECTIONS.items():
        index = body.find(f"{number}. {title}", position)
        if index < 0:
            missing.append(f"{number}. {title}")
            continue
        starts.append(index)
        position = index + 1
    if missing:
        raise PolicyLoadError(
            f"Could not split the manual into its {len(SECTIONS)} sections. "
            f"Headings not found (in order): {missing}"
        )

    sections = {}
    bounds = zip(starts, [*starts[1:], len(body)])
    for (number, (topic, title)), (start, end) in zip(SECTIONS.items(), bounds):
        text = _format_section(body[start:end])
        if len(text) <= len(f"{number}. {title}"):
            raise PolicyLoadError(f"Section {number} ({title}) is empty.")
        sections[number] = PolicySection(number, topic, title, text)
    return PolicyManual(version=version, cover=cover, sections=sections)


def load_policy_manual(path: Path | str = MANUAL_PATH) -> PolicyManual:
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (OSError, ValueError) as exc:
        raise PolicyLoadError(f"Could not read the policy manual at {path}: {exc}") from exc
    return parse_manual(pages)
