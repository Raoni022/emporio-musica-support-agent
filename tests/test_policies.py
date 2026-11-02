import pytest

from emporio_agent.policies import (
    OFFICIAL_PHONE,
    SECTIONS,
    TOPICS,
    PolicyLoadError,
    load_policy_manual,
    parse_manual,
)
from emporio_agent.tools import get_policy


@pytest.fixture(scope="module")
def manual():
    return load_policy_manual()


# --- Real manual -------------------------------------------------------------


def test_manual_splits_into_ten_sections_in_order(manual):
    assert list(manual.sections) == list(range(1, 11))
    for number, section in manual.sections.items():
        assert section.text.startswith(f"{number}. {section.title}")


def test_version_is_extracted(manual):
    assert manual.version == "Versão 2.1 — Junho de 2025"


def test_list_items_inside_a_section_do_not_split_it(manual):
    # Section 7.2 has a numbered list "1. Saudação ... 5. Fechamento".
    service = manual.by_topic("customer_service").text
    assert "1. Saudação" in service
    assert "5. Fechamento" in service
    assert "7.3 Situações Especiais" in service


def test_page_headers_and_footers_are_removed(manual):
    for section in manual.sections.values():
        assert "Página" not in section.text
        assert "Manual de Políticas e Procedimentos" not in section.text


def test_text_spans_page_breaks(manual):
    # Section 5.2 starts on page 4 and continues on page 5.
    assert "Utilizamos os serviços dos Correios" in manual.by_topic("shipping").text


def test_key_facts_land_in_the_right_section(manual):
    assert "Rua 14 de Maio, 3200" in manual.by_topic("about_and_contact").text
    assert "(67) 3341-4444" in manual.by_topic("about_and_contact").text  # "3341 - 4444" rejoined
    assert "09:00 às 13:00" in manual.by_topic("opening_hours").text
    assert "5% de desconto" in manual.by_topic("payment").text
    assert "7 (sete) dias corridos" in manual.by_topic("returns_and_exchanges").text
    assert "rain check" in manual.by_topic("promotions").text
    assert "90 (noventa) dias" in manual.by_topic("warranty").text
    assert "LGPD" in manual.by_topic("privacy").text
    assert "consultar a gerência" in manual.by_topic("final_provisions").text


def test_unreadable_manual_fails_with_clear_error(tmp_path):
    with pytest.raises(PolicyLoadError, match="Could not read"):
        load_policy_manual(tmp_path / "missing.pdf")


# --- Splitting rules (synthetic manual) --------------------------------------


def _pages(skip: int | None = None, cover: str = "MANUAL Versão 2.1 — Junho de 2025"):
    body = [
        f"{n}. {title} Texto da seção {n}."
        for n, (_, title) in SECTIONS.items()
        if n != skip
    ]
    return [cover, " ".join(body)]


def test_synthetic_manual_parses():
    manual = parse_manual(_pages())
    assert len(manual.sections) == 10
    assert manual.by_topic("warranty").text == "8. Garantia Texto da seção 8."


def test_missing_section_fails_at_load():
    with pytest.raises(PolicyLoadError, match="8. Garantia"):
        parse_manual(_pages(skip=8))


def test_out_of_order_sections_fail_at_load():
    pages = _pages()
    swapped = pages[1].replace("7. Atendimento via WhatsApp", "TMP").replace(
        "8. Garantia", "7. Atendimento via WhatsApp"
    ).replace("TMP", "8. Garantia")
    with pytest.raises(PolicyLoadError, match="not found"):
        parse_manual([pages[0], swapped])


def test_missing_version_fails_at_load():
    with pytest.raises(PolicyLoadError, match="version"):
        parse_manual(_pages(cover="MANUAL sem versão"))


# --- get_policy --------------------------------------------------------------


def test_get_policy_returns_section_with_version(manual):
    result = get_policy(manual, "returns_and_exchanges")
    assert result["section"] == 4
    assert result["version"] == "Versão 2.1 — Junho de 2025"
    assert result["text"].startswith("4. Política de Trocas e Devoluções")
    assert result["notes"] == []


def test_every_topic_is_reachable(manual):
    assert sorted(get_policy(manual, t)["section"] for t in TOPICS) == list(range(1, 11))


def test_get_policy_notes_resolve_manual_contradictions(manual):
    notes = " ".join(get_policy(manual, "customer_service")["notes"])
    assert OFFICIAL_PHONE in notes
    assert "lojas parceiras" in notes


def test_unknown_topic_returns_structured_error(manual):
    result = get_policy(manual, "refunds")
    assert "error" in result
    assert result["retryable"] is False
    assert "returns_and_exchanges" in result["valid_topics"]
