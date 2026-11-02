import json
from datetime import date

import pytest

from emporio_agent.data import load_store_data
from emporio_agent.tools import get_order_status

TODAY = date(2026, 3, 25)


@pytest.fixture(scope="module")
def store():
    return load_store_data()


def status(store, order_id, contact, today=TODAY):
    return get_order_status(store, order_id, contact, today)


# --- Verification ------------------------------------------------------------


@pytest.mark.parametrize(
    "contact",
    [
        "pedro.oliveira@jmail.com",
        "  Pedro.Oliveira@JMAIL.com ",
        "(67) 98432-1098",
        "67984321098",
        "+55 67 98432-1098",
    ],
)
def test_order_owner_is_verified_by_email_or_phone(store, contact):
    result = status(store, 1, contact)
    assert result["verified"] is True
    assert result["order_id"] == 1


def test_order_number_with_hash_is_accepted(store):
    assert status(store, "#1", "pedro.oliveira@jmail.com")["verified"] is True


def test_all_verification_failures_look_identical(store):
    # Unknown order, someone else's contact, wrong contact, malformed input,
    # phone without area code: the response never reveals whether the order exists.
    failures = [
        status(store, 999, "pedro.oliveira@jmail.com"),
        status(store, 1, "lucas.mendes@jmail.com"),  # real customer, not this order's
        status(store, 1, "nobody@example.com"),
        status(store, "abc", "pedro.oliveira@jmail.com"),
        status(store, 1, "98432-1098"),
        status(store, 1, ""),
    ]
    assert all(f == failures[0] for f in failures)
    assert failures[0]["verified"] is False
    assert "error" in failures[0]


def test_shared_email_still_verifies_its_own_order(store):
    # Customers 11 and 43 share an e-mail; order 10 belongs to customer 11.
    assert status(store, 10, "bruno.martins@jmail.com")["verified"] is True


# --- Content -----------------------------------------------------------------


def test_order_details(store):
    result = status(store, 2, "thiago.barbosa@jmail.com")
    assert result["customer_first_name"] == "Thiago"
    assert result["status_label"] == "entregue"
    assert result["items"] == [{"name": "Taylor 110e Dreadnought Natural", "quantity": 1}]
    assert result["payment_method"] == "cartão de crédito em 6x"
    assert result["tracking_code"] == "BRCX9876543BR"
    assert result["estimated_delivery"] == "2025-11-15"


def test_recorded_total_is_shown_not_recalculated(store):
    # 2 x R$ 1.749,00 = R$ 3.498,00, but R$ 3.450,00 is what was recorded.
    assert status(store, 3, "isabela.moreira@jmail.com")["total"] == 3450.0


def test_cancelled_order_shows_reason(store):
    result = status(store, 17, "matheus.correia@hayoo.com.br")
    assert result["status_label"] == "cancelado"
    assert result["notes"] == "Pedido cancelado pelo cliente antes do processamento"


def test_response_has_no_contact_data(store):
    text = json.dumps(status(store, 1, "pedro.oliveira@jmail.com"), ensure_ascii=False)
    assert "jmail" not in text
    assert "98432" not in text
    assert "Oliveira" not in text  # first name only


# --- Delays ------------------------------------------------------------------


def test_shipped_order_past_estimate_is_late(store):
    result = status(store, 8, "anacarol.ferreira@coldmail.com")
    assert result["late"] is True
    assert result["days_past_estimate"] == 25
    assert result["suggest_handoff"] is True


def test_open_order_for_weeks_is_late(store):
    result = status(store, 9, "juliana.almeida@hayoo.com.br")
    assert result["status"] == "confirmed"
    assert result["late"] is True
    assert result["days_in_current_status_since_order"] == 33


def test_recent_open_order_is_not_late(store):
    result = status(store, 16, "camila.duarte@inlook.com")
    assert result["late"] is False
    assert result["suggest_handoff"] is False


def test_delay_depends_on_reference_date(store):
    result = status(store, 8, "anacarol.ferreira@coldmail.com", today=date(2026, 2, 27))
    assert result["late"] is False
