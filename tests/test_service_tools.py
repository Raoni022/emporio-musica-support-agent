import json
from datetime import date, datetime, timezone

import pytest

from emporio_agent.data import load_store_data
from emporio_agent.tools import (
    STORE_TZ,
    check_return_eligibility,
    get_shipping_options,
    get_store_status,
    register_handoff,
)

# Order 7: Letícia, ordered 2026-02-03, delivered (estimated) 2026-02-17.
ORDER, CONTACT = 7, "leticia.rocha@jmail.com"


@pytest.fixture(scope="module")
def store():
    return load_store_data()


def local(*args):
    return datetime(*args, tzinfo=STORE_TZ)


# --- check_return_eligibility ------------------------------------------------


def test_regret_within_7_days_of_receipt(store):
    result = check_return_eligibility(store, ORDER, CONTACT, "regret", date(2026, 2, 20))
    assert result["eligible"] is True
    assert result["deadline"] == "2026-02-24"
    assert result["days_remaining"] == 4
    assert result["receipt_date_assumed"] == "2026-02-17"
    assert any("10 business days" in c for c in result["conditions"])


def test_regret_after_the_window(store):
    result = check_return_eligibility(store, ORDER, CONTACT, "regret", date(2026, 3, 25))
    assert result["eligible"] is False
    assert result["days_past_deadline"] == 29


def test_regret_on_the_last_day_is_still_eligible(store):
    assert check_return_eligibility(store, ORDER, CONTACT, "regret", date(2026, 2, 24))["eligible"]


def test_defect_counts_from_purchase_date(store):
    result = check_return_eligibility(store, ORDER, CONTACT, "defect", date(2026, 3, 5))
    assert result["eligible"] is True
    assert result["deadline"] == "2026-03-05"  # 30 days after 2026-02-03


def test_defect_after_30_days_points_to_warranty(store):
    result = check_return_eligibility(store, ORDER, CONTACT, "defect", date(2026, 3, 25))
    assert result["eligible"] is False
    warranty = result["after_exchange_window"]
    assert warranty["legal_warranty"] == {"covered": True, "until": "2026-05-18"}
    assert warranty["store_can_intermediate"] is True


def test_preference_exchange_window(store):
    result = check_return_eligibility(store, ORDER, CONTACT, "preference", date(2026, 2, 20))
    assert result["eligible"] is True
    assert any("availability" in c for c in result["conditions"])


def test_exclusions_are_always_listed_for_the_agent_to_ask(store):
    result = check_return_eligibility(store, ORDER, CONTACT, "regret", date(2026, 2, 20))
    assert len(result["exclusions_to_check_with_customer"]) == 3


def test_undelivered_order_requires_human(store):
    result = check_return_eligibility(store, 8, "anacarol.ferreira@coldmail.com", "regret",
                                      date(2026, 3, 25))
    assert result["eligible"] is None
    assert result["requires_human"] is True


def test_cancelled_order(store):
    result = check_return_eligibility(store, 17, "matheus.correia@hayoo.com.br", "regret",
                                      date(2026, 3, 25))
    assert result["eligible"] is False
    assert "cancelled" in result["message"]


def test_return_check_requires_verification(store):
    wrong = check_return_eligibility(store, ORDER, "lucas.mendes@jmail.com", "regret", date(2026, 2, 20))
    missing = check_return_eligibility(store, 999, CONTACT, "regret", date(2026, 2, 20))
    assert wrong == missing
    assert wrong["verified"] is False


def test_invalid_return_reason(store):
    result = check_return_eligibility(store, ORDER, CONTACT, "changed_my_mind", date(2026, 2, 20))
    assert result["retryable"] is False


# --- get_shipping_options ----------------------------------------------------


@pytest.mark.parametrize(
    "value, price, free",
    [(600, 0.0, True), (500.01, 0.0, True), (500, 35.0, False), (429.9, 35.0, False)],
)
def test_campo_grande_shipping(store, value, price, free):
    [option] = get_shipping_options(store, "Campo Grande", value)["options"]
    assert option["price"] == price
    assert option["free_shipping"] is free
    assert option["estimated_business_days"] == "1 a 3"


def test_campo_grande_matches_without_case_or_spaces(store):
    assert get_shipping_options(store, "  campo GRANDE ", 100)["region"] == "Campo Grande"


def test_other_cities_get_carriers_without_price(store):
    result = get_shipping_options(store, "Dourados", 3000)
    assert result["region"] == "other cities"
    assert [o["service"] for o in result["options"]] == [
        "PAC (Correios)", "SEDEX (Correios)", "Jadlog (.package)",
    ]
    assert all(o["price"] is None for o in result["options"])


def test_acoustic_drum_kit_needs_special_freight(store):
    result = get_shipping_options(store, "Campo Grande", 4439, product_ids=[139])
    assert result["special_freight"]["products"] == ["Bateria Acústica Yamaha Kit 1 Studio"]


def test_electric_bass_ships_normally(store):
    assert "special_freight" not in get_shipping_options(store, "Dourados", 8732, product_ids=[135])


@pytest.mark.parametrize(
    "kwargs", [{"city": "", "order_value": 100}, {"city": "Dourados", "order_value": -1},
               {"city": "Dourados", "order_value": 100, "product_ids": [9999]}],
)
def test_invalid_shipping_input(store, kwargs):
    assert "error" in get_shipping_options(store, **kwargs)


# --- get_store_status --------------------------------------------------------


def test_open_on_a_weekday():
    result = get_store_status(local(2026, 3, 25, 10, 0))  # Wednesday
    assert result["is_open"] is True
    assert result["weekday"] == "quarta-feira"
    assert result["closes_at"] == "18:00"


def test_closed_on_sunday_reopens_monday():
    result = get_store_status(local(2026, 3, 22, 10, 0))
    assert result["is_open"] is False
    assert result["next_opening"] == {"date": "2026-03-23", "weekday": "segunda-feira",
                                      "time": "09:00"}


def test_saturday_closes_at_13():
    assert get_store_status(local(2026, 3, 28, 12, 59))["is_open"] is True
    closed = get_store_status(local(2026, 3, 28, 13, 0))
    assert closed["is_open"] is False
    assert closed["next_opening"]["date"] == "2026-03-30"


def test_holiday_on_a_weekday():
    # Monday 2026-11-02 is Finados: closed, reopens Tuesday.
    result = get_store_status(local(2026, 11, 2, 10, 0))
    assert result["is_open"] is False
    assert result["holiday"] == "Finados"
    assert result["next_opening"]["date"] == "2026-11-03"


def test_before_opening_reopens_same_day():
    result = get_store_status(local(2026, 3, 25, 8, 0))
    assert result["next_opening"] == {"date": "2026-03-25", "weekday": "quarta-feira",
                                      "time": "09:00"}


def test_closing_time_is_exclusive():
    result = get_store_status(local(2026, 3, 25, 18, 0))
    assert result["is_open"] is False
    assert result["next_opening"]["date"] == "2026-03-26"


def test_utc_input_is_converted_to_store_time():
    # 13:00 UTC = 09:00 in Campo Grande.
    assert get_store_status(datetime(2026, 3, 25, 13, 0, tzinfo=timezone.utc))["is_open"] is True


# --- register_handoff --------------------------------------------------------


def test_handoff_is_written_with_verified_identity(store, tmp_path):
    path = tmp_path / "handoffs.jsonl"
    result = register_handoff(
        store, path, "complaint", "Cliente reclamou do atraso.", local(2026, 3, 25, 10, 0),
        contact=CONTACT, order_id=ORDER, tools_consulted=["get_order_status"],
    )
    assert result["registered"] is True
    assert result["identity_verified"] is True
    assert result["response_time"] == "até 24 horas úteis"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["handoff_id"] == result["handoff_id"]
    assert record["tools_consulted"] == ["get_order_status"]


def test_handoff_identity_is_computed_not_trusted(store, tmp_path):
    result = register_handoff(store, tmp_path / "h.jsonl", "complaint", "x",
                              local(2026, 3, 25, 10, 0), contact="wrong@x.com", order_id=ORDER)
    assert result["identity_verified"] is False


def test_handoff_outside_hours_says_when_team_is_back(store, tmp_path):
    result = register_handoff(store, tmp_path / "h.jsonl", "purchase_intent", "Quer comprar.",
                              local(2026, 3, 22, 10, 0))
    assert result["team_available_now"] is False
    assert result["team_back_at"]["date"] == "2026-03-23"


def test_handoff_appends(store, tmp_path):
    path = tmp_path / "h.jsonl"
    for _ in range(2):
        register_handoff(store, path, "other", "x", local(2026, 3, 25, 10, 0))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.parametrize("reason, summary", [("angry", "x"), ("complaint", "  ")])
def test_invalid_handoff(store, tmp_path, reason, summary):
    result = register_handoff(store, tmp_path / "h.jsonl", reason, summary,
                              local(2026, 3, 25, 10, 0))
    assert result["retryable"] is False


def test_handoff_write_failure_is_a_retryable_error(store, tmp_path):
    result = register_handoff(store, tmp_path, "other", "x", local(2026, 3, 25, 10, 0))
    assert result["retryable"] is True


def test_deadline_is_returned_ready_to_show(store):
    result = check_return_eligibility(store, ORDER, CONTACT, "regret", date(2026, 2, 20))
    assert result["deadline_display"] == "24/02/2026 (terça-feira)"
