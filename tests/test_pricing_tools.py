import json

import pytest

from emporio_agent.data import load_store_data, money
from emporio_agent.tools import calculate_installments, get_product_details, list_active_promotions


@pytest.fixture(scope="module")
def store():
    return load_store_data()


# --- PIX ---------------------------------------------------------------------


def test_pix_is_5_percent_off_list_price(store):
    # "Quanto custa o Takamine GD20?" R$ 2.199,00; PIX R$ 2.089,05.
    product = get_product_details(store, "Takamine GD20")["product"]
    assert product["price"] == 2199.0
    assert product["pix_price"] == 2089.05
    assert "pix_note" not in product


def test_pix_does_not_stack_on_promotion(store):
    # Crafter HT-100: 18% promotion -> R$ 1.967,18. 5% on top would be R$ 1.868,82.
    product = get_product_details(store, "Crafter HT-100")["product"]
    assert product["final_price"] == 1967.18
    assert product["pix_price"] is None
    assert "does not apply" in product["pix_note"]


# --- Installments ------------------------------------------------------------


def max_installments(amount):
    return calculate_installments(amount)["max_installments"]


@pytest.mark.parametrize(
    "amount, expected",
    [
        (40, 1),  # 1x is always allowed
        (120, 2),  # 3x would be R$ 40 < R$ 50
        (150, 3),  # 3x of R$ 50: exactly the minimum
        (159.9, 3),  # 4x would be R$ 39,98 < R$ 80
        (480, 6),  # 6x of R$ 80: exactly the minimum
        (599.9, 6),  # 7x would be R$ 85,70 < R$ 100
        (1200, 12),  # 12x of R$ 100: exactly the minimum
        (5999, 12),  # never more than 12x
    ],
)
def test_installment_minimums(amount, expected):
    assert max_installments(amount) == expected


def test_installment_values_are_rounded_to_cents():
    options = {o["installments"]: o for o in calculate_installments(1967.18)["options"]}
    assert options[10]["installment_value"] == 196.72
    assert options[10]["total"] == 1967.18
    assert calculate_installments(1967.18)["interest_free"] is True


def test_installment_options_are_contiguous():
    counts = [o["installments"] for o in calculate_installments(1000)["options"]]
    assert counts == list(range(1, 11))  # 10x of R$ 100; 11x would be R$ 90,91


@pytest.mark.parametrize("amount, allowed", [(2000, False), (2000.01, True), (2199, True)])
def test_combined_payment_only_above_2000(amount, allowed):
    assert calculate_installments(amount)["can_combine_payments"] is allowed


@pytest.mark.parametrize("amount", [0, -10, None])
def test_invalid_amount_returns_structured_error(amount):
    assert calculate_installments(amount)["retryable"] is False


def test_installments_for_a_product_use_its_final_price(store):
    # Real run: the model passed the previous product's price (2199) for the Crafter.
    result = calculate_installments(store=store, product="Crafter HT-100")
    assert result["amount"] == 1967.18  # promotional price, not the list price
    assert result["price_basis"]["product"] == "Crafter HT-100 Folk Aço Natural"
    assert result["price_basis"]["promotion"]["discount_percent"] == 18
    options = {o["installments"]: o["installment_value"] for o in result["options"]}
    assert options[10] == 196.72


def test_product_wins_over_a_wrong_amount(store):
    assert calculate_installments(2199, store=store, product="Crafter HT-100")["amount"] == 1967.18


def test_installments_for_an_ambiguous_product_return_candidates(store):
    assert calculate_installments(store=store, product="Kala tenor")["ambiguous"] is True


def test_installments_for_an_unknown_product_return_an_error(store):
    assert "error" in calculate_installments(store=store, product="Fender Precision")


# --- Promotions --------------------------------------------------------------


def test_list_active_promotions(store):
    result = list_active_promotions(store)
    assert result["count"] == 4
    assert [p["product_id"] for p in result["promotions"]] == [121, 90, 127, 94]  # by discount
    crafter = next(p for p in result["promotions"] if p["product_id"] == 90)
    assert crafter["original_price"] == 2399.0
    assert crafter["discount_percent"] == 18
    assert crafter["promo_price"] == 1967.18
    json.dumps(result)


def test_inactive_promotions_are_not_listed(store):
    listed = {p["product_id"] for p in list_active_promotions(store)["promotions"]}
    assert 95 not in listed  # Takamine GD20, inactive 12%


# --- Rounding ----------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [(0.125, 0.13), (2.675, 2.68), (1.005, 1.01)])
def test_money_rounds_half_up(value, expected):
    # Plain round() gives 0.12, 2.67 and 1.0 for these.
    assert money(value) == expected


# --- Real CLI session: two products, and "PIX in installments" -----------------


def test_installments_for_several_products_add_their_final_prices(store):
    # Real run: the agent computed each ukulele separately and said 6x was not
    # possible for both together. 439,20 + 323,10 = 762,30, so 6x of 127,05.
    result = calculate_installments(store=store, products=["Ohana CK-20", "Kalani KAL-700T"])
    assert result["amount"] == 762.3
    assert len(result["price_basis"]["items"]) == 2
    options = {o["installments"]: o["installment_value"] for o in result["options"]}
    assert options[6] == 127.05


def test_unknown_product_in_a_list_is_reported(store):
    result = calculate_installments(store=store, products=["Ohana CK-20", "Fender Precision"])
    assert "error" in result and result["product_query"] == "Fender Precision"


def test_installment_notes_say_credit_only_and_what_combining_means():
    notes = " ".join(calculate_installments(1000)["notes"])
    assert "credit card only" in notes
    assert "nothing to do with buying several products" in notes
