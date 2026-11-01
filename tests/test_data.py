from datetime import date

import pandas as pd
import pytest

from emporio_agent.data import (
    DataLoadError,
    availability,
    find_table_file,
    load_store_data,
    normalize_text,
    parse_specs,
    pick_active_promotions,
)


@pytest.fixture(scope="module")
def store():
    return load_store_data()


@pytest.fixture(scope="module")
def products(store):
    return store.products.set_index("product_id")


@pytest.fixture(scope="module")
def orders(store):
    return store.orders.set_index("order_id")


def test_all_tables_load_with_expected_sizes(store):
    assert len(store.categories) == 9
    assert len(store.customers) == 50
    assert len(store.products) == 65
    assert len(store.orders) == 20
    assert len(store.order_items) == 22
    assert len(store.promotions) == 25


def test_table_is_found_by_suffix_not_by_prefix():
    assert find_table_file("orders").name.endswith(" - orders.csv")
    assert find_table_file("order_items").name.endswith(" - order_items.csv")


def test_missing_table_fails_with_clear_error(tmp_path):
    with pytest.raises(DataLoadError, match=r"Expected exactly one file matching '\* - \w+\.csv'"):
        load_store_data(tmp_path)


# --- Customers ---------------------------------------------------------------


def test_customer_contact_is_normalized(store):
    lucas = store.customers.set_index("customer_id").loc[1]
    assert lucas["email"] == "lucas.mendes@jmail.com"
    assert lucas["phone_digits"] == "67998123456"
    assert lucas["first_name"] == "Lucas"


# --- Specs -------------------------------------------------------------------


def test_strings_material_becomes_string_type(products):
    assert products.loc[81, "string_type"] == "nylon"
    assert products.loc[90, "string_type"] == "steel"
    assert products.loc[81, "string_count"] is None


def test_strings_count_becomes_string_count(products):
    assert products.loc[145, "string_count"] == 6
    assert products.loc[135, "string_count"] == 4
    assert products.loc[145, "string_type"] is None


def test_raw_strings_key_is_never_exposed(products):
    assert all("strings" not in specs for specs in products["specs"])


def test_specs_json_with_spaces_is_parsed(products):
    # Basses and drums use '{"body": "Ash", ...}' with spaces after separators.
    assert products.loc[135, "specs"]["body"] == "Ash"
    assert products.loc[139, "specs"]["pieces"] == 5


def test_electronics_is_boolean(products):
    assert products.loc[94, "electronics"] is True
    assert products.loc[81, "electronics"] is False
    assert products.loc[145, "electronics"] is None


@pytest.mark.parametrize(
    "raw", ['{"strings": "metal"}', '{"electronics": "maybe"}', "{not json"]
)
def test_unexpected_specs_fail_loudly(raw):
    with pytest.raises(DataLoadError):
        parse_specs(raw)


# --- Availability and descriptions ------------------------------------------


@pytest.mark.parametrize(
    "product_id, expected",
    [(81, "available"), (96, "out_of_stock"), (113, "discontinued"), (130, "coming_soon")],
)
def test_availability(products, product_id, expected):
    assert products.loc[product_id, "availability"] == expected


def test_status_wins_over_stock():
    assert availability("coming_soon", 2) == "coming_soon"
    assert availability("discontinued", 13) == "discontinued"
    assert availability("active", 0) == "out_of_stock"


def test_contradicting_descriptions_are_suppressed(products):
    for category_id in (2, 3, 4):
        assert products[products["category_id"] == category_id]["description"].isna().all()
    assert products.loc[95, "description"].startswith("Violão dreadnought Takamine")


def test_search_text_is_lowercase_without_accents(products):
    assert "bateria acustica yamaha" in products.loc[139, "search_text"]
    assert "aco" in products.loc[90, "search_text"]  # steel strings, searchable in Portuguese


def test_normalize_text():
    assert normalize_text("  Violão  AÇO ") == "violao aco"


# --- Promotions --------------------------------------------------------------


def test_active_promotion_with_promo_price(products):
    crafter = products.loc[90]
    assert crafter["discount_percent"] == 18
    assert crafter["promo_price"] == 1967.18


def test_inactive_promotion_is_not_applied(products):
    # Takamine GD20 has a 12% promotion that is not active.
    assert products.loc[95, "discount_percent"] is None
    assert products.loc[95, "promo_price"] is None


def test_only_four_promotions_are_active(store):
    assert sorted(store.active_promotions["product_id"]) == [90, 94, 121, 127]


def test_largest_active_discount_wins_when_duplicated():
    promotions = pd.DataFrame(
        {
            "promotion_id": [1, 2, 3],
            "product_id": [10, 10, 10],
            "discount_percent": [10, 25, 40],
            "description": ["a", "b", "c"],
            "is_active": [True, True, False],
        }
    )
    picked = pick_active_promotions(promotions)
    assert len(picked) == 1
    assert picked.loc[0, "promotion_id"] == 2


# --- Orders ------------------------------------------------------------------


def test_order_fields_are_typed(orders):
    assert orders.loc[1, "order_date"] == date(2025, 10, 15)
    assert orders.loc[1, "estimated_delivery"] == date(2025, 10, 25)
    assert orders.loc[4, "total_brl"] == 799.9


def test_empty_order_fields_are_none(orders):
    assert orders.loc[9, "estimated_delivery"] is None
    assert orders.loc[9, "tracking_code"] is None
    assert orders.loc[1, "notes"] is None


def test_order_status_is_translated(orders):
    assert orders.loc[9, "status_label"] == "confirmado, em preparação"
    assert orders.loc[17, "status_label"] == "cancelado"


def test_recorded_total_is_kept_not_recalculated(orders):
    # Items x price would give 3.498,00; the recorded total is what was charged.
    assert orders.loc[3, "total_brl"] == 3450.0
