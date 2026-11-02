import json

import pytest

from emporio_agent.data import load_store_data
from emporio_agent.tools import (
    SEARCH_LIMIT,
    get_product_details,
    list_categories,
    search_products,
)


@pytest.fixture(scope="module")
def store():
    return load_store_data()


def ids(result):
    return [p["product_id"] for p in result["results"]]


# --- search_products ---------------------------------------------------------


def test_brief_question_violoes_up_to_1000(store):
    # "Quais opções de violões disponíveis custando até R$1000?"
    result = search_products(store, category="Violões", max_price=1000)
    assert result["total_count"] == 12
    assert result["truncated"] is True
    assert len(result["results"]) == SEARCH_LIMIT
    assert result["refine_by"]["string_type"] == {"nylon": 5, "steel": 7}


def test_results_are_sorted_by_final_price(store):
    prices = [p["final_price"] for p in search_products(store, category="Violões")["results"]]
    assert prices == sorted(prices)


def test_only_sellable_products_by_default(store):
    result = search_products(store, category="Violões", max_price=1000)
    assert {p["availability"] for p in result["results"]} == {"available"}
    everything = search_products(store, category="Violões", max_price=1000, include_unavailable=True)
    assert everything["total_count"] == 14
    narrow = search_products(
        store, category="Violões", min_price=740, max_price=800, include_unavailable=True
    )
    assert {96: "out_of_stock", 113: "discontinued"}.items() <= {
        p["product_id"]: p["availability"] for p in narrow["results"]
    }.items()


def test_coming_soon_is_not_sellable_even_with_stock(store):
    assert 130 not in ids(search_products(store, category="Ukuleles"))


def test_unavailable_matches_are_counted(store):
    result = search_products(store, query="Giannini GF-3D")
    assert result["total_count"] == 0
    assert result["unavailable_matches"] == 1


def test_category_and_query_ignore_accents_and_plurals(store):
    assert search_products(store, category="violao")["total_count"] == 33
    assert search_products(store, category="VIOLÕES")["total_count"] == 33
    assert search_products(store, query="violões")["total_count"] == 33


def test_partial_category_name(store):
    result = search_products(store, category="teclados")
    assert {p["category"] for p in result["results"]} == {"Teclados e Pianos"}


def test_unknown_and_ambiguous_category(store):
    unknown = search_products(store, category="flautas doces")
    assert "error" in unknown and "Violões" in unknown["valid_categories"]
    ambiguous = search_products(store, category="instrumentos de sopro")
    assert len(ambiguous["matching_categories"]) == 2


def test_empty_category_returns_no_results(store):
    result = search_products(store, category="Cordas Orquestrais")
    assert result == {"total_count": 0, "truncated": False, "results": []}


def test_string_type_and_electronics_filters(store):
    nylon_electric = search_products(store, category="Violões", string_type="nylon", electronics=True)
    assert nylon_electric["total_count"] > 0
    for p in nylon_electric["results"]:
        assert p["string_type"] == "nylon" and p["electronics"] is True


def test_steel_is_searchable_as_aco(store):
    result = search_products(store, query="aço", category="Violões")
    assert {p["string_type"] for p in result["results"]} == {"steel"}


def test_price_filter_uses_promotional_price(store):
    # Crafter HT-100: R$ 2.399,00 with an active 18% promotion -> R$ 1.967,18.
    assert 90 in ids(search_products(store, category="Violões", max_price=2000, min_price=1900))


def test_numbers_match_exactly_not_by_prefix(store):
    # "7" must not match the "70" in other names or the "650mm" scale.
    result = search_products(store, query="7 cordas")
    assert result["total_count"] > 0
    assert all("7 Cordas" in p["name"] for p in result["results"])


@pytest.mark.parametrize(
    "kwargs",
    [{"min_price": 500, "max_price": 100}, {"max_price": -1}, {"string_type": "metal"}],
)
def test_invalid_filters_return_structured_error(store, kwargs):
    assert search_products(store, **kwargs)["retryable"] is False


def test_search_result_is_json_serializable(store):
    json.dumps(search_products(store, category="Violões"))


# --- get_product_details -----------------------------------------------------


@pytest.mark.parametrize(
    "name, product_id",
    [
        ("Takamine GD20", 95),
        ("takamine gd-20", 95),  # different separator
        ("Crafter HT100", 90),
        ("95", 95),
        (95, 95),
        ("Martin D-28", 93),  # not the D-28E
        ("Martin D-28E", 108),
    ],
)
def test_product_lookup(store, name, product_id):
    result = get_product_details(store, name)
    assert result["product"]["product_id"] == product_id
    assert result["approximate_match"] is False


def test_typo_is_flagged_as_approximate(store):
    result = get_product_details(store, "takamini gd20")
    assert result["product"]["product_id"] == 95
    assert result["approximate_match"] is True


def test_ambiguous_name_returns_candidates(store):
    result = get_product_details(store, "Kala tenor")
    assert result["ambiguous"] is True
    assert {c["product_id"] for c in result["candidates"]} == {124, 126}  # not the Kalani


def test_category_word_helps_lookup(store):
    result = get_product_details(store, "violão Martin")
    assert {c["product_id"] for c in result["candidates"]} == {93, 108}


def test_unknown_product_returns_error(store):
    result = get_product_details(store, "Fender Precision")
    assert "error" in result and "hint" in result


def test_details_include_inactive_promotion_as_none(store):
    # Takamine GD20 has a 12% promotion that is not active.
    product = get_product_details(store, "Takamine GD20")["product"]
    assert product["price"] == 2199.0
    assert product["promotion"] is None
    assert product["final_price"] == 2199.0


def test_details_include_active_promotion(store):
    promotion = get_product_details(store, "Crafter HT-100")["product"]["promotion"]
    assert promotion == {
        "description": "Desconto Exclusivo",
        "discount_percent": 18,
        "original_price": 2399.0,
        "promo_price": 1967.18,
        "display": "de R$ 2.399,00 por R$ 1.967,18 (18% de desconto)",
    }


def test_out_of_stock_product_comes_with_alternatives(store):
    result = get_product_details(store, "Giannini GF-3D")
    assert result["product"]["availability"] == "out_of_stock"
    alternatives = result["alternatives"]
    assert 0 < len(alternatives) <= 3
    for alt in alternatives:
        assert alt["availability"] == "available"
        assert alt["category"] == "Violões"
        assert alt["string_type"] == "steel"
        assert abs(alt["final_price"] - 799.9) <= 799.9 * 0.3


def test_available_product_has_no_alternatives(store):
    assert "alternatives" not in get_product_details(store, "Takamine GD20")


def test_suppressed_description_stays_hidden(store):
    product = get_product_details(store, "Music Man Bass 1X")["product"]
    assert product["description"] is None
    assert product["specs"]["string_count"] == 4


# --- list_categories ---------------------------------------------------------


def test_list_categories_counts_available_products(store):
    categories = {c["name"]: c["available_products"] for c in list_categories(store)["categories"]}
    assert len(categories) == 9
    assert categories["Violões"] == 33
    assert categories["Cordas Orquestrais"] == 0
    assert categories["Ukuleles"] == 14  # 15 products, one coming_soon


# --- Real-run regression: "violão clássico" returned nothing ------------------


def test_classico_matches_nylon_guitars(store):
    # Real run: query="violão clássico" + string_type=nylon gave 0 results and the
    # agent told the customer the store had no classical guitars (there are 15).
    result = search_products(store, query="violão clássico", category="Violões", string_type="nylon")
    assert result["total_count"] == 15
    assert "query_relaxed" not in result


@pytest.mark.parametrize(
    "query, field, value",
    [("clássicos", "string_type", "nylon"), ("folk", "string_type", "steel"),
     ("eletroacústico", "electronics", True)],
)
def test_customer_words_derived_from_specs(store, query, field, value):
    result = search_products(store, query=query, category="Violões")
    assert result["total_count"] > 3
    assert {p[field] for p in result["results"]} == {value}


def test_unmatched_words_fall_back_to_the_filters(store):
    result = search_products(store, query="para iniciante barato", category="Violões", max_price=500)
    assert result["query_relaxed"] is True
    assert result["total_count"] == 2  # Tagima AC-39 and Rozini RC-104
    assert "note" in result


def test_unmatched_words_without_filters_still_return_nothing(store):
    result = search_products(store, query="theremin")
    assert result["total_count"] == 0
    assert "query_relaxed" not in result


# --- Instructions next to the data (from eval failures) ----------------------


def test_truncated_search_tells_the_total(store):
    result = search_products(store, category="Violões", max_price=1000)
    assert "12 options in total" in result["instruction"]


def test_unavailable_product_instructs_to_show_alternatives(store):
    result = get_product_details(store, "Giannini GF-3D")
    assert "do not ask whether to show them" in result["instruction"]


def test_available_product_has_no_instruction(store):
    assert "instruction" not in get_product_details(store, "Takamine GD20")
