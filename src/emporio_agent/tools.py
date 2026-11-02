"""Business tools called by the agent.

Each tool is a plain function that takes its data dependencies explicitly
and returns a JSON-serializable dict. Errors are returned as
{"error": ..., "retryable": bool}, never raised to the LLM as a stack trace.
Prices are returned as numbers; formatting as "R$ 1.234,90" is the agent's job.
"""

from __future__ import annotations

import difflib
import re

from decimal import Decimal

from emporio_agent.data import StoreData, apply_discount, money, normalize_text
from emporio_agent.policies import SECTION_NOTES, TOPICS, PolicyManual

SEARCH_LIMIT = 8
MAX_CANDIDATES = 5
MAX_ALTERNATIVES = 3
ALTERNATIVE_PRICE_BAND = 0.30  # alternatives within ±30% of the product's price
FUZZY_CUTOFF = 0.75  # per-word similarity for typo-tolerant name lookup

# Payment rules (manual, section 3).
PIX_DISCOUNT_PERCENT = 5
MAX_INSTALLMENTS = 12
# (up to N installments, minimum installment value). "Até 3x sem valor mínimo
# (exceto abaixo de R$ 50,00)" is read as a R$ 50 minimum. Paying in 1x is always allowed.
INSTALLMENT_MINIMUMS = ((3, 50.0), (6, 80.0), (12, 100.0))
COMBINED_PAYMENT_THRESHOLD = 2000.0  # combining methods only "acima de R$ 2.000,00"

STRING_TYPES = ("nylon", "steel")
_STOPWORDS = frozenset({"a", "as", "o", "os", "de", "da", "do", "das", "dos", "e", "em",
                        "com", "para", "um", "uma"})


def _error(message: str, **extra) -> dict:
    return {"error": message, "retryable": False, **extra}


# --- Text matching -----------------------------------------------------------


def _stem(token: str) -> str:
    """Very small Portuguese plural stripper: violões -> violao, digitais -> digital."""
    if len(token) > 4 and token.endswith("oes"):
        return token[:-3] + "ao"
    if len(token) > 4 and token.endswith("ais"):
        return token[:-3] + "al"
    if len(token) > 4 and token.endswith("res"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    words = re.split(r"[^a-z0-9]+", normalize_text(text))
    return [_stem(w) for w in words if w and w not in _STOPWORDS]


def _matches(query_tokens: list[str], text_tokens: list[str]) -> bool:
    """Every query token must match a text token: numbers exactly, words by prefix."""
    for q in query_tokens:
        if q.isdigit():
            if q not in text_tokens:
                return False
        elif not any(t.startswith(q) for t in text_tokens):
            return False
    return True


def _name_tokens(product: dict) -> list[str]:
    return _tokens(f"{product['name']} {product['category_name']}")


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_text(text))


def _fuzzy_matches(query_tokens: list[str], text_tokens: list[str]) -> bool:
    return all(
        difflib.get_close_matches(q, text_tokens, n=1, cutoff=FUZZY_CUTOFF) for q in query_tokens
    )


# --- Serialization -----------------------------------------------------------


def _final_price(product: dict) -> float:
    promo = product["promo_price"]
    return float(promo if promo is not None else product["price_brl"])


def _promotion(product: dict) -> dict | None:
    if product["promotion_id"] is None:
        return None
    return {
        "description": product["promotion_description"],
        "discount_percent": int(product["discount_percent"]),
        "original_price": float(product["price_brl"]),
        "promo_price": float(product["promo_price"]),
    }


def _summary(product: dict) -> dict:
    return {
        "product_id": int(product["product_id"]),
        "name": product["name"],
        "category": product["category_name"],
        "price": float(product["price_brl"]),
        "promotion": _promotion(product),
        "final_price": _final_price(product),
        "availability": product["availability"],
        "string_type": product["string_type"],
        "electronics": product["electronics"],
    }


def _details(product: dict) -> dict:
    details = {
        **_summary(product),
        "pix_price": _pix_price(product),
        "stock_quantity": int(product["stock_quantity"]),
        "specs": product["specs"],
        "description": product["description"],
    }
    if details["pix_price"] is None:
        details["pix_note"] = (
            "The PIX discount does not apply on top of a promotional price (manual 6.2); "
            "the PIX price is the promotional price."
        )
    return details


def _pix_price(product: dict) -> float | None:
    """5% off the list price; None when a promotion is active (not cumulative)."""
    if product["promotion_id"] is not None:
        return None
    return apply_discount(float(product["price_brl"]), PIX_DISCOUNT_PERCENT)


def _products(store: StoreData) -> list[dict]:
    return store.products.to_dict("records")


# --- Catalog tools -----------------------------------------------------------


def _resolve_category(store: StoreData, category: str) -> tuple[str | None, dict | None]:
    """Match a category name accent- and plural-insensitively. Returns (name, error)."""
    names = list(store.categories["name"])
    wanted = _tokens(category)
    exact = [n for n in names if _tokens(n) == wanted]
    partial = [n for n in names if wanted and all(w in _tokens(n) for w in wanted)]
    found = exact or partial
    if len(found) == 1:
        return found[0], None
    if not found:
        return None, _error(f"Unknown category: {category!r}.", valid_categories=names)
    return None, _error(f"Ambiguous category: {category!r}.", matching_categories=found)


def search_products(
    store: StoreData,
    query: str | None = None,
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    string_type: str | None = None,
    electronics: bool | None = None,
    include_unavailable: bool = False,
) -> dict:
    """Search the catalog. Price filters and ordering use the final price
    (promotional price when a promotion is active), i.e. what the customer pays."""
    if min_price is not None and min_price < 0 or max_price is not None and max_price < 0:
        return _error("Prices must not be negative.")
    if min_price is not None and max_price is not None and min_price > max_price:
        return _error("min_price is greater than max_price.")
    if string_type is not None and string_type not in STRING_TYPES:
        return _error(f"string_type must be one of {list(STRING_TYPES)}.")

    category_name = None
    if category:
        category_name, error = _resolve_category(store, category)
        if error:
            return error
    query_tokens = _tokens(query) if query else []

    matches = []
    for product in _products(store):
        price = _final_price(product)
        if category_name and product["category_name"] != category_name:
            continue
        if query_tokens and not _matches(query_tokens, _tokens(product["search_text"])):
            continue
        if min_price is not None and price < min_price:
            continue
        if max_price is not None and price > max_price:
            continue
        if string_type is not None and product["string_type"] != string_type:
            continue
        if electronics is not None and product["electronics"] is not electronics:
            continue
        matches.append(product)

    unavailable = [p for p in matches if p["availability"] != "available"]
    if not include_unavailable:
        matches = [p for p in matches if p["availability"] == "available"]
    matches.sort(key=lambda p: (_final_price(p), p["name"]))

    result = {
        "total_count": len(matches),
        "truncated": len(matches) > SEARCH_LIMIT,
        "results": [_summary(p) for p in matches[:SEARCH_LIMIT]],
    }
    if not include_unavailable and unavailable:
        # Lets the agent say "it exists but is out of stock" instead of "not found".
        result["unavailable_matches"] = len(unavailable)
    if result["truncated"]:
        # Ways to narrow the search, counted over all matches.
        result["refine_by"] = {
            "string_type": _count(p["string_type"] for p in matches),
            "electronics": _count(p["electronics"] for p in matches),
            "price_range": [_final_price(matches[0]), _final_price(matches[-1])],
        }
    return result


def _count(values) -> dict:
    counts: dict = {}
    for value in values:
        if value is not None:
            key = str(value).lower()
            counts[key] = counts.get(key, 0) + 1
    return counts


def _alternatives(store: StoreData, product: dict) -> list[dict]:
    """Similar available products: same category, price within the band, closest first."""
    price = float(product["price_brl"])
    pool = [
        p for p in _products(store)
        if p["category_id"] == product["category_id"]
        and p["product_id"] != product["product_id"]
        and p["availability"] == "available"
    ]
    if product["string_type"]:
        same_strings = [p for p in pool if p["string_type"] == product["string_type"]]
        pool = same_strings or pool
    in_band = [p for p in pool if abs(_final_price(p) - price) <= price * ALTERNATIVE_PRICE_BAND]
    pool = in_band or pool
    pool.sort(key=lambda p: abs(_final_price(p) - price))
    return [_summary(p) for p in pool[:MAX_ALTERNATIVES]]


def get_product_details(store: StoreData, product: str | int) -> dict:
    """Find one product by id or (approximate) name and return its full details."""
    products = _products(store)
    found: list[dict] = []
    approximate = False

    text = str(product).strip()
    if text.isdigit():
        found = [p for p in products if p["product_id"] == int(text)]
    else:
        wanted = _tokens(text)
        # 1. Every word matches name or category ("Takamine GD20", "violão Martin").
        #    An exact full name wins, then whole words over prefixes ("Kala" vs "Kalani").
        found = [p for p in products if wanted and _matches(wanted, _name_tokens(p))]
        exact = [p for p in found if normalize_text(p["name"]) == normalize_text(text)]
        whole_words = [p for p in found if set(wanted) <= set(_name_tokens(p))]
        found = exact or whole_words or found
        # 2. Same characters, different separators ("GD-20" vs "GD20").
        if not found and _compact(text):
            found = [p for p in products if _compact(text) in _compact(p["name"])]
        # 3. Typos, word by word ("takamini" vs "takamine").
        if not found and wanted:
            found = [p for p in products if _fuzzy_matches(wanted, _name_tokens(p))]
            approximate = bool(found)

    if not found:
        return _error(
            f"No product matches {product!r}.",
            hint="Try search_products with a broader query or a category.",
        )
    if len(found) > 1:
        return {
            "ambiguous": True,
            "message": "More than one product matches; ask the customer which one.",
            "total_candidates": len(found),
            "candidates": [_summary(p) for p in found[:MAX_CANDIDATES]],
        }

    match = found[0]
    result = {"product": _details(match), "approximate_match": approximate}
    if match["availability"] != "available":
        result["alternatives"] = _alternatives(store, match)
    return result


def list_categories(store: StoreData) -> dict:
    """All categories with how many products are available to buy in each."""
    products = _products(store)
    categories = []
    for row in store.categories.to_dict("records"):
        available = sum(
            1 for p in products
            if p["category_id"] == row["category_id"] and p["availability"] == "available"
        )
        categories.append(
            {
                "category_id": int(row["category_id"]),
                "name": row["name"],
                "description": row["description"],
                "available_products": available,
            }
        )
    return {"categories": categories}


# --- Pricing tools -----------------------------------------------------------


def _minimum_installment(installments: int) -> float:
    for up_to, minimum in INSTALLMENT_MINIMUMS:
        if installments <= up_to:
            return minimum
    raise ValueError(installments)


def calculate_installments(amount: float) -> dict:
    """Interest-free credit card options for an amount (manual, section 3.1).

    Values are rounded to cents; the card operator may adjust the last
    installment by a few cents so the total matches the amount exactly.
    """
    if amount is None or amount <= 0:
        return _error("amount must be greater than zero.")
    amount = money(amount)
    options = []
    for n in range(1, MAX_INSTALLMENTS + 1):
        value = money(Decimal(str(amount)) / n)
        if n > 1 and value < _minimum_installment(n):
            continue
        options.append({"installments": n, "installment_value": value, "total": amount})
    return {
        "amount": amount,
        "interest_free": True,
        "max_installments": options[-1]["installments"],
        "options": options,
        "can_combine_payments": amount > COMBINED_PAYMENT_THRESHOLD,
    }


def list_active_promotions(store: StoreData) -> dict:
    """Products with an active promotion, largest discount first."""
    promoted = [p for p in _products(store) if p["promotion_id"] is not None]
    promoted.sort(key=lambda p: (-int(p["discount_percent"]), p["name"]))
    return {
        "count": len(promoted),
        "promotions": [
            {
                "product_id": int(p["product_id"]),
                "name": p["name"],
                "category": p["category_name"],
                "availability": p["availability"],
                **_promotion(p),
            }
            for p in promoted
        ],
    }


# --- Policies ----------------------------------------------------------------


def get_policy(manual: PolicyManual, topic: str) -> dict:
    """Return one section of the store policy manual, with its version."""
    if topic not in TOPICS:
        return _error(f"Unknown policy topic: {topic!r}.", valid_topics=sorted(TOPICS))
    section = manual.by_topic(topic)
    return {
        "section": section.number,
        "title": section.title,
        "version": manual.version,
        "text": section.text,
        "notes": SECTION_NOTES.get(section.number, []),
    }
