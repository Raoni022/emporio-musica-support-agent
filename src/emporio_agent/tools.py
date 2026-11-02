"""Business tools called by the agent.

Each tool is a plain function that takes its data dependencies explicitly
and returns a JSON-serializable dict. Errors are returned as
{"error": ..., "retryable": bool}, never raised to the LLM as a stack trace.
Prices are returned as numbers; formatting as "R$ 1.234,90" is the agent's job.
"""

from __future__ import annotations

import difflib
import functools
import json
import re
import uuid
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import holidays

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

    def find(tokens: list[str]) -> list[dict]:
        found = []
        for product in _products(store):
            price = _final_price(product)
            if category_name and product["category_name"] != category_name:
                continue
            if tokens and not _matches(tokens, _tokens(product["search_text"])):
                continue
            if min_price is not None and price < min_price:
                continue
            if max_price is not None and price > max_price:
                continue
            if string_type is not None and product["string_type"] != string_type:
                continue
            if electronics is not None and product["electronics"] is not electronics:
                continue
            found.append(product)
        return found

    matches = find(query_tokens)
    # The model often repeats the customer's words in `query` next to structured
    # filters ("violão clássico" + nylon). If those words match nothing but the
    # filters do, answer with the filters instead of a false "we don't have it".
    has_filters = any(v is not None for v in (category_name, min_price, max_price,
                                              string_type, electronics))
    query_relaxed = False
    if not matches and query_tokens and has_filters:
        matches = find([])
        query_relaxed = bool(matches)

    unavailable = [p for p in matches if p["availability"] != "available"]
    if not include_unavailable:
        matches = [p for p in matches if p["availability"] == "available"]
    matches.sort(key=lambda p: (_final_price(p), p["name"]))

    result = {
        "total_count": len(matches),
        "truncated": len(matches) > SEARCH_LIMIT,
        "results": [_summary(p) for p in matches[:SEARCH_LIMIT]],
    }
    if query_relaxed:
        result["query_relaxed"] = True
        result["note"] = (f"No product matched the words {query!r}; these results match the "
                          "other filters only.")
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


def calculate_installments(
    amount: float | None = None, *, store: StoreData | None = None, product: str | None = None
) -> dict:
    """Interest-free credit card options (manual, section 3.1).

    For a product, the tool looks up the final price itself (promotional when
    active), so the model never has to know it first: in a real run it called
    this in parallel with get_product_details and passed the previous product's
    price. `amount` stays for other values (e.g. two items together).
    Values are rounded to cents; the card operator may adjust the last
    installment by a few cents so the total matches the amount exactly.
    """
    basis = None
    if product is not None:
        if store is None:
            raise ValueError("store is required to look up a product")
        found = get_product_details(store, product)
        if "product" not in found:
            return found  # error or ambiguous candidates, for the agent to resolve
        details = found["product"]
        amount = details["final_price"]
        basis = {
            "product": details["name"],
            "price": details["price"],
            "promotion": details["promotion"],
            "final_price": details["final_price"],
        }
    if amount is None or amount <= 0:
        return _error("Pass a product, or an amount greater than zero.")
    amount = money(amount)
    options = []
    for n in range(1, MAX_INSTALLMENTS + 1):
        value = money(Decimal(str(amount)) / n)
        if n > 1 and value < _minimum_installment(n):
            continue
        options.append({"installments": n, "installment_value": value, "total": amount})
    result = {
        "amount": amount,
        "interest_free": True,
        "max_installments": options[-1]["installments"],
        "options": options,
        "can_combine_payments": amount > COMBINED_PAYMENT_THRESHOLD,
    }
    if basis:
        result = {"price_basis": basis, **result}
    return result


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


# --- Orders ------------------------------------------------------------------

# Same message for unknown order, wrong contact and malformed input, so the
# response never reveals whether an order exists (LGPD, manual section 9).
ORDER_NOT_VERIFIED = (
    "No order matches this order number and contact. Ask the customer to check the "
    "order number and to use the e-mail or the phone number (with area code) registered "
    "with the store."
)
STALE_ORDER_DAYS = 7  # pending/confirmed for longer than this is flagged as possibly delayed

_PAYMENT_LABELS = {
    "pix": "PIX",
    "debit": "cartão de débito",
    "boleto": "boleto bancário",
}


def _payment_label(method: str) -> str:
    credit = re.fullmatch(r"credit_(\d+)x", method)
    if credit:
        return f"cartão de crédito em {credit.group(1)}x"
    return _PAYMENT_LABELS.get(method, method)


def _contact_matches(contact: str, customer: dict) -> bool:
    """E-mail (case-insensitive) or phone digits, with or without the +55 country code.
    The area code (DDD) is required."""
    contact = (contact or "").strip()
    if "@" in contact:
        return contact.lower() == customer["email"]
    digits = re.sub(r"\D", "", contact)
    if len(digits) in (12, 13) and digits.startswith("55"):
        digits = digits[2:]
    return bool(digits) and digits == customer["phone_digits"]


def verify_order_owner(store: StoreData, order_id, contact: str) -> tuple[dict, dict] | None:
    """Return (order, customer) only if the contact belongs to the order's customer."""
    match = re.fullmatch(r"\s*#?\s*(\d+)\s*", str(order_id))
    if not match:
        return None
    orders = store.orders[store.orders["order_id"] == int(match.group(1))].to_dict("records")
    if not orders:
        return None
    order = orders[0]
    customers = store.customers[store.customers["customer_id"] == order["customer_id"]]
    if customers.empty:
        return None
    customer = customers.to_dict("records")[0]
    if not _contact_matches(contact, customer):
        return None
    return order, customer


def _order_items(store: StoreData, order_id: int) -> list[dict]:
    names = dict(zip(store.products["product_id"], store.products["name"]))
    items = store.order_items[store.order_items["order_id"] == order_id]
    return [
        {"name": names.get(int(item["product_id"]), "produto fora do catálogo atual"),
         "quantity": int(item["quantity"])}
        for item in items.to_dict("records")
    ]


def _delay(order: dict, today: date) -> dict:
    """Flags for orders that look late, so the agent can offer a handoff."""
    if order["status"] == "shipped" and order["estimated_delivery"]:
        days = (today - order["estimated_delivery"]).days
        if days > 0:
            return {"late": True, "days_past_estimate": days}
    if order["status"] in ("pending", "confirmed"):
        days = (today - order["order_date"]).days
        if days > STALE_ORDER_DAYS:
            return {"late": True, "days_in_current_status_since_order": days}
    return {"late": False}


def get_order_status(store: StoreData, order_id, contact: str, today: date) -> dict:
    """Order status, only after checking the contact against the order's customer."""
    verified = verify_order_owner(store, order_id, contact)
    if verified is None:
        return _error(ORDER_NOT_VERIFIED, verified=False)
    order, customer = verified
    estimated = order["estimated_delivery"]
    delay = _delay(order, today)
    return {
        "verified": True,
        "customer_first_name": customer["first_name"],
        "order_id": int(order["order_id"]),
        "status": order["status"],
        "status_label": order["status_label"],
        "order_date": order["order_date"].isoformat(),
        "items": _order_items(store, int(order["order_id"])),
        "total": float(order["total_brl"]),  # as recorded (what was charged); never recalculated
        "payment_method": _payment_label(order["payment_method"]),
        "tracking_code": order["tracking_code"],
        "estimated_delivery": estimated.isoformat() if estimated else None,
        "notes": order["notes"],
        **delay,
        "suggest_handoff": delay["late"],
    }


# --- Returns and exchanges (manual, section 4) -------------------------------

RETURN_REASONS = ("regret", "defect", "preference")
REGRET_DAYS = 7  # after receipt (4.1)
DEFECT_DAYS = 30  # after purchase (4.2)
PREFERENCE_DAYS = 7  # 4.3 does not say from when; counted from receipt, like regret
LEGAL_WARRANTY_DAYS = 90  # after receipt (8.1)

# 4.4: the data has no marker for these, so the agent must ask instead of assuming.
RETURN_EXCLUSIONS = [
    "Instruments with customization or made-to-order adjustments (setup, special regulation).",
    "Products bought in clearance sales marked as 'final sale'.",
    "Mouthpieces of wind instruments (hygiene).",
]

_RETURN_CONDITIONS = {
    "regret": [
        "Original packaging, no signs of use, with all accessories and manuals.",
        "Refund in the original payment method within 10 business days.",
        "Return shipping is paid by the store.",
    ],
    "defect": [
        "Manufacturing defects only; misuse, falls, excessive humidity or unauthorized "
        "modifications are not covered.",
    ],
    "preference": [
        "Product in perfect condition and in the original packaging.",
        "Subject to availability of the desired product.",
        "Price differences are charged or refunded.",
    ],
}


def _window(start: date, days: int, today: date) -> dict:
    deadline = start + timedelta(days=days)
    remaining = (deadline - today).days
    return {
        "eligible": remaining >= 0,
        "deadline": deadline.isoformat(),
        "days_remaining": max(remaining, 0),
        "days_past_deadline": max(-remaining, 0),
    }


def check_return_eligibility(
    store: StoreData, order_id, contact: str, reason: str, today: date
) -> dict:
    """Deterministic return/exchange eligibility for a verified order (manual, section 4)."""
    if reason not in RETURN_REASONS:
        return _error(f"reason must be one of {list(RETURN_REASONS)}.")
    verified = verify_order_owner(store, order_id, contact)
    if verified is None:
        return _error(ORDER_NOT_VERIFIED, verified=False)
    order, customer = verified
    base = {
        "verified": True,
        "customer_first_name": customer["first_name"],
        "order_id": int(order["order_id"]),
        "order_status": order["status_label"],
        "reason": reason,
    }

    if order["status"] == "cancelled":
        return {**base, "eligible": False, "message": "The order is already cancelled.",
                "notes": order["notes"]}
    if order["status"] != "delivered" or order["estimated_delivery"] is None:
        # The manual does not cover cancelling or returning an order before delivery.
        return {
            **base,
            "eligible": None,
            "requires_human": True,
            "message": "The order has not been delivered yet; the policy does not cover this "
                       "case. Offer to hand it to the team (reason: cancellation).",
        }

    # There is no actual delivery date in the data; the estimated delivery is used.
    received = order["estimated_delivery"]
    start = order["order_date"] if reason == "defect" else received
    days = {"regret": REGRET_DAYS, "defect": DEFECT_DAYS, "preference": PREFERENCE_DAYS}[reason]
    result = {
        **base,
        **_window(start, days, today),
        "window": f"{days} calendar days after {'purchase' if reason == 'defect' else 'receipt'}",
        "order_date": order["order_date"].isoformat(),
        "receipt_date_assumed": received.isoformat(),
        "receipt_date_note": "Estimated delivery date used as the receipt date; ask the "
                             "customer if they received it on another day.",
        "conditions": _RETURN_CONDITIONS[reason],
        "exclusions_to_check_with_customer": RETURN_EXCLUSIONS,
    }
    if reason == "defect" and not result["eligible"]:
        legal = _window(received, LEGAL_WARRANTY_DAYS, today)
        result["after_exchange_window"] = {
            "legal_warranty": {"covered": legal["eligible"], "until": legal["deadline"]},
            "manufacturer_warranty": "6 months to 2 years depending on the manufacturer; "
                                     "see the product's warranty certificate.",
            "store_can_intermediate": True,
        }
    if result["eligible"]:
        result["next_step"] = "Offer to hand the request to the team to start the process."
    return result


# --- Shipping (manual, section 5) --------------------------------------------

LOCAL_CITIES = ("campo grande",)  # the manual does not list the metropolitan region's cities
FREE_SHIPPING_ABOVE = 500.0
LOCAL_SHIPPING_FEE = 35.0
CARRIERS = [
    {"service": "PAC (Correios)", "estimated_business_days": "5 a 12"},
    {"service": "SEDEX (Correios)", "estimated_business_days": "2 a 5"},
    {"service": "Jadlog (.package)", "estimated_business_days": "3 a 8"},
]


def _is_large(product: dict) -> bool:
    """Acoustic drum kits, digital pianos, acoustic double basses (manual 5.2)."""
    name = normalize_text(product["name"])
    return (
        (product["category_id"] == 3 and "acustica" in name)
        or (product["category_id"] == 4 and "piano" in name)
        or (product["category_id"] == 8 and "contrabaixo" in name)
    )


def get_shipping_options(
    store: StoreData, city: str, order_value: float, product_ids: list[int] | None = None
) -> dict:
    """Delivery options and costs. Outside Campo Grande the cost depends on the CEP,
    so no value is returned for it."""
    if not city or not city.strip():
        return _error("city is required.")
    if order_value is None or order_value < 0:
        return _error("order_value must not be negative.")

    products = {int(p["product_id"]): p for p in _products(store)}
    unknown = [pid for pid in product_ids or [] if pid not in products]
    if unknown:
        return _error(f"Unknown product ids: {unknown}.")
    large = [products[pid]["name"] for pid in product_ids or [] if _is_large(products[pid])]

    common = {
        "city": city.strip(),
        "order_value": money(order_value),
        "damaged_on_arrival": "Refuse the delivery and contact the store immediately.",
    }
    if large:
        common["special_freight"] = {
            "products": large,
            "note": "Large instruments may need special freight with an individual quote; "
                    "offer to hand it to the team.",
        }

    if normalize_text(city) in LOCAL_CITIES:
        free = order_value > FREE_SHIPPING_ABOVE
        return {
            **common,
            "region": "Campo Grande",
            "options": [{
                "service": "Motoboy da loja",
                "price": 0.0 if free else LOCAL_SHIPPING_FEE,
                "free_shipping": free,
                "estimated_business_days": "1 a 3",
                "note": "The customer is called before the delivery.",
            }],
        }
    return {
        **common,
        "region": "other cities",
        "options": [{**c, "price": None, "tracking": True, "insurance": True} for c in CARRIERS],
        "price_note": "Calculated from the destination CEP, weight and dimensions at checkout; "
                      "do not quote a value.",
    }


# --- Store hours (manual, section 2) -----------------------------------------

# Campo Grande has had no daylight saving time since 2019: a fixed UTC-4 offset
# avoids depending on the IANA database (not bundled with Python on Windows).
STORE_TZ = timezone(timedelta(hours=-4), "America/Campo_Grande")
OPENING_HOURS = {  # weekday() -> (open, close); Sunday closed
    0: (time(9), time(18)), 1: (time(9), time(18)), 2: (time(9), time(18)),
    3: (time(9), time(18)), 4: (time(9), time(18)), 5: (time(9), time(13)),
}
WEEKDAYS_PT = ("segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
               "sexta-feira", "sábado", "domingo")


@functools.lru_cache(maxsize=8)
def _holiday_calendar(year: int) -> holidays.HolidayBase:
    # National + Mato Grosso do Sul holidays; municipal holidays are not covered.
    return holidays.Brazil(subdiv="MS", years=year)


def _holiday(day: date) -> str | None:
    return _holiday_calendar(day.year).get(day)


def _hours(day: date) -> tuple[time, time] | None:
    if _holiday(day):
        return None
    return OPENING_HOURS.get(day.weekday())


def get_store_status(now: datetime) -> dict:
    """Whether the store (and WhatsApp team) is open now and, if not, when it reopens."""
    now = now.astimezone(STORE_TZ) if now.tzinfo else now.replace(tzinfo=STORE_TZ)
    today = now.date()
    hours = _hours(today)
    is_open = bool(hours) and hours[0] <= now.time() < hours[1]
    result = {
        "now": now.strftime("%Y-%m-%d %H:%M"),
        "weekday": WEEKDAYS_PT[today.weekday()],
        "timezone": "America/Campo_Grande (UTC-4)",
        "is_open": is_open,
        "today_hours": f"{hours[0]:%H:%M} às {hours[1]:%H:%M}" if hours else None,
        "holiday": _holiday(today),
        "special_dates_note": "Hours may be extended on special dates (Black Friday, store "
                              "anniversary, Christmas), announced on social media.",
    }
    if is_open:
        result["closes_at"] = f"{hours[1]:%H:%M}"
        return result

    day = today if hours and now.time() < hours[0] else today + timedelta(days=1)
    while _hours(day) is None:
        day += timedelta(days=1)
    result["next_opening"] = {
        "date": day.isoformat(),
        "weekday": WEEKDAYS_PT[day.weekday()],
        "time": f"{_hours(day)[0]:%H:%M}",
    }
    return result


# --- Handoff to the team (manual, section 7.3) -------------------------------

HANDOFF_REASONS = (
    "complaint", "purchase_intent", "late_order", "cancellation",
    "data_deletion", "unanswered", "other",
)
HANDOFF_RESPONSE_TIME = "até 24 horas úteis"
MAX_SUMMARY_CHARS = 1000


def register_handoff(
    store: StoreData,
    path: Path,
    reason: str,
    summary: str,
    now: datetime,
    contact: str | None = None,
    order_id=None,
    suggested_next_action: str | None = None,
    tools_consulted: list[str] | None = None,
) -> dict:
    """Record a case for a human in a JSONL queue (stand-in for a helpdesk).

    identity_verified is computed here, never taken from the LLM. tools_consulted
    is filled by the agent loop, so the attendant sees what was already checked.
    """
    if reason not in HANDOFF_REASONS:
        return _error(f"reason must be one of {list(HANDOFF_REASONS)}.")
    if not summary or not summary.strip():
        return _error("summary is required.")

    verified = bool(order_id and contact and verify_order_owner(store, order_id, contact))
    handoff_id = f"H-{now:%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
    record = {
        "handoff_id": handoff_id,
        "created_at": now.isoformat(timespec="seconds"),
        "reason": reason,
        "summary": summary.strip()[:MAX_SUMMARY_CHARS],
        # Kept unmasked on purpose: the team needs it to call back. The file is
        # outside git; the traces, unlike this queue, are masked.
        "contact": contact,
        "order_id": order_id,
        "identity_verified": verified,
        "suggested_next_action": suggested_next_action,
        "tools_consulted": tools_consulted or [],
    }
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        return {"error": f"Could not register the handoff: {exc}", "retryable": True}

    status = get_store_status(now)
    return {
        "registered": True,
        "handoff_id": handoff_id,
        "reason": reason,
        "identity_verified": verified,
        "response_time": HANDOFF_RESPONSE_TIME,
        "team_available_now": status["is_open"],
        "team_back_at": None if status["is_open"] else status["next_opening"],
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
