"""Load the store's CSV files and normalize them for the tools.

The files in data/ are never modified. This module only normalizes formats
(types, casing, phone digits, parsed specs) and adds derived fields
(availability, active promotion, search text). Business facts such as prices,
order totals and statuses are kept exactly as recorded.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

TABLES = ("categories", "customers", "products", "orders", "order_items", "promotions")

# In these categories (Baixos, Baterias e Percussão, Teclados e Pianos) the
# description names a different brand or model than the product name, e.g.
# "Music Man Bass 1X" described as a Fender Jazz Bass. Descriptions there are
# never shown to the LLM; name, category, price and specs are the source of truth.
SUPPRESSED_DESCRIPTION_CATEGORIES = frozenset({2, 3, 4})

PRODUCT_STATUSES = frozenset({"active", "discontinued", "coming_soon"})

ORDER_STATUS_LABELS = {
    "pending": "aguardando confirmação",
    "confirmed": "confirmado, em preparação",
    "shipped": "enviado",
    "delivered": "entregue",
    "cancelled": "cancelado",
}

# Portuguese search terms for the normalized string material.
STRING_TYPE_TERMS = {"nylon": "nylon", "steel": "aco"}


class DataLoadError(RuntimeError):
    """Raised when the raw data cannot be loaded or has an unexpected shape."""


@dataclass(frozen=True)
class StoreData:
    categories: pd.DataFrame
    customers: pd.DataFrame
    products: pd.DataFrame
    orders: pd.DataFrame
    order_items: pd.DataFrame
    promotions: pd.DataFrame
    active_promotions: pd.DataFrame


# --- Small normalizers -------------------------------------------------------


def normalize_text(value: object) -> str:
    """Lowercase and strip accents, for accent-insensitive matching."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip().lower()


def phone_digits(value: object) -> str:
    return re.sub(r"\D", "", str(value))


def _parse_date(value: str) -> date | None:
    return date.fromisoformat(value) if value else None


def _object_column(values: list) -> pd.Series:
    # object dtype keeps None as None instead of turning it into NaN.
    return pd.Series(values, dtype=object)


# --- Raw access --------------------------------------------------------------


def find_table_file(table: str, data_dir: Path = DATA_DIR) -> Path:
    """Find a table by its suffix, e.g. 'desafio_tecnico_ai_eng - orders.csv'."""
    matches = sorted(Path(data_dir).glob(f"* - {table}.csv"))
    if len(matches) != 1:
        raise DataLoadError(
            f"Expected exactly one file matching '* - {table}.csv' in {data_dir}, "
            f"found {len(matches)}."
        )
    return matches[0]


def read_raw_table(table: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read a CSV as-is: every column as text, empty cells as ''."""
    return pd.read_csv(
        find_table_file(table, data_dir), dtype=str, keep_default_na=False, encoding="utf-8"
    )


# --- Specs -------------------------------------------------------------------


def parse_specs(raw: str) -> dict:
    """Parse the specs JSON and normalize its fields.

    The raw 'strings' field means two different things: the material in
    Violões and Ukuleles ("nylon"/"steel") and the count in Guitarras and
    Baixos ("6"/"4"). It is split into 'string_type' and 'string_count' and
    the ambiguous raw key is dropped.
    """
    try:
        specs = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"Invalid specs JSON: {raw!r}") from exc

    normalized: dict = {}
    for key, value in specs.items():
        value = str(value).strip()
        if key == "strings":
            if value.isdigit():
                normalized["string_count"] = int(value)
            elif value.lower() in STRING_TYPE_TERMS:
                normalized["string_type"] = value.lower()
            else:
                raise DataLoadError(f"Unknown 'strings' spec value: {value!r}")
        elif key == "electronics":
            if value.lower() not in ("yes", "no"):
                raise DataLoadError(f"Unknown 'electronics' spec value: {value!r}")
            normalized["electronics"] = value.lower() == "yes"
        elif value.isdigit():
            normalized[key] = int(value)
        else:
            normalized[key] = value
    return normalized


# --- Derived fields ----------------------------------------------------------


def availability(status: str, stock_quantity: int) -> str:
    """Status wins over stock: a coming_soon product is not sellable even with stock."""
    if status == "discontinued":
        return "discontinued"
    if status == "coming_soon":
        return "coming_soon"
    if stock_quantity <= 0:
        return "out_of_stock"
    return "available"


def pick_active_promotions(promotions: pd.DataFrame) -> pd.DataFrame:
    """One active promotion per product.

    Promotions are not cumulative (manual, section 6), so if a product ever
    has more than one active promotion only the largest discount applies.
    """
    active = promotions[promotions["is_active"]]
    active = active.sort_values(["discount_percent", "promotion_id"], ascending=[False, True])
    return active.drop_duplicates("product_id", keep="first").reset_index(drop=True)


def money(value: float | Decimal) -> float:
    """Round to cents, half up (float round() is half-even and binary-inexact)."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def apply_discount(price: float, discount_percent: int) -> float:
    return money(Decimal(str(price)) * (100 - discount_percent) / 100)


def promo_price(price: float, discount_percent: int) -> float:
    return apply_discount(price, discount_percent)


# --- Table loaders -----------------------------------------------------------


def _load_categories(data_dir: Path) -> pd.DataFrame:
    df = read_raw_table("categories", data_dir)
    df["category_id"] = df["category_id"].astype(int)
    return df


def _load_customers(data_dir: Path) -> pd.DataFrame:
    df = read_raw_table("customers", data_dir)
    df["customer_id"] = df["customer_id"].astype(int)
    df["email"] = df["email"].str.strip().str.lower()
    df["phone_digits"] = df["phone"].map(phone_digits)
    df["first_name"] = df["name"].str.split().str[0]
    return df


def _load_promotions(data_dir: Path) -> pd.DataFrame:
    df = read_raw_table("promotions", data_dir)
    for col in ("promotion_id", "product_id", "discount_percent"):
        df[col] = df[col].astype(int)
    df["is_active"] = df["is_active"].map({"1": True, "0": False})
    if df["is_active"].isna().any():
        raise DataLoadError("promotions.is_active must be 0 or 1.")
    df["is_active"] = df["is_active"].astype(bool)
    return df


def _load_products(
    data_dir: Path, categories: pd.DataFrame, active_promotions: pd.DataFrame
) -> pd.DataFrame:
    df = read_raw_table("products", data_dir)
    df["product_id"] = df["product_id"].astype(int)
    df["category_id"] = df["category_id"].astype(int)
    df["price_brl"] = df["price_brl"].astype(float)
    df["stock_quantity"] = df["stock_quantity"].astype(int)
    df["created_at"] = _object_column([_parse_date(v) for v in df["created_at"]])

    unknown = set(df["status"]) - PRODUCT_STATUSES
    if unknown:
        raise DataLoadError(f"Unknown product status: {sorted(unknown)}")

    category_names = dict(zip(categories["category_id"], categories["name"]))
    df["category_name"] = df["category_id"].map(category_names)

    specs = [parse_specs(raw) for raw in df["specs"]]
    df["specs"] = _object_column(specs)
    df["string_type"] = _object_column([s.get("string_type") for s in specs])
    df["string_count"] = _object_column([s.get("string_count") for s in specs])
    df["electronics"] = _object_column([s.get("electronics") for s in specs])

    df["description"] = _object_column(
        [
            None if cat in SUPPRESSED_DESCRIPTION_CATEGORIES else desc
            for cat, desc in zip(df["category_id"], df["description"])
        ]
    )
    df["availability"] = [
        availability(status, stock) for status, stock in zip(df["status"], df["stock_quantity"])
    ]

    search_text = []
    for name, category, spec in zip(df["name"], df["category_name"], specs):
        parts = [name, category or ""]
        parts += [str(v) for k, v in spec.items() if k not in ("electronics", "string_type")]
        if spec.get("string_type"):
            parts.append(STRING_TYPE_TERMS[spec["string_type"]])
        search_text.append(normalize_text(" ".join(parts)))
    df["search_text"] = search_text

    promos = active_promotions.set_index("product_id")
    promo_rows = [promos.loc[pid] if pid in promos.index else None for pid in df["product_id"]]
    df["promotion_id"] = _object_column(
        [None if p is None else int(p["promotion_id"]) for p in promo_rows]
    )
    df["discount_percent"] = _object_column(
        [None if p is None else int(p["discount_percent"]) for p in promo_rows]
    )
    df["promotion_description"] = _object_column(
        [None if p is None else p["description"] for p in promo_rows]
    )
    df["promo_price"] = _object_column(
        [
            None if p is None else promo_price(price, int(p["discount_percent"]))
            for p, price in zip(promo_rows, df["price_brl"])
        ]
    )
    return df


def _load_orders(data_dir: Path) -> pd.DataFrame:
    df = read_raw_table("orders", data_dir)
    df["order_id"] = df["order_id"].astype(int)
    df["customer_id"] = df["customer_id"].astype(int)
    df["total_brl"] = df["total_brl"].astype(float)

    unknown = set(df["status"]) - set(ORDER_STATUS_LABELS)
    if unknown:
        raise DataLoadError(f"Unknown order status: {sorted(unknown)}")
    df["status_label"] = df["status"].map(ORDER_STATUS_LABELS)

    df["order_date"] = _object_column([_parse_date(v) for v in df["order_date"]])
    df["estimated_delivery"] = _object_column(
        [_parse_date(v) for v in df["estimated_delivery"]]
    )
    for col in ("tracking_code", "notes"):
        df[col] = _object_column([v or None for v in df[col]])
    return df


def _load_order_items(data_dir: Path) -> pd.DataFrame:
    df = read_raw_table("order_items", data_dir)
    for col in ("order_id", "quantity", "product_id"):
        df[col] = df[col].astype(int)
    return df


def load_store_data(data_dir: Path | str = DATA_DIR) -> StoreData:
    """Load and normalize every table. Fails loudly on unexpected data."""
    data_dir = Path(data_dir)
    categories = _load_categories(data_dir)
    promotions = _load_promotions(data_dir)
    active_promotions = pick_active_promotions(promotions)
    return StoreData(
        categories=categories,
        customers=_load_customers(data_dir),
        products=_load_products(data_dir, categories, active_promotions),
        orders=_load_orders(data_dir),
        order_items=_load_order_items(data_dir),
        promotions=promotions,
        active_promotions=active_promotions,
    )
