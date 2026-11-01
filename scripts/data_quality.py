"""Generate docs/data_quality.md: a diagnosis of the store's data, written for the store.

Every check reads the raw files through emporio_agent.data and nothing is
fixed here. Each finding says what was found, how the agent handles it and
what the store should correct at the source.

Usage: python scripts/data_quality.py [--reference-date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emporio_agent.data import (  # noqa: E402
    SUPPRESSED_DESCRIPTION_CATEGORIES,
    StoreData,
    load_store_data,
    normalize_text,
    read_raw_table,
)

OUTPUT = ROOT / "docs" / "data_quality.md"
DEFAULT_REFERENCE_DATE = date(2026, 3, 25)
STALE_ORDER_DAYS = 7

KNOWN_BRANDS = (
    "Crafter", "Fender", "Giannini", "Gibson", "Ibanez", "Kala", "Kalani", "Korg",
    "Martin", "Music Man", "Nord", "Ohana", "Pearl", "PRS", "Roland", "Rozini",
    "Shelby", "Tagima", "Takamine", "Tama", "Taylor", "Yamaha",
)
# Model names that identify a brand even when the brand itself is not written.
MODEL_BRANDS = {
    "Jazz Bass": "Fender", "Precision": "Fender", "Stratocaster": "Fender",
    "Telecaster": "Fender", "Les Paul": "Gibson", "StingRay": "Music Man",
    "Stage Custom": "Yamaha",
}
SHELL_WOODS = ("birch", "maple", "mogno", "mahogany", "poplar")


@dataclass
class Finding:
    title: str
    evidence: list[str]
    agent_handling: str
    recommendation: str
    severity: str = "medium"  # high: can lead to a wrong answer; medium/low otherwise


@dataclass
class CheckResult:
    name: str
    findings: list[Finding] = field(default_factory=list)


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[:1]}***@{domain}"


def _brands_in(text: str) -> set[str]:
    norm = normalize_text(text)
    brands = {b for b in KNOWN_BRANDS if re.search(rf"\b{normalize_text(b)}\b", norm)}
    brands |= {
        brand for model, brand in MODEL_BRANDS.items()
        if re.search(rf"\b{normalize_text(model)}\b", norm)
    }
    return brands


def _brl(value: float) -> str:
    text = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {text}"


# --- Customers ---------------------------------------------------------------


def check_duplicate_emails(store: StoreData) -> CheckResult:
    result = CheckResult("Customers sharing an e-mail")
    customers = store.customers
    for email, group in customers.groupby("email"):
        if len(group) < 2:
            continue
        rows = [
            f"customer {r.customer_id}: {r['name']}, {r.city}" for _, r in group.iterrows()
        ]
        result.findings.append(
            Finding(
                title=f"{len(group)} customer records share {mask_email(email)}",
                evidence=rows,
                agent_handling=(
                    "Order verification still works: the owner of the e-mail is the same "
                    "person, so either record's orders are shown to whoever proves that e-mail."
                ),
                recommendation="Probable duplicate registration. Merge the records.",
                severity="low",
            )
        )
    return result


# --- Products ----------------------------------------------------------------


def check_strings_meaning(store: StoreData) -> CheckResult:
    result = CheckResult("Ambiguous 'strings' spec")
    raw = read_raw_table("products")
    by_category: dict[str, set[str]] = {}
    categories = dict(zip(store.categories["category_id"].astype(str), store.categories["name"]))
    for _, row in raw.iterrows():
        value = json.loads(row["specs"]).get("strings")
        if value is not None:
            by_category.setdefault(categories[row["category_id"]], set()).add(value)
    if any(v.isdigit() for vals in by_category.values() for v in vals) and any(
        not v.isdigit() for vals in by_category.values() for v in vals
    ):
        result.findings.append(
            Finding(
                title="The 'strings' field holds a material in some categories and a count in others",
                evidence=[
                    f"{cat}: {', '.join(sorted(vals))}" for cat, vals in sorted(by_category.items())
                ],
                agent_handling=(
                    "Split at load time into 'string_type' (nylon/steel) and 'string_count'; "
                    "the raw field is never shown."
                ),
                recommendation="Use two separate fields in the catalog.",
            )
        )
    return result


def check_description_contradictions(store: StoreData) -> CheckResult:
    result = CheckResult("Descriptions contradicting name or specs")
    raw = read_raw_table("products").set_index("product_id")
    evidence: list[str] = []
    affected_categories: set[int] = set()
    for _, product in store.products.iterrows():
        desc = raw.loc[str(product.product_id), "description"]
        name = product["name"]
        problems = []

        other_brands = _brands_in(desc) - _brands_in(name)
        if other_brands:
            problems.append(f"description names {', '.join(sorted(other_brands))}")

        desc_norm = normalize_text(desc)
        strings = re.search(r"(\d+)\s+cordas", desc_norm)
        if strings and product.string_count is not None:
            if int(strings.group(1)) != product.string_count:
                problems.append(
                    f"specs: {product.string_count} strings, description: {strings.group(1)}"
                )
        keys = re.search(r"(\d+)\s+teclas", desc_norm)
        if keys and "keys" in product.specs and int(keys.group(1)) != product.specs["keys"]:
            problems.append(f"specs: {product.specs['keys']} keys, description: {keys.group(1)}")
        shells = product.specs.get("shells")
        if shells:
            mentioned = {w for w in SHELL_WOODS if re.search(rf"\b{w}\b", desc_norm)}
            if mentioned and normalize_text(shells) not in mentioned:
                problems.append(
                    f"specs shells: {shells}, description: {', '.join(sorted(mentioned))}"
                )

        if problems:
            affected_categories.add(int(product.category_id))
            evidence.append(f"{product.product_id} {name}: {'; '.join(problems)}")

    if evidence:
        suppressed = ", ".join(
            store.categories.set_index("category_id").loc[sorted(SUPPRESSED_DESCRIPTION_CATEGORIES), "name"]
        )
        result.findings.append(
            Finding(
                title=f"{len(evidence)} product descriptions contradict the product name or specs",
                evidence=evidence,
                agent_handling=(
                    "Name, category, price and specs are treated as the source of truth. "
                    f"Descriptions are never sent to the LLM in: {suppressed}."
                ),
                recommendation=(
                    "Rewrite these descriptions; they appear to be copied from other products."
                ),
                severity="high",
            )
        )
        uncovered = affected_categories - SUPPRESSED_DESCRIPTION_CATEGORIES
        if uncovered:
            result.findings.append(
                Finding(
                    title="Contradicting descriptions outside the suppressed categories",
                    evidence=[f"category {c}" for c in sorted(uncovered)],
                    agent_handling="Not handled yet: these descriptions still reach the LLM.",
                    recommendation="Review and decide whether to suppress these categories too.",
                    severity="high",
                )
            )
    return result


def check_empty_categories(store: StoreData) -> CheckResult:
    result = CheckResult("Categories without products")
    used = set(store.products["category_id"])
    empty = store.categories[~store.categories["category_id"].isin(used)]
    if len(empty):
        result.findings.append(
            Finding(
                title=f"{len(empty)} categories have no products",
                evidence=[f"{r.category_id} {r['name']}" for _, r in empty.iterrows()],
                agent_handling=(
                    "The agent says there are no items in these categories right now, "
                    "without inventing products."
                ),
                recommendation="Add products or hide these categories.",
                severity="low",
            )
        )
    return result


def check_id_order(store: StoreData) -> CheckResult:
    result = CheckResult("Product id order")
    ids = list(store.products["product_id"])
    out_of_order = [(a, b) for a, b in zip(ids, ids[1:]) if b < a]
    if out_of_order:
        result.findings.append(
            Finding(
                title="Product ids are not in ascending order in the file",
                evidence=[f"id {b} comes right after id {a}" for a, b in out_of_order],
                agent_handling="Nothing depends on id order.",
                recommendation="None needed.",
                severity="low",
            )
        )
    return result


# --- Promotions --------------------------------------------------------------


def check_duplicate_promotions(store: StoreData) -> CheckResult:
    result = CheckResult("Several promotions for one product")
    promos = store.promotions
    counts = promos.groupby("product_id").size()
    duplicated = counts[counts > 1].index
    if len(duplicated):
        evidence = []
        for pid in duplicated:
            rows = promos[promos["product_id"] == pid]
            parts = [
                f"#{r.promotion_id} {r.discount_percent}% {r.description} "
                f"({'active' if r.is_active else 'inactive'})"
                for _, r in rows.iterrows()
            ]
            evidence.append(f"product {pid}: " + "; ".join(parts))
        active_counts = promos[promos["is_active"]].groupby("product_id").size()
        n_active_dupes = int((active_counts > 1).sum())
        result.findings.append(
            Finding(
                title=f"{len(duplicated)} products have more than one promotion "
                      f"({n_active_dupes} with more than one active)",
                evidence=evidence,
                agent_handling=(
                    "If more than one is active, only the largest discount applies "
                    "(promotions are not cumulative)."
                ),
                recommendation="Give promotions start and end dates instead of only an is_active flag.",
                severity="low" if n_active_dupes == 0 else "high",
            )
        )
    return result


def check_promotions_on_unavailable(store: StoreData) -> CheckResult:
    result = CheckResult("Active promotions on unavailable products")
    unavailable = store.products[
        store.products["promotion_id"].notna() & (store.products["availability"] != "available")
    ]
    if len(unavailable):
        result.findings.append(
            Finding(
                title=f"{len(unavailable)} active promotions on products that cannot be bought",
                evidence=[f"{r.product_id} {r['name']}: {r.availability}"
                          for _, r in unavailable.iterrows()],
                agent_handling="Availability is always shown; there is no rain check (manual 6.2).",
                recommendation="Deactivate these promotions.",
            )
        )
    return result


# --- Orders ------------------------------------------------------------------


def check_order_totals(store: StoreData) -> CheckResult:
    result = CheckResult("Order totals vs. items x current price")
    prices = store.products.set_index("product_id")["price_brl"]
    evidence = []
    for _, order in store.orders.iterrows():
        items = store.order_items[store.order_items["order_id"] == order.order_id]
        if items.empty or not items["product_id"].isin(prices.index).all():
            continue
        expected = round(sum(prices[r.product_id] * r.quantity for _, r in items.iterrows()), 2)
        if abs(expected - order.total_brl) > 0.01:
            evidence.append(
                f"order {order.order_id} ({order.status}): recorded {_brl(order.total_brl)}, "
                f"items x current price {_brl(expected)}"
            )
    if evidence:
        result.findings.append(
            Finding(
                title=f"{len(evidence)} order totals differ from items x current price",
                evidence=evidence,
                agent_handling=(
                    "The recorded total is shown, since it reflects what was charged. "
                    "It is never recalculated."
                ),
                recommendation=(
                    "Store the unit price paid and any discount per order item, so totals can "
                    "be audited."
                ),
            )
        )
    return result


def check_orders_with_unsellable_products(store: StoreData) -> CheckResult:
    result = CheckResult("Orders containing products not on sale")
    products = store.products.set_index("product_id")
    evidence = []
    for _, item in store.order_items.iterrows():
        if item.product_id not in products.index:
            continue
        product = products.loc[item.product_id]
        if product["status"] != "active":
            order = store.orders.set_index("order_id").loc[item.order_id]
            evidence.append(
                f"order {item.order_id} ({order['status']}) contains product "
                f"{item.product_id} {product['name']}, now '{product['status']}'"
            )
    if evidence:
        result.findings.append(
            Finding(
                title=f"{len(evidence)} order items reference products that are not active",
                evidence=evidence,
                agent_handling=(
                    "Order status is reported as recorded. A coming_soon product in an order "
                    "may be a pre-sale; the agent does not speculate."
                ),
                recommendation="Confirm whether pre-sales exist and flag them explicitly.",
            )
        )
    return result


def check_stale_orders(store: StoreData, reference_date: date) -> CheckResult:
    result = CheckResult(f"Late or stale orders (reference date {reference_date.isoformat()})")
    evidence = []
    for _, order in store.orders.iterrows():
        if order.status in ("pending", "confirmed"):
            age = (reference_date - order.order_date).days
            if age > STALE_ORDER_DAYS:
                evidence.append(
                    f"order {order.order_id}: '{order.status}' since "
                    f"{order.order_date.isoformat()} ({age} days)"
                )
        elif order.status == "shipped" and order.estimated_delivery:
            late = (reference_date - order.estimated_delivery).days
            if late > 0:
                evidence.append(
                    f"order {order.order_id}: shipped, estimated delivery "
                    f"{order.estimated_delivery.isoformat()} ({late} days late)"
                )
    if evidence:
        result.findings.append(
            Finding(
                title=f"{len(evidence)} orders look late (open for more than "
                      f"{STALE_ORDER_DAYS} days, or past the estimated delivery)",
                evidence=evidence,
                agent_handling=(
                    "The agent reports the system status and, if the order is clearly late, "
                    "offers to hand it to the team."
                ),
                recommendation="Review these orders; update their status or contact the customers.",
            )
        )
    return result


def check_order_field_consistency(store: StoreData) -> CheckResult:
    result = CheckResult("Tracking and delivery fields vs. order status")
    evidence = []
    for _, order in store.orders.iterrows():
        shipped = order.status in ("shipped", "delivered")
        if shipped and (not order.tracking_code or not order.estimated_delivery):
            evidence.append(f"order {order.order_id}: {order.status} without tracking or estimate")
        if not shipped and order.tracking_code:
            evidence.append(f"order {order.order_id}: {order.status} with a tracking code")
        if order.status == "cancelled" and not order.notes:
            evidence.append(f"order {order.order_id}: cancelled without a reason")
    if evidence:
        result.findings.append(
            Finding(
                title="Order fields inconsistent with status",
                evidence=evidence,
                agent_handling="Missing fields are reported as unavailable, never guessed.",
                recommendation="Fill in the missing fields.",
            )
        )
    return result


def check_references(store: StoreData) -> CheckResult:
    result = CheckResult("Referential integrity")
    evidence = []
    customers = set(store.customers["customer_id"])
    products = set(store.products["product_id"])
    orders = set(store.orders["order_id"])
    categories = set(store.categories["category_id"])
    evidence += [f"order {o} -> missing customer {c}"
                 for o, c in zip(store.orders["order_id"], store.orders["customer_id"])
                 if c not in customers]
    evidence += [f"order item -> missing order {o}" for o in store.order_items["order_id"]
                 if o not in orders]
    evidence += [f"order item -> missing product {p}" for p in store.order_items["product_id"]
                 if p not in products]
    evidence += [f"promotion -> missing product {p}" for p in store.promotions["product_id"]
                 if p not in products]
    evidence += [f"product -> missing category {c}" for c in store.products["category_id"]
                 if c not in categories]
    evidence += [f"order {o} has no items" for o in orders
                 if o not in set(store.order_items["order_id"])]
    if evidence:
        result.findings.append(
            Finding(
                title="Broken references between tables",
                evidence=evidence,
                agent_handling="Tools return a structured error instead of failing.",
                recommendation="Fix the references at the source.",
                severity="high",
            )
        )
    return result


# --- Report ------------------------------------------------------------------


def run_checks(store: StoreData, reference_date: date) -> list[CheckResult]:
    return [
        check_description_contradictions(store),
        check_strings_meaning(store),
        check_order_totals(store),
        check_orders_with_unsellable_products(store),
        check_stale_orders(store, reference_date),
        check_order_field_consistency(store),
        check_duplicate_promotions(store),
        check_promotions_on_unavailable(store),
        check_duplicate_emails(store),
        check_empty_categories(store),
        check_id_order(store),
        check_references(store),
    ]


def render(results: list[CheckResult], reference_date: date) -> str:
    findings = [f for r in results for f in r.findings]
    lines = [
        "# Data quality report — Empório da Música",
        "",
        "Generated by `python scripts/data_quality.py`. Do not edit by hand.",
        "",
        f"Reference date: {reference_date.isoformat()}. "
        "The raw files in `data/` are never modified; this report lists what was found, "
        "how the agent handles each issue and what the store should fix at the source. "
        "E-mails are masked.",
        "",
        "## Summary",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for r in results:
        status = (
            "no issues" if not r.findings
            else "; ".join(f"**{f.severity}**: {f.title}" for f in r.findings)
        )
        lines.append(f"| {r.name} | {status} |")
    lines += ["", f"{len(findings)} findings in {len(results)} checks.", "", "## Findings", ""]
    for r in results:
        for f in r.findings:
            lines += [f"### {f.title}", "", f"*Check: {r.name}. Severity: {f.severity}.*", ""]
            lines += [f"- {e}" for e in f.evidence]
            lines += [
                "",
                f"**How the agent handles it:** {f.agent_handling}",
                "",
                f"**Recommendation to the store:** {f.recommendation}",
                "",
            ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-date", type=date.fromisoformat, default=DEFAULT_REFERENCE_DATE)
    args = parser.parse_args()
    store = load_store_data()
    results = run_checks(store, args.reference_date)
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(render(results, args.reference_date), encoding="utf-8")
    n = sum(len(r.findings) for r in results)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({n} findings).")


if __name__ == "__main__":
    main()
