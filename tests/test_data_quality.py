from datetime import date

import pytest

from emporio_agent.data import load_store_data
from scripts.data_quality import (
    check_description_contradictions,
    check_duplicate_emails,
    check_order_totals,
    check_stale_orders,
    mask_email,
    render,
    run_checks,
)


@pytest.fixture(scope="module")
def store():
    return load_store_data()


def test_duplicate_email_is_reported_masked(store):
    [finding] = check_duplicate_emails(store).findings
    assert "b***@jmail.com" in finding.title
    assert "bruno.martins" not in finding.title


def test_order_total_mismatches_are_found(store):
    [finding] = check_order_totals(store).findings
    assert any(e.startswith("order 3 ") for e in finding.evidence)
    assert any(e.startswith("order 20 ") for e in finding.evidence)
    assert len(finding.evidence) == 2


def test_description_contradictions_stay_in_suppressed_categories(store):
    findings = check_description_contradictions(store).findings
    assert len(findings) == 1  # no second finding about uncovered categories
    evidence = " ".join(findings[0].evidence)
    assert "4 strings, description: 5" in evidence
    assert "61 keys, description: 88" in evidence


def test_stale_orders_depend_on_reference_date(store):
    [finding] = check_stale_orders(store, date(2026, 3, 25)).findings
    assert any("order 8: shipped" in e for e in finding.evidence)
    assert check_stale_orders(store, date(2026, 2, 21)).findings == []


def test_report_has_no_raw_emails(store):
    report = render(run_checks(store, date(2026, 3, 25)), date(2026, 3, 25))
    assert "@jmail.com" not in report.replace("***@jmail.com", "")


def test_mask_email():
    assert mask_email("lucas.mendes@jmail.com") == "l***@jmail.com"
