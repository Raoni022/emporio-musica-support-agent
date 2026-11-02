from datetime import date

import pytest

from emporio_agent.data import load_store_data
from emporio_agent.policies import load_policy_manual
from scripts.data_quality import (
    check_description_contradictions,
    check_duplicate_emails,
    check_manual_contacts,
    check_manual_installments,
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


@pytest.fixture(scope="module")
def manual():
    return load_policy_manual()


def test_manual_contact_contradictions_are_found(manual):
    titles = [f.title for f in check_manual_contacts(manual).findings]
    assert any("2 different phone" in t for t in titles)
    assert any("e-mail 2 different ways" in t for t in titles)


def test_manual_installment_contradiction_is_found(manual):
    [finding] = check_manual_installments(manual).findings
    assert len(finding.evidence) == 4  # table + three tiers of 3.1


def test_report_has_no_raw_customer_emails(store, manual):
    reference_date = date(2026, 3, 25)
    report = render(run_checks(store, manual, reference_date), reference_date, manual.version)
    assert "@jmail.com" not in report.replace("***@jmail.com", "")


def test_mask_email():
    assert mask_email("lucas.mendes@jmail.com") == "l***@jmail.com"
