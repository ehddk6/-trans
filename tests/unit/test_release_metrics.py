from __future__ import annotations

from translation_forensics.release_metrics import (
    calculate_review_budget_metrics,
    validate_release_gate,
)


def _records() -> list[dict[str, object]]:
    return [
        {"record_id": "r-1", "rank": 1, "reviewer_label": "error", "error_type": "omission", "severity": "critical"},
        {"record_id": "r-2", "rank": 2, "reviewer_label": "no-error"},
        {"record_id": "r-3", "rank": 3, "reviewer_label": "error", "error_type": "register", "severity": "major"},
        {"record_id": "r-4", "rank": 4, "reviewer_label": "no-error"},
        {"record_id": "r-5", "rank": 5, "reviewer_label": "no-error"},
        {"record_id": "r-6", "rank": 6, "reviewer_label": "no-error"},
        {"record_id": "r-7", "rank": 7, "reviewer_label": "no-error"},
        {"record_id": "r-8", "rank": 8, "reviewer_label": "no-error"},
        {"record_id": "r-9", "rank": 9, "reviewer_label": "no-error"},
        {"record_id": "r-10", "rank": 10, "reviewer_label": "no-error"},
    ]


def test_metrics_are_withheld_without_completed_reviewer_labels() -> None:
    records = _records()
    records[-1]["reviewer_label"] = "unreviewed"

    report = calculate_review_budget_metrics(records, gold_positive_count=2)

    assert report["metrics_status"] == "not-demonstrated"
    assert report["recall"] is None
    assert report["precision"] is None
    assert report["budget_metrics"] == {}


def test_metrics_are_withheld_without_reconciled_gold_denominator() -> None:
    report = calculate_review_budget_metrics(_records(), gold_positive_count=None)
    assert report["metrics_status"] == "not-demonstrated"

    mismatched = calculate_review_budget_metrics(_records(), gold_positive_count=3)
    assert mismatched["metrics_status"] == "not-demonstrated"
    assert any("does not reconcile" in reason for reason in mismatched["reasons"])


def test_budget_metrics_use_locked_labels_and_explicit_denominator() -> None:
    report = calculate_review_budget_metrics(_records(), gold_positive_count=2, budgets=(10, 20, 100))

    assert report["metrics_status"] == "demonstrated"
    assert report["budget_metrics"]["10"]["recall"] == 0.5
    assert report["budget_metrics"]["10"]["precision"] == 1.0
    assert report["budget_metrics"]["20"]["precision"] == 0.5
    assert report["budget_metrics"]["100"]["recall"] == 1.0
    assert report["budget_metrics"]["100"]["error_type_recall"]["omission"]["recall"] == 1.0


def test_release_gate_refuses_not_demonstrated_metrics_and_missing_human_evidence() -> None:
    report = validate_release_gate({
        "gold_suite": {"status": "pass", "adjudicated_records": 1, "locked_test_records": 1},
        "blind_review": {"evaluation_status": "human-reviewed", "review_complete": True, "reviewed_blocks": 2},
        "audit_summary": {"review_complete": True, "reviewed_blocks": 2},
        "review_metrics": {"metrics_status": "not-demonstrated"},
        "release_approval": {"approved": False},
    })

    assert report["release_allowed"] is False
    assert report["checks"]["measured_review_efficiency"] is False
    assert report["checks"]["human_release_approval"] is False


def test_release_gate_passes_only_complete_evidence_contract() -> None:
    report = validate_release_gate({
        "gold_suite": {"status": "pass", "adjudicated_records": 2, "locked_test_records": 1},
        "blind_review": {"evaluation_status": "human-reviewed", "review_complete": True, "reviewed_blocks": 10},
        "audit_summary": {"review_complete": True, "reviewed_blocks": 5},
        "review_metrics": {"metrics_status": "demonstrated"},
        "release_approval": {"approved": True, "approved_by": "reviewer-1"},
    })

    assert report["status"] == "pass"
    assert report["release_allowed"] is True
