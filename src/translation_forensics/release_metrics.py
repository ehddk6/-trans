"""Evidence-bounded evaluation metrics and final-release gate checks.

This module intentionally does not infer correctness from a review priority,
an unreviewed audit row, or an empty gold layout.  It only reports numerical
recall/precision for a fully labelled evaluation universe whose positive
denominator has been supplied and reconciles with those labels.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


REVIEWER_LABELS = frozenset({"error", "no-error"})
DEFAULT_BUDGETS = (5, 10, 20)


def _not_demonstrated(*, reasons: list[str], record_count: int) -> dict[str, Any]:
    """Return the common shape without leaking partially supported metrics."""
    return {
        "schema_name": "translation-forensics/review-budget-metrics",
        "schema_version": "1",
        "metrics_status": "not-demonstrated",
        "record_count": record_count,
        "recall": None,
        "precision": None,
        "budget_metrics": {},
        "error_type_metrics": {},
        "reasons": reasons,
        "note": (
            "Recall and precision are withheld unless every record has a human "
            "reviewer label and the supplied gold positive denominator reconciles "
            "with the labelled evaluation universe."
        ),
    }


def _normalise_budgets(budgets: Sequence[int]) -> tuple[int, ...]:
    values = tuple(sorted(set(budgets)))
    if not values or any(not isinstance(value, int) or value <= 0 or value > 100 for value in values):
        raise ValueError("budgets must be unique integer percentages from 1 through 100")
    return values


def calculate_review_budget_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    gold_positive_count: int | None,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
) -> dict[str, Any]:
    """Calculate locked-set recall/precision at explicit review budgets.

    Required record fields are ``record_id``, a unique positive integer
    ``rank`` (1 is reviewed first), and ``reviewer_label`` of ``error`` or
    ``no-error``.  Positive records must also identify ``error_type`` and may
    identify ``severity`` as ``critical``.  ``gold_positive_count`` is not
    guessed from an audit sample: it is the adjudicated positive denominator
    for exactly this evaluation universe and must equal the labelled count.

    The function is deliberately all-or-nothing.  A partially reviewed P3/P4
    sample can still be described elsewhere, but it cannot yield a recall or
    precision claim here.
    """
    rows = [dict(record) for record in records]
    budget_values = _normalise_budgets(budgets)
    reasons: list[str] = []
    if not rows:
        reasons.append("evaluation universe contains no records")
    if not isinstance(gold_positive_count, int) or isinstance(gold_positive_count, bool) or gold_positive_count < 1:
        reasons.append("a positive adjudicated gold denominator is required")

    ids: set[str] = set()
    ranks: set[int] = set()
    for index, row in enumerate(rows, 1):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            reasons.append(f"record {index} has no stable record_id")
        elif record_id in ids:
            reasons.append(f"duplicate record_id: {record_id}")
        else:
            ids.add(record_id)

        rank = row.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            reasons.append(f"record {index} has no positive integer rank")
        elif rank in ranks:
            reasons.append(f"duplicate rank: {rank}")
        else:
            ranks.add(rank)

        if row.get("reviewer_label") not in REVIEWER_LABELS:
            reasons.append(f"record {index} lacks a completed reviewer_label")
        if row.get("reviewer_label") == "error" and not str(row.get("error_type", "")).strip():
            reasons.append(f"error record {index} lacks error_type")

    if ranks and ranks != set(range(1, len(rows) + 1)):
        reasons.append("ranks must cover the evaluation universe consecutively from 1")
    labelled_positive_count = sum(row.get("reviewer_label") == "error" for row in rows)
    if isinstance(gold_positive_count, int) and not isinstance(gold_positive_count, bool) and gold_positive_count >= 1:
        if labelled_positive_count != gold_positive_count:
            reasons.append(
                "gold_positive_count does not reconcile with completed reviewer labels "
                f"({gold_positive_count} supplied, {labelled_positive_count} labelled)"
            )
    if reasons:
        return _not_demonstrated(reasons=reasons, record_count=len(rows))

    ordered = sorted(rows, key=lambda row: int(row["rank"]))
    by_error_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        if row["reviewer_label"] == "error":
            by_error_type[str(row["error_type"]).strip()].append(row)

    budget_metrics: dict[str, dict[str, Any]] = {}
    for budget in budget_values:
        reviewed_count = min(len(ordered), math.ceil(len(ordered) * budget / 100))
        reviewed = ordered[:reviewed_count]
        true_positive_rows = [row for row in reviewed if row["reviewer_label"] == "error"]
        true_positive = len(true_positive_rows)
        false_positive = reviewed_count - true_positive
        critical_denominator = sum(
            row["reviewer_label"] == "error" and str(row.get("severity", "")).lower() == "critical"
            for row in ordered
        )
        critical_found = sum(str(row.get("severity", "")).lower() == "critical" for row in true_positive_rows)
        type_metrics = {
            error_type: {
                "gold_positive_count": len(error_rows),
                "found": sum(row in reviewed for row in error_rows),
                "recall": sum(row in reviewed for row in error_rows) / len(error_rows),
            }
            for error_type, error_rows in sorted(by_error_type.items())
        }
        budget_metrics[str(budget)] = {
            "review_budget_percent": budget,
            "reviewed_count": reviewed_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": gold_positive_count - true_positive,
            "gold_positive_count": gold_positive_count,
            "recall": true_positive / gold_positive_count,
            "precision": true_positive / reviewed_count,
            "critical_gold_positive_count": critical_denominator,
            "critical_found": critical_found,
            "critical_recall": critical_found / critical_denominator if critical_denominator else None,
            "error_type_recall": type_metrics,
        }

    return {
        "schema_name": "translation-forensics/review-budget-metrics",
        "schema_version": "1",
        "metrics_status": "demonstrated",
        "scope": "fully-labelled supplied evaluation universe only",
        "record_count": len(ordered),
        "gold_positive_count": gold_positive_count,
        "reviewer_labels_complete": True,
        "budget_metrics": budget_metrics,
        "error_type_metrics": {
            error_type: {"gold_positive_count": len(error_rows)}
            for error_type, error_rows in sorted(by_error_type.items())
        },
        "reasons": [],
    }


def validate_release_gate(
    evidence: Mapping[str, Mapping[str, Any] | None],
    *,
    require_human_approval: bool = True,
    require_sol_experiment: bool = False,
) -> dict[str, Any]:
    """Validate evidence required before a release can claim final status.

    Expected keys are ``gold_suite``, ``blind_review``, ``audit_summary``,
    ``review_metrics``, and, when human approval is required,
    ``release_approval``.  A Sol experiment is optional in the plan, so it is
    only a requirement when the caller explicitly elects to make it one.
    """
    errors: list[str] = []
    checks: dict[str, bool] = {}

    gold = evidence.get("gold_suite") or {}
    checks["adjudicated_gold"] = (
        gold.get("status") == "pass"
        and isinstance(gold.get("adjudicated_records"), int)
        and gold["adjudicated_records"] > 0
        and isinstance(gold.get("locked_test_records"), int)
        and gold["locked_test_records"] > 0
    )
    if not checks["adjudicated_gold"]:
        errors.append("release requires passing adjudicated gold evidence with at least one locked-test record")

    blind = evidence.get("blind_review") or {}
    checks["blind_human_review"] = (
        blind.get("evaluation_status") == "human-reviewed"
        and blind.get("review_complete") is True
        and isinstance(blind.get("reviewed_blocks"), int)
        and blind["reviewed_blocks"] > 0
    )
    if not checks["blind_human_review"]:
        errors.append("release requires a complete human blind-review summary")

    audit = evidence.get("audit_summary") or {}
    checks["p3_p4_audit"] = (
        audit.get("review_complete") is True
        and isinstance(audit.get("reviewed_blocks"), int)
        and audit["reviewed_blocks"] > 0
    )
    if not checks["p3_p4_audit"]:
        errors.append("release requires a completed P3/P4 audit with reviewer labels")

    metrics = evidence.get("review_metrics") or {}
    checks["measured_review_efficiency"] = metrics.get("metrics_status") == "demonstrated"
    if not checks["measured_review_efficiency"]:
        errors.append("release cannot claim measured recall/precision without demonstrated review metrics")

    approval = evidence.get("release_approval") or {}
    checks["human_release_approval"] = not require_human_approval or (
        approval.get("approved") is True and bool(str(approval.get("approved_by", "")).strip())
    )
    if not checks["human_release_approval"]:
        errors.append("release requires explicit accountable human approval")

    sol = evidence.get("sol_experiment") or {}
    checks["sol_experiment"] = not require_sol_experiment or sol.get("status") == "completed"
    if not checks["sol_experiment"]:
        errors.append("the configured release policy requires a completed Sol falsification experiment")

    return {
        "schema_name": "translation-forensics/release-gate-report",
        "schema_version": "1",
        "status": "pass" if not errors else "fail",
        "release_allowed": not errors,
        "checks": checks,
        "errors": errors,
        "note": "A pass validates supplied evidence contracts; it does not create human listening, gold labels, or approval evidence.",
    }
