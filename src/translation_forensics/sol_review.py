"""Guarded record format for optional Sol counterexample reviews.

Sol is deliberately not a translation generator or final arbiter in this
workflow.  This module only creates and validates audit records for an
optional, separately approved counterexample review.  It does not call a
provider, send content outside the project, or change translation decisions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_NAME = "translation-forensics/sol-counterexample-review"
SCHEMA_VERSION = "1"
REVIEW_STATUSES = {
    "not-requested",
    "approved-for-model-review",
    "model-reviewed",
    "completed",
    "cancelled",
}
HUMAN_VERDICTS = {
    "accept-terra",
    "accept-sol-counterexample",
    "hold",
    "reject-sol-counterexample",
    "needs-human-listening",
    "unresolved",
}
MODEL_REVIEW_STATUSES = {"model-reviewed", "completed"}
FORBIDDEN_FINAL_FIELDS = {
    "final_decision",
    "automatic_final_decision",
    "automatic_final_verdict",
    "model_final_decision",
    "final_promotion_allowed",
}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: Sol review record must be a JSON object")
        rows.append(value)
    return rows


def initialize_sol_review_records(
    queue_path: Path,
    output_path: Path,
    *,
    title_id: str,
) -> dict[str, Any]:
    """Make a no-transmission Sol-review template from selected queue rows.

    Queue rows must identify a positive integer ``block_number``.  A template
    starts as ``not-requested``; creating it is neither provider approval nor
    a model invocation.
    """
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace existing Sol review records: {output_path}")
    if not title_id.strip():
        raise ValueError("title_id is required")

    queue = _read_jsonl(queue_path)
    seen: set[int] = set()
    rows: list[dict[str, Any]] = []
    for source in queue:
        block_number = source.get("block_number")
        if not isinstance(block_number, int) or block_number < 1:
            raise ValueError("every queue row needs a positive integer block_number")
        if block_number in seen:
            raise ValueError(f"duplicate block_number in Sol review queue: {block_number}")
        seen.add(block_number)
        rows.append({
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "review_id": f"SOL-{title_id}-{block_number:04d}",
            "title_id": title_id,
            "block_number": block_number,
            "review_scope": "counterexample-only",
            "eligible_reason": "",
            "terra_decision_ref": "",
            "semantic_conflict_refs": [],
            "reviewer_model": "gpt-5.6-sol",
            "review_status": "not-requested",
            "provider": "",
            "provider_request_id": "",
            "external_transfer_allowed": False,
            "external_transfer_approval_id": "",
            "external_transfer_approved_by": "",
            "external_transfer_approved_at": "",
            "cost_currency": "USD",
            "cost_budget_limit": None,
            "estimated_cost": None,
            "actual_cost": None,
            "cost_approval_id": "",
            "cost_approved_by": "",
            "cost_approved_at": "",
            "reviewer_model_output": "",
            "reviewer_model_recommendation": "",
            "review_reason": "",
            "disagreement_type": "",
            "model_completed_at": "",
            "human_verdict": "",
            "human_reviewer": "",
            "human_reviewed_at": "",
            "human_rationale": "",
            "human_evidence_refs": [],
            "human_direct_listening": False,
            "final_decision_authority": "human-with-direct-evidence",
            "note": "Sol may propose a counterexample only. It cannot make or promote a final translation decision.",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    _write_jsonl(output_path, rows)
    return {
        "status": "sol-review-template-created",
        "output": str(output_path),
        "title_id": title_id,
        "records": len(rows),
        "provider_called": False,
        "review_status": "not-requested",
    }


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: object, *, positive: bool = False) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and (value > 0 if positive else value >= 0)


def _append_missing(errors: list[str], row_label: str, row: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        if not _nonempty(row.get(field)):
            errors.append(f"{row_label}: {field} is required")


def _validate_model_metadata(errors: list[str], row_label: str, row: dict[str, Any]) -> None:
    _append_missing(errors, row_label, row, (
        "provider", "provider_request_id", "external_transfer_approval_id",
        "external_transfer_approved_by", "external_transfer_approved_at",
        "cost_approval_id", "cost_approved_by", "cost_approved_at",
        "model_completed_at", "reviewer_model_output", "review_reason",
    ))
    if row.get("external_transfer_allowed") is not True:
        errors.append(f"{row_label}: a model-reviewed record requires external_transfer_allowed=true")
    if not _nonempty(row.get("cost_currency")):
        errors.append(f"{row_label}: cost_currency is required")
    if not _number(row.get("cost_budget_limit"), positive=True):
        errors.append(f"{row_label}: cost_budget_limit must be a positive number")
    if not _number(row.get("actual_cost")):
        errors.append(f"{row_label}: actual_cost must be a non-negative number")
    budget, actual = row.get("cost_budget_limit"), row.get("actual_cost")
    if _number(budget, positive=True) and _number(actual) and actual > budget:
        errors.append(f"{row_label}: actual_cost exceeds approved cost_budget_limit")


def validate_sol_review_records(path: Path) -> dict[str, Any]:
    """Validate that Sol evidence remains optional and human-controlled."""
    try:
        rows = _read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "fail", "records": 0, "completed_records": 0, "errors": [str(exc)]}

    errors: list[str] = []
    review_ids: set[str] = set()
    seen_blocks: set[tuple[str, int]] = set()
    completed = 0
    model_reviewed = 0
    for index, row in enumerate(rows, 1):
        label = f"row {index}"
        if row.get("schema_name") != SCHEMA_NAME or row.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{label}: unsupported Sol review schema")
        _append_missing(errors, label, row, ("review_id", "title_id", "review_scope", "reviewer_model", "final_decision_authority"))
        review_id = row.get("review_id")
        if _nonempty(review_id):
            if review_id in review_ids:
                errors.append(f"{label}: duplicate review_id {review_id}")
            review_ids.add(review_id)
        block_number = row.get("block_number")
        if not isinstance(block_number, int) or block_number < 1:
            errors.append(f"{label}: block_number must be a positive integer")
        elif _nonempty(row.get("title_id")):
            key = (str(row["title_id"]), block_number)
            if key in seen_blocks:
                errors.append(f"{label}: duplicate title_id/block_number")
            seen_blocks.add(key)
        if row.get("review_scope") != "counterexample-only":
            errors.append(f"{label}: Sol review_scope must be counterexample-only")
        if row.get("reviewer_model") != "gpt-5.6-sol":
            errors.append(f"{label}: reviewer_model must be gpt-5.6-sol")
        if row.get("final_decision_authority") != "human-with-direct-evidence":
            errors.append(f"{label}: final decision authority must remain human-with-direct-evidence")
        for field in FORBIDDEN_FINAL_FIELDS:
            if field in row:
                errors.append(f"{label}: {field} is forbidden; Sol cannot make an automatic final decision")

        status = row.get("review_status")
        if status not in REVIEW_STATUSES:
            errors.append(f"{label}: invalid review_status")
            continue
        if status in MODEL_REVIEW_STATUSES:
            model_reviewed += 1
            _validate_model_metadata(errors, label, row)
        if status == "completed":
            completed += 1
            _append_missing(errors, label, row, (
                "human_verdict", "human_reviewer", "human_reviewed_at", "human_rationale",
            ))
            if row.get("human_verdict") not in HUMAN_VERDICTS:
                errors.append(f"{label}: invalid human_verdict")
            evidence_refs = row.get("human_evidence_refs")
            if not isinstance(evidence_refs, list) or not evidence_refs or not all(_nonempty(item) for item in evidence_refs):
                errors.append(f"{label}: completed records require non-empty human_evidence_refs")
            if row.get("human_verdict") == "accept-sol-counterexample" and row.get("human_direct_listening") is not True:
                errors.append(f"{label}: accepting a Sol counterexample requires human_direct_listening=true")

    return {
        "status": "pass" if not errors else "fail",
        "sol_review_records": str(path),
        "records": len(rows),
        "model_reviewed_records": model_reviewed,
        "completed_records": completed,
        "automatic_final_decision_prohibited": True,
        "errors": errors,
    }
