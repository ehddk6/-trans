from __future__ import annotations

import json
from pathlib import Path

from translation_forensics.sol_review import (
    initialize_sol_review_records,
    validate_sol_review_records,
)


def _queue(path: Path) -> None:
    path.write_text(json.dumps({"block_number": 7}) + "\n", encoding="utf-8")


def _record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _complete_model_metadata(record: dict) -> None:
    record.update({
        "review_status": "completed",
        "provider": "approved-provider",
        "provider_request_id": "req-123",
        "external_transfer_allowed": True,
        "external_transfer_approval_id": "transfer-approval-1",
        "external_transfer_approved_by": "privacy-owner",
        "external_transfer_approved_at": "2026-07-26T01:00:00Z",
        "cost_budget_limit": 2.0,
        "actual_cost": 0.4,
        "cost_approval_id": "cost-approval-1",
        "cost_approved_by": "budget-owner",
        "cost_approved_at": "2026-07-26T01:00:00Z",
        "reviewer_model_output": "Potential polarity conflict.",
        "review_reason": "P1 semantic conflict",
        "model_completed_at": "2026-07-26T01:02:00Z",
        "human_verdict": "accept-sol-counterexample",
        "human_reviewer": "reviewer-1",
        "human_reviewed_at": "2026-07-26T01:03:00Z",
        "human_rationale": "Direct listening confirms the counterexample.",
        "human_evidence_refs": ["audio:00:12-00:14"],
        "human_direct_listening": True,
    })


def test_sol_template_is_not_a_provider_call_or_final_decision(tmp_path: Path) -> None:
    queue, output = tmp_path / "queue.jsonl", tmp_path / "sol.jsonl"
    _queue(queue)
    result = initialize_sol_review_records(queue, output, title_id="TITLE")
    record = _record(output)
    assert result["provider_called"] is False
    assert record["review_status"] == "not-requested"
    assert record["final_decision_authority"] == "human-with-direct-evidence"
    assert validate_sol_review_records(output)["status"] == "pass"


def test_sol_automatic_final_decision_field_is_rejected(tmp_path: Path) -> None:
    queue, output = tmp_path / "queue.jsonl", tmp_path / "sol.jsonl"
    _queue(queue); initialize_sol_review_records(queue, output, title_id="TITLE")
    record = _record(output); record["automatic_final_decision"] = False; _write(output, record)
    report = validate_sol_review_records(output)
    assert report["status"] == "fail"
    assert any("automatic final decision" in error for error in report["errors"])


def test_completed_sol_record_requires_human_verdict_fields(tmp_path: Path) -> None:
    queue, output = tmp_path / "queue.jsonl", tmp_path / "sol.jsonl"
    _queue(queue); initialize_sol_review_records(queue, output, title_id="TITLE")
    record = _record(output); _complete_model_metadata(record)
    record["human_reviewer"] = ""; _write(output, record)
    report = validate_sol_review_records(output)
    assert report["status"] == "fail"
    assert any("human_reviewer is required" in error for error in report["errors"])


def test_completed_sol_record_requires_approved_provider_and_cost_metadata(tmp_path: Path) -> None:
    queue, output = tmp_path / "queue.jsonl", tmp_path / "sol.jsonl"
    _queue(queue); initialize_sol_review_records(queue, output, title_id="TITLE")
    record = _record(output); _complete_model_metadata(record)
    record["actual_cost"] = 2.5; _write(output, record)
    report = validate_sol_review_records(output)
    assert report["status"] == "fail"
    assert any("exceeds approved" in error for error in report["errors"])


def test_completed_sol_record_can_only_be_closed_by_human_direct_evidence(tmp_path: Path) -> None:
    queue, output = tmp_path / "queue.jsonl", tmp_path / "sol.jsonl"
    _queue(queue); initialize_sol_review_records(queue, output, title_id="TITLE")
    record = _record(output); _complete_model_metadata(record); _write(output, record)
    report = validate_sol_review_records(output)
    assert report["status"] == "pass"
    assert report["completed_records"] == 1
