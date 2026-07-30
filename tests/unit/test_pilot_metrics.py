from __future__ import annotations

import json
from pathlib import Path

import pytest

from translation_forensics.cli import main
from translation_forensics.pilot_metrics import (
    PilotMetricsError,
    evaluate_new_critical_semantic_errors,
)


def _error(
    candidate: str,
    severity: str,
    *,
    error_type: str = "question-statement-flip",
    evidence_ref: str = "audio:block-0001",
) -> dict[str, object]:
    return {
        "candidate": candidate,
        "dimension": "semantic_fidelity",
        "severity": severity,
        "error_type": error_type,
        "detail": f"{candidate} {severity} semantic error",
        "evidence_refs": [evidence_ref],
    }


def _row(number: int, *, gate: str = "both-pass", errors: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "schema_name": "translation-forensics/pilot-adjudication-record",
        "schema_version": "1",
        "title_id": "SSIS-908",
        "block_number": number,
        "adjudication_status": "agreed",
        "reviewer_ids": ["listener-a", "listener-b"],
        "direct_listening_completed_by": ["listener-a", "listener-b"],
        "semantic_gate": gate,
        "context_gate": "both-pass",
        "naturalness_outcome": "tie",
        "nonverbal": False,
        "mqm_errors": errors or [],
        "consensus_recorded": False,
        "internal_key_unsealed_for_adjudication": True,
        "human_reviewed": True,
        "reason": "independent reviewer agreement",
    }


def test_zero_new_critical_semantic_errors_passes_only_complete_adjudication() -> None:
    report = evaluate_new_critical_semantic_errors(
        [_row(1), _row(2), _row(3)],
        expected_blocks=3,
    )

    assert report["status"] == "pass"
    assert report["new_critical_semantic_error_count"] == 0
    assert report["new_critical_semantic_error_blocks"] == []
    assert report["zero_new_critical_semantic_errors"] is True
    assert report["gate_passed"] is True


def test_improvement_critical_without_baseline_counterpart_fails_gate() -> None:
    rows = [
        _row(1, gate="baseline-only", errors=[_error("improvement", "critical")]),
        _row(2),
        _row(3),
    ]

    report = evaluate_new_critical_semantic_errors(rows, expected_blocks=3)

    assert report["status"] == "fail"
    assert report["new_critical_semantic_error_count"] == 1
    assert report["new_critical_semantic_error_blocks"] == [1]
    assert report["zero_new_critical_semantic_errors"] is False
    assert report["gate_passed"] is False
    assert report["new_critical_semantic_errors"][0]["candidate"] == "improvement"


def test_shared_critical_is_not_new_but_major_to_critical_regression_is() -> None:
    shared_ref = "audio:block-0001"
    shared = [
        _error("baseline", "critical", evidence_ref=shared_ref),
        _error("improvement", "critical", evidence_ref=shared_ref),
    ]
    severity_regression = [
        _error("baseline", "major", evidence_ref="audio:block-0002"),
        _error("improvement", "critical", evidence_ref="audio:block-0002"),
    ]
    report = evaluate_new_critical_semantic_errors(
        [
            _row(1, gate="both-fail", errors=shared),
            _row(2, gate="both-fail", errors=severity_regression),
            _row(3),
        ],
        expected_blocks=3,
    )

    assert report["new_critical_semantic_error_count"] == 1
    assert report["new_critical_semantic_error_blocks"] == [2]


def test_unresolved_adjudication_never_reports_a_numeric_zero() -> None:
    rows = [_row(1), _row(2), _row(3)]
    rows[1].update({
        "adjudication_status": "unresolved",
        "reviewer_ids": [],
        "direct_listening_completed_by": [],
        "semantic_gate": "unresolved",
        "nonverbal": None,
        "internal_key_unsealed_for_adjudication": False,
        "human_reviewed": False,
    })

    report = evaluate_new_critical_semantic_errors(rows, expected_blocks=3)

    assert report["status"] == "awaiting-human-review"
    assert report["blocking_blocks"] == [2]
    assert report["new_critical_semantic_error_count"] is None
    assert report["zero_new_critical_semantic_errors"] is None
    assert report["gate_passed"] is None


def test_gate_and_error_ledger_must_reconcile() -> None:
    rows = [_row(1, errors=[_error("improvement", "critical")]), _row(2), _row(3)]

    with pytest.raises(PilotMetricsError, match="both-pass gate conflicts"):
        evaluate_new_critical_semantic_errors(rows, expected_blocks=3)


def test_cli_writes_failed_gate_receipt_without_overwriting(tmp_path: Path) -> None:
    rows = [
        _row(1, gate="baseline-only", errors=[_error("improvement", "critical")]),
        _row(2),
        _row(3),
    ]
    adjudication = tmp_path / "final-decisions.jsonl"
    adjudication.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    output = tmp_path / "new-critical-result.json"

    assert main([
        "evaluate-pilot-new-critical",
        "--adjudication", str(adjudication),
        "--expected-blocks", "3",
        "--output", str(output),
    ]) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["new_critical_semantic_error_count"] == 1
    assert main([
        "evaluate-pilot-new-critical",
        "--adjudication", str(adjudication),
        "--expected-blocks", "3",
        "--output", str(output),
    ]) == 2
