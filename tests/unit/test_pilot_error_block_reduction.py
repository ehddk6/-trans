from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from translation_forensics.cli import main
from translation_forensics.pilot_metrics import (
    PilotMetricsError,
    evaluate_critical_major_error_block_reduction,
)


def _error(
    candidate: str,
    dimension: str,
    severity: str,
    *,
    error_type: str,
) -> dict[str, object]:
    return {
        "candidate": candidate,
        "dimension": dimension,
        "severity": severity,
        "error_type": error_type,
        "detail": f"{candidate} {dimension} {severity} error",
        "evidence_refs": [f"audio:{error_type}"],
    }


def _row(
    number: int,
    *,
    semantic_gate: str = "both-pass",
    context_gate: str = "both-pass",
    errors: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "schema_name": "translation-forensics/pilot-adjudication-record",
        "schema_version": "1",
        "title_id": "SSIS-908",
        "block_number": number,
        "adjudication_status": "agreed",
        "reviewer_ids": ["listener-a", "listener-b"],
        "direct_listening_completed_by": ["listener-a", "listener-b"],
        "semantic_gate": semantic_gate,
        "context_gate": context_gate,
        "naturalness_outcome": "tie",
        "nonverbal": False,
        "mqm_errors": errors or [],
        "consensus_recorded": False,
        "internal_key_unsealed_for_adjudication": True,
        "human_reviewed": True,
        "reason": "independent reviewer agreement",
    }


def _fifty_percent_rows() -> list[dict[str, object]]:
    return [
        _row(
            1,
            semantic_gate="improvement-only",
            context_gate="improvement-only",
            errors=[
                _error("baseline", "semantic_fidelity", "critical", error_type="polarity"),
                _error("baseline", "contextual_consistency", "major", error_type="speaker"),
            ],
        ),
        _row(
            2,
            semantic_gate="improvement-only",
            context_gate="baseline-only",
            errors=[
                _error("baseline", "semantic_fidelity", "major", error_type="speech-act"),
                _error("improvement", "contextual_consistency", "major", error_type="register"),
            ],
        ),
        _row(
            3,
            context_gate="improvement-only",
            errors=[
                _error("baseline", "contextual_consistency", "critical", error_type="scene-flow"),
            ],
        ),
        _row(
            4,
            semantic_gate="improvement-only",
            errors=[
                _error("baseline", "semantic_fidelity", "critical", error_type="actor"),
            ],
        ),
        _row(
            5,
            semantic_gate="baseline-only",
            errors=[
                _error("improvement", "semantic_fidelity", "major", error_type="target"),
            ],
        ),
        _row(6),
    ]


def test_counts_unique_semantic_and_context_error_blocks_and_accepts_exactly_fifty_percent() -> None:
    report = evaluate_critical_major_error_block_reduction(
        _fifty_percent_rows(),
        expected_blocks=6,
    )

    assert report["status"] == "pass"
    assert report["baseline_error_block_count"] == 4
    assert report["baseline_error_blocks"] == [1, 2, 3, 4]
    assert report["improvement_error_block_count"] == 2
    assert report["improvement_error_blocks"] == [2, 5]
    assert report["error_block_reduction_count"] == 2
    assert report["error_block_reduction_rate"] == 0.5
    assert report["error_block_reduction_percent"] == 50.0
    assert report["meets_minimum_reduction"] is True
    assert report["gate_passed"] is True


def test_less_than_fifty_percent_fails() -> None:
    rows = _fifty_percent_rows()
    rows[5] = _row(
        6,
        context_gate="baseline-only",
        errors=[
            _error("improvement", "contextual_consistency", "critical", error_type="honorific"),
        ],
    )

    report = evaluate_critical_major_error_block_reduction(rows, expected_blocks=6)

    assert report["status"] == "fail"
    assert report["error_block_reduction_percent"] == 25.0
    assert report["meets_minimum_reduction"] is False
    assert report["gate_passed"] is False


def test_zero_error_baseline_is_inconclusive_instead_of_claiming_reduction() -> None:
    report = evaluate_critical_major_error_block_reduction(
        [_row(1), _row(2), _row(3)],
        expected_blocks=3,
    )

    assert report["status"] == "inconclusive"
    assert report["baseline_error_block_count"] == 0
    assert report["error_block_reduction_rate"] is None
    assert report["error_block_reduction_percent"] is None
    assert report["meets_minimum_reduction"] is None
    assert report["gate_passed"] is False


def test_recorded_unresolved_adjudication_blocks_pass_as_inconclusive() -> None:
    rows = [_row(1), _row(2), _row(3)]
    rows[1].update({
        "adjudication_status": "unresolved",
        "semantic_gate": "unresolved",
        "context_gate": "unresolved",
        "naturalness_outcome": "unresolved",
        "nonverbal": None,
        "human_reviewed": False,
    })

    report = evaluate_critical_major_error_block_reduction(rows, expected_blocks=3)

    assert report["status"] == "inconclusive"
    assert report["blocking_blocks"] == [2]
    assert report["unresolved_blocks"] == [2]
    assert report["baseline_error_block_count"] is None
    assert report["gate_passed"] is False


def test_unreturned_review_remains_awaiting_human_review() -> None:
    rows = [_row(1), _row(2), _row(3)]
    rows[1].update({
        "adjudication_status": "unresolved",
        "reviewer_ids": [],
        "direct_listening_completed_by": [],
        "semantic_gate": "unresolved",
        "context_gate": "unresolved",
        "naturalness_outcome": "unresolved",
        "nonverbal": None,
        "internal_key_unsealed_for_adjudication": False,
        "human_reviewed": False,
    })

    report = evaluate_critical_major_error_block_reduction(rows, expected_blocks=3)

    assert report["status"] == "awaiting-human-review"
    assert report["blocking_blocks"] == [2]
    assert report["unresolved_blocks"] == []
    assert report["error_block_reduction_percent"] is None
    assert report["gate_passed"] is None


def test_context_gate_must_reconcile_with_contextual_error_ledger() -> None:
    rows = [
        _row(
            1,
            errors=[
                _error("baseline", "contextual_consistency", "major", error_type="speaker"),
            ],
        ),
        _row(2),
        _row(3),
    ]

    with pytest.raises(PilotMetricsError, match="both-pass gate conflicts.*context_gate"):
        evaluate_critical_major_error_block_reduction(rows, expected_blocks=3)


def test_cli_writes_schema_valid_receipt_and_refuses_overwrite(tmp_path: Path) -> None:
    adjudication = tmp_path / "final-decisions.jsonl"
    adjudication.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _fifty_percent_rows()),
        encoding="utf-8",
        newline="\n",
    )
    output = tmp_path / "error-block-reduction.json"

    assert main([
        "evaluate-pilot-error-reduction",
        "--adjudication", str(adjudication),
        "--expected-blocks", "6",
        "--output", str(output),
    ]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    schema_path = Path(__file__).resolve().parents[2] / "schemas" / "pilot-error-block-reduction.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)
    assert report["error_block_reduction_percent"] == 50.0

    assert main([
        "evaluate-pilot-error-reduction",
        "--adjudication", str(adjudication),
        "--expected-blocks", "6",
        "--output", str(output),
    ]) == 2
