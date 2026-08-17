from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from translation_forensics.cli import main
from translation_forensics.pilot_metrics import (
    PilotMetricsError,
    evaluate_naturalness_comparison,
)


def _row(number: int, outcome: str = "tie") -> dict[str, object]:
    nonverbal = outcome == "nonverbal"
    return {
        "schema_name": "translation-forensics/pilot-adjudication-record",
        "schema_version": "1",
        "title_id": "SSIS-908",
        "block_number": number,
        "adjudication_status": "agreed",
        "reviewer_ids": ["listener-a", "listener-b"],
        "direct_listening_completed_by": ["listener-a", "listener-b"],
        "semantic_gate": "nonverbal" if nonverbal else "both-pass",
        "context_gate": "nonverbal" if nonverbal else "both-pass",
        "naturalness_outcome": outcome,
        "nonverbal": nonverbal,
        "mqm_errors": [],
        "consensus_recorded": False,
        "internal_key_unsealed_for_adjudication": True,
        "human_reviewed": True,
        "reason": "independent reviewer agreement",
    }


def _rows(*, wins: int, losses: int, ties: int, nonverbal_ties: int = 0) -> list[dict[str, object]]:
    outcomes = (
        ["improvement"] * wins
        + ["baseline"] * losses
        + ["tie"] * ties
        + ["nonverbal"] * nonverbal_ties
    )
    return [_row(number, outcome) for number, outcome in enumerate(outcomes, 1)]


def test_298_block_fixed_denominator_contract_passes_at_integer_boundary() -> None:
    report = evaluate_naturalness_comparison(
        _rows(wins=194, losses=44, ties=60),
    )

    assert report["status"] == "pass"
    assert report["denominator_block_count"] == 298
    assert report["improvement_win_count"] == 194
    assert report["improvement_loss_count"] == 44
    assert report["tie_count"] == 60
    assert report["improvement_win_percent"] == 65.100671
    assert report["improvement_loss_percent"] == 14.765101
    assert report["meets_minimum_improvement_win_rate"] is True
    assert report["meets_maximum_improvement_loss_rate"] is True
    assert report["gate_passed"] is True


@pytest.mark.parametrize(
    ("wins", "losses", "ties", "failed_field"),
    [
        (12, 3, 5, "meets_minimum_improvement_win_rate"),
        (13, 4, 3, "meets_maximum_improvement_loss_rate"),
    ],
)
def test_either_numeric_threshold_failure_fails_gate(
    wins: int,
    losses: int,
    ties: int,
    failed_field: str,
) -> None:
    report = evaluate_naturalness_comparison(
        _rows(wins=wins, losses=losses, ties=ties),
        expected_blocks=20,
    )

    assert report["status"] == "fail"
    assert report[failed_field] is False
    assert report["gate_passed"] is False


def test_ties_and_agreed_nonverbal_ties_remain_in_full_denominator() -> None:
    report = evaluate_naturalness_comparison(
        _rows(wins=13, losses=3, ties=3, nonverbal_ties=1),
        expected_blocks=20,
    )

    assert report["improvement_win_rate"] == 0.65
    assert report["improvement_loss_rate"] == 0.15
    assert report["tie_count"] == 4
    assert report["nonverbal_tie_count"] == 1
    assert report["nonverbal_tie_blocks"] == [20]
    assert report["gate_passed"] is True


def test_recorded_unresolved_outcome_stays_in_denominator_and_blocks_gate() -> None:
    rows = _rows(wins=13, losses=3, ties=4)
    rows[-1].update({
        "adjudication_status": "unresolved",
        "semantic_gate": "unresolved",
        "context_gate": "unresolved",
        "naturalness_outcome": "unresolved",
        "nonverbal": None,
        "human_reviewed": False,
        "reason": "reviewer disagreement remains unresolved",
    })

    report = evaluate_naturalness_comparison(rows, expected_blocks=20)

    assert report["status"] == "inconclusive"
    assert report["denominator_block_count"] == 20
    assert report["unresolved_blocks"] == [20]
    assert report["unresolved_count"] == 1
    assert report["improvement_win_percent"] == 65.0
    assert report["improvement_loss_percent"] == 15.0
    assert report["gate_passed"] is False


def test_unreturned_review_keeps_rates_pending() -> None:
    rows = _rows(wins=13, losses=3, ties=4)
    rows[-1].update({
        "adjudication_status": "unresolved",
        "reviewer_ids": [],
        "direct_listening_completed_by": [],
        "semantic_gate": "unresolved",
        "context_gate": "unresolved",
        "naturalness_outcome": "unresolved",
        "nonverbal": None,
        "internal_key_unsealed_for_adjudication": False,
        "human_reviewed": False,
        "reason": "review not returned",
    })

    report = evaluate_naturalness_comparison(rows, expected_blocks=20)

    assert report["status"] == "awaiting-human-review"
    assert report["blocking_blocks"] == [20]
    assert report["improvement_win_rate"] is None
    assert report["improvement_loss_rate"] is None
    assert report["gate_passed"] is None


def test_adjudicated_nonverbal_tie_requires_recorded_consensus() -> None:
    rows = _rows(wins=13, losses=3, ties=3, nonverbal_ties=1)
    rows[-1]["adjudication_status"] = "adjudicated"

    with pytest.raises(PilotMetricsError, match="requires recorded reviewer consensus"):
        evaluate_naturalness_comparison(rows, expected_blocks=20)


def test_cli_writes_schema_valid_receipt_and_refuses_overwrite(tmp_path: Path) -> None:
    adjudication = tmp_path / "final-decisions.jsonl"
    adjudication.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in _rows(wins=13, losses=3, ties=4)
        ),
        encoding="utf-8",
        newline="\n",
    )
    output = tmp_path / "naturalness-comparison.json"

    assert main([
        "evaluate-pilot-naturalness",
        "--adjudication", str(adjudication),
        "--expected-blocks", "20",
        "--output", str(output),
    ]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    schema_path = Path(__file__).resolve().parents[2] / "schemas" / "pilot-naturalness-comparison.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)
    assert report["improvement_win_percent"] == 65.0
    assert report["improvement_loss_percent"] == 15.0

    assert main([
        "evaluate-pilot-naturalness",
        "--adjudication", str(adjudication),
        "--expected-blocks", "20",
        "--output", str(output),
    ]) == 2
