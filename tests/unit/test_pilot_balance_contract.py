from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from translation_forensics.cli import main
from translation_forensics.pilot_metrics import evaluate_pilot_balance_contract


ROOT = Path(__file__).resolve().parents[2]


def _error(
    candidate: str,
    dimension: str,
    severity: str,
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


def _row(number: int, outcome: str) -> dict[str, object]:
    return {
        "schema_name": "translation-forensics/pilot-adjudication-record",
        "schema_version": "1",
        "title_id": "SSIS-908",
        "block_number": number,
        "adjudication_status": "agreed",
        "reviewer_ids": ["listener-a", "listener-b"],
        "direct_listening_completed_by": ["listener-a", "listener-b"],
        "semantic_gate": "both-pass",
        "context_gate": "both-pass",
        "naturalness_outcome": outcome,
        "nonverbal": False,
        "mqm_errors": [],
        "consensus_recorded": False,
        "internal_key_unsealed_for_adjudication": True,
        "human_reviewed": True,
        "reason": "independent reviewer agreement",
    }


def _passing_rows() -> list[dict[str, object]]:
    outcomes = ["improvement"] * 13 + ["baseline"] * 3 + ["tie"] * 4
    rows = [_row(number, outcome) for number, outcome in enumerate(outcomes, 1)]
    rows[0].update({
        "semantic_gate": "improvement-only",
        "mqm_errors": [_error("baseline", "semantic_fidelity", "critical", "polarity")],
    })
    rows[1].update({
        "context_gate": "improvement-only",
        "mqm_errors": [_error("baseline", "contextual_consistency", "major", "speaker")],
    })
    rows[2].update({
        "semantic_gate": "improvement-only",
        "context_gate": "baseline-only",
        "mqm_errors": [
            _error("baseline", "semantic_fidelity", "major", "speech-act"),
            _error("improvement", "contextual_consistency", "major", "register"),
        ],
    })
    rows[3].update({
        "context_gate": "improvement-only",
        "mqm_errors": [_error("baseline", "contextual_consistency", "critical", "scene-flow")],
    })
    rows[4].update({
        "semantic_gate": "baseline-only",
        "mqm_errors": [_error("improvement", "semantic_fidelity", "major", "target")],
    })
    return rows


def test_pass_requires_all_three_non_compensating_gates() -> None:
    metrics, ledger = evaluate_pilot_balance_contract(_passing_rows(), expected_blocks=20)

    assert metrics["status"] == "pilot-evaluated"
    assert metrics["result"] == "pass"
    assert metrics["lifecycle_status"] == "pilot-evaluated/pass"
    assert metrics["passed_gate_count"] == 3
    assert metrics["all_required_gates_passed"] is True
    assert all(gate["passed"] is True for gate in metrics["gates"].values())
    assert metrics["gates"]["zero_new_critical_semantic_errors"]["observed_count"] == 0
    assert metrics["gates"]["critical_major_error_block_reduction"]["observed_percent"] == 50.0
    assert metrics["gates"]["naturalness_balance"]["observed_win_percent"] == 65.0
    assert metrics["gates"]["naturalness_balance"]["observed_loss_percent"] == 15.0
    assert len(ledger) == 20
    assert metrics["error_ledger_record_count"] == 20


@pytest.mark.parametrize("failed_gate", ["new-critical", "error-reduction", "naturalness"])
def test_any_single_gate_failure_forces_pilot_fail(failed_gate: str) -> None:
    rows = _passing_rows()
    if failed_gate == "new-critical":
        rows[4]["mqm_errors"] = [
            _error("improvement", "semantic_fidelity", "critical", "target")
        ]
    elif failed_gate == "error-reduction":
        rows[5].update({
            "context_gate": "baseline-only",
            "mqm_errors": [
                _error("improvement", "contextual_consistency", "major", "honorific")
            ],
        })
    else:
        rows[12]["naturalness_outcome"] = "tie"

    metrics, _ = evaluate_pilot_balance_contract(rows, expected_blocks=20)

    assert metrics["status"] == "pilot-evaluated"
    assert metrics["result"] == "fail"
    assert metrics["lifecycle_status"] == "pilot-evaluated/fail"
    assert metrics["all_required_gates_passed"] is False
    assert metrics["passed_gate_count"] == 2


def test_unresolved_and_pending_reviews_never_become_pass() -> None:
    unresolved = _passing_rows()
    unresolved[-1].update({
        "adjudication_status": "unresolved",
        "semantic_gate": "unresolved",
        "context_gate": "unresolved",
        "naturalness_outcome": "unresolved",
        "nonverbal": None,
        "human_reviewed": False,
        "reason": "reviewer disagreement remains unresolved",
    })
    inconclusive, _ = evaluate_pilot_balance_contract(unresolved, expected_blocks=20)
    assert inconclusive["lifecycle_status"] == "pilot-evaluated/inconclusive"
    assert inconclusive["unresolved_blocks"] == [20]
    assert inconclusive["all_required_gates_passed"] is False

    pending = _passing_rows()
    pending[-1].update({
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
    awaiting, _ = evaluate_pilot_balance_contract(pending, expected_blocks=20)
    assert awaiting["status"] == "awaiting-human-review"
    assert awaiting["result"] is None
    assert awaiting["blocking_blocks"] == [20]
    assert awaiting["all_required_gates_passed"] is None


def test_zero_error_baseline_keeps_aggregate_inconclusive() -> None:
    outcomes = ["improvement"] * 12 + ["baseline"] * 3 + ["tie"] * 5
    rows = [_row(number, outcome) for number, outcome in enumerate(outcomes, 1)]

    metrics, _ = evaluate_pilot_balance_contract(rows, expected_blocks=20)

    assert metrics["lifecycle_status"] == "pilot-evaluated/inconclusive"
    assert metrics["gates"]["critical_major_error_block_reduction"]["status"] == "inconclusive"
    assert metrics["gates"]["naturalness_balance"]["status"] == "fail"
    assert metrics["all_required_gates_passed"] is False


def test_cli_writes_linked_schema_valid_artifacts_and_refuses_overwrite(tmp_path: Path) -> None:
    adjudication = tmp_path / "final-decisions.jsonl"
    adjudication.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _passing_rows()),
        encoding="utf-8",
        newline="\n",
    )
    metrics_path = tmp_path / "results" / "metrics.json"
    ledger_path = tmp_path / "results" / "error-ledger.jsonl"
    command = [
        "evaluate-pilot-balance-contract",
        "--adjudication", str(adjudication),
        "--expected-blocks", "20",
        "--metrics-output", str(metrics_path),
        "--error-ledger-output", str(ledger_path),
    ]

    assert main(command) == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    metrics_schema = json.loads((ROOT / "schemas" / "pilot-metrics.schema.json").read_text(encoding="utf-8"))
    ledger_schema = json.loads((ROOT / "schemas" / "pilot-error-ledger-record.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(metrics_schema)
    Draft202012Validator.check_schema(ledger_schema)
    Draft202012Validator(metrics_schema).validate(metrics)
    for record in ledger:
        Draft202012Validator(ledger_schema).validate(record)
    assert hashlib.sha256(ledger_path.read_bytes()).hexdigest() == metrics["error_ledger_sha256"]
    assert main(command) == 2
