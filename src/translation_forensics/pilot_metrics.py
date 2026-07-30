from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ADJUDICATION_SCHEMA_NAME = "translation-forensics/pilot-adjudication-record"
METRIC_SCHEMA_NAME = "translation-forensics/pilot-new-critical-semantic-errors"
ERROR_BLOCK_REDUCTION_SCHEMA_NAME = "translation-forensics/pilot-error-block-reduction"
NATURALNESS_SCHEMA_NAME = "translation-forensics/pilot-naturalness-comparison"
PILOT_METRICS_SCHEMA_NAME = "translation-forensics/pilot-metrics"
ERROR_LEDGER_RECORD_SCHEMA_NAME = "translation-forensics/pilot-error-ledger-record"
SCHEMA_VERSION = "1"
TITLE_ID = "SSIS-908"
VARIANTS = frozenset({"baseline", "improvement"})
DIMENSIONS = frozenset({"semantic_fidelity", "contextual_consistency", "natural_korean"})
SEVERITIES = frozenset({"critical", "major", "minor"})
REDUCTION_DIMENSIONS = frozenset({"semantic_fidelity", "contextual_consistency"})
REDUCTION_SEVERITIES = frozenset({"critical", "major"})
MINIMUM_ERROR_BLOCK_REDUCTION_PERCENT = 50.0
MINIMUM_NATURALNESS_WIN_PERCENT = 65.0
MAXIMUM_NATURALNESS_LOSS_PERCENT = 15.0
NATURALNESS_OUTCOMES = frozenset(
    {"baseline", "improvement", "tie", "unresolved", "nonverbal"}
)
SEMANTIC_GATES = frozenset(
    {"both-pass", "baseline-only", "improvement-only", "both-fail", "unresolved", "nonverbal"}
)


class PilotMetricsError(ValueError):
    """Raised when adjudication evidence cannot support a pilot metric."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise PilotMetricsError(f"{path}:{line_number}: JSON object required")
        rows.append(value)
    return rows


def _normalize_error(error: Any, *, block_number: int, index: int) -> dict[str, Any]:
    label = f"Block {block_number} MQM error {index}"
    if not isinstance(error, Mapping):
        raise PilotMetricsError(f"{label} must be an object")
    candidate = error.get("candidate")
    dimension = error.get("dimension")
    severity = error.get("severity")
    error_type = str(error.get("error_type", "")).strip()
    detail = str(error.get("detail", "")).strip()
    evidence_refs = error.get("evidence_refs")
    if candidate not in VARIANTS:
        raise PilotMetricsError(f"{label} has an invalid candidate")
    if dimension not in DIMENSIONS or severity not in SEVERITIES:
        raise PilotMetricsError(f"{label} has an invalid dimension or severity")
    if not error_type or not detail:
        raise PilotMetricsError(f"{label} lacks error_type or detail")
    if (
        not isinstance(evidence_refs, list)
        or not evidence_refs
        or not all(isinstance(ref, str) and ref.strip() for ref in evidence_refs)
    ):
        raise PilotMetricsError(f"{label} lacks evidence_refs")
    return {
        "candidate": candidate,
        "dimension": dimension,
        "severity": severity,
        "error_type": error_type,
        "detail": detail,
        "evidence_refs": sorted({ref.strip() for ref in evidence_refs}),
    }


def _normalize_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_blocks: int,
) -> list[dict[str, Any]]:
    if not isinstance(expected_blocks, int) or isinstance(expected_blocks, bool) or expected_blocks < 1:
        raise PilotMetricsError("expected_blocks must be a positive integer")
    rows = [dict(record) for record in records]
    if len(rows) != expected_blocks:
        raise PilotMetricsError(f"Pilot metric requires exactly {expected_blocks} adjudication records")
    numbers = [row.get("block_number") for row in rows]
    if numbers != list(range(1, expected_blocks + 1)):
        raise PilotMetricsError("Adjudication block numbers must be unique and consecutive from 1")

    normalized: list[dict[str, Any]] = []
    for row in rows:
        number = int(row["block_number"])
        if row.get("schema_name") != ADJUDICATION_SCHEMA_NAME or row.get("schema_version") != SCHEMA_VERSION:
            raise PilotMetricsError(f"Block {number} has an invalid adjudication schema")
        if row.get("title_id") != TITLE_ID:
            raise PilotMetricsError(f"Block {number} is outside the SSIS-908 pilot")
        if row.get("adjudication_status") not in {"agreed", "adjudicated", "unresolved"}:
            raise PilotMetricsError(f"Block {number} has an invalid adjudication_status")
        if row.get("semantic_gate") not in SEMANTIC_GATES:
            raise PilotMetricsError(f"Block {number} has an invalid semantic_gate")
        if row.get("context_gate") not in SEMANTIC_GATES:
            raise PilotMetricsError(f"Block {number} has an invalid context_gate")
        errors_value = row.get("mqm_errors")
        if not isinstance(errors_value, list):
            raise PilotMetricsError(f"Block {number} mqm_errors must be an array")
        errors = [
            _normalize_error(error, block_number=number, index=index)
            for index, error in enumerate(errors_value, 1)
        ]
        signatures = [_canonical_bytes(error) for error in errors]
        if len(signatures) != len(set(signatures)):
            raise PilotMetricsError(f"Block {number} contains duplicate MQM errors")
        normalized.append({**row, "mqm_errors": errors})
    return normalized


def _is_complete(row: Mapping[str, Any]) -> bool:
    reviewer_ids = row.get("reviewer_ids")
    listeners = row.get("direct_listening_completed_by")
    return (
        row.get("adjudication_status") in {"agreed", "adjudicated"}
        and row.get("human_reviewed") is True
        and row.get("internal_key_unsealed_for_adjudication") is True
        and row.get("semantic_gate") != "unresolved"
        and row.get("context_gate") != "unresolved"
        and isinstance(reviewer_ids, list)
        and len(reviewer_ids) == 2
        and len(set(reviewer_ids)) == 2
        and isinstance(listeners, list)
        and set(listeners) == set(reviewer_ids)
    )


def _severe_errors(
    row: Mapping[str, Any],
    candidate: str,
    *,
    dimensions: frozenset[str],
) -> list[dict[str, Any]]:
    return [
        error
        for error in row["mqm_errors"]
        if error["candidate"] == candidate
        and error["dimension"] in dimensions
        and error["severity"] in REDUCTION_SEVERITIES
    ]


def _semantic_severe_errors(row: Mapping[str, Any], candidate: str) -> list[dict[str, Any]]:
    return _severe_errors(row, candidate, dimensions=frozenset({"semantic_fidelity"}))


def _validate_quality_gate(
    row: Mapping[str, Any],
    *,
    gate_field: str,
    dimension: str,
) -> None:
    number = int(row["block_number"])
    gate = str(row[gate_field])
    dimensions = frozenset({dimension})
    if gate == "nonverbal":
        if (
            row.get("nonverbal") is not True
            or _severe_errors(row, "baseline", dimensions=dimensions)
            or _severe_errors(row, "improvement", dimensions=dimensions)
        ):
            raise PilotMetricsError(f"Block {number} {gate_field} nonverbal gate conflicts with MQM errors")
        return
    if row.get("nonverbal") is not False:
        raise PilotMetricsError(f"Block {number} verbal {gate_field} requires nonverbal=false")

    failing = {
        "both-pass": set(),
        "baseline-only": {"improvement"},
        "improvement-only": {"baseline"},
        "both-fail": {"baseline", "improvement"},
    }[gate]
    for candidate in VARIANTS:
        severe = _severe_errors(row, candidate, dimensions=dimensions)
        if candidate in failing and not severe:
            raise PilotMetricsError(
                f"Block {number} {gate} gate lacks a critical/major {dimension} error "
                f"for {candidate} ({gate_field})"
            )
        if candidate not in failing and severe:
            raise PilotMetricsError(
                f"Block {number} {gate} gate conflicts with a critical/major {dimension} error "
                f"for {candidate} ({gate_field})"
            )


def _validate_semantic_gate(row: Mapping[str, Any]) -> None:
    _validate_quality_gate(row, gate_field="semantic_gate", dimension="semantic_fidelity")


def _validate_reduction_gates(row: Mapping[str, Any]) -> None:
    _validate_semantic_gate(row)
    _validate_quality_gate(
        row,
        gate_field="context_gate",
        dimension="contextual_consistency",
    )


def _has_completed_unresolved_review(row: Mapping[str, Any]) -> bool:
    reviewer_ids = row.get("reviewer_ids")
    listeners = row.get("direct_listening_completed_by")
    return (
        row.get("adjudication_status") == "unresolved"
        and row.get("internal_key_unsealed_for_adjudication") is True
        and isinstance(reviewer_ids, list)
        and len(reviewer_ids) == 2
        and len(set(reviewer_ids)) == 2
        and isinstance(listeners, list)
        and set(listeners) == set(reviewer_ids)
    )


def _comparison_signature(error: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    return (
        str(error["error_type"]).casefold(),
        tuple(str(ref).casefold() for ref in error["evidence_refs"]),
    )


def _new_error_record(block_number: int, error: Mapping[str, Any]) -> dict[str, Any]:
    identity = {"block_number": block_number, **dict(error)}
    return {
        "error_id": "new-critical-semantic-" + hashlib.sha256(_canonical_bytes(identity)).hexdigest()[:16],
        "block_number": block_number,
        **dict(error),
        "new_relative_to_baseline": True,
    }


def evaluate_new_critical_semantic_errors(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_blocks: int = 298,
) -> dict[str, Any]:
    """Count improvement-only critical semantic/speech-act MQM errors.

    Newness is evaluated within each adjudicated block.  A critical improvement
    error is not new only when the baseline has a critical semantic error with
    the same normalized error type and evidence references.  Consequently a
    baseline major error that becomes critical is conservatively counted as new.
    Incomplete or unresolved human adjudication never yields a numeric zero.
    """
    rows = _normalize_rows(records, expected_blocks=expected_blocks)
    blocking_blocks = [int(row["block_number"]) for row in rows if not _is_complete(row)]
    if blocking_blocks:
        return {
            "schema_name": METRIC_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "title_id": TITLE_ID,
            "status": "awaiting-human-review",
            "expected_block_count": expected_blocks,
            "adjudicated_block_count": expected_blocks - len(blocking_blocks),
            "blocking_blocks": blocking_blocks,
            "new_critical_semantic_error_count": None,
            "new_critical_semantic_error_blocks": [],
            "zero_new_critical_semantic_errors": None,
            "gate_passed": None,
            "new_critical_semantic_errors": [],
        }

    reviewer_sets = {tuple(sorted(str(value) for value in row["reviewer_ids"])) for row in rows}
    if len(reviewer_sets) != 1:
        raise PilotMetricsError("All adjudication records must use the same two reviewers")

    new_errors: list[dict[str, Any]] = []
    for row in rows:
        _validate_semantic_gate(row)
        baseline_critical = {
            _comparison_signature(error)
            for error in row["mqm_errors"]
            if error["candidate"] == "baseline"
            and error["dimension"] == "semantic_fidelity"
            and error["severity"] == "critical"
        }
        for error in row["mqm_errors"]:
            if (
                error["candidate"] == "improvement"
                and error["dimension"] == "semantic_fidelity"
                and error["severity"] == "critical"
                and _comparison_signature(error) not in baseline_critical
            ):
                new_errors.append(_new_error_record(int(row["block_number"]), error))

    new_errors.sort(key=lambda error: (int(error["block_number"]), str(error["error_id"])))
    blocks = sorted({int(error["block_number"]) for error in new_errors})
    passed = not new_errors
    return {
        "schema_name": METRIC_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "title_id": TITLE_ID,
        "status": "pass" if passed else "fail",
        "expected_block_count": expected_blocks,
        "adjudicated_block_count": expected_blocks,
        "blocking_blocks": [],
        "counting_unit": "unique adjudicated MQM error records",
        "comparison_rule": "same block + critical semantic_fidelity + normalized error_type + evidence_refs",
        "new_critical_semantic_error_count": len(new_errors),
        "new_critical_semantic_error_blocks": blocks,
        "zero_new_critical_semantic_errors": passed,
        "gate_passed": passed,
        "new_critical_semantic_errors": new_errors,
    }


def evaluate_new_critical_semantic_errors_file(
    adjudication_path: Path,
    *,
    expected_blocks: int = 298,
) -> dict[str, Any]:
    return evaluate_new_critical_semantic_errors(
        _read_jsonl(adjudication_path),
        expected_blocks=expected_blocks,
    )


def _empty_error_block_reduction_result(
    *,
    expected_blocks: int,
    adjudicated_blocks: int,
    status: str,
    blocking_blocks: list[int],
    unresolved_blocks: list[int],
    gate_passed: bool | None,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_name": ERROR_BLOCK_REDUCTION_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "title_id": TITLE_ID,
        "status": status,
        "expected_block_count": expected_blocks,
        "adjudicated_block_count": adjudicated_blocks,
        "blocking_blocks": blocking_blocks,
        "unresolved_blocks": unresolved_blocks,
        "counting_unit": "unique blocks with one or more qualifying MQM errors",
        "qualifying_dimensions": sorted(REDUCTION_DIMENSIONS),
        "qualifying_severities": sorted(REDUCTION_SEVERITIES),
        "minimum_reduction_percent": MINIMUM_ERROR_BLOCK_REDUCTION_PERCENT,
        "baseline_error_block_count": None,
        "baseline_error_blocks": [],
        "improvement_error_block_count": None,
        "improvement_error_blocks": [],
        "error_block_reduction_count": None,
        "error_block_reduction_rate": None,
        "error_block_reduction_percent": None,
        "meets_minimum_reduction": None,
        "gate_passed": gate_passed,
        "reason": reason,
    }


def evaluate_critical_major_error_block_reduction(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_blocks: int = 298,
) -> dict[str, Any]:
    """Compare unique severe meaning/context error blocks for the two variants.

    Each candidate contributes at most once per block even when reviewers record
    several critical/major semantic or contextual MQM errors in that block.  A
    recorded unresolved adjudication blocks the threshold gate and is reported
    as inconclusive; an unreturned/incomplete review remains awaiting review.
    A zero-error baseline cannot provide a reduction denominator.
    """
    rows = _normalize_rows(records, expected_blocks=expected_blocks)
    complete_blocks = [int(row["block_number"]) for row in rows if _is_complete(row)]
    unresolved_blocks = [
        int(row["block_number"])
        for row in rows
        if not _is_complete(row) and _has_completed_unresolved_review(row)
    ]
    pending_blocks = [
        int(row["block_number"])
        for row in rows
        if not _is_complete(row) and not _has_completed_unresolved_review(row)
    ]
    if pending_blocks:
        return _empty_error_block_reduction_result(
            expected_blocks=expected_blocks,
            adjudicated_blocks=len(complete_blocks),
            status="awaiting-human-review",
            blocking_blocks=sorted(pending_blocks + unresolved_blocks),
            unresolved_blocks=unresolved_blocks,
            gate_passed=None,
            reason="Every block requires completed two-reviewer adjudication before reduction can be measured.",
        )
    if unresolved_blocks:
        return _empty_error_block_reduction_result(
            expected_blocks=expected_blocks,
            adjudicated_blocks=len(complete_blocks),
            status="inconclusive",
            blocking_blocks=unresolved_blocks,
            unresolved_blocks=unresolved_blocks,
            gate_passed=False,
            reason="Recorded reviewer disagreements remain unresolved and block the reduction gate.",
        )

    reviewer_sets = {tuple(sorted(str(value) for value in row["reviewer_ids"])) for row in rows}
    if len(reviewer_sets) != 1:
        raise PilotMetricsError("All adjudication records must use the same two reviewers")

    for row in rows:
        _validate_reduction_gates(row)
    baseline_blocks = sorted(
        int(row["block_number"])
        for row in rows
        if _severe_errors(row, "baseline", dimensions=REDUCTION_DIMENSIONS)
    )
    improvement_blocks = sorted(
        int(row["block_number"])
        for row in rows
        if _severe_errors(row, "improvement", dimensions=REDUCTION_DIMENSIONS)
    )
    baseline_count = len(baseline_blocks)
    improvement_count = len(improvement_blocks)
    reduction_count = baseline_count - improvement_count
    common = {
        "schema_name": ERROR_BLOCK_REDUCTION_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "title_id": TITLE_ID,
        "expected_block_count": expected_blocks,
        "adjudicated_block_count": expected_blocks,
        "blocking_blocks": [],
        "unresolved_blocks": [],
        "counting_unit": "unique blocks with one or more qualifying MQM errors",
        "qualifying_dimensions": sorted(REDUCTION_DIMENSIONS),
        "qualifying_severities": sorted(REDUCTION_SEVERITIES),
        "minimum_reduction_percent": MINIMUM_ERROR_BLOCK_REDUCTION_PERCENT,
        "baseline_error_block_count": baseline_count,
        "baseline_error_blocks": baseline_blocks,
        "improvement_error_block_count": improvement_count,
        "improvement_error_blocks": improvement_blocks,
        "error_block_reduction_count": reduction_count,
    }
    if baseline_count == 0:
        return {
            **common,
            "status": "inconclusive",
            "error_block_reduction_rate": None,
            "error_block_reduction_percent": None,
            "meets_minimum_reduction": None,
            "gate_passed": False,
            "reason": "The baseline has zero qualifying error blocks, so a reduction rate cannot be claimed.",
        }

    reduction_rate = reduction_count / baseline_count
    reduction_percent = round(reduction_rate * 100, 6)
    passed = (
        reduction_count * 100
        >= MINIMUM_ERROR_BLOCK_REDUCTION_PERCENT * baseline_count
    )
    return {
        **common,
        "status": "pass" if passed else "fail",
        "error_block_reduction_rate": round(reduction_rate, 8),
        "error_block_reduction_percent": reduction_percent,
        "meets_minimum_reduction": passed,
        "gate_passed": passed,
        "reason": (
            f"Qualifying error blocks decreased by {reduction_percent}% "
            f"against the required {MINIMUM_ERROR_BLOCK_REDUCTION_PERCENT}%."
        ),
    }


def evaluate_critical_major_error_block_reduction_file(
    adjudication_path: Path,
    *,
    expected_blocks: int = 298,
) -> dict[str, Any]:
    return evaluate_critical_major_error_block_reduction(
        _read_jsonl(adjudication_path),
        expected_blocks=expected_blocks,
    )


def _validate_naturalness_outcome(row: Mapping[str, Any]) -> str:
    number = int(row["block_number"])
    outcome = row.get("naturalness_outcome")
    if outcome not in NATURALNESS_OUTCOMES:
        raise PilotMetricsError(f"Block {number} has an invalid naturalness_outcome")

    if not _is_complete(row):
        if outcome != "unresolved" or row.get("nonverbal") is not None:
            raise PilotMetricsError(
                f"Block {number} incomplete adjudication must keep naturalness unresolved"
            )
        return str(outcome)

    if row.get("nonverbal") is True:
        if outcome != "nonverbal":
            raise PilotMetricsError(
                f"Block {number} nonverbal adjudication must use a nonverbal naturalness outcome"
            )
        if row.get("semantic_gate") != "nonverbal" or row.get("context_gate") != "nonverbal":
            raise PilotMetricsError(
                f"Block {number} nonverbal naturalness conflicts with quality gates"
            )
        if row.get("adjudication_status") == "adjudicated" and row.get("consensus_recorded") is not True:
            raise PilotMetricsError(
                f"Block {number} adjudicated nonverbal tie requires recorded reviewer consensus"
            )
        return str(outcome)

    if row.get("nonverbal") is not False or outcome in {"nonverbal", "unresolved"}:
        raise PilotMetricsError(
            f"Block {number} completed verbal adjudication has an invalid naturalness outcome"
        )
    return str(outcome)


def _empty_naturalness_result(
    *,
    expected_blocks: int,
    adjudicated_blocks: int,
    blocking_blocks: list[int],
    unresolved_blocks: list[int],
) -> dict[str, Any]:
    return {
        "schema_name": NATURALNESS_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "title_id": TITLE_ID,
        "status": "awaiting-human-review",
        "expected_block_count": expected_blocks,
        "adjudicated_block_count": adjudicated_blocks,
        "denominator_block_count": expected_blocks,
        "blocking_blocks": blocking_blocks,
        "unresolved_blocks": unresolved_blocks,
        "counting_unit": "all final adjudicated blocks including ties",
        "minimum_improvement_win_percent": MINIMUM_NATURALNESS_WIN_PERCENT,
        "maximum_improvement_loss_percent": MAXIMUM_NATURALNESS_LOSS_PERCENT,
        "improvement_win_count": None,
        "improvement_win_blocks": [],
        "improvement_loss_count": None,
        "improvement_loss_blocks": [],
        "tie_count": None,
        "tie_blocks": [],
        "nonverbal_tie_count": None,
        "nonverbal_tie_blocks": [],
        "unresolved_count": None,
        "improvement_win_rate": None,
        "improvement_win_percent": None,
        "improvement_loss_rate": None,
        "improvement_loss_percent": None,
        "meets_minimum_improvement_win_rate": None,
        "meets_maximum_improvement_loss_rate": None,
        "gate_passed": None,
        "reason": "Every block requires completed two-reviewer adjudication before naturalness rates can be measured.",
    }


def evaluate_naturalness_comparison(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_blocks: int = 298,
) -> dict[str, Any]:
    """Evaluate the fixed-denominator naturalness win/loss contract.

    Improvement wins, improvement losses, and ties all use the full expected
    block count as their denominator.  A nonverbal block contributes a tie only
    after complete two-reviewer agreement or recorded consensus.  Returned but
    unresolved disagreements remain in the denominator and block the gate;
    submissions that have not been returned keep the metric pending.
    """
    rows = _normalize_rows(records, expected_blocks=expected_blocks)
    outcomes = {
        int(row["block_number"]): _validate_naturalness_outcome(row)
        for row in rows
    }
    complete_blocks = [int(row["block_number"]) for row in rows if _is_complete(row)]
    unresolved_blocks = [
        int(row["block_number"])
        for row in rows
        if not _is_complete(row) and _has_completed_unresolved_review(row)
    ]
    pending_blocks = [
        int(row["block_number"])
        for row in rows
        if not _is_complete(row) and not _has_completed_unresolved_review(row)
    ]
    if pending_blocks:
        return _empty_naturalness_result(
            expected_blocks=expected_blocks,
            adjudicated_blocks=len(complete_blocks),
            blocking_blocks=sorted(pending_blocks + unresolved_blocks),
            unresolved_blocks=unresolved_blocks,
        )

    reviewer_sets = {
        tuple(sorted(str(value) for value in row["reviewer_ids"]))
        for row in rows
    }
    if len(reviewer_sets) != 1:
        raise PilotMetricsError("All adjudication records must use the same two reviewers")

    improvement_win_blocks = sorted(
        number for number, outcome in outcomes.items() if outcome == "improvement"
    )
    improvement_loss_blocks = sorted(
        number for number, outcome in outcomes.items() if outcome == "baseline"
    )
    nonverbal_tie_blocks = sorted(
        number for number, outcome in outcomes.items() if outcome == "nonverbal"
    )
    tie_blocks = sorted(
        number for number, outcome in outcomes.items() if outcome in {"tie", "nonverbal"}
    )
    win_count = len(improvement_win_blocks)
    loss_count = len(improvement_loss_blocks)
    tie_count = len(tie_blocks)
    unresolved_count = len(unresolved_blocks)
    if win_count + loss_count + tie_count + unresolved_count != expected_blocks:
        raise PilotMetricsError("Naturalness outcomes do not cover the fixed denominator")

    win_rate = win_count / expected_blocks
    loss_rate = loss_count / expected_blocks
    win_percent = round(win_rate * 100, 6)
    loss_percent = round(loss_rate * 100, 6)
    meets_win = win_count * 100 >= MINIMUM_NATURALNESS_WIN_PERCENT * expected_blocks
    meets_loss = loss_count * 100 <= MAXIMUM_NATURALNESS_LOSS_PERCENT * expected_blocks
    threshold_passed = meets_win and meets_loss
    common = {
        "schema_name": NATURALNESS_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "title_id": TITLE_ID,
        "expected_block_count": expected_blocks,
        "adjudicated_block_count": len(complete_blocks),
        "denominator_block_count": expected_blocks,
        "blocking_blocks": unresolved_blocks,
        "unresolved_blocks": unresolved_blocks,
        "counting_unit": "all final adjudicated blocks including ties",
        "minimum_improvement_win_percent": MINIMUM_NATURALNESS_WIN_PERCENT,
        "maximum_improvement_loss_percent": MAXIMUM_NATURALNESS_LOSS_PERCENT,
        "improvement_win_count": win_count,
        "improvement_win_blocks": improvement_win_blocks,
        "improvement_loss_count": loss_count,
        "improvement_loss_blocks": improvement_loss_blocks,
        "tie_count": tie_count,
        "tie_blocks": tie_blocks,
        "nonverbal_tie_count": len(nonverbal_tie_blocks),
        "nonverbal_tie_blocks": nonverbal_tie_blocks,
        "unresolved_count": unresolved_count,
        "improvement_win_rate": round(win_rate, 8),
        "improvement_win_percent": win_percent,
        "improvement_loss_rate": round(loss_rate, 8),
        "improvement_loss_percent": loss_percent,
        "meets_minimum_improvement_win_rate": meets_win,
        "meets_maximum_improvement_loss_rate": meets_loss,
    }
    if unresolved_blocks:
        return {
            **common,
            "status": "inconclusive",
            "gate_passed": False,
            "reason": "Recorded reviewer disagreements remain unresolved and block the naturalness gate.",
        }

    return {
        **common,
        "status": "pass" if threshold_passed else "fail",
        "gate_passed": threshold_passed,
        "reason": (
            f"Improvement naturalness win rate is {win_percent}% "
            f"(required >= {MINIMUM_NATURALNESS_WIN_PERCENT}%) and loss rate is "
            f"{loss_percent}% (required <= {MAXIMUM_NATURALNESS_LOSS_PERCENT}%)."
        ),
    }


def evaluate_naturalness_comparison_file(
    adjudication_path: Path,
    *,
    expected_blocks: int = 298,
) -> dict[str, Any]:
    return evaluate_naturalness_comparison(
        _read_jsonl(adjudication_path),
        expected_blocks=expected_blocks,
    )


def _error_ledger_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(dict(record)) + b"\n" for record in records)


def build_pilot_error_ledger(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_blocks: int = 298,
) -> list[dict[str, Any]]:
    """Build a stable, block-complete projection of final MQM adjudication.

    The ledger retains all expected blocks, including pending and unresolved
    ones, so an empty MQM list cannot be mistaken for completed error-free
    review.  Metrics are computed from the same normalized rows.
    """
    rows = _normalize_rows(records, expected_blocks=expected_blocks)
    return [
        {
            "schema_name": ERROR_LEDGER_RECORD_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "title_id": TITLE_ID,
            "block_number": int(row["block_number"]),
            "adjudication_status": row["adjudication_status"],
            "human_reviewed": row.get("human_reviewed") is True,
            "semantic_gate": row["semantic_gate"],
            "context_gate": row["context_gate"],
            "naturalness_outcome": row.get("naturalness_outcome"),
            "nonverbal": row.get("nonverbal"),
            "mqm_errors": row["mqm_errors"],
        }
        for row in rows
    ]


def evaluate_pilot_balance_contract(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_blocks: int = 298,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate the three non-compensating SSIS-908 pilot quality gates.

    A pass requires all of: zero new critical semantic errors, at least 50%
    fewer critical/major meaning-context error blocks, and both naturalness
    thresholds.  A later-stage score can never offset an earlier gate failure.
    """
    rows = _normalize_rows(records, expected_blocks=expected_blocks)
    error_ledger = build_pilot_error_ledger(rows, expected_blocks=expected_blocks)
    new_critical = evaluate_new_critical_semantic_errors(
        rows,
        expected_blocks=expected_blocks,
    )
    error_reduction = evaluate_critical_major_error_block_reduction(
        rows,
        expected_blocks=expected_blocks,
    )
    naturalness = evaluate_naturalness_comparison(
        rows,
        expected_blocks=expected_blocks,
    )

    complete_blocks = [int(row["block_number"]) for row in rows if _is_complete(row)]
    unresolved_blocks = sorted(
        int(row["block_number"])
        for row in rows
        if not _is_complete(row) and _has_completed_unresolved_review(row)
    )
    pending_blocks = sorted(
        int(row["block_number"])
        for row in rows
        if not _is_complete(row) and not _has_completed_unresolved_review(row)
    )

    raw_gates = {
        "zero_new_critical_semantic_errors": new_critical,
        "critical_major_error_block_reduction": error_reduction,
        "naturalness_balance": naturalness,
    }
    if pending_blocks:
        status = "awaiting-human-review"
        result: str | None = None
        gate_passed: bool | None = None
    elif unresolved_blocks:
        status = "pilot-evaluated"
        result = "inconclusive"
        gate_passed = False
    elif any(gate["status"] == "inconclusive" for gate in raw_gates.values()):
        status = "pilot-evaluated"
        result = "inconclusive"
        gate_passed = False
    elif any(gate["status"] == "fail" for gate in raw_gates.values()):
        status = "pilot-evaluated"
        result = "fail"
        gate_passed = False
    elif all(gate.get("gate_passed") is True for gate in raw_gates.values()):
        status = "pilot-evaluated"
        result = "pass"
        gate_passed = True
    else:
        raise PilotMetricsError("Pilot gates produced an inconsistent aggregate state")

    if pending_blocks:
        reason = "Two independent human reviews and final adjudication are required for every block."
    elif unresolved_blocks:
        reason = "Recorded reviewer disagreements remain unresolved, so the pilot is inconclusive."
    elif result == "pass":
        reason = "All three non-compensating meaning and naturalness gates passed."
    elif result == "fail":
        reason = "At least one required meaning or naturalness gate failed."
    else:
        reason = "A required metric has no valid comparison denominator, so the pilot is inconclusive."

    def gate_status(metric: Mapping[str, Any]) -> str:
        if pending_blocks:
            return "awaiting-human-review"
        if unresolved_blocks:
            return "inconclusive"
        return str(metric["status"])

    gates = {
        "zero_new_critical_semantic_errors": {
            "status": gate_status(new_critical),
            "passed": None if pending_blocks else bool(new_critical.get("gate_passed")) and not unresolved_blocks,
            "required_count": 0,
            "observed_count": new_critical["new_critical_semantic_error_count"],
            "reason": (
                "No improvement-only critical semantic or speech-act MQM errors are allowed."
            ),
        },
        "critical_major_error_block_reduction": {
            "status": gate_status(error_reduction),
            "passed": None if pending_blocks else bool(error_reduction.get("gate_passed")) and not unresolved_blocks,
            "minimum_percent": MINIMUM_ERROR_BLOCK_REDUCTION_PERCENT,
            "observed_percent": error_reduction["error_block_reduction_percent"],
            "baseline_error_block_count": error_reduction["baseline_error_block_count"],
            "improvement_error_block_count": error_reduction["improvement_error_block_count"],
            "reason": error_reduction["reason"],
        },
        "naturalness_balance": {
            "status": gate_status(naturalness),
            "passed": None if pending_blocks else bool(naturalness.get("gate_passed")) and not unresolved_blocks,
            "minimum_win_percent": MINIMUM_NATURALNESS_WIN_PERCENT,
            "maximum_loss_percent": MAXIMUM_NATURALNESS_LOSS_PERCENT,
            "observed_win_percent": naturalness["improvement_win_percent"],
            "observed_loss_percent": naturalness["improvement_loss_percent"],
            "win_count": naturalness["improvement_win_count"],
            "loss_count": naturalness["improvement_loss_count"],
            "tie_count": naturalness["tie_count"],
            "reason": naturalness["reason"],
        },
    }
    passed_gate_count = sum(gate["passed"] is True for gate in gates.values())
    ledger_bytes = _error_ledger_bytes(error_ledger)
    metrics = {
        "schema_name": PILOT_METRICS_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "title_id": TITLE_ID,
        "status": status,
        "result": result,
        "lifecycle_status": status if result is None else f"{status}/{result}",
        "expected_block_count": expected_blocks,
        "adjudicated_block_count": len(complete_blocks),
        "blocking_blocks": sorted(pending_blocks + unresolved_blocks),
        "unresolved_blocks": unresolved_blocks,
        "gate_order": [
            "zero_new_critical_semantic_errors",
            "critical_major_error_block_reduction",
            "naturalness_balance",
        ],
        "required_gate_count": 3,
        "passed_gate_count": passed_gate_count,
        "all_required_gates_passed": gate_passed,
        "gates": gates,
        "error_ledger_record_count": len(error_ledger),
        "mqm_error_count": sum(len(row["mqm_errors"]) for row in error_ledger),
        "error_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "reason": reason,
    }
    return metrics, error_ledger


def evaluate_pilot_balance_contract_file(
    adjudication_path: Path,
    *,
    expected_blocks: int = 298,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return evaluate_pilot_balance_contract(
        _read_jsonl(adjudication_path),
        expected_blocks=expected_blocks,
    )


def write_pilot_balance_contract_results(
    metrics_path: Path,
    error_ledger_path: Path,
    metrics: Mapping[str, Any],
    error_ledger: Iterable[Mapping[str, Any]],
) -> None:
    """Write the linked aggregate receipt and ledger without overwriting either."""
    for path in (metrics_path, error_ledger_path):
        if path.exists():
            raise FileExistsError(f"Existing pilot result will not be overwritten: {path}")
    ledger_rows = [dict(record) for record in error_ledger]
    ledger_bytes = _error_ledger_bytes(ledger_rows)
    if metrics.get("error_ledger_sha256") != hashlib.sha256(ledger_bytes).hexdigest():
        raise PilotMetricsError("Pilot metrics do not match the error ledger hash")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    error_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(dict(metrics), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    error_ledger_path.write_bytes(ledger_bytes)


def write_new_critical_semantic_error_result(path: Path, result: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Existing pilot metric will not be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_error_block_reduction_result(path: Path, result: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Existing pilot metric will not be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_naturalness_comparison_result(path: Path, result: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Existing pilot metric will not be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
