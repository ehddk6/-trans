from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .manifest import sha256_file
from .srt import SubtitleBlock, compare_structure, has_japanese, parse_srt, write_srt


TERRA_SCHEMA_NAME = "translation-forensics/pilot-terra-decision"
SOL_SCHEMA_NAME = "translation-forensics/pilot-sol-review"
PROVENANCE_SCHEMA_NAME = "translation-forensics/pilot-candidate-provenance"
SCHEMA_VERSION = "1"
TERRA_MODEL = "gpt-5.6-terra"
SOL_MODEL = "gpt-5.6-sol"
PIPELINE_ORDER = ["terra-translation-decision", "sol-independent-critique-repair", "semantic-recheck"]
SEMANTIC_RECHECK_KEYS = {
    "semantic_fidelity",
    "speech_act_and_polarity",
    "speaker_actor_target",
    "time_direction_intensity",
    "contextual_consistency",
    "naturalness_only_after_semantics",
    "no_unsupported_additions",
}


class PilotCandidateError(ValueError):
    """Raised when the pilot candidate lineage cannot pass a hard gate."""


def canonical_record_sha256(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise PilotCandidateError(f"{path}:{line_number}: JSON object required")
        records.append(value)
    return records


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
        newline="\n",
    )


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(project_root.expanduser().resolve()).as_posix()
    except ValueError:
        return resolved.name


def _file_ref(path: Path, project_root: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": _portable_path(resolved, project_root),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _resolve_artifact_ref(ref: Any, project_root: Path, label: str) -> Path:
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise PilotCandidateError(f"{label} artifact reference is invalid")
    raw = Path(ref["path"])
    if raw.is_absolute():
        raise PilotCandidateError(f"{label} artifact path must be project-relative")
    root = project_root.expanduser().resolve()
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PilotCandidateError(f"{label} artifact path escapes the project root") from exc
    if not resolved.is_file():
        raise PilotCandidateError(f"{label} artifact is missing")
    if ref.get("sha256") != sha256_file(resolved):
        raise PilotCandidateError(f"{label} artifact hash mismatch")
    return resolved


def _validate_locked_structure(
    structure_path: Path,
    baseline_path: Path,
    *,
    expected_blocks: int,
    expected_baseline_sha256: str,
) -> list[SubtitleBlock]:
    if sha256_file(baseline_path) != expected_baseline_sha256:
        raise PilotCandidateError("Frozen baseline SHA-256 mismatch")
    structure, _, _ = parse_srt(structure_path)
    baseline, _, _ = parse_srt(baseline_path)
    if len(structure) != expected_blocks or len(baseline) != expected_blocks:
        raise PilotCandidateError(f"Locked pilot requires exactly {expected_blocks} blocks")
    if [block.number for block in structure] != list(range(1, expected_blocks + 1)):
        raise PilotCandidateError("Locked block numbers must be consecutive from 1")
    if not compare_structure(structure, baseline)["pass"]:
        raise PilotCandidateError("Frozen baseline does not match the locked structure")
    return structure


def _records_by_block(
    records: list[dict[str, Any]],
    *,
    expected_blocks: int,
    label: str,
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        try:
            number = int(record.get("block_number"))
        except (TypeError, ValueError) as exc:
            raise PilotCandidateError(f"{label} block_number must be an integer") from exc
        if number in result:
            raise PilotCandidateError(f"Duplicate {label} record for block {number}")
        result[number] = record
    expected = set(range(1, expected_blocks + 1))
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise PilotCandidateError(f"{label} coverage mismatch; missing={missing[:10]} extra={extra[:10]}")
    return result


def _korean_text(record: dict[str, Any], field: str, block_number: int) -> str:
    value = record.get(field)
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        raise PilotCandidateError(f"Block {block_number}: {field} is empty")
    if has_japanese(text):
        raise PilotCandidateError(f"Block {block_number}: {field} contains Japanese text")
    return text


def _validate_terra_record(record: dict[str, Any], block_number: int) -> tuple[str, str]:
    if record.get("schema_name") != TERRA_SCHEMA_NAME or record.get("schema_version") != SCHEMA_VERSION:
        raise PilotCandidateError(f"Block {block_number}: invalid Terra decision schema")
    if record.get("title_id") != "SSIS-908" or record.get("translation_model") != TERRA_MODEL:
        raise PilotCandidateError(f"Block {block_number}: Terra identity/model mismatch")
    if record.get("confidence") not in {"high", "medium", "low"}:
        raise PilotCandidateError(f"Block {block_number}: invalid Terra confidence")
    evidence_refs = record.get("evidence_refs")
    if not isinstance(evidence_refs, list) or f"japanese-srt:{block_number}" not in evidence_refs:
        raise PilotCandidateError(f"Block {block_number}: Terra decision lacks Japanese evidence")
    if not isinstance(record.get("risk_codes"), list) or not str(record.get("reason") or "").strip():
        raise PilotCandidateError(f"Block {block_number}: incomplete Terra rationale")
    return (
        _korean_text(record, "source_faithful_korean", block_number),
        _korean_text(record, "viewer_natural_korean", block_number),
    )


def _validate_recheck(record: dict[str, Any], block_number: int) -> None:
    recheck = record.get("semantic_recheck")
    if not isinstance(recheck, dict) or set(recheck) != SEMANTIC_RECHECK_KEYS:
        raise PilotCandidateError(f"Block {block_number}: semantic recheck contract is incomplete")
    failed = sorted(key for key, value in recheck.items() if value is not True)
    if failed:
        raise PilotCandidateError(f"Block {block_number}: semantic recheck failed: {', '.join(failed)}")


def write_pilot_sol_reviews(
    *,
    terra_decisions_path: Path,
    review_plan_path: Path,
    output_path: Path,
    expected_blocks: int = 298,
) -> dict[str, Any]:
    """Materialize an already-completed independent Sol review plan.

    This function does not call a model and does not infer acceptance. The plan
    must explicitly enumerate every reviewed block and attest every semantic
    recheck field before records can be written.
    """
    if output_path.exists():
        raise FileExistsError(f"Existing Sol review records will not be overwritten: {output_path}")
    terra_records = _read_jsonl(terra_decisions_path)
    terra_by_block = _records_by_block(terra_records, expected_blocks=expected_blocks, label="Terra decision")
    plan = json.loads(review_plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema_name") != "translation-forensics/pilot-sol-review-plan":
        raise PilotCandidateError("Invalid Sol review plan schema")
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("title_id") != "SSIS-908":
        raise PilotCandidateError("Sol review plan identity mismatch")
    if plan.get("reviewer_model") != SOL_MODEL or plan.get("independent_from_terra_creation") is not True:
        raise PilotCandidateError("Sol review plan does not establish an independent gpt-5.6-sol review")
    reviewed = plan.get("reviewed_block_numbers")
    if reviewed != list(range(1, expected_blocks + 1)):
        raise PilotCandidateError("Sol review plan must explicitly enumerate every locked block in order")
    default_reason = str(plan.get("accepted_review_reason") or "").strip()
    if not default_reason:
        raise PilotCandidateError("Sol review plan lacks an acceptance rationale")
    recheck = plan.get("semantic_recheck")
    if not isinstance(recheck, dict) or set(recheck) != SEMANTIC_RECHECK_KEYS or not all(value is True for value in recheck.values()):
        raise PilotCandidateError("Sol review plan semantic recheck attestation is incomplete")

    raw_repairs = plan.get("repairs", [])
    if not isinstance(raw_repairs, list):
        raise PilotCandidateError("Sol review plan repairs must be an array")
    repairs: dict[int, dict[str, Any]] = {}
    for repair in raw_repairs:
        if not isinstance(repair, dict):
            raise PilotCandidateError("Sol repair must be an object")
        try:
            number = int(repair.get("block_number"))
        except (TypeError, ValueError) as exc:
            raise PilotCandidateError("Sol repair block_number must be an integer") from exc
        if number not in terra_by_block or number in repairs:
            raise PilotCandidateError(f"Invalid or duplicate Sol repair for block {number}")
        _korean_text(repair, "repaired_source_faithful_korean", number)
        _korean_text(repair, "repaired_viewer_natural_korean", number)
        if not isinstance(repair.get("resolved_issues"), list) or not repair["resolved_issues"]:
            raise PilotCandidateError(f"Block {number}: repair plan must identify resolved issues")
        if not str(repair.get("reason") or "").strip():
            raise PilotCandidateError(f"Block {number}: repair plan reason is empty")
        repairs[number] = repair

    plan_hash = sha256_file(review_plan_path)
    records: list[dict[str, Any]] = []
    for number in range(1, expected_blocks + 1):
        terra = terra_by_block[number]
        repair = repairs.get(number)
        record: dict[str, Any] = {
            "schema_name": SOL_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "title_id": "SSIS-908",
            "block_number": number,
            "reviewer_model": SOL_MODEL,
            "independent_from_terra_creation": True,
            "terra_decision_sha256": canonical_record_sha256(terra),
            "review_plan_sha256": plan_hash,
            "verdict": "repair" if repair else "accept",
            "critical_slot_conflicts": list(repair.get("critical_slot_conflicts", [])) if repair else [],
            "unsupported_additions": list(repair.get("unsupported_additions", [])) if repair else [],
            "risk_codes": list(repair.get("risk_codes", [])) if repair else list(terra.get("risk_codes", [])),
            "resolved_issues": list(repair["resolved_issues"]) if repair else [],
            "semantic_recheck": dict(recheck),
            "reason": str(repair["reason"]) if repair else default_reason,
        }
        if repair:
            record["repaired_source_faithful_korean"] = repair["repaired_source_faithful_korean"]
            record["repaired_viewer_natural_korean"] = repair["repaired_viewer_natural_korean"]
        records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, records)
    return {
        "status": "recorded",
        "title_id": "SSIS-908",
        "reviewer_model": SOL_MODEL,
        "reviewed_blocks": expected_blocks,
        "repairs": len(repairs),
        "output": str(output_path),
        "external_paid_api_call": False,
    }


def _apply_sol_review(
    terra: dict[str, Any],
    review: dict[str, Any],
    block_number: int,
) -> tuple[str, str, dict[str, Any]]:
    terra_source, terra_viewer = _validate_terra_record(terra, block_number)
    if review.get("schema_name") != SOL_SCHEMA_NAME or review.get("schema_version") != SCHEMA_VERSION:
        raise PilotCandidateError(f"Block {block_number}: invalid Sol review schema")
    if review.get("title_id") != "SSIS-908" or review.get("reviewer_model") != SOL_MODEL:
        raise PilotCandidateError(f"Block {block_number}: Sol identity/model mismatch")
    if review.get("independent_from_terra_creation") is not True:
        raise PilotCandidateError(f"Block {block_number}: Sol review is not marked independent")
    terra_hash = canonical_record_sha256(terra)
    if review.get("terra_decision_sha256") != terra_hash:
        raise PilotCandidateError(f"Block {block_number}: Sol review does not reference the Terra decision")
    if not str(review.get("reason") or "").strip():
        raise PilotCandidateError(f"Block {block_number}: Sol review reason is empty")
    for field in ("critical_slot_conflicts", "unsupported_additions", "risk_codes"):
        if not isinstance(review.get(field), list):
            raise PilotCandidateError(f"Block {block_number}: Sol {field} must be an array")
    _validate_recheck(review, block_number)

    verdict = review.get("verdict")
    if verdict == "accept":
        if review["critical_slot_conflicts"] or review["unsupported_additions"]:
            raise PilotCandidateError(f"Block {block_number}: accept cannot retain semantic conflicts")
        source, viewer = terra_source, terra_viewer
    elif verdict == "repair":
        source = _korean_text(review, "repaired_source_faithful_korean", block_number)
        viewer = _korean_text(review, "repaired_viewer_natural_korean", block_number)
        resolved = review.get("resolved_issues")
        if not isinstance(resolved, list) or not resolved:
            raise PilotCandidateError(f"Block {block_number}: repair must record resolved issues")
    elif verdict == "quarantine":
        raise PilotCandidateError(f"Block {block_number}: quarantined Sol review blocks candidate generation")
    else:
        raise PilotCandidateError(f"Block {block_number}: invalid Sol verdict")

    recheck_record = {
        "schema_name": "translation-forensics/pilot-semantic-recheck",
        "schema_version": SCHEMA_VERSION,
        "title_id": "SSIS-908",
        "block_number": block_number,
        "terra_decision_sha256": terra_hash,
        "sol_review_sha256": canonical_record_sha256(review),
        "source_faithful_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "viewer_natural_sha256": hashlib.sha256(viewer.encode("utf-8")).hexdigest(),
        "hard_gate": "pass",
        "checks": {key: True for key in sorted(SEMANTIC_RECHECK_KEYS)},
    }
    return source, viewer, recheck_record


def build_pilot_candidate(
    *,
    title_id: str,
    structure_path: Path,
    baseline_path: Path,
    terra_decisions_path: Path,
    sol_reviews_path: Path,
    terra_prompt_path: Path,
    sol_prompt_path: Path,
    provenance_schema_path: Path,
    output_dir: Path,
    project_root: Path,
    expected_blocks: int = 298,
    expected_baseline_sha256: str = "8653a42dc952152994c75e9d43265c49eddc1b7d581cf05d8012fd87e3c3e25b",
) -> dict[str, Any]:
    """Build a locked pilot candidate only after all three model/gate stages pass."""
    if title_id != "SSIS-908":
        raise PilotCandidateError("This frozen pilot contract is limited to SSIS-908")
    output_dir = output_dir.expanduser().resolve()
    targets = {
        "source_faithful": output_dir / "source-faithful.srt",
        "viewer_natural": output_dir / "viewer-natural.srt",
        "semantic_recheck": output_dir / "semantic-recheck.jsonl",
        "provenance": output_dir / "provenance.json",
    }
    existing = [path.name for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Pilot candidate outputs already exist and will not be overwritten: {', '.join(existing)}")

    structure = _validate_locked_structure(
        structure_path,
        baseline_path,
        expected_blocks=expected_blocks,
        expected_baseline_sha256=expected_baseline_sha256,
    )
    terra_records = _read_jsonl(terra_decisions_path)
    sol_records = _read_jsonl(sol_reviews_path)
    terra_by_block = _records_by_block(terra_records, expected_blocks=expected_blocks, label="Terra decision")
    sol_by_block = _records_by_block(sol_records, expected_blocks=expected_blocks, label="Sol review")

    source_blocks: list[SubtitleBlock] = []
    viewer_blocks: list[SubtitleBlock] = []
    rechecks: list[dict[str, Any]] = []
    repair_blocks: list[int] = []
    for block in structure:
        source, viewer, recheck = _apply_sol_review(
            terra_by_block[block.number], sol_by_block[block.number], block.number
        )
        source_blocks.append(SubtitleBlock(block.number, block.start, block.end, source, block.start_seconds, block.end_seconds))
        viewer_blocks.append(SubtitleBlock(block.number, block.start, block.end, viewer, block.start_seconds, block.end_seconds))
        rechecks.append(recheck)
        if sol_by_block[block.number].get("verdict") == "repair":
            repair_blocks.append(block.number)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_srt(targets["source_faithful"], source_blocks)
    write_srt(targets["viewer_natural"], viewer_blocks)
    _write_jsonl(targets["semantic_recheck"], rechecks)

    output_source, source_encoding, source_newline = parse_srt(targets["source_faithful"])
    output_viewer, viewer_encoding, viewer_newline = parse_srt(targets["viewer_natural"])
    if not compare_structure(structure, output_source)["pass"] or not compare_structure(structure, output_viewer)["pass"]:
        raise PilotCandidateError("Generated SRT structure changed after serialization")
    if (source_encoding, source_newline, viewer_encoding, viewer_newline) != ("utf-8", "LF", "utf-8", "LF"):
        raise PilotCandidateError("Generated SRT files must be UTF-8 without BOM and LF")

    provenance = {
        "schema_name": PROVENANCE_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "title_id": title_id,
        "status": "candidate-generated",
        "candidate_status": "pilot-candidate",
        "block_count": expected_blocks,
        "structure_preserved": True,
        "pipeline_order": PIPELINE_ORDER,
        "models": {"translation_decision": TERRA_MODEL, "independent_critique_repair": SOL_MODEL},
        "inputs": {
            "locked_structure": _file_ref(structure_path, project_root),
            "frozen_baseline": _file_ref(baseline_path, project_root),
            "terra_prompt": _file_ref(terra_prompt_path, project_root),
            "sol_prompt": _file_ref(sol_prompt_path, project_root),
        },
        "stages": [
            {
                "stage": PIPELINE_ORDER[0],
                "model": TERRA_MODEL,
                "records": expected_blocks,
                "artifact": _file_ref(terra_decisions_path, project_root),
            },
            {
                "stage": PIPELINE_ORDER[1],
                "model": SOL_MODEL,
                "independent": True,
                "records": expected_blocks,
                "repairs": len(repair_blocks),
                "repair_blocks": repair_blocks,
                "artifact": _file_ref(sol_reviews_path, project_root),
            },
            {
                "stage": PIPELINE_ORDER[2],
                "gate": "pipeline-hard-gate",
                "records": expected_blocks,
                "passed": expected_blocks,
                "failed": 0,
                "artifact": _file_ref(targets["semantic_recheck"], project_root),
            },
        ],
        "outputs": {
            "source_faithful": _file_ref(targets["source_faithful"], output_dir),
            "viewer_natural": _file_ref(targets["viewer_natural"], output_dir),
        },
        "human_reviewed": False,
        "pilot_evaluated": False,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
        "claim_scope": "SSIS-908 only; candidate generation is not a quality-improvement result",
    }
    schema = json.loads(provenance_schema_path.read_text(encoding="utf-8"))
    first_error = next(iter(Draft202012Validator(schema).iter_errors(provenance)), None)
    if first_error is not None:
        raise PilotCandidateError(f"Candidate provenance schema validation failed: {first_error.message}")
    _write_json(targets["provenance"], provenance)
    return provenance


def validate_pilot_candidate(
    *,
    output_dir: Path,
    structure_path: Path,
    baseline_path: Path,
    provenance_schema_path: Path,
    project_root: Path,
    expected_blocks: int = 298,
    expected_baseline_sha256: str = "8653a42dc952152994c75e9d43265c49eddc1b7d581cf05d8012fd87e3c3e25b",
) -> dict[str, Any]:
    errors: list[str] = []
    output_dir = output_dir.expanduser().resolve()
    try:
        structure = _validate_locked_structure(
            structure_path,
            baseline_path,
            expected_blocks=expected_blocks,
            expected_baseline_sha256=expected_baseline_sha256,
        )
    except (OSError, ValueError) as exc:
        return {"status": "fail", "errors": [str(exc)], "final_promotion_allowed": False}

    paths = {
        "source_faithful": output_dir / "source-faithful.srt",
        "viewer_natural": output_dir / "viewer-natural.srt",
        "semantic_recheck": output_dir / "semantic-recheck.jsonl",
        "provenance": output_dir / "provenance.json",
    }
    for label, path in paths.items():
        if not path.is_file():
            errors.append(f"Missing candidate artifact: {label}")
    if errors:
        return {"status": "fail", "errors": errors, "final_promotion_allowed": False}

    try:
        source, source_encoding, source_newline = parse_srt(paths["source_faithful"])
        viewer, viewer_encoding, viewer_newline = parse_srt(paths["viewer_natural"])
        for label, candidate in (("source-faithful", source), ("viewer-natural", viewer)):
            comparison = compare_structure(structure, candidate)
            if not comparison["pass"]:
                errors.append(f"{label} structure mismatch")
            if any(not block.text.strip() or has_japanese(block.text) for block in candidate):
                errors.append(f"{label} contains empty or Japanese text")
        if (source_encoding, source_newline, viewer_encoding, viewer_newline) != ("utf-8", "LF", "utf-8", "LF"):
            errors.append("Candidate SRT encoding/newline contract failed")

        provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
        schema = json.loads(provenance_schema_path.read_text(encoding="utf-8"))
        schema_error = next(iter(Draft202012Validator(schema).iter_errors(provenance)), None)
        if schema_error is not None:
            errors.append(f"Provenance schema: {schema_error.message}")
        if provenance.get("pipeline_order") != PIPELINE_ORDER:
            errors.append("Candidate pipeline order is invalid")
        if provenance.get("models") != {"translation_decision": TERRA_MODEL, "independent_critique_repair": SOL_MODEL}:
            errors.append("Candidate model lineage is invalid")
        if provenance.get("inputs", {}).get("frozen_baseline", {}).get("sha256") != expected_baseline_sha256:
            errors.append("Provenance frozen baseline hash mismatch")
        for label in ("source_faithful", "viewer_natural"):
            if provenance.get("outputs", {}).get(label, {}).get("sha256") != sha256_file(paths[label]):
                errors.append(f"Provenance {label} output hash mismatch")

        rechecks = _read_jsonl(paths["semantic_recheck"])
        rechecks_by_block = _records_by_block(rechecks, expected_blocks=expected_blocks, label="semantic recheck")
        if any(record.get("hard_gate") != "pass" or set(record.get("checks", {})) != SEMANTIC_RECHECK_KEYS or not all(record["checks"].values()) for record in rechecks_by_block.values()):
            errors.append("Semantic recheck hard gate is incomplete or failing")
        if provenance.get("stages", [None, None, {}])[2].get("artifact", {}).get("sha256") != sha256_file(paths["semantic_recheck"]):
            errors.append("Provenance semantic recheck hash mismatch")

        stages = provenance.get("stages")
        if not isinstance(stages, list) or len(stages) != 3:
            raise PilotCandidateError("Candidate must record exactly three pipeline stages")
        terra_path = _resolve_artifact_ref(stages[0].get("artifact"), project_root, "Terra decision")
        sol_path = _resolve_artifact_ref(stages[1].get("artifact"), project_root, "Sol review")
        semantic_path = _resolve_artifact_ref(stages[2].get("artifact"), project_root, "semantic recheck")
        if semantic_path != paths["semantic_recheck"]:
            errors.append("Provenance semantic recheck path does not identify the candidate artifact")
        terra_by_block = _records_by_block(
            _read_jsonl(terra_path), expected_blocks=expected_blocks, label="Terra decision"
        )
        sol_by_block = _records_by_block(
            _read_jsonl(sol_path), expected_blocks=expected_blocks, label="Sol review"
        )
        expected_rechecks: list[dict[str, Any]] = []
        expected_source: list[str] = []
        expected_viewer: list[str] = []
        for block_number in range(1, expected_blocks + 1):
            checked_source, checked_viewer, checked_record = _apply_sol_review(
                terra_by_block[block_number], sol_by_block[block_number], block_number
            )
            expected_source.append(checked_source)
            expected_viewer.append(checked_viewer)
            expected_rechecks.append(checked_record)
        if rechecks != expected_rechecks:
            errors.append("Semantic recheck records do not match the Terra/Sol lineage")
        if [block.text for block in source] != expected_source:
            errors.append("Source-faithful SRT does not match the reviewed lineage")
        if [block.text for block in viewer] != expected_viewer:
            errors.append("Viewer-natural SRT does not match the reviewed lineage")
    except (OSError, ValueError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        errors.append(str(exc))

    return {
        "status": "pass" if not errors else "fail",
        "title_id": "SSIS-908",
        "blocks": expected_blocks,
        "errors": errors,
        "human_reviewed": False,
        "pilot_evaluated": False,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
    }
