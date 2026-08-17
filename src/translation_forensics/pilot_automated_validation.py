from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .manifest import sha256_file
from .pilot_candidate import validate_pilot_candidate
from .prompt_contract import validate_prompt_contract
from .srt import compare_structure, parse_srt
from .translation_quality import validate_quality_regression_suite
from .validation import validate_pair


SCHEMA_NAME = "translation-forensics/pilot-automated-validation"
SCHEMA_VERSION = "1"
READABILITY_ISSUE_CODES = {"too_many_lines", "long_line", "high_cps"}
CHARACTER_SAFETY_ISSUE_CODES = {
    "parse",
    "encoding",
    "newline",
    "empty_text",
    "japanese_residue",
    "placeholder",
    "work_tag",
    "broken_character",
    "control_character",
}


class PilotAutomatedValidationError(ValueError):
    """Raised when the automated pilot validation contract cannot be evaluated."""


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise PilotAutomatedValidationError(f"{path}:{line_number}: JSON object required")
        records.append(record)
    return records


def _resolve_project_artifact(ref: Any, project_root: Path, label: str) -> Path:
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise PilotAutomatedValidationError(f"{label} artifact reference is invalid")
    raw = Path(ref["path"])
    if raw.is_absolute():
        raise PilotAutomatedValidationError(f"{label} artifact path must be project-relative")
    root = project_root.expanduser().resolve()
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PilotAutomatedValidationError(f"{label} artifact path escapes the project root") from exc
    if not resolved.is_file():
        raise PilotAutomatedValidationError(f"{label} artifact is missing")
    if ref.get("sha256") != sha256_file(resolved):
        raise PilotAutomatedValidationError(f"{label} artifact hash mismatch")
    return resolved


def _issue_rows(pair_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in ("source_faithful", "viewer_natural"):
        report = pair_report.get(variant)
        if not isinstance(report, dict):
            rows.append({"variant": variant, "code": "parse", "severity": "error", "block": None})
            continue
        for issue in report.get("issues", []):
            if isinstance(issue, dict):
                rows.append({"variant": variant, **issue})
    return rows


def _prompt_contract_gate(
    *,
    terra_manifest_path: Path,
    autonomous_manifest_path: Path,
    provenance: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    terra_report = validate_prompt_contract(terra_manifest_path)
    autonomous_report = validate_prompt_contract(autonomous_manifest_path)
    terra_manifest = json.loads(terra_manifest_path.read_text(encoding="utf-8"))
    autonomous_manifest = json.loads(autonomous_manifest_path.read_text(encoding="utf-8"))
    errors = [
        *(f"Terra contract: {error}" for error in terra_report.get("errors", [])),
        *(f"Sol contract: {error}" for error in autonomous_report.get("errors", [])),
    ]
    inputs = provenance.get("inputs", {})
    if inputs.get("terra_prompt", {}).get("sha256") != terra_manifest.get("prompt_sha256"):
        errors.append("Candidate Terra prompt hash does not match the frozen Terra contract")
    if inputs.get("sol_prompt", {}).get("sha256") != autonomous_manifest.get("critic_prompt_sha256"):
        errors.append("Candidate Sol prompt hash does not match the frozen critic contract")
    return {
        "status": "pass" if not errors else "fail",
        "contracts": [
            {
                "role": "terra-translation-decision",
                "manifest": _file_ref(terra_manifest_path, project_root),
                "test_case_count": terra_report.get("test_case_count", 0),
            },
            {
                "role": "sol-independent-critique-repair",
                "manifest": _file_ref(autonomous_manifest_path, project_root),
                "test_case_count": autonomous_report.get("test_case_count", 0),
            },
        ],
        "errors": errors,
    }


def _evidence_reference_gate(
    *,
    provenance: dict[str, Any],
    structure_path: Path,
    project_root: Path,
    expected_blocks: int,
) -> dict[str, Any]:
    errors: list[str] = []
    inputs = provenance.get("inputs", {})
    locked_ref = inputs.get("locked_structure", {})
    structure_hash = sha256_file(structure_path)
    if locked_ref.get("sha256") != structure_hash:
        errors.append("Provenance locked-structure hash does not match the Japanese evidence source")
    stages = provenance.get("stages")
    terra_records: list[dict[str, Any]] = []
    if not isinstance(stages, list) or not stages:
        errors.append("Provenance Terra stage is missing")
    else:
        try:
            terra_path = _resolve_project_artifact(stages[0].get("artifact"), project_root, "Terra decision")
            terra_records = _read_jsonl(terra_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))

    expected_numbers = set(range(1, expected_blocks + 1))
    seen_numbers: set[int] = set()
    checked_refs = 0
    for record in terra_records:
        raw_number = record.get("block_number")
        if not isinstance(raw_number, int):
            errors.append("Terra evidence record has a non-integer block_number")
            continue
        if raw_number in seen_numbers:
            errors.append(f"Duplicate Terra evidence record for block {raw_number}")
        seen_numbers.add(raw_number)
        refs = record.get("evidence_refs")
        required_ref = f"japanese-srt:{raw_number}"
        if not isinstance(refs, list) or not refs:
            errors.append(f"Block {raw_number}: evidence_refs is empty")
            continue
        normalized = [str(ref) for ref in refs]
        checked_refs += len(normalized)
        if required_ref not in normalized:
            errors.append(f"Block {raw_number}: missing evidence ID {required_ref}")
        unknown = sorted(ref for ref in normalized if ref not in {required_ref})
        if unknown:
            errors.append(f"Block {raw_number}: unregistered evidence IDs: {', '.join(unknown)}")
    if seen_numbers != expected_numbers:
        missing = sorted(expected_numbers - seen_numbers)
        extra = sorted(seen_numbers - expected_numbers)
        errors.append(f"Terra evidence coverage mismatch; missing={missing[:10]} extra={extra[:10]}")
    return {
        "status": "pass" if not errors else "fail",
        "evidence_source": {
            "kind": "japanese-srt",
            "sha256": structure_hash,
            "blocks": expected_blocks,
        },
        "records": len(terra_records),
        "checked_refs": checked_refs,
        "errors": errors,
    }


def validate_pilot_automated(
    *,
    title_id: str,
    structure_path: Path,
    baseline_path: Path,
    candidate_dir: Path,
    provenance_schema_path: Path,
    terra_manifest_path: Path,
    autonomous_manifest_path: Path,
    regression_suite_path: Path,
    report_schema_path: Path,
    project_root: Path,
    expected_blocks: int = 298,
    expected_baseline_sha256: str = "8653a42dc952152994c75e9d43265c49eddc1b7d581cf05d8012fd87e3c3e25b",
) -> dict[str, Any]:
    """Run every deterministic SSIS-908 pilot gate without making a human-quality claim."""
    if title_id != "SSIS-908":
        raise PilotAutomatedValidationError("This automated pilot contract is limited to SSIS-908")
    root = project_root.expanduser().resolve()
    structure_path = structure_path.expanduser().resolve()
    baseline_path = baseline_path.expanduser().resolve()
    candidate_dir = candidate_dir.expanduser().resolve()
    source_path = candidate_dir / "source-faithful.srt"
    viewer_path = candidate_dir / "viewer-natural.srt"
    provenance_path = candidate_dir / "provenance.json"

    structure, _, _ = parse_srt(structure_path)
    baseline, _, _ = parse_srt(baseline_path)
    structure_errors: list[str] = []
    if sha256_file(baseline_path) != expected_baseline_sha256:
        structure_errors.append("Frozen baseline SHA-256 mismatch")
    if len(structure) != expected_blocks or len(baseline) != expected_blocks:
        structure_errors.append(f"Locked pilot requires exactly {expected_blocks} blocks")
    if [block.number for block in structure] != list(range(1, expected_blocks + 1)):
        structure_errors.append("Locked block numbers must be consecutive from 1")
    if not compare_structure(structure, baseline)["pass"]:
        structure_errors.append("Japanese evidence source and frozen baseline structures differ")

    pair_report = validate_pair(structure_path, source_path, viewer_path, project_root=root)
    if pair_report.get("structure_same") is not True:
        structure_errors.append("Candidate SRT pair does not preserve the locked structure")
    issue_rows = _issue_rows(pair_report)
    character_violations = [row for row in issue_rows if row.get("code") in CHARACTER_SAFETY_ISSUE_CODES]
    readability_violations = [row for row in issue_rows if row.get("code") in READABILITY_ISSUE_CODES]
    warning_counts = Counter(
        str(row.get("code"))
        for row in issue_rows
        if row.get("severity") == "warning" and row.get("code") not in READABILITY_ISSUE_CODES
    )

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance_report = validate_pilot_candidate(
        output_dir=candidate_dir,
        structure_path=structure_path,
        baseline_path=baseline_path,
        provenance_schema_path=provenance_schema_path,
        project_root=root,
        expected_blocks=expected_blocks,
        expected_baseline_sha256=expected_baseline_sha256,
    )
    prompt_gate = _prompt_contract_gate(
        terra_manifest_path=terra_manifest_path,
        autonomous_manifest_path=autonomous_manifest_path,
        provenance=provenance,
        project_root=root,
    )
    evidence_gate = _evidence_reference_gate(
        provenance=provenance,
        structure_path=structure_path,
        project_root=root,
        expected_blocks=expected_blocks,
    )
    regression_report = validate_quality_regression_suite(regression_suite_path)

    checks: dict[str, Any] = {
        "locked_srt_structure": {
            "status": "pass" if not structure_errors else "fail",
            "expected_blocks": expected_blocks,
            "source_blocks": len(structure),
            "baseline_blocks": len(baseline),
            "candidate_pair_preserved": pair_report.get("structure_same") is True,
            "errors": structure_errors,
        },
        "character_safety": {
            "status": "pass" if not character_violations else "fail",
            "requirements": ["utf-8-no-bom", "lf", "non-empty-korean", "no-work-markers", "no-replacement-character"],
            "violations": character_violations,
        },
        "readability": {
            "status": "pass" if not readability_violations else "fail",
            "limits": {"max_lines": 2, "max_line_characters": 42, "max_cps": 16},
            "violations": readability_violations,
        },
        "provenance": {
            "status": provenance_report.get("status", "fail"),
            "pipeline_order": provenance.get("pipeline_order", []),
            "errors": provenance_report.get("errors", []),
        },
        "evidence_references": evidence_gate,
        "prompt_contracts": prompt_gate,
        "synthetic_quality_regressions": {
            "status": regression_report.get("status", "fail"),
            "cases": regression_report.get("cases", 0),
            "required_pilot_error_types": regression_report.get("required_pilot_error_types", []),
            "covered_pilot_error_types": regression_report.get("covered_pilot_error_types", []),
            "missing_pilot_error_types": regression_report.get("missing_pilot_error_types", []),
            "errors": regression_report.get("errors", []),
        },
    }
    failed_checks = sorted(name for name, value in checks.items() if value.get("status") != "pass")
    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "title_id": title_id,
        "status": "pass" if not failed_checks else "fail",
        "automated_validation_passed": not failed_checks,
        "expected_block_count": expected_blocks,
        "frozen_baseline": _file_ref(baseline_path, root),
        "japanese_evidence": _file_ref(structure_path, root),
        "candidate_provenance": _file_ref(provenance_path, root),
        "checks": checks,
        "failed_checks": failed_checks,
        "non_blocking_warning_counts": dict(sorted(warning_counts.items())),
        "human_reviewed": False,
        "pilot_evaluated": False,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
        "claim_scope": "SSIS-908 deterministic validation only; no human quality or cross-title improvement claim",
    }
    schema = json.loads(report_schema_path.read_text(encoding="utf-8"))
    schema_error = next(iter(Draft202012Validator(schema).iter_errors(report)), None)
    if schema_error is not None:
        raise PilotAutomatedValidationError(f"Automated validation report schema failed: {schema_error.message}")
    return report


def write_pilot_automated_validation(output_path: Path, report: dict[str, Any]) -> None:
    if output_path.exists():
        raise FileExistsError(f"Existing automated validation report will not be overwritten: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
