from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


REQUIRED_OUTPUT_FIELDS = {
    "block_number", "source_faithful_korean", "viewer_natural_korean",
    "translation_method", "translation_model", "status", "confidence",
    "evidence_refs", "consistency_refs", "consistency_conflicts",
    "preserved_meaning", "review_required_reasons", "uncertain_slots", "review_note",
}

AUTONOMOUS_REQUIRED_OUTPUT_FIELDS = {
    "title_id", "block_number", "source_faithful_korean",
    "viewer_natural_korean", "source_status", "viewer_status", "confidence",
    "evidence_refs", "semantic_slots", "inferred_slots",
    "competing_interpretations", "risk_codes", "reason",
}


def sha256_text(path: Path) -> str:
    """Hash UTF-8 text after universal-newline normalization.

    Git may materialize CRLF on Windows even when a manifest was created from
    the LF repository blob. Prompt and JSON contracts are text, so their
    integrity identity is canonical UTF-8 with LF newlines rather than an
    operating-system checkout detail.
    """

    with path.open("r", encoding="utf-8-sig", newline=None) as handle:
        canonical = handle.read()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_hashed_file(
    *,
    manifest: dict[str, Any],
    base: Path,
    path_field: str,
    hash_field: str,
    errors: list[str],
) -> None:
    path = (base / str(manifest.get(path_field, ""))).resolve()
    if not path.exists():
        errors.append(f"{path_field} does not exist.")
        return
    if str(manifest.get(hash_field, "")) != sha256_text(path):
        errors.append(f"{hash_field} does not match {path_field}.")


def validate_prompt_contract(manifest_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = root or manifest_path.parent
    errors: list[str] = []
    prompt_path = (base / manifest.get("prompt_file", "")).resolve()
    if not prompt_path.exists():
        errors.append("prompt_file이 존재하지 않습니다.")
    else:
        expected = str(manifest.get("prompt_sha256", ""))
        actual = sha256_text(prompt_path)
        if expected != actual:
            errors.append("prompt_sha256가 프롬프트 본문과 일치하지 않습니다.")
    if manifest.get("translation_model") != "gpt-5.6-terra":
        errors.append("공식 의미 번역 프롬프트 모델은 gpt-5.6-terra여야 합니다.")
    contract_type = str(manifest.get("contract_type", "semantic-translation"))
    if contract_type == "autonomous-subtitle-decision":
        if manifest.get("critic_model") != "gpt-5.6-sol":
            errors.append("autonomous-release critic model must be gpt-5.6-sol.")
        for path_field, hash_field in (
            ("critic_prompt_file", "critic_prompt_sha256"),
            ("decision_schema_file", "decision_schema_sha256"),
            ("critique_schema_file", "critique_schema_sha256"),
        ):
            _validate_hashed_file(
                manifest=manifest,
                base=base,
                path_field=path_field,
                hash_field=hash_field,
                errors=errors,
            )
    elif contract_type == "semantic-translation":
        _validate_hashed_file(
            manifest=manifest,
            base=base,
            path_field="decision_schema_file",
            hash_field="decision_schema_sha256",
            errors=errors,
        )
        schema_path = (base / str(manifest.get("decision_schema_file", ""))).resolve()
        if schema_path.is_file():
            try:
                schema = json.loads(schema_path.read_text(encoding="utf-8"))
                Draft202012Validator.check_schema(schema)
                validator = Draft202012Validator(schema)
                common = {
                    "block_number": 1,
                    "translation_method": "semantic_review_from_japanese",
                    "translation_model": "gpt-5.6-terra",
                    "confidence": "unknown",
                    "evidence_refs": [],
                    "consistency_refs": [],
                    "consistency_conflicts": [],
                    "preserved_meaning": [],
                    "review_required_reasons": [],
                    "uncertain_slots": [],
                    "review_note": "missing evidence",
                }
                unresolved = {
                    **common,
                    "status": "unresolved",
                    "source_faithful_korean": "",
                    "viewer_natural_korean": "",
                }
                translated = {
                    **common,
                    "status": "translated",
                    "confidence": "high",
                    "source_faithful_korean": "번역",
                    "viewer_natural_korean": "번역",
                }
                if list(validator.iter_errors(unresolved)):
                    errors.append("decision schema rejects the documented unresolved output")
                if list(validator.iter_errors(translated)):
                    errors.append("decision schema rejects the documented translated output")
                invalid_translated = {
                    **translated,
                    "source_faithful_korean": "",
                    "viewer_natural_korean": "",
                }
                if not list(validator.iter_errors(invalid_translated)):
                    errors.append("decision schema permits empty translated output")
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                errors.append(f"decision schema is invalid: {exc}")
    fields = set(manifest.get("required_output_fields", []))
    required_fields = AUTONOMOUS_REQUIRED_OUTPUT_FIELDS if contract_type == "autonomous-subtitle-decision" else REQUIRED_OUTPUT_FIELDS
    missing = sorted(required_fields - fields)
    if missing:
        errors.append(f"필수 결정 필드 누락: {', '.join(missing)}")
    tests_path = (base / manifest.get("test_cases_file", "")).resolve()
    cases: list[dict[str, Any]] = []
    if not tests_path.exists():
        errors.append("test_cases_file이 존재하지 않습니다.")
    else:
        for line in tests_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(json.loads(line))
        kinds = {str(case.get("case_type")) for case in cases}
        required = {"normal", "missing-evidence", "conflicting-evidence", "missing-file"}
        if contract_type == "autonomous-subtitle-decision":
            required.update({"utf8-corruption", "prompt-injection"})
        else:
            required.update({"consistency-context", "ambiguous-subject"})
        if required - kinds:
            errors.append(f"필수 프롬프트 시험 유형 누락: {', '.join(sorted(required - kinds))}")
        for case in cases:
            if not case.get("expected_status") or not case.get("expected_behavior"):
                errors.append(f"프롬프트 시험 사례에 기대 결과가 없습니다: {case.get('case_id', '?')}")
    return {
        "status": "pass" if not errors else "fail",
        "manifest": str(manifest_path),
        "prompt": str(prompt_path),
        "tests": str(tests_path),
        "test_case_count": len(cases),
        "errors": errors,
    }
