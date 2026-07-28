from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


REQUIRED_INPUT_FIELDS = {
    "source_japanese", "before_context", "after_context", "speaker_id", "addressee_id",
    "relationship", "existing_register", "address_terms", "terminology", "asr_candidates",
    "uncertainty", "duration_seconds",
}
_POLITE_ENDING = re.compile(r"(?:요|니다|습니까|세요|죠)[.!…?]?$", re.MULTILINE)
_CASUAL_ENDING = re.compile(r"(?:해|야|지|어|아|거든|네)[.!…?]?$", re.MULTILINE)


def _text(candidate: dict[str, Any], target: str) -> str:
    field = "source_faithful_korean" if target == "source" else "viewer_natural_korean"
    return str(candidate.get(field) or "")


def _register(text: str) -> str:
    last = text.strip().split("\n")[-1] if text.strip() else ""
    if _POLITE_ENDING.search(last):
        return "polite"
    if _CASUAL_ENDING.search(last):
        return "casual"
    return "unknown"


def evaluate_quality_case(case: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Evaluate only deterministic regression constraints, never fluency as a score."""
    checks = case.get("checks", {})
    errors: list[dict[str, str]] = []
    for target, terms in checks.get("required_text", {}).items():
        text = _text(candidate, target)
        for term in terms:
            if str(term) not in text:
                errors.append({"code": "required_text_missing", "detail": f"{target}:{term}"})
    for target, terms in checks.get("forbidden_text", {}).items():
        text = _text(candidate, target)
        for term in terms:
            if str(term) in text:
                errors.append({"code": "forbidden_text_present", "detail": f"{target}:{term}"})
    for target, expected in checks.get("required_register", {}).items():
        actual = _register(_text(candidate, target))
        if actual != expected:
            errors.append({"code": "register_mismatch", "detail": f"{target}:{actual}->{expected}"})
    required_refs = set(checks.get("required_consistency_refs", []))
    actual_refs = {str(value) for value in candidate.get("consistency_refs", []) if isinstance(value, str)}
    if required_refs - actual_refs:
        errors.append({"code": "consistency_ref_missing", "detail": ",".join(sorted(required_refs - actual_refs))})
    forbidden_slots = set(checks.get("forbidden_inferred_slots", []))
    actual_slots = {str(value) for value in candidate.get("inferred_slots", []) if isinstance(value, str)}
    if forbidden_slots & actual_slots:
        errors.append({"code": "unsupported_inference", "detail": ",".join(sorted(forbidden_slots & actual_slots))})
    if checks.get("must_abstain"):
        status = str(candidate.get("status") or "")
        if status not in {"abstained", "unresolved", "hold"} or _text(candidate, "source") or _text(candidate, "viewer"):
            errors.append({"code": "abstention_required", "detail": status or "missing-status"})
    minimum_families = checks.get("minimum_independent_families")
    if isinstance(minimum_families, int):
        families = {str(value) for value in candidate.get("source_families", []) if isinstance(value, str) and value}
        if len(families) < minimum_families:
            errors.append({"code": "insufficient_independent_families", "detail": f"{len(families)}<{minimum_families}"})
    for target, maximum in checks.get("max_cps", {}).items():
        duration = float(case["input"]["duration_seconds"])
        chars = len(re.sub(r"\s+", "", _text(candidate, target)))
        if duration <= 0 or chars / duration > float(maximum):
            errors.append({"code": "readability_cps_exceeded", "detail": f"{target}:{chars / max(duration, 0.001):.2f}>{maximum}"})
    return {
        "case_id": str(case.get("case_id") or ""),
        "errors": errors,
        "human_review_required": list(case.get("human_review_required", [])),
        "automatic_quality_claim": False,
    }


def validate_quality_regression_suite(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    cases = value.get("cases") if isinstance(value, dict) else None
    default_input = value.get("default_input", {}) if isinstance(value, dict) else {}
    if not isinstance(cases, list):
        raise ValueError("quality regression suite는 cases 배열을 포함해야 합니다")
    if not isinstance(default_input, dict):
        raise ValueError("quality regression suite의 default_input은 객체여야 합니다")
    errors: list[str] = []
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            errors.append(f"{index}: case는 객체여야 합니다")
            continue
        case_id = str(case.get("case_id") or "")
        if not case_id or case_id in seen:
            errors.append(f"{index}: case_id가 없거나 중복됩니다")
            continue
        seen.add(case_id)
        raw_payload = case.get("input")
        if not isinstance(raw_payload, dict):
            errors.append(f"{case_id}: input 객체가 필요합니다")
            continue
        payload = {**default_input, **raw_payload}
        case = {**case, "input": payload}
        missing = REQUIRED_INPUT_FIELDS - set(payload)
        if missing:
            errors.append(f"{case_id}: input 필드 누락: {', '.join(sorted(missing))}")
        passing = case.get("passing_candidate")
        failing = case.get("regression_candidate")
        expected = set(case.get("expected_regression_codes", []))
        if not isinstance(passing, dict) or not isinstance(failing, dict) or not expected:
            errors.append(f"{case_id}: passing_candidate, regression_candidate, expected_regression_codes가 필요합니다")
            continue
        passing_report = evaluate_quality_case(case, passing)
        failing_report = evaluate_quality_case(case, failing)
        passing_codes = {item["code"] for item in passing_report["errors"]}
        failing_codes = {item["code"] for item in failing_report["errors"]}
        if passing_codes:
            errors.append(f"{case_id}: passing candidate가 자동 제약을 위반합니다: {', '.join(sorted(passing_codes))}")
        if not expected <= failing_codes:
            errors.append(f"{case_id}: regression candidate 탐지 누락: {', '.join(sorted(expected - failing_codes))}")
        results.append({"case_id": case_id, "passing_codes": sorted(passing_codes), "failing_codes": sorted(failing_codes), "human_review_required": passing_report["human_review_required"]})
    return {
        "status": "pass" if not errors else "fail",
        "suite": str(path),
        "cases": len(cases),
        "errors": errors,
        "results": results,
        "automatic_quality_claim": False,
    }
