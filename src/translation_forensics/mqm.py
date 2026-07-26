from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


MQM_ERROR_TYPES = {
    "mistranslation", "omission", "addition", "unsupported-specificity", "polarity-flip",
    "question-statement-flip", "permission-prohibition-flip", "temporal-result-error",
    "speaker-error", "addressee-error", "actor-error", "target-error", "relationship-error",
    "action-error", "location-error", "body-part-invention", "sexual-meaning-loss",
    "sexual-meaning-intensification", "coercion-invention", "emotion-invention",
    "broken-question-answer", "scene-state-conflict", "register-drift", "address-term-drift",
    "duplicate-meaning", "premature-information", "delayed-information", "translationese",
    "unnatural-order", "fragment", "overexplicit-pronoun", "awkward-ending", "inconsistent-register",
    "excessive-reading-load", "bad-line-break", "three-plus-lines", "boundary-leak", "empty-subtitle",
    "structural-mismatch", "evidence-overclaim", "false-verification-status", "unresolved-hidden",
    "dependent-evidence-double-counted",
}
SEVERITIES = {"critical", "major", "minor", "neutral"}
REQUIRED_COLUMNS = ("block_number", "error_type", "severity", "evidence_refs", "review_status")


def validate_mqm_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(set(REQUIRED_COLUMNS) - fields)
        rows = list(reader)
    errors = [f"필수 열 누락: {', '.join(missing)}"] if missing else []
    critical = 0
    for index, row in enumerate(rows, 2):
        try:
            if int(row.get("block_number", "")) < 1:
                raise ValueError
        except ValueError:
            errors.append(f"행 {index}: block_number는 1 이상의 정수여야 합니다.")
        if row.get("error_type") not in MQM_ERROR_TYPES:
            errors.append(f"행 {index}: 알 수 없는 error_type입니다: {row.get('error_type')!r}")
        if row.get("severity") not in SEVERITIES:
            errors.append(f"행 {index}: 알 수 없는 severity입니다: {row.get('severity')!r}")
        if row.get("severity") == "critical":
            critical += 1
        if row.get("review_status") not in {"reviewed", "unresolved", "dismissed"}:
            errors.append(f"행 {index}: review_status는 reviewed/unresolved/dismissed여야 합니다.")
    return {"status": "pass" if not errors else "fail", "rows": len(rows), "critical_errors": critical, "errors": errors}
