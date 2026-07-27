from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .forensic_model import read_jsonl


SCOPES = {"global", "title", "scene", "speaker", "relationship", "temporary", "deprecated"}
APPROVALS = {"candidate", "approved", "rejected", "deprecated"}
REQUIRED = {"terminology_id", "japanese", "approved_korean", "scope", "approval_status", "evidence_refs", "approved_by"}


def initialize_terminology(path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"기존 용어집을 덮어쓰지 않습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8", newline="\n")
    return {"status": "terminology-ledger-created", "output": str(path), "entries": 0, "note": "빈 용어집은 자동 치환 규칙을 만들지 않습니다."}


def validate_terminology(path: Path) -> dict[str, Any]:
    records = read_jsonl(path)
    errors: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(records, 1):
        missing = REQUIRED - set(row)
        if missing:
            errors.append(f"{index}: 필수 필드 누락: {', '.join(sorted(missing))}")
            continue
        identifier = str(row.get("terminology_id", ""))
        if not identifier or identifier in seen:
            errors.append(f"{index}: terminology_id가 없거나 중복됩니다.")
        seen.add(identifier)
        if row.get("scope") not in SCOPES:
            errors.append(f"{identifier}: 잘못된 scope")
        if row.get("approval_status") not in APPROVALS:
            errors.append(f"{identifier}: 잘못된 approval_status")
        if not isinstance(row.get("evidence_refs"), list):
            errors.append(f"{identifier}: evidence_refs는 배열이어야 합니다.")
        if row.get("approval_status") == "approved":
            if not str(row.get("approved_korean", "")).strip() or not str(row.get("approved_by", "")).strip() or not row.get("evidence_refs"):
                errors.append(f"{identifier}: 승인 항목에는 한국어·승인자·근거가 필요합니다.")
    return {"status": "pass" if not errors else "fail", "entries": len(records), "approved_entries": sum(row.get("approval_status") == "approved" for row in records), "errors": errors, "automatic_substitution": False}


def terminology_conflicts(path: Path, output_path: Path) -> dict[str, Any]:
    records = read_jsonl(path)
    by_term: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        if row.get("approval_status") == "approved":
            by_term.setdefault((str(row.get("japanese", "")), str(row.get("scope", ""))), []).append(row)
    conflicts = []
    for (japanese, scope), values in by_term.items():
        variants = sorted({str(value.get("approved_korean", "")) for value in values})
        if len(variants) > 1:
            conflicts.append({"japanese": japanese, "scope": scope, "approved_korean_variants": variants, "terminology_ids": [value.get("terminology_id") for value in values], "status": "needs-human-resolution"})
    if output_path.exists():
        raise FileExistsError(f"기존 용어 충돌 보고서를 덮어쓰지 않습니다: {output_path}")
    value = {"schema_name": "translation-forensics/terminology-conflicts", "schema_version": "1", "created_at": datetime.now(timezone.utc).isoformat(), "entries": len(records), "conflicts": conflicts, "automatic_replacement_applied": False}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"status": "pass", "output": str(output_path), "conflicts": len(conflicts), "automatic_replacement_applied": False}
