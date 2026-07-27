from __future__ import annotations

from pathlib import Path

from .forensic_model import read_jsonl


KINDS = {"scene", "translation", "error", "style-policy"}
SCOPES = {"global", "title", "scene", "speaker", "relationship", "temporary", "deprecated"}


def initialize_memory_ledger(path: Path, *, kind: str) -> dict[str, object]:
    if kind not in KINDS:
        raise ValueError(f"지원하지 않는 memory ledger 종류: {kind}")
    if path.exists():
        raise FileExistsError(f"기존 memory ledger를 덮어쓰지 않습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8", newline="\n")
    return {"status": "memory-ledger-created", "kind": kind, "output": str(path), "entries": 0, "automatic_application": False}


def validate_memory_ledger(path: Path, *, kind: str) -> dict[str, object]:
    if kind not in KINDS:
        raise ValueError(f"지원하지 않는 memory ledger 종류: {kind}")
    records = read_jsonl(path)
    errors: list[str] = []; identifiers: set[str] = set()
    for index, row in enumerate(records, 1):
        identifier = str(row.get("memory_id", ""))
        if not identifier or identifier in identifiers:
            errors.append(f"{index}: memory_id가 없거나 중복됩니다.")
        identifiers.add(identifier)
        if row.get("memory_kind") != kind:
            errors.append(f"{identifier}: memory_kind이 {kind}과 일치하지 않습니다.")
        if row.get("scope") not in SCOPES:
            errors.append(f"{identifier}: 범위가 잘못되었습니다.")
        if not isinstance(row.get("evidence_refs"), list):
            errors.append(f"{identifier}: evidence_refs는 배열이어야 합니다.")
        if row.get("scope") == "global":
            if not str(row.get("approved_by", "")).strip() or not str(row.get("approval_status", "")).strip() or not row.get("evidence_refs"):
                errors.append(f"{identifier}: global 항목에는 승인자·상태·근거가 필요합니다.")
        if row.get("automatic_application") is True:
            errors.append(f"{identifier}: memory ledger는 자동 적용할 수 없습니다.")
    return {"status": "pass" if not errors else "fail", "kind": kind, "entries": len(records), "errors": errors, "automatic_application": False}
