from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evidence_artifacts import validate_speaker_state
from .forensic_model import read_jsonl
from .terminology import validate_terminology


ENTRY_TYPES = {"register", "address_term", "proper_noun", "terminology", "translation_choice"}
SCOPES = {"title", "scene", "speaker", "relationship", "temporary"}
STATUSES = {"candidate", "confirmed", "unresolved", "deprecated"}
REQUIRED_FIELDS = {
    "consistency_id", "title_id", "entry_type", "scope", "key", "source_japanese",
    "korean", "status", "evidence_refs", "speaker_id", "addressee_id", "scene_ids",
    "start_block", "end_block", "last_changed_block", "note",
}


def initialize_consistency_ledger(path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"기존 translation consistency ledger를 덮어쓰지 않습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8", newline="\n")
    return {
        "status": "consistency-ledger-created",
        "output": str(path),
        "entries": 0,
        "automatic_application": False,
        "note": "confirmed 항목만 번역 큐의 참고 문맥으로 선택됩니다.",
    }


def _entry_errors(record: dict[str, Any], *, index: int, seen: set[str]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        return [f"{index}: 필수 필드 누락: {', '.join(sorted(missing))}"]
    identifier = str(record.get("consistency_id") or "").strip()
    if not identifier or identifier in seen:
        errors.append(f"{index}: consistency_id가 없거나 중복됩니다")
    seen.add(identifier)
    if not str(record.get("title_id") or "").strip():
        errors.append(f"{identifier}: title_id가 필요합니다")
    if record.get("entry_type") not in ENTRY_TYPES:
        errors.append(f"{identifier}: 지원하지 않는 entry_type")
    if record.get("scope") not in SCOPES:
        errors.append(f"{identifier}: 지원하지 않는 scope")
    if record.get("status") not in STATUSES:
        errors.append(f"{identifier}: 지원하지 않는 status")
    if not isinstance(record.get("scene_ids"), list) or not all(isinstance(value, str) for value in record.get("scene_ids", [])):
        errors.append(f"{identifier}: scene_ids는 문자열 배열이어야 합니다")
    if not isinstance(record.get("evidence_refs"), list) or not all(isinstance(value, str) and value.strip() for value in record.get("evidence_refs", [])):
        errors.append(f"{identifier}: evidence_refs는 빈 값 없는 문자열 배열이어야 합니다")
    for field in ("start_block", "end_block", "last_changed_block"):
        value = record.get(field)
        if value is not None and (not isinstance(value, int) or value < 1):
            errors.append(f"{identifier}: {field}는 양의 정수 또는 null이어야 합니다")
    start, end = record.get("start_block"), record.get("end_block")
    if isinstance(start, int) and isinstance(end, int) and end < start:
        errors.append(f"{identifier}: end_block이 start_block보다 빠릅니다")
    if record.get("status") == "confirmed":
        if not str(record.get("korean") or "").strip() or not record.get("evidence_refs"):
            errors.append(f"{identifier}: confirmed 항목에는 한국어 값과 근거가 필요합니다")
    return errors


def _conflict_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(record.get("entry_type") or ""),
        str(record.get("scope") or ""),
        str(record.get("key") or ""),
        str(record.get("speaker_id") or ""),
        str(record.get("addressee_id") or ""),
    )


def consistency_conflicts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("status") == "confirmed":
            grouped.setdefault(_conflict_key(record), []).append(record)
    conflicts: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        korean_values = sorted({str(value.get("korean") or "") for value in values})
        if len(korean_values) > 1:
            conflicts.append({
                "entry_type": key[0],
                "scope": key[1],
                "key": key[2],
                "speaker_id": key[3],
                "addressee_id": key[4],
                "entry_ids": sorted(str(value.get("consistency_id") or "") for value in values),
                "korean_variants": korean_values,
                "status": "needs-human-resolution",
            })
    return conflicts


def validate_consistency_ledger(path: Path) -> dict[str, Any]:
    records = read_jsonl(path)
    seen: set[str] = set()
    errors = [error for index, record in enumerate(records, 1) for error in _entry_errors(record, index=index, seen=seen)]
    conflicts = consistency_conflicts(records) if not errors else []
    return {
        "status": "pass" if not errors else "fail",
        "entries": len(records),
        "confirmed_entries": sum(record.get("status") == "confirmed" for record in records),
        "conflicts": conflicts,
        "conflict_count": len(conflicts),
        "errors": errors,
        "automatic_application": False,
    }


def read_consistency_ledger(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = validate_consistency_ledger(path)
    if report["status"] != "pass":
        raise ValueError("translation consistency ledger 검증 실패: " + "; ".join(report["errors"][:5]))
    return read_jsonl(path), report


def terminology_consistency_records(path: Path, *, title_id: str) -> list[dict[str, Any]]:
    """Adapt the existing approved terminology ledger without copying it to disk."""
    report = validate_terminology(path)
    if report["status"] != "pass":
        raise ValueError("terminology ledger 검증 실패: " + "; ".join(report["errors"][:5]))
    records: list[dict[str, Any]] = []
    for item in read_jsonl(path):
        if item.get("approval_status") != "approved":
            continue
        scope = str(item.get("scope") or "title")
        if scope == "global":
            scope = "title"
        if scope not in SCOPES:
            continue
        identifier = str(item.get("terminology_id") or "").strip()
        if not identifier:
            continue
        records.append({
            "consistency_id": f"terminology:{identifier}",
            "title_id": title_id,
            "entry_type": "terminology",
            "scope": scope,
            "key": str(item.get("japanese") or ""),
            "source_japanese": str(item.get("japanese") or ""),
            "korean": str(item.get("approved_korean") or ""),
            "status": "confirmed",
            "evidence_refs": list(item.get("evidence_refs") or []),
            "speaker_id": str(item.get("speaker_id") or ""),
            "addressee_id": str(item.get("addressee_id") or ""),
            "scene_ids": list(item.get("scene_ids") or []),
            "start_block": item.get("start_block"),
            "end_block": item.get("end_block"),
            "last_changed_block": item.get("last_changed_block"),
            "note": f"approved terminology ledger: {identifier}",
        })
    return records


def speaker_context_by_scene(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    validation = validate_speaker_state(path)
    if validation["status"] != "pass":
        raise ValueError("speaker-state 검증 실패: " + "; ".join(validation["errors"][:5]))
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    records = value if isinstance(value, list) else value.get("scenes", []) if isinstance(value, dict) else []
    if not isinstance(records, list):
        raise ValueError("speaker-state는 JSON 배열 또는 scenes 배열을 가진 객체여야 합니다")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        scene_id = str(record.get("scene_id") or "").strip()
        if not scene_id or scene_id in result:
            continue
        confidence = str(record.get("speaker_confidence") or "unknown")
        review_status = str(record.get("review_status") or "unreviewed")
        trusted = confidence == "confirmed" or review_status == "reviewed"
        result[scene_id] = {
            "scene_id": scene_id,
            "speaker_id": str(record.get("current_speaker") or "unknown"),
            "addressee_id": str(record.get("addressee") or "unknown"),
            "relationship": str(record.get("relationship") or "unknown"),
            "trusted": trusted,
            "source_status": review_status,
        }
    return result


def _matches_scope(record: dict[str, Any], *, block_number: int, scene_ids: set[str], speaker_context: list[dict[str, Any]]) -> bool:
    start, end = record.get("start_block"), record.get("end_block")
    if isinstance(start, int) and block_number < start:
        return False
    if isinstance(end, int) and block_number > end:
        return False
    scope = str(record.get("scope") or "")
    if scope == "title":
        return True
    if scope == "temporary":
        return True
    if scope == "scene":
        return bool(scene_ids & set(record.get("scene_ids") or []))
    trusted = [item for item in speaker_context if item.get("trusted")]
    if scope == "speaker":
        speaker = str(record.get("speaker_id") or "")
        return bool(speaker and any(item.get("speaker_id") == speaker for item in trusted))
    if scope == "relationship":
        speaker = str(record.get("speaker_id") or "")
        addressee = str(record.get("addressee_id") or "")
        return bool(
            speaker and addressee
            and any(item.get("speaker_id") == speaker and item.get("addressee_id") == addressee for item in trusted)
        )
    return False


def _compact(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "consistency_id": str(record.get("consistency_id") or ""),
        "entry_type": str(record.get("entry_type") or ""),
        "scope": str(record.get("scope") or ""),
        "key": str(record.get("key") or ""),
        "source_japanese": str(record.get("source_japanese") or ""),
        "korean": str(record.get("korean") or ""),
        "evidence_refs": list(record.get("evidence_refs") or []),
        "last_changed_block": record.get("last_changed_block"),
    }


def select_consistency_context(
    records: list[dict[str, Any]],
    *,
    block_number: int,
    scene_ids: list[str],
    speaker_context: list[dict[str, Any]],
) -> dict[str, Any]:
    relevant = [
        record for record in records
        if _matches_scope(record, block_number=block_number, scene_ids=set(scene_ids), speaker_context=speaker_context)
    ]
    active_conflicts = consistency_conflicts(relevant)
    conflicting_ids = {identifier for conflict in active_conflicts for identifier in conflict["entry_ids"]}
    applied = [
        _compact(record) for record in relevant
        if record.get("status") == "confirmed" and str(record.get("consistency_id") or "") not in conflicting_ids
    ]
    unresolved = [
        _compact(record) for record in relevant
        if record.get("status") in {"candidate", "unresolved"}
    ]
    return {
        "status": "available",
        "applied_entries": sorted(applied, key=lambda item: item["consistency_id"]),
        "unresolved_entries": sorted(unresolved, key=lambda item: item["consistency_id"]),
        "conflicts": active_conflicts,
        "speaker_context": speaker_context,
        "human_final_evidence": False,
    }
