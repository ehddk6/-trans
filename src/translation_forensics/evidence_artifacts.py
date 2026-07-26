from __future__ import annotations

"""Validation for reviewer-authored evidence artifacts.

These checks validate provenance and completeness only. They do not infer
identities, semantic slots, or translations from diarization, ASR, or timing.
"""

import csv
import json
from pathlib import Path
from typing import Any


SPEAKER_STATE_FIELDS = (
    "scene_id", "participants", "current_speaker", "previous_speaker",
    "addressee", "speaker_confidence", "relationship", "register_by_speaker",
    "address_terms", "current_action_by_participant", "question_owner",
    "expected_responder",
)
ALIGNMENT_FIELDS = (
    "token", "start", "end", "confidence", "candidate_source",
    "block_overlap", "boundary_warning",
)
BACKTRANSLATION_FIELDS = ("block_number", "status", "difference_types", "review_status")
BACKTRANSLATION_STATUSES = {
    "equivalent", "acceptable-ellipsis", "meaning-loss", "meaning-addition",
    "meaning-flip", "unresolved",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_speaker_state(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    records = value if isinstance(value, list) else value.get("scenes", []) if isinstance(value, dict) else []
    errors: list[str] = []
    if not isinstance(records, list):
        errors.append("speaker-state must be a JSON list or an object with a scenes list")
        records = []
    seen: set[str] = set()
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            errors.append(f"scene {index}: record must be an object")
            continue
        missing = [field for field in SPEAKER_STATE_FIELDS if field not in record]
        if missing:
            errors.append(f"scene {index}: missing fields: {', '.join(missing)}")
        scene_id = str(record.get("scene_id", ""))
        if not scene_id:
            errors.append(f"scene {index}: scene_id is required")
        elif scene_id in seen:
            errors.append(f"scene {index}: duplicate scene_id {scene_id}")
        seen.add(scene_id)
        confidence = record.get("speaker_confidence")
        if confidence not in {"confirmed", "unknown", "low", "medium", "high", None}:
            errors.append(f"scene {index}: invalid speaker_confidence")
        if record.get("current_speaker") not in {None, "", "unknown"} and not record.get("identity_evidence_refs"):
            errors.append(f"scene {index}: named current_speaker requires identity_evidence_refs")
    return {"status": "pass" if not errors else "fail", "scenes": len(records), "errors": errors}


def validate_alignment_evidence(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    errors = [f"missing columns: {', '.join(sorted(set(ALIGNMENT_FIELDS) - fields))}"]
    if set(ALIGNMENT_FIELDS) <= fields:
        errors = []
    for index, row in enumerate(rows, 2):
        try:
            start, end = float(row["start"]), float(row["end"])
            if end < start:
                raise ValueError
        except (KeyError, ValueError, TypeError):
            errors.append(f"row {index}: start/end must be ordered numbers")
        if row.get("boundary_warning") not in {"", "none", "warning", "unresolved"}:
            errors.append(f"row {index}: invalid boundary_warning")
    return {"status": "pass" if not errors else "fail", "rows": len(rows), "errors": errors,
            "note": "Alignment timing is boundary evidence, not a semantic decision."}


def validate_backtranslation_check(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    errors = [f"missing columns: {', '.join(sorted(set(BACKTRANSLATION_FIELDS) - fields))}"]
    if set(BACKTRANSLATION_FIELDS) <= fields:
        errors = []
    failures = 0
    for index, row in enumerate(rows, 2):
        if row.get("status") not in BACKTRANSLATION_STATUSES:
            errors.append(f"row {index}: invalid status")
        if row.get("review_status") not in {"reviewed", "unresolved"}:
            errors.append(f"row {index}: invalid review_status")
        if row.get("status") in {"meaning-flip", "meaning-addition"}:
            failures += 1
    return {"status": "pass" if not errors and not failures else "fail", "rows": len(rows),
            "blocking_differences": failures, "errors": errors}
