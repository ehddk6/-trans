from __future__ import annotations

"""Evidence-safe semantic review primitives.

This module deliberately does not infer a speaker, action, or translation from
an ASR string.  It validates a reviewer-supplied record, finds explicit slot
conflicts, and keeps evidence families from being counted as independent.
"""

import json
from pathlib import Path
from typing import Any, Iterable


FRAME_SLOTS = (
    "speaker", "addressee", "speech_act", "polarity", "interrogative",
    "request_strength", "permission", "prohibition", "action", "actor",
    "target", "body_part", "location", "direction", "temporal_state",
    "completion_state", "result", "emotion", "sexual_semantic_class", "register",
)
SLOT_STATES = {"null", "unknown", "confirmed", "contradicted"}
HYPOTHESIS_STATUSES = {"candidate", "supported", "rejected", "dominant", "unresolved"}
SOURCE_FAMILIES = {
    "reference-japanese", "whisper-family", "independent-asr-family",
    "phonetic-analysis", "forced-alignment", "scene-context", "visual-reference",
    "human-listening", "korean-candidate",
}
CRITICAL_SLOTS = {
    "polarity", "interrogative", "request_strength", "permission", "prohibition",
    "actor", "target", "location", "completion_state", "sexual_semantic_class", "speaker",
}


def empty_semantic_frame(block_number: int) -> dict[str, Any]:
    """Return an explicitly incomplete frame; it is not a semantic decision."""
    frame: dict[str, Any] = {"block_number": block_number}
    frame.update({slot: {"value": None, "state": "null", "evidence_refs": []} for slot in FRAME_SLOTS})
    frame.update({"confirmed_slots": [], "uncertain_slots": [], "unsupported_slots": [], "review_status": "unreviewed"})
    return frame


def _slot(record: dict[str, Any], name: str) -> tuple[object, str]:
    value = record.get(name, {})
    if not isinstance(value, dict):
        return value, "invalid"
    return value.get("value"), str(value.get("state", ""))


def validate_semantic_frame(frame: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(frame.get("block_number"), int) or frame["block_number"] < 1:
        errors.append("block_number는 1 이상의 정수여야 합니다.")
    for name in FRAME_SLOTS:
        value, state = _slot(frame, name)
        if state not in SLOT_STATES:
            errors.append(f"{name}: state는 {sorted(SLOT_STATES)} 중 하나여야 합니다.")
            continue
        if state == "null" and value is not None:
            errors.append(f"{name}: null 상태의 value는 null이어야 합니다.")
        if state in {"confirmed", "contradicted"} and (value is None or value == ""):
            errors.append(f"{name}: {state} 상태에는 명시적인 value가 필요합니다.")
        slot = frame.get(name, {})
        if isinstance(slot, dict) and not isinstance(slot.get("evidence_refs", []), list):
            errors.append(f"{name}: evidence_refs는 배열이어야 합니다.")
    return errors


def frame_conflicts(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, str]]:
    """Compare only confirmed slots; unknown information is never a conflict."""
    conflicts: list[dict[str, str]] = []
    for name in FRAME_SLOTS:
        left_value, left_state = _slot(left, name)
        right_value, right_state = _slot(right, name)
        if left_state == right_state == "confirmed" and str(left_value).strip() != str(right_value).strip():
            conflicts.append({"slot": name, "severity": "critical" if name in CRITICAL_SLOTS else "major", "left": str(left_value), "right": str(right_value)})
    return conflicts


def requires_escalation(frames: Iterable[dict[str, Any]]) -> tuple[bool, list[dict[str, str]]]:
    values = list(frames)
    conflicts: list[dict[str, str]] = []
    for index, left in enumerate(values):
        for right in values[index + 1:]:
            conflicts.extend(item for item in frame_conflicts(left, right) if item["severity"] == "critical")
    return bool(conflicts), conflicts


def evidence_independence(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize support by source family, never by raw ASR pass count."""
    families: dict[str, list[str]] = {}
    invalid: list[str] = []
    for row in records:
        family = str(row.get("source_family", ""))
        ref = str(row.get("evidence_id", ""))
        if family not in SOURCE_FAMILIES:
            invalid.append(family or "<missing>")
            continue
        families.setdefault(family, []).append(ref)
    return {
        "independent_family_count": len(families),
        "families": {family: sorted(set(refs)) for family, refs in sorted(families.items())},
        "invalid_families": sorted(set(invalid)),
        "note": "같은 source_family의 복수 결과는 독립 근거로 중복 계산하지 않습니다.",
    }


def validate_hypotheses(records: Iterable[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: set[tuple[int, str]] = set()
    for row in records:
        try:
            number = int(row["block_number"])
        except (KeyError, TypeError, ValueError):
            errors.append("hypothesis ledger에 block_number가 없습니다.")
            continue
        hypothesis_id = str(row.get("hypothesis_id", ""))
        if not hypothesis_id:
            errors.append(f"block {number}: hypothesis_id가 없습니다.")
        if (number, hypothesis_id) in seen:
            errors.append(f"block {number}: hypothesis_id가 중복됩니다: {hypothesis_id}")
        seen.add((number, hypothesis_id))
        if row.get("status") not in HYPOTHESIS_STATUSES:
            errors.append(f"block {number}: 잘못된 hypothesis status입니다.")
        frame = row.get("semantic_frame")
        if not isinstance(frame, dict):
            errors.append(f"block {number}: semantic_frame 객체가 필요합니다.")
        else:
            errors.extend(f"block {number}: {error}" for error in validate_semantic_frame({"block_number": number, **frame}))
        for key in ("supported_by", "contradicted_by", "unsupported_specificity"):
            if not isinstance(row.get(key), list):
                errors.append(f"block {number}: {key} must be a list")
        if row.get("status") in {"supported", "dominant"} and not row.get("supported_by"):
            errors.append(f"block {number}: {row.get('status')} hypothesis needs explicit supporting evidence")
    return errors


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8", newline="\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: JSON 객체여야 합니다.")
        records.append(value)
    return records


def initialize_forensic_records(queue_path: Path, frame_path: Path, hypothesis_path: Path) -> dict[str, Any]:
    """Create honest, unreviewed records from a translation queue.

    Hypotheses are intentionally empty.  A candidate must be supplied by a
    reviewer with evidence; manufacturing two readings here would be evidence
    fabrication.
    """
    queue = read_jsonl(queue_path)
    numbers: list[int] = []
    for row in queue:
        try:
            number = int(row["block_number"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("translation queue의 block_number가 잘못되었습니다.") from exc
        numbers.append(number)
    if len(numbers) != len(set(numbers)):
        raise ValueError("translation queue의 block_number가 중복됩니다.")
    write_jsonl(frame_path, (empty_semantic_frame(number) for number in numbers))
    write_jsonl(hypothesis_path, [])
    return {
        "status": "forensic-record-template",
        "blocks": len(numbers),
        "semantic_frames": str(frame_path),
        "hypothesis_ledger": str(hypothesis_path),
        "note": "semantic frame은 unreviewed, hypothesis ledger는 빈 상태입니다. 의미 판정이나 검증 완료를 뜻하지 않습니다.",
    }


def validate_forensic_records(frame_path: Path, hypothesis_path: Path) -> dict[str, Any]:
    frames = read_jsonl(frame_path)
    hypotheses = read_jsonl(hypothesis_path)
    errors = [f"semantic frame: {error}" for frame in frames for error in validate_semantic_frame(frame)]
    errors.extend(f"hypothesis: {error}" for error in validate_hypotheses(hypotheses))
    by_block: dict[int, list[dict[str, Any]]] = {}
    for hypothesis in hypotheses:
        try:
            by_block.setdefault(int(hypothesis["block_number"]), []).append(hypothesis)
        except (KeyError, TypeError, ValueError):
            continue
    escalations: dict[str, list[dict[str, str]]] = {}
    for number, records in by_block.items():
        candidates = [row.get("semantic_frame", {}) for row in records if isinstance(row.get("semantic_frame"), dict)]
        escalate, conflicts = requires_escalation(candidates)
        if escalate:
            escalations[str(number)] = conflicts
    reviewed = sum(frame.get("review_status") == "reviewed" for frame in frames)
    unresolved = sum(frame.get("review_status") == "unresolved" for frame in frames)
    return {
        "status": "pass" if not errors else "fail",
        "frames": len(frames),
        "reviewed_frames": reviewed,
        "unresolved_frames": unresolved,
        "hypotheses": len(hypotheses),
        "critical_conflict_escalations": escalations,
        "errors": errors,
    }
