from __future__ import annotations

import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


ASR_FIELDS = ["scene_id", "model", "profile", "audio", "text", "avg_logprob", "no_speech_prob", "language_probability", "error"]
OPTIONAL_ASR_FIELDS = ("source_family", "semantic_slots_json")
PROFILES = ("original_unbiased", "dialogue_unbiased", "original_no_vad", "original_prompted")
BASE_PROFILES = {"original_unbiased", "dialogue_unbiased"}


def normalize_japanese(text: str) -> str:
    return re.sub(r"[\s。、！？!?・…ー]+", "", text or "")


def text_similarity(left: str, right: str) -> float:
    a, b = normalize_japanese(left), normalize_japanese(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def as_float(value: object, default: float | None = None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def declared_slot_conflicts(rows: list[dict[str, str]]) -> list[str]:
    """Return explicit confirmed-slot disagreements, never inferred ones.

    ``semantic_slots_json`` is reviewer-supplied provenance, not an NLP
    prediction. A conflict in a critical slot is therefore a reason to
    escalate for review, not a reason to choose the majority ASR text.
    """
    critical = {"polarity", "interrogative", "request_strength", "permission", "prohibition", "actor", "target", "location", "completion_state", "sexual_semantic_class", "speaker"}
    values: dict[str, set[str]] = {}
    for row in rows:
        raw = row.get("semantic_slots_json", "")
        if not raw:
            continue
        try:
            slots = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(slots, dict):
            continue
        for name, claim in slots.items():
            if name not in critical or not isinstance(claim, dict):
                continue
            if claim.get("state") == "confirmed" and claim.get("value") not in {None, ""}:
                values.setdefault(name, set()).add(str(claim["value"]).strip())
    return sorted(name for name, candidates in values.items() if len(candidates) > 1)


def should_escalate(rows: list[dict[str, str]], threshold: float = 0.82) -> tuple[bool, str]:
    by_profile = {row.get("profile", ""): row for row in rows}
    left = by_profile.get("original_unbiased", {})
    right = by_profile.get("dialogue_unbiased", {})
    reasons: list[str] = []
    if not left.get("text", "") or not right.get("text", ""):
        reasons.append("empty")
    similarity = text_similarity(left.get("text", ""), right.get("text", ""))
    if similarity < threshold:
        reasons.append(f"disagree:{similarity:.2f}")
    for row in (left, right):
        logprob = as_float(row.get("avg_logprob"))
        no_speech = as_float(row.get("no_speech_prob"))
        if logprob is not None and logprob < -1.0:
            reasons.append("low_logprob")
        if no_speech is not None and no_speech > 0.60:
            reasons.append("high_no_speech")
        if row.get("error"):
            reasons.append("error")
    reasons.extend(f"semantic_conflict:{slot}" for slot in declared_slot_conflicts(rows))
    return bool(reasons), ",".join(sorted(set(reasons))) or "stable"


def classify_scene(rows: list[dict[str, str]]) -> str:
    by_profile = {row.get("profile", ""): row for row in rows}
    base = [by_profile.get(name, {}) for name in BASE_PROFILES]
    texts = [row.get("text", "").strip() for row in base]
    if all(not text for text in texts) and all((row.get("error") or "") for row in base if row):
        return "unresolved"
    if texts and all(not text for text in texts):
        return "nonverbal"
    if len(texts) == 2 and texts[0] and texts[1]:
        similarity = text_similarity(texts[0], texts[1])
        if similarity >= 0.92:
            return "stable"
        if similarity >= 0.75:
            return "normalized"
    if any(row.get("profile") in {"original_no_vad", "original_prompted"} and row.get("text", "").strip() for row in rows):
        return "context-resolved"
    return "unresolved"


def evidence_weight(profile: str) -> str:
    if profile == "original_prompted":
        return "low_prompt_influenced"
    if profile in BASE_PROFILES or profile == "original_no_vad":
        return "candidate_not_independent"
    return "unknown"


def validate_asr_rows(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    for index, row in enumerate(rows, 2):
        missing = [field for field in ASR_FIELDS if field not in row]
        if missing:
            errors.append(f"행 {index}: 필수 열 누락 {missing}")
        if row.get("profile") not in PROFILES:
            errors.append(f"행 {index}: 알 수 없는 profile {row.get('profile')!r}")
        family = row.get("source_family", "whisper-family") or "whisper-family"
        if family not in {"whisper-family", "independent-asr-family", "human-listening", "reference-japanese"}:
            errors.append(f"row {index}: unsupported source_family {family!r}")
        raw_slots = row.get("semantic_slots_json", "")
        if raw_slots:
            try:
                decoded = json.loads(raw_slots)
                if not isinstance(decoded, dict):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                errors.append(f"row {index}: semantic_slots_json must be an object")
    return errors


def read_asr_candidates(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        # Existing runner output has no family column.  Its profiles are one
        # Whisper family, never several independent pieces of evidence.
        row.setdefault("source_family", "whisper-family")
        row.setdefault("semantic_slots_json", "")
    errors = validate_asr_rows(rows)
    if errors:
        raise ValueError("ASR CSV 규격 오류: " + "; ".join(errors[:5]))
    return rows
