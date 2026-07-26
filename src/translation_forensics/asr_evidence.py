from __future__ import annotations

import csv
import re
from difflib import SequenceMatcher
from pathlib import Path


ASR_FIELDS = ["scene_id", "model", "profile", "audio", "text", "avg_logprob", "no_speech_prob", "language_probability", "error"]
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
    return errors


def read_asr_candidates(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors = validate_asr_rows(rows)
    if errors:
        raise ValueError("ASR CSV 규격 오류: " + "; ".join(errors[:5]))
    return rows
