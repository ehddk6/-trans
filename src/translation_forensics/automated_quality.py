from __future__ import annotations

"""Deterministic quality gates used when a process is run without human review.

The checks intentionally validate invariants that can be proven locally.  They
do not pretend to establish literary quality; low-confidence decisions remain
machine-uncertain and use the source-faithful draft as their safe fallback.
"""

import re
from typing import Any, Iterable, Mapping

from .srt import has_japanese


_JAPANESE_NEGATION_RE = re.compile(r"(?:ない|ません|ぬ|ず|なかった|じゃない|ではない|いや|だめ|駄目)")
_KOREAN_NEGATION_RE = re.compile(
    r"(?:^안\s|\s안\s|^못\s|\s못\s|없(?:다|어|어요|습니다)|아니|아닌|아닙|않|말아|말고|싫|금지|안돼|안 돼|하지 마)"
)
_JAPANESE_QUESTION_RE = re.compile(r"[?？]|(?:か|の)\s*$")
_KOREAN_QUESTION_RE = re.compile(r"[?？]|(?:까|나요|어요|습니까|는지|지)\s*[?？]?$", re.IGNORECASE)
_NUMBER_RE = re.compile(
    r"(?P<value>\d+(?:[.,:]\d+)?|[一二三四五六七八九十百千万億]+)\s*"
    r"(?P<unit>人|名|分|秒|時間|円|%|％|回|日|명|분|초|시간|원|회|일)?"
)
_EMPTY_RE = re.compile(r"^[\s.…・･ー\-~～!?！？]+$")
_KANJI_DIGITS = {
    "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
    "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
}
_UNIT_GROUPS = {
    "人": "person", "名": "person", "명": "person",
    "分": "minute", "분": "minute", "秒": "second", "초": "second",
    "時間": "hour", "시간": "hour", "円": "currency", "원": "currency",
    "%": "percent", "％": "percent", "回": "count", "회": "count",
    "日": "day", "일": "day",
}
_JAPANESE_STOP_RE = re.compile(r"(?:やめて|やめろ|やめなさい|止めて|止めろ)")
_KOREAN_STOP_RE = re.compile(r"(?:그만|멈춰|중단|하지\s*마)")
_JAPANESE_CONTINUE_RE = re.compile(r"(?:続けて|続けろ|続けなさい|もっと)")
_KOREAN_CONTINUE_RE = re.compile(r"(?:계속|이어가|더(?:\s|[.!?！？…]|$))")
_JAPANESE_REFUSAL_RE = re.compile(r"(?:できない|無理|だめ|駄目|いや)")
_KOREAN_REFUSAL_RE = re.compile(r"(?:못|불가|안\s*돼|안돼|싫|아니)")
_JAPANESE_PERMISSION_RE = re.compile(
    r"(?:ても|でも)\s*(?:いい|良い|よい|構わない|かまわない)|"
    r"(?:て|で)\s*いい|(?:いい|良い|よい)\s*よ\s*[。.!！…]*$|許(?:す|して)"
)
_KOREAN_PERMISSION_RE = re.compile(
    r"(?:(?:아도|어도|해도)\s*(?:돼|괜찮|좋)|허락|가능|"
    r"(?:^|\s)(?:돼|괜찮아|괜찮아요|좋아|좋아요)\s*[.!?！？…]*$)"
)
_JAPANESE_REQUEST_COMMAND_RE = re.compile(
    r"(?:ください|下さい|くれ|ちょうだい|なさい)\s*[。.!！…]*$|"
    r"^(?:やって|して|入れて|出して|見て|来て|聞いて|待って|早く|ゆっくり)\s*[。.!！…]*$"
)
_KOREAN_REQUEST_COMMAND_RE = re.compile(
    r"(?:해(?:\s*줘)?|해라|하세요|줘|주세요|넣어|빼|봐|보세요|와|오세요|가|가세요|"
    r"기다려|기다리세요|빨리|천천히)\s*[.!?！？…]*$"
)
_JAPANESE_DEICTIC_LOCATION_RE = re.compile(
    r"(?:ここ|そこ|あそこ|こちら|そちら|あちら|こっち|そっち|あっち|どこ|どちら|どっち)"
)
_KOREAN_DEICTIC_LOCATION_RE = re.compile(r"(?:여기|거기|저기|이쪽|그쪽|저쪽|어디)")
# Compression may be harmless for a short breath/moan, but kana-only is much
# too broad: lexical cues such as ここ, もっと, and やって are also kana-only.
# Keep this deliberately closed to non-lexical vowel/nasal vocalizations.
_SHORT_VOCALIZATION_RE = re.compile(
    r"^(?:[あぁ]+っ?|[うぅ]+っ?|[えぇ]+っ?|[おぉ]+っ?|ん+っ?|"
    r"は[ぁあ]+|ふ[ぅう]+|ひ[ぃい]+|く[ぅう]+|"
    r"[アァ]+ッ?|[ウゥ]+ッ?|[エェ]+ッ?|[オォ]+ッ?|ン+ッ?|"
    r"ハ[ァア]+|フ[ゥウ]+|ヒ[ィイ]+|ク[ゥウ]+)$"
)
_VOCALIZATION_PUNCTUATION_RE = re.compile(r"[\s\u3000。、！？!?…・･ー\-~～]+")
_RECOVERY_CLASSIFICATIONS = frozenset({"RELIABLE", "FUNCTIONAL_RECOVERY", "UNRESOLVED"})


def _numbers(value: str) -> tuple[str, ...]:
    normalized = value.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    tokens: list[str] = []
    for match in _NUMBER_RE.finditer(normalized):
        raw_value = match.group("value")
        value_token = _KANJI_DIGITS.get(raw_value, raw_value)
        unit = _UNIT_GROUPS.get(match.group("unit") or "", "number")
        tokens.append(f"{value_token}:{unit}")
    return tuple(tokens)


def _question(value: str, *, japanese: bool) -> bool:
    return bool((_JAPANESE_QUESTION_RE if japanese else _KOREAN_QUESTION_RE).search(value.strip()))


def _is_short_vocalization(value: str) -> bool:
    normalized = _VOCALIZATION_PUNCTUATION_RE.sub("", value)
    return 1 <= len(normalized) <= 5 and bool(_SHORT_VOCALIZATION_RE.fullmatch(normalized))


def audit_translation_decision(
    *,
    unit_id: str,
    source_japanese: str,
    source_faithful_korean: str,
    viewer_natural_korean: str,
    source_quality_status: str = "suspect",
    confidence: str = "low",
    existing_reasons: Iterable[str] = (),
    recovery_classification: str = "RELIABLE",
    recovery_basis: Iterable[str] = (),
) -> dict[str, Any]:
    """Return a machine-verifiable decision without requiring human approval."""

    source = str(source_japanese or "").strip()
    faithful = str(source_faithful_korean or "").strip()
    natural = str(viewer_natural_korean or "").strip()
    reasons = [str(value) for value in existing_reasons]
    recovery = str(recovery_classification or "RELIABLE").upper()
    if recovery not in _RECOVERY_CLASSIFICATIONS:
        raise ValueError(f"unsupported recovery classification: {recovery_classification}")
    hard_failures: list[str] = []
    faithful_hard_failures: list[str] = []
    natural_hard_failures: list[str] = []
    warnings: list[str] = []

    if not faithful or _EMPTY_RE.fullmatch(faithful):
        faithful_hard_failures.append("empty_source_faithful_translation")
    if not natural or _EMPTY_RE.fullmatch(natural):
        natural_hard_failures.append("empty_viewer_translation")
    if has_japanese(faithful):
        faithful_hard_failures.append("japanese_residual")
    if has_japanese(natural):
        natural_hard_failures.append("japanese_residual")
    if source and _question(source, japanese=True):
        if not _question(faithful, japanese=False):
            faithful_hard_failures.append("question_force_not_preserved")
        if not _question(natural, japanese=False):
            natural_hard_failures.append("question_force_not_preserved")
    source_negative = bool(_JAPANESE_NEGATION_RE.search(source))
    faithful_negative = bool(_KOREAN_NEGATION_RE.search(faithful))
    natural_negative = bool(_KOREAN_NEGATION_RE.search(natural))
    if source_negative and not faithful_negative:
        faithful_hard_failures.append("negation_not_preserved")
    if source_negative and not natural_negative:
        natural_hard_failures.append("negation_not_preserved")
    if not source_negative and faithful_negative:
        warnings.append("possible_added_negation")
    if not source_negative and natural_negative:
        warnings.append("possible_added_negation")
    source_numbers = _numbers(source)
    if source_numbers and not all(token in _numbers(faithful) for token in source_numbers):
        faithful_hard_failures.append("numeric_token_not_preserved")
    if source_numbers and not all(token in _numbers(natural) for token in source_numbers):
        natural_hard_failures.append("numeric_token_not_preserved")
    if _JAPANESE_STOP_RE.search(source) and not _KOREAN_STOP_RE.search(faithful):
        faithful_hard_failures.append("stop_command_not_preserved")
    if _JAPANESE_STOP_RE.search(source) and not _KOREAN_STOP_RE.search(natural):
        natural_hard_failures.append("stop_command_not_preserved")
    if _JAPANESE_CONTINUE_RE.search(source) and not _KOREAN_CONTINUE_RE.search(faithful):
        faithful_hard_failures.append("continue_command_not_preserved")
    if _JAPANESE_CONTINUE_RE.search(source) and not _KOREAN_CONTINUE_RE.search(natural):
        natural_hard_failures.append("continue_command_not_preserved")
    if _JAPANESE_REFUSAL_RE.search(source) and not _KOREAN_REFUSAL_RE.search(faithful):
        faithful_hard_failures.append("refusal_not_preserved")
    if _JAPANESE_REFUSAL_RE.search(source) and not _KOREAN_REFUSAL_RE.search(natural):
        natural_hard_failures.append("refusal_not_preserved")
    if _JAPANESE_PERMISSION_RE.search(source) and not _KOREAN_PERMISSION_RE.search(faithful):
        faithful_hard_failures.append("permission_not_preserved")
    if _JAPANESE_PERMISSION_RE.search(source) and not _KOREAN_PERMISSION_RE.search(natural):
        natural_hard_failures.append("permission_not_preserved")
    if _JAPANESE_REQUEST_COMMAND_RE.search(source) and not _KOREAN_REQUEST_COMMAND_RE.search(faithful):
        faithful_hard_failures.append("request_command_not_preserved")
    if _JAPANESE_REQUEST_COMMAND_RE.search(source) and not _KOREAN_REQUEST_COMMAND_RE.search(natural):
        natural_hard_failures.append("request_command_not_preserved")
    if _JAPANESE_DEICTIC_LOCATION_RE.search(source) and not _KOREAN_DEICTIC_LOCATION_RE.search(faithful):
        faithful_hard_failures.append("deictic_location_not_preserved")
    if _JAPANESE_DEICTIC_LOCATION_RE.search(source) and not _KOREAN_DEICTIC_LOCATION_RE.search(natural):
        natural_hard_failures.append("deictic_location_not_preserved")
    hard_failures.extend([*faithful_hard_failures, *natural_hard_failures])

    semantic_guard_present = any(
        (
            _question(source, japanese=True),
            source_negative,
            bool(source_numbers),
            bool(_JAPANESE_STOP_RE.search(source)),
            bool(_JAPANESE_CONTINUE_RE.search(source)),
            bool(_JAPANESE_REFUSAL_RE.search(source)),
            bool(_JAPANESE_PERMISSION_RE.search(source)),
            bool(_JAPANESE_REQUEST_COMMAND_RE.search(source)),
            bool(_JAPANESE_DEICTIC_LOCATION_RE.search(source)),
        )
    )
    suppressed_warnings: list[str] = []
    if "high_compression_ratio" in reasons and _is_short_vocalization(source) and not semantic_guard_present:
        reasons = [reason for reason in reasons if reason != "high_compression_ratio"]
        suppressed_warnings.append("high_compression_ratio_short_vocalization")

    if source_quality_status == "unusable":
        warnings.append("unusable_source_quality")
    elif source_quality_status != "trusted":
        warnings.append("source_quality_not_independently_resolved")
    if recovery != "RELIABLE":
        warnings.append("recovery_classification_requires_machine_uncertain")
    if confidence == "low":
        warnings.append("low_model_confidence")
    if reasons:
        warnings.append("model_reported_uncertainty")

    faithful_hard_failures = list(dict.fromkeys(faithful_hard_failures))
    natural_hard_failures = list(dict.fromkeys(natural_hard_failures))
    hard_failures = list(dict.fromkeys(hard_failures))
    warnings = list(dict.fromkeys(warnings))
    reasons = list(dict.fromkeys([*reasons, *hard_failures, *warnings]))
    score = 1.0
    score -= min(0.75, 0.25 * len(hard_failures))
    score -= min(0.25, 0.05 * len(warnings))
    score = max(0.0, round(score, 3))
    status = "fallback" if hard_failures or warnings else "passed"
    return {
        "unit_id": unit_id,
        "status": status,
        "score": score,
        "hard_failures": hard_failures,
        "faithful_hard_failures": faithful_hard_failures,
        "natural_hard_failures": natural_hard_failures,
        "faithful_safe": not faithful_hard_failures,
        "natural_safe": not natural_hard_failures,
        "warnings": warnings,
        "suppressed_warnings": suppressed_warnings,
        "reasons": reasons,
        "question_preserved": not "question_force_not_preserved" in hard_failures,
        "negation_preserved": not "negation_not_preserved" in hard_failures,
        "numeric_tokens_preserved": not "numeric_token_not_preserved" in hard_failures,
        "stop_command_preserved": not "stop_command_not_preserved" in hard_failures,
        "continue_command_preserved": not "continue_command_not_preserved" in hard_failures,
        "refusal_preserved": not "refusal_not_preserved" in hard_failures,
        "permission_preserved": not "permission_not_preserved" in hard_failures,
        "request_command_preserved": not "request_command_not_preserved" in hard_failures,
        "deictic_location_preserved": not "deictic_location_not_preserved" in hard_failures,
        "source_quality_status": source_quality_status,
        "confidence": confidence,
        "recovery_classification": recovery,
        "recovery_basis": [str(value) for value in recovery_basis],
        "machine_evidence_id": f"automated-quality:{unit_id}",
    }


def audit_translation_decisions(
    units: Iterable[Mapping[str, Any]], decisions: Iterable[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Annotate decisions and return ``(decisions, per-unit audit records)``."""

    unit_map = {str(unit["unit_id"]): unit for unit in units}
    updated: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for raw in decisions:
        decision = dict(raw)
        unit_id = str(decision.get("unit_id") or "")
        unit = unit_map.get(unit_id)
        if unit is None:
            raise ValueError(f"automated quality received unknown unit_id: {unit_id}")
        record = audit_translation_decision(
            unit_id=unit_id,
            source_japanese=str(unit.get("source_japanese", unit.get("text_raw", ""))),
            source_faithful_korean=str(decision.get("source_faithful_korean") or ""),
            viewer_natural_korean=str(decision.get("viewer_natural_korean") or ""),
            source_quality_status=str(decision.get("source_quality_status") or unit.get("quality_status", "suspect")),
            confidence=str(decision.get("confidence") or "low"),
            existing_reasons=decision.get("review_required_reasons", ()),
            recovery_classification=str(decision.get("recovery_classification") or "RELIABLE"),
            recovery_basis=decision.get("recovery_basis", ()),
        )
        decision["automated_quality_status"] = record["status"]
        decision["automated_quality_score"] = record["score"]
        decision["automated_quality_reasons"] = record["reasons"]
        decision["automated_quality_evidence_id"] = record["machine_evidence_id"]
        decision["automated_quality_faithful_safe"] = record["faithful_safe"]
        decision["automated_quality_natural_safe"] = record["natural_safe"]
        decision["translation_status"] = "machine_verified" if record["status"] == "passed" else "machine_uncertain"
        updated.append(decision)
        records.append(record)
    return updated, records


__all__ = ["audit_translation_decision", "audit_translation_decisions"]
