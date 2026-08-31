from __future__ import annotations

"""Deterministic semantic gates used when a process is run without human review.

The checks deliberately separate concrete meaning failures from evidence and
confidence warnings. They do not score literary quality or rewrite a natural
viewer line into a literal baseline.
"""

import re
from collections import Counter
from typing import Any, Iterable, Mapping

from .evidence_bridge import source_evidence_reason_codes
from .srt import has_japanese


_JAPANESE_NEGATION_RE = re.compile(r"(?:ない|ません|なかった|じゃない|ではない|だめ|駄目)")
_KOREAN_NEGATION_RE = re.compile(
    r"(?:^안\s|\s안\s|^못(?:\s|[가-힣])|\s못(?:\s|[가-힣])|"
    r"없(?:다|어|어요|습니다|는|던|었|겠)|아니|아닌|아닙|아뇨|아냐|않|"
    r"말아|말고|지\s*마|싫|금지|안돼|안 돼|하지 마|몰랐|모르|모자라|부족|잖아)"
)
_JAPANESE_QUESTION_RE = re.compile(r"(?:[?？]|(?:か|の))\s*$")
_KOREAN_QUESTION_RE = re.compile(r"[?？]|(?:까|나요|어요|습니까|는지|지)\s*[?？]?$", re.IGNORECASE)
_ARABIC_NUMBER_RE = re.compile(
    r"(?P<value>\d+(?:[.,:]\d+)?)\s*"
    r"(?P<unit>人分|인분|사람\s*분|人|名|사람|명|時間|시간|秒|초|分|분|"
    r"円|원|%|％|回目|回|회|本目|本|번|번째|째|つ目|つ|개|日|일)?"
)
_JAPANESE_KANJI_NUMBER_RE = re.compile(
    r"(?P<value>[一二三四五六七八九十百千万億]+)\s*"
    r"(?P<unit>人分|人|名|時間|秒|分|円|%|％|回目|回|本目|本|番目|番|"
    r"つ目|つ|個目|個|日)"
)
_KOREAN_NUMBER_WORD_WITH_UNIT_RE = re.compile(
    r"(?<![가-힣])(?P<value>첫째|첫|하나|한|둘|두|셋|세|넷|네|다섯|여섯|"
    r"일곱|여덟|아홉|열|스무|일|이|삼|사|오|육|칠|팔|구|십)\s*"
    r"(?P<unit>인분|사람\s*분|사람|명|시간|초|분|원|회|번|번째|째|개|일)"
)
_KOREAN_BARE_CARDINAL_RE = re.compile(
    r"(?<![가-힣])(?P<value>하나|둘|셋|넷|다섯|여섯|일곱|여덟|아홉|열)(?:만|도|뿐)?"
)
_EMPTY_RE = re.compile(r"^[\s.…・･ー\-~～!?！？]+$")
_KANJI_DIGITS = {
    "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
    "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
}
_KOREAN_NUMBER_WORDS = {
    "첫": "1", "첫째": "1", "한": "1", "하나": "1",
    "두": "2", "둘": "2", "세": "3", "셋": "3", "네": "4", "넷": "4",
    "다섯": "5", "여섯": "6", "일곱": "7", "여덟": "8", "아홉": "9",
    "열": "10", "스무": "20", "일": "1", "이": "2", "삼": "3", "사": "4",
    "오": "5", "육": "6", "칠": "7", "팔": "8", "구": "9", "십": "10",
}
_UNIT_GROUPS = {
    "人分": "person", "인분": "person", "사람분": "person", "사람": "person",
    "人": "person", "名": "person", "명": "person",
    "分": "minute", "분": "minute", "秒": "second", "초": "second",
    "時間": "hour", "시간": "hour", "円": "currency", "원": "currency",
    "%": "percent", "％": "percent", "回": "count", "回目": "count",
    "本": "count", "本目": "count", "番": "count", "番目": "count",
    "つ": "count", "つ目": "count", "個": "count", "個目": "count",
    "회": "count", "번": "count", "번째": "count", "째": "count", "개": "count",
    "日": "day", "일": "day",
}
_JAPANESE_STOP_RE = re.compile(r"(?:やめて|やめろ|やめなさい|止めて|止めろ)")
_KOREAN_STOP_RE = re.compile(r"(?:그만|멈춰|중단|하지\s*마)")
_JAPANESE_CONTINUE_RE = re.compile(r"(?:続けて|続けろ|続けなさい|もっと)")
_KOREAN_CONTINUE_RE = re.compile(r"(?:계속|이어가|더(?:\s|[.!?！？…]|$))")
_JAPANESE_REFUSAL_RE = re.compile(r"(?:できない|無理|だめ|駄目|いや)")
_KOREAN_REFUSAL_RE = re.compile(r"(?:못|불가|안\s*돼|안돼|싫|아니|아뇨|아냐)")
_JAPANESE_PERMISSION_RE = re.compile(
    r"(?:ても|でも)\s*(?:いい|良い|よい|構わない|かまわない)|"
    r"(?:て|で)\s*いい|(?:いい|良い|よい)\s*よ\s*[。.!！…]*$|許(?:す|して)"
)
_KOREAN_PERMISSION_RE = re.compile(
    r"(?:(?:아도|어도|해도)\s*(?:돼|괜찮|좋)|허락|가능|"
    r"(?:^|\s)(?:돼|괜찮아|괜찮아요|좋아|좋아요)\s*[.!?！？…]*$)"
)
_JAPANESE_REQUEST_COMMAND_RE = re.compile(
    r"(?:ください|下さい|くれ|ちょうだい)\s*[。.!！…]*$|"
    r"^(?:やって|して|入れて|出して|見て|来て|聞いて|待って|早く|ゆっくり)\s*[。.!！…]*$"
)
_KOREAN_REQUEST_COMMAND_RE = re.compile(
    r"(?:해(?:\s*줘)?|해라|하세요|줘|주세요|넣어|빼|봐|보세요|와|오세요|가|가세요|"
    r"기다려|기다리세요|빨리|천천히|쉬어|쉬어요|[가-힣]+(?:세요|십시오)|잠깐만요)\s*[.!?！？…]*$"
)
_JAPANESE_DEICTIC_LOCATION_RE = re.compile(
    r"(?:ここ|そこ|あそこ|こちら|そちら|あちら|こっち|そっち|あっち|どこ|どちら|どっち)"
)
_KOREAN_DEICTIC_LOCATION_RE = re.compile(r"(?:여기|여긴|거기|거긴|저기|저긴|이쪽|그쪽|저쪽|어디)")
_JAPANESE_MOMENTARY_RE = re.compile(r"一瞬")
_KOREAN_MOMENTARY_RE = re.compile(r"(?:순간|잠깐|잠시|찰나|순식간)")
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


def _numeric_token(raw_value: str, raw_unit: str) -> str:
    value = _KANJI_DIGITS.get(raw_value, _KOREAN_NUMBER_WORDS.get(raw_value, raw_value))
    unit = _UNIT_GROUPS.get(re.sub(r"\s+", "", raw_unit), "number")
    return f"{value}:{unit}"


def _source_numbers(value: str) -> tuple[str, ...]:
    """Extract only explicit Japanese numeric expressions.

    A bare kanji such as ``一`` in ``一緒`` or ``一面`` is lexical content, not
    automatically verifiable numeric evidence. Kanji numerals therefore require a
    counter, while Arabic digits remain explicit numeric evidence on their own.
    """

    normalized = value.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    tokens: list[str] = []
    for match in _ARABIC_NUMBER_RE.finditer(normalized):
        tokens.append(_numeric_token(match.group("value"), match.group("unit") or ""))
    for match in _JAPANESE_KANJI_NUMBER_RE.finditer(normalized):
        tokens.append(_numeric_token(match.group("value"), match.group("unit")))
    return tuple(tokens)


def _korean_numbers(value: str) -> tuple[str, ...]:
    """Extract numeric forms that naturally occur in Korean subtitle dialogue."""

    normalized = value.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    tokens: list[str] = []
    for match in _ARABIC_NUMBER_RE.finditer(normalized):
        tokens.append(_numeric_token(match.group("value"), match.group("unit") or ""))
    for match in _KOREAN_NUMBER_WORD_WITH_UNIT_RE.finditer(normalized):
        tokens.append(_numeric_token(match.group("value"), match.group("unit")))
    for match in _KOREAN_BARE_CARDINAL_RE.finditer(normalized):
        tokens.append(_numeric_token(match.group("value"), ""))
    return tuple(tokens)


def _numeric_tokens_preserved(source_tokens: tuple[str, ...], target_tokens: tuple[str, ...]) -> bool:
    return not (Counter(source_tokens) - Counter(target_tokens))


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
    source_evidence: Mapping[str, Any] | None = None,
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
    source_numbers = _source_numbers(source)
    faithful_numbers = _korean_numbers(faithful)
    natural_numbers = _korean_numbers(natural)
    if source_numbers and not _numeric_tokens_preserved(source_numbers, faithful_numbers):
        faithful_hard_failures.append("numeric_token_not_preserved")
    if source_numbers and not _numeric_tokens_preserved(source_numbers, natural_numbers):
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
    if _JAPANESE_MOMENTARY_RE.search(source) and not _KOREAN_MOMENTARY_RE.search(faithful):
        faithful_hard_failures.append("momentary_aspect_not_preserved")
    if _JAPANESE_MOMENTARY_RE.search(source) and not _KOREAN_MOMENTARY_RE.search(natural):
        natural_hard_failures.append("momentary_aspect_not_preserved")
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
            bool(_JAPANESE_MOMENTARY_RE.search(source)),
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
    evidence = source_evidence if isinstance(source_evidence, Mapping) else {}
    evidence_reasons = source_evidence_reason_codes(evidence)
    warnings.extend(evidence_reasons)

    faithful_hard_failures = list(dict.fromkeys(faithful_hard_failures))
    natural_hard_failures = list(dict.fromkeys(natural_hard_failures))
    hard_failures = list(dict.fromkeys(hard_failures))
    warnings = list(dict.fromkeys(warnings))
    reasons = list(dict.fromkeys([*reasons, *hard_failures, *warnings]))
    score = 1.0
    score -= min(0.75, 0.25 * len(hard_failures))
    score -= min(0.25, 0.05 * len(warnings))
    score = max(0.0, round(score, 3))
    # Cross-language pattern checks remain useful provenance and fallback metadata,
    # but a model audit does not repair their output.  They also cannot distinguish
    # natural equivalents such as ``한 번쯤``, ``괜찮아``, or ``이리 와`` from a
    # semantic loss. Reserve independent model calls for an explicit translator
    # conflict or independent-source disagreement instead of fanning out from a
    # heuristic fallback alone.
    semantic_audit_selection_reasons: list[str] = []
    if evidence.get("critical_risk_slots"):
        semantic_audit_selection_reasons.append(
            "source_evidence_critical_slot_conflict"
        )

    # Diagnostics such as a suspect ASR source or low model confidence are useful
    # for provenance, but are not evidence that an otherwise complete Korean line
    # has a semantic error. Only a concrete semantic/structural failure changes the
    # deterministic gate result.
    status = "fallback" if hard_failures else "passed"
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
        "source_evidence_state": str(
            (evidence.get("asr_fusion") or {}).get("state") or "empty"
        ),
        "source_evidence_risk_codes": [
            str(value)
            for value in (evidence.get("asr_fusion") or {}).get("risk_codes", [])
        ],
        "source_evidence_critical_slots": [
            str(value) for value in evidence.get("critical_risk_slots", [])
        ],
        "semantic_audit_recommended": bool(semantic_audit_selection_reasons),
        "semantic_audit_selection_reasons": semantic_audit_selection_reasons,
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
            source_evidence=(
                decision.get("source_evidence")
                if isinstance(decision.get("source_evidence"), Mapping)
                else unit.get("source_evidence")
                if isinstance(unit.get("source_evidence"), Mapping)
                else None
            ),
        )
        decision["automated_quality_status"] = record["status"]
        decision["automated_quality_score"] = record["score"]
        decision["automated_quality_reasons"] = record["reasons"]
        decision["automated_quality_evidence_id"] = record["machine_evidence_id"]
        decision["automated_quality_faithful_safe"] = record["faithful_safe"]
        decision["automated_quality_natural_safe"] = record["natural_safe"]
        decision["automated_quality_source_evidence_state"] = record[
            "source_evidence_state"
        ]
        decision["automated_quality_source_evidence_risk_codes"] = record[
            "source_evidence_risk_codes"
        ]
        # A deterministic pass is a candidate baseline. An independent semantic
        # audit may later establish ``pass``, ``issue``, or ``inconclusive``; it
        # must never be inferred from confidence or style alone.
        decision["semantic_audit_status"] = "not-selected"
        decision["semantic_audit_selection_reasons"] = list(
            record["semantic_audit_selection_reasons"]
        )
        decision["translation_status"] = (
            "machine_verified"
            if record["status"] == "passed" and not record["warnings"]
            else "machine_uncertain"
        )
        record["semantic_audit_status"] = "not-selected"
        updated.append(decision)
        records.append(record)
    return updated, records


__all__ = ["audit_translation_decision", "audit_translation_decisions"]
