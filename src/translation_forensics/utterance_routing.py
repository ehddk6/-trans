from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

UTTERANCE_KINDS = frozenset({"lexical_speech", "vocalization", "nonverbal", "unknown"})

_MEANINGFUL_SHORT = frozenset(
    {
        "いや",
        "いいえ",
        "だめ",
        "ダメ",
        "やめて",
        "やめろ",
        "お願い",
        "おねがい",
        "はい",
        "痛い",
        "いたい",
        "待って",
        "まって",
        "来て",
        "きて",
        "行って",
        "いって",
        "ごめん",
        "ごめんなさい",
        "ありがとう",
        "助けて",
        "たすけて",
        "違う",
        "ちがう",
        "無理",
        "むり",
    }
)

_VOCALIZATIONS = frozenset(
    {
        "あ",
        "ああ",
        "あっ",
        "う",
        "うう",
        "うっ",
        "うー",
        "ん",
        "んん",
        "んっ",
        "はぁ",
        "はあ",
        "ふぅ",
        "ふう",
        "えっ",
        "おっ",
        "わあ",
        "わぁ",
        "きゃ",
        "きゃあ",
        "あはは",
        "はは",
        "ふふ",
        "へへ",
    }
)

_NONVERBAL_LABELS = frozenset(
    {
        "music",
        "applause",
        "laughter",
        "laughs",
        "sigh",
        "breathing",
        "moan",
        "groan",
        "音楽",
        "拍手",
        "笑い",
        "ため息",
        "息",
    }
)

_CORRUPTION_MARKERS = ("\ufffd", "���", "\x00")


def _compact(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^[\[\(【（]\s*|\s*[\]\)】）]$", "", text)
    return re.sub(r"[\s。、！？!?…・~〜～]+", "", text)


def _contains_corruption(value: Any) -> bool:
    text = str(value or "")
    return any(marker in text for marker in _CORRUPTION_MARKERS)


def _surface_kind(value: Any) -> str:
    if _contains_corruption(value):
        return "corrupt"
    compact = _compact(value)
    if not compact:
        return "empty"
    if compact in _MEANINGFUL_SHORT:
        return "lexical"
    if compact.casefold() in _NONVERBAL_LABELS:
        return "nonverbal"
    if compact in _VOCALIZATIONS:
        return "vocalization"
    return "lexical"


def _fusion_surfaces(acoustic_record: Mapping[str, Any]) -> list[str]:
    fusion = acoustic_record.get("asr_fusion") or {}
    hypotheses = fusion.get("family_hypotheses") or []
    return [
        str(row.get("text") or "")
        for row in hypotheses
        if isinstance(row, Mapping) and str(row.get("text") or "").strip()
    ]


def classify_utterance_kind(
    *,
    source_text: str,
    agreement: Mapping[str, Any],
    acoustic_record: Mapping[str, Any],
    source_quality_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify lexical speech separately from supported non-lexical audio.

    A vocalization or nonverbal route requires two independent signal classes
    among source text, dual-family ASR fusion, and the dual-frame arbitration.
    """

    source_kind = _surface_kind(source_text)
    source_quality = str(
        (source_quality_record or {}).get("source_quality_status")
        or "trusted"
    )
    source_is_trusted = source_quality == "trusted"
    selected_frame = agreement.get("selected_frame") or {}
    speech_act = str(selected_frame.get("speech_act") or "")
    frame_kind = (
        "vocalization"
        if speech_act == "vocalization"
        else "nonverbal"
        if speech_act == "nonverbal"
        else "lexical"
        if speech_act
        else "unknown"
    )

    fusion = acoustic_record.get("asr_fusion") or {}
    fusion_state = str(fusion.get("state") or "empty")
    surfaces = _fusion_surfaces(acoustic_record)
    surface_kinds = [_surface_kind(text) for text in surfaces]
    dual_fusion = fusion_state in {"dual_agreement", "dual_compatible"}
    fusion_kind = "unknown"
    if dual_fusion and surface_kinds and all(kind == "vocalization" for kind in surface_kinds):
        fusion_kind = "vocalization"
    elif dual_fusion and surface_kinds and all(kind == "nonverbal" for kind in surface_kinds):
        fusion_kind = "nonverbal"
    elif any(kind == "lexical" for kind in surface_kinds):
        fusion_kind = "lexical"

    reason_codes: list[str] = []
    if source_kind == "corrupt":
        reason_codes.append("source-text-corruption")
    elif not source_is_trusted and source_kind not in {"empty"}:
        reason_codes.append("untrusted-source-kind-ignored")

    # Explicit meaningful short words remain speech only when the source text is
    # trusted. For damaged source, independent frame or acoustic evidence must
    # carry the lexical classification.
    trusted_meaningful_short = (
        source_is_trusted and _compact(source_text) in _MEANINGFUL_SHORT
    )
    if trusted_meaningful_short or frame_kind == "lexical" or fusion_kind == "lexical":
        return {
            "utterance_kind": "lexical_speech",
            "confidence": (
                "high"
                if (
                    source_is_trusted
                    and source_kind == "lexical"
                    and frame_kind == "lexical"
                )
                else "medium"
            ),
            "reason_codes": sorted(set(reason_codes) | {"lexical-content-present"}),
            "evidence_refs": [],
        }

    signals = {
        "source": (
            source_kind
            if source_is_trusted and source_kind not in {"corrupt", "empty"}
            else "unknown"
        ),
        "frame": frame_kind,
        "fusion": fusion_kind,
    }
    for candidate in ("vocalization", "nonverbal"):
        supporting = sorted(name for name, kind in signals.items() if kind == candidate)
        if len(supporting) >= 2:
            refs = sorted(
                {
                    str(ref)
                    for row in (fusion.get("family_hypotheses") or [])
                    if isinstance(row, Mapping)
                    for ref in (row.get("evidence_refs") or [])
                }
            )
            return {
                "utterance_kind": candidate,
                "confidence": "high" if "fusion" in supporting else "medium",
                "reason_codes": sorted(
                    set(reason_codes)
                    | {f"{candidate}-supported-by-{'-and-'.join(supporting)}"}
                ),
                "evidence_refs": refs,
            }

    return {
        "utterance_kind": "unknown",
        "confidence": "low",
        "reason_codes": sorted(
            set(reason_codes) | {"insufficient-nonlexical-support"}
        ),
        "evidence_refs": [],
    }


def render_nonlexical(kind: str, evidence_text: str) -> str:
    """Render only broad audible form; never infer action, cause, or emotion."""
    compact = _compact(evidence_text)
    if kind == "nonverbal":
        return "[비언어음]"
    mapping = {
        "あ": "아…",
        "ああ": "아…",
        "あっ": "앗!",
        "う": "으…",
        "うう": "으…",
        "うっ": "윽…",
        "うー": "으…",
        "ん": "음…",
        "んん": "음…",
        "んっ": "음…",
        "はぁ": "하…",
        "はあ": "하…",
        "ふぅ": "후…",
        "ふう": "후…",
        "えっ": "어?",
        "おっ": "오!",
        "わあ": "와…",
        "わぁ": "와…",
        "きゃ": "꺄!",
        "きゃあ": "꺄!",
        "あはは": "아하하…",
        "はは": "하하…",
        "ふふ": "후후…",
        "へへ": "헤헤…",
    }
    return mapping.get(compact, "…")


def apply_utterance_route(
    agreement: Mapping[str, Any],
    *,
    source_text: str,
    acoustic_record: Mapping[str, Any],
    source_quality_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach a route to one arbitration record without mutating source evidence."""
    output = dict(agreement)
    route = classify_utterance_kind(
        source_text=source_text,
        agreement=agreement,
        acoustic_record=acoustic_record,
        source_quality_record=source_quality_record,
    )
    output["utterance_kind"] = route["utterance_kind"]
    output["utterance_kind_confidence"] = route["confidence"]
    output["utterance_kind_reason_codes"] = route["reason_codes"]
    output["utterance_kind_evidence_refs"] = route["evidence_refs"]

    original_state = str(agreement.get("recovery_state") or "")
    if (
        original_state == "vocalization"
        and route["utterance_kind"] not in {"vocalization", "nonverbal"}
    ):
        output["recovery_state"] = "abstained"
        output["controlled_nonlexical_korean"] = ""
        output["uncertainty_codes"] = sorted(
            set(agreement.get("uncertainty_codes") or [])
            | {"lexical-nonlexical-route-mismatch"}
        )
        output["coherence_findings"] = list(
            agreement.get("coherence_findings") or []
        ) + [
            {
                "code": "lexical-nonlexical-route-mismatch",
                "severity": "high",
                "slots": ["speech_act"],
                "remediation": "abstain",
            }
        ]
        return output

    if route["utterance_kind"] in {"vocalization", "nonverbal"}:
        fusion = acoustic_record.get("asr_fusion") or {}
        hypotheses = fusion.get("family_hypotheses") or []
        evidence_text = (
            source_text
            if _surface_kind(source_text) == route["utterance_kind"]
            else ""
        )
        if not evidence_text:
            for row in hypotheses:
                if not isinstance(row, Mapping):
                    continue
                candidate = str(row.get("text") or "")
                if _surface_kind(candidate) == route["utterance_kind"]:
                    evidence_text = candidate
                    break
        rendered = render_nonlexical(route["utterance_kind"], evidence_text)
        if rendered != "…":
            output["recovery_state"] = "vocalization"
            output["controlled_nonlexical_korean"] = rendered
            output["rendered_slots"] = ["speech_act"]
            output["omitted_slots"] = [
                slot
                for slot in (
                    "question",
                    "polarity",
                    "refusal_permission",
                    "command_strength",
                    "speaker",
                    "actor",
                    "action",
                    "target",
                    "location",
                    "tense_aspect",
                    "direction",
                    "intensity",
                )
            ]
    return output
