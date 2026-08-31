from __future__ import annotations

"""Canonical, additive bridge between subtitle ASR evidence and quality gates.

The bridge never rewrites source text.  It normalizes family aliases only in a
copied evidence record so repeated runs of one model family cannot become
independent votes merely because their labels differ.
"""

import re
from collections.abc import Iterable, Mapping
from typing import Any

from .asr_fusion import add_asr_fusion
from .utterance_routing import classify_utterance_kind


EVIDENCE_BRIDGE_SCHEMA_VERSION = "1"

_KNOWN_FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("whisper", re.compile(r"whisper", re.IGNORECASE)),
    ("reazon", re.compile(r"reazon", re.IGNORECASE)),
    ("qwen", re.compile(r"qwen", re.IGNORECASE)),
    ("openai-transcribe", re.compile(r"openai[-_ ]?transcrib", re.IGNORECASE)),
)

FUSION_RISK_SLOTS: dict[str, tuple[str, ...]] = {
    "polarity-marker-divergence": ("polarity",),
    "question-marker-divergence": ("question", "speech_act"),
    "refusal-permission-marker-divergence": ("refusal_permission",),
    "stop-continue-marker-divergence": ("action", "command_strength"),
    "direction-up-down-divergence": ("direction",),
    "direction-in-out-divergence": ("direction",),
}


def canonical_asr_family(value: object) -> str:
    """Collapse known aliases; all unregistered labels share one fail-closed family."""

    raw = str(value or "").strip()
    for canonical, pattern in _KNOWN_FAMILY_PATTERNS:
        if pattern.search(raw):
            return canonical
    return "unverified"


def _normalized_transcript(row: Mapping[str, Any]) -> dict[str, Any]:
    raw_family = str(row.get("source_family") or "").strip()
    canonical_family = canonical_asr_family(raw_family)
    normalized = dict(row)
    normalized["source_family"] = canonical_family
    normalized["declared_source_family"] = raw_family or None
    return normalized


def _critical_slots(risk_codes: Iterable[object]) -> list[str]:
    return sorted(
        {
            slot
            for raw_code in risk_codes
            for slot in FUSION_RISK_SLOTS.get(str(raw_code), ())
        }
    )


def prepare_acoustic_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized copy with recomputed family counts and ASR fusion.

    Unknown family labels collapse to one ``unverified`` family.  They remain in
    the raw copied rows and may reveal a conflict, but cannot manufacture a
    multi-family agreement.
    """

    output = dict(record)
    transcripts = [
        _normalized_transcript(row)
        for row in (record.get("transcripts") or [])
        if isinstance(row, Mapping)
    ]
    output["transcripts"] = transcripts
    verified_families = sorted(
        {
            str(row["source_family"])
            for row in transcripts
            if row["source_family"] != "unverified"
        }
    )
    output["independent_source_families"] = verified_families
    output["family_lineage"] = [
        {
            "canonical_family": family,
            "declared_labels": sorted(
                {
                    str(row.get("declared_source_family") or "")
                    for row in transcripts
                    if row["source_family"] == family
                    and str(row.get("declared_source_family") or "")
                }
            ),
        }
        for family in sorted({str(row["source_family"]) for row in transcripts})
    ]
    output["family_lineage_status"] = (
        "unavailable"
        if not transcripts
        else "contains-unverified-family"
        if any(row["source_family"] == "unverified" for row in transcripts)
        else "known-family-aliases-normalized"
    )
    output = add_asr_fusion(output)
    risks = list(output["asr_fusion"].get("risk_codes") or [])
    output["critical_risk_slots"] = _critical_slots(risks)
    output["evidence_bridge_schema_version"] = EVIDENCE_BRIDGE_SCHEMA_VERSION
    return output


def _alternative_texts(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    alternatives: list[str] = []
    for item in value:
        text = (
            str(item.get("text") or "").strip()
            if isinstance(item, Mapping)
            else str(item or "").strip()
        )
        if text and text not in alternatives:
            alternatives.append(text)
    return alternatives


def build_unit_source_evidence(
    *,
    unit_id: str,
    start: float,
    end: float,
    source_text: str,
    asr_metrics: Mapping[str, Any] | None,
    source_quality_status: str,
    evidence_ids: Iterable[object] = (),
) -> dict[str, Any]:
    """Build one process-title evidence record from a transcript JSON row."""

    metrics = dict(asr_metrics or {})
    backend = str(metrics.get("backend") or "").strip()
    transcripts: list[dict[str, Any]] = []
    if backend and source_text.strip():
        transcripts.append(
            {
                "source_family": backend,
                "text": source_text,
                "utterance_id": unit_id,
                "start_seconds": float(start),
                "end_seconds": float(end),
                "alignment_scope": "utterance-timestamp",
                "alignment_confidence": 1.0,
                "evidence_ids": [str(value) for value in evidence_ids],
            }
        )

    alternatives = _alternative_texts(metrics.get("qwen_alternatives"))
    # qwen_alternatives are n-best alternatives for the same span, not
    # sequential utterances. Preserve all in raw_asr_metrics, but admit one
    # stable family hypothesis into fusion.
    if alternatives and canonical_asr_family(backend) != "qwen":
        transcripts.append(
            {
                "source_family": "qwen",
                "text": alternatives[0],
                "utterance_id": unit_id,
                "start_seconds": float(start),
                "end_seconds": float(end),
                "alignment_scope": "utterance-timestamp",
                "alignment_confidence": 1.0,
                "candidate_rank": 1,
                "evidence_ids": [str(value) for value in evidence_ids],
            }
        )

    prepared = prepare_acoustic_evidence(
        {
            "schema_name": "translation-forensics/canonical-source-evidence",
            "schema_version": EVIDENCE_BRIDGE_SCHEMA_VERSION,
            "unit_id": unit_id,
            "start_seconds": float(start),
            "end_seconds": float(end),
            "source_text": source_text,
            "source_quality_status": source_quality_status,
            "declared_evidence_ids": [str(value) for value in evidence_ids],
            "raw_asr_metrics": metrics,
            "transcripts": transcripts,
        }
    )
    route = classify_utterance_kind(
        source_text=source_text,
        agreement={"selected_frame": {}},
        acoustic_record=prepared,
        source_quality_record={"source_quality_status": source_quality_status},
    )
    prepared["utterance_route"] = route
    prepared["bridge_state"] = (
        "unavailable"
        if prepared["asr_fusion"]["state"] == "empty"
        else "conflict"
        if prepared["asr_fusion"]["state"] == "dual_conflict"
        else "available"
    )
    return prepared


def source_evidence_reason_codes(evidence: Mapping[str, Any] | None) -> list[str]:
    """Return stable deterministic reason codes for quality-gate consumption."""

    if not isinstance(evidence, Mapping):
        return []
    fusion = evidence.get("asr_fusion") or {}
    if not isinstance(fusion, Mapping):
        return []
    reasons = [
        "asr_" + str(code).replace("-", "_")
        for code in (fusion.get("risk_codes") or [])
    ]
    if fusion.get("state") == "dual_conflict":
        reasons.append("asr_family_conflict")
    return list(dict.fromkeys(reasons))

__all__ = [
    "EVIDENCE_BRIDGE_SCHEMA_VERSION",
    "FUSION_RISK_SLOTS",
    "build_unit_source_evidence",
    "canonical_asr_family",
    "prepare_acoustic_evidence",
    "source_evidence_reason_codes",
]
