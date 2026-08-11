from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Protocol

from .codex_exec_provider import CodexUsageLimitError


_JAPANESE_RE = re.compile(r"[\u3040-\u30ff]")
_ONLY_HOLD_RE = re.compile(r"^[\s.…・･ー\-~～!?！？]+$")
_UNRESOLVED_MARKER = "[불명]"
_RECOVERY_CLASSIFICATIONS = frozenset({"RELIABLE", "FUNCTIONAL_RECOVERY", "UNRESOLVED"})
_ASR_INFERENCE_MARKERS = (
    "asr",
    "오인식",
    "잘못 전사",
    "전사 오류",
    "원음 확인",
    "misrecogn",
    "transcription error",
)

TERRA_TRANSLATION_PROMPT = """\
You restore and translate Japanese dialogue subtitles into Korean. Return one decision
for every input unit, in the same order and with the exact unit_id. The source
transcript is the authority. Preserve questions, negation, refusal, request/command
force, speaker, action, target, location, tense, and intensity.

For each unit classify recovery_classification as RELIABLE, FUNCTIONAL_RECOVERY, or
UNRESOLVED. RELIABLE translates confirmed source meaning. FUNCTIONAL_RECOVERY may use
only neighboring turns, recurrence, timing, and a broad utterance function; it must
not invent an action, body part, relationship, or proposition. UNRESOLVED is allowed
only when source_quality_status is unusable and no broad utterance function can be
recovered. Only then may either Korean field be exactly [불명].

First write a readable source-faithful Korean line, then a natural viewer line that
does not add meaning. Do not copy a previous translation as truth. Do not omit
low-confidence units or use an ellipsis as a substitute for translation. Report the
specific recovery_basis and uncertainty instead of inventing facts. Images are not
present in this pass.
"""

SOL_VISUAL_REVIEW_PROMPT = """\
Independently review one Japanese-to-Korean subtitle decision using only the given
Japanese text, neighboring transcript context, the draft translations, and up to
three attached timestamp frames. Pixels may resolve only these slots: speaker,
addressee, deictic_location, on_screen_text, and scene_continuity. Never add an
action, body part, relationship, or spoken proposition merely because it is visible.
If a visual observation would materially change a critical semantic slot, set
critical_visual_impact=true and escalate for human review. Keep or repair the Korean
text only when it remains licensed by the Japanese source.
"""

AUTOMATED_AUDIT_PROMPT = """\
Audit the Korean subtitle decisions independently from the Japanese source.
Compare source_japanese with both Korean candidates and use back-translation
only as supporting evidence. Do not rewrite either Korean candidate. Return
pass, fail, or unknown, plus invariant checks and evidence. Use unknown when
the context is genuinely ambiguous or the evidence is insufficient. Never
invent a repair; the caller will choose a conservative fallback. Treat
FUNCTIONAL_RECOVERY and UNRESOLVED as machine-uncertain classifications: do not
promote them to a final semantic claim merely because a structural invariant passes.
"""

TRANSLATION_BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["translations"],
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "unit_id",
                    "source_faithful_korean",
                    "viewer_natural_korean",
                    "confidence",
                    "uncertain_slots",
                    "review_required_reasons",
                    "recovery_classification",
                    "recovery_basis",
                ],
                "properties": {
                    "unit_id": {"type": "string", "minLength": 1},
                    "source_faithful_korean": {"type": "string"},
                    "viewer_natural_korean": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "uncertain_slots": {"type": "array", "items": {"type": "string"}},
                    "review_required_reasons": {"type": "array", "items": {"type": "string"}},
                    "recovery_classification": {
                        "type": "string",
                        "enum": ["RELIABLE", "FUNCTIONAL_RECOVERY", "UNRESOLVED"],
                    },
                    "recovery_basis": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}

VISUAL_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "unit_id",
        "verdict",
        "source_faithful_korean",
        "viewer_natural_korean",
        "confidence",
        "critical_visual_impact",
        "visual_slots",
        "review_required_reasons",
    ],
    "properties": {
        "unit_id": {"type": "string", "minLength": 1},
        "verdict": {"type": "string", "enum": ["keep", "repair", "escalate"]},
        "source_faithful_korean": {"type": "string"},
        "viewer_natural_korean": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "critical_visual_impact": {"type": "boolean"},
        "visual_slots": {
            "type": "object",
            "required": [
                "speaker",
                "addressee",
                "deictic_location",
                "on_screen_text",
                "scene_continuity",
            ],
            "properties": {
                "speaker": {"type": "array", "items": {"type": "string"}},
                "addressee": {"type": "array", "items": {"type": "string"}},
                "deictic_location": {"type": "array", "items": {"type": "string"}},
                "on_screen_text": {"type": "array", "items": {"type": "string"}},
                "scene_continuity": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
        "review_required_reasons": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

AUTOMATED_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["audits"],
    "properties": {
        "audits": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "unit_id",
                    "verdict",
                    "backtranslation_japanese",
                    "source_fidelity_score",
                    "question_preserved",
                    "negation_preserved",
                    "numeric_tokens_preserved",
                    "reasons",
                    "meaning_flip",
                    "uncertainty_codes",
                ],
                "properties": {
                    "unit_id": {"type": "string", "minLength": 1},
                    "verdict": {"type": "string", "enum": ["pass", "fail", "unknown"]},
                    "backtranslation_japanese": {"type": "string"},
                    "source_fidelity_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "question_preserved": {"type": "boolean"},
                    "negation_preserved": {"type": "boolean"},
                    "numeric_tokens_preserved": {"type": "boolean"},
                    "reasons": {"type": "array", "items": {"type": "string"}},
                    "meaning_flip": {"type": "boolean"},
                    "uncertainty_codes": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}


class StructuredProvider(Protocol):
    def run_structured(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]: ...


def _chunks(
    units: list[dict[str, Any]], *, batch_size: int, max_source_characters: int
) -> Iterable[list[dict[str, Any]]]:
    if batch_size < 1 or max_source_characters < 1:
        raise ValueError("batch_size and max_source_characters must be positive")
    batch: list[dict[str, Any]] = []
    characters = 0
    for unit in units:
        length = len(str(unit.get("source_japanese") or unit.get("text_raw") or ""))
        if batch and (len(batch) >= batch_size or characters + length > max_source_characters):
            yield batch
            batch = []
            characters = 0
        batch.append(unit)
        characters += length
    if batch:
        yield batch


def _source_profile(units: Iterable[dict[str, Any]]) -> str:
    """Return the title-level ASR profile used as model context, never as a final claim."""

    values = list(units)
    if not values:
        return "NOT_VERIFIED"
    warnings = {
        str(reason)
        for unit in values
        for reason in unit.get("asr_warnings", unit.get("warnings", []))
    }
    if any(str(unit.get("quality_status") or "") == "unusable" for unit in values) or warnings.intersection(
        {"possible_runaway_repetition", "possible_periodic_repetition", "review_required"}
    ):
        return "NOISY_ASR"
    if any(str(unit.get("quality_status") or "") == "suspect" for unit in values):
        return "MIXED"
    return "CLEAN"


def _recovery_classification(value: object, *, source_quality_status: str) -> str:
    classification = str(value or "").strip().upper()
    if classification in _RECOVERY_CLASSIFICATIONS:
        return classification
    if source_quality_status == "unusable":
        return "UNRESOLVED"
    if source_quality_status == "suspect":
        return "FUNCTIONAL_RECOVERY"
    return "RELIABLE"


def _complete_text(
    value: object,
    *,
    source: str,
    unresolved_marker_allowed: bool,
) -> tuple[str, list[str]]:
    text = str(value or "").strip()
    reasons: list[str] = []
    if not text or _ONLY_HOLD_RE.fullmatch(text):
        if not unresolved_marker_allowed:
            raise ValueError("empty_or_ellipsis_translation_without_unresolved_authorization")
        text = _UNRESOLVED_MARKER
        reasons.append("unresolved_marker_inserted")
    if text == _UNRESOLVED_MARKER and not unresolved_marker_allowed:
        raise ValueError("unresolved_marker_without_unusable_source_and_unresolved_classification")
    if _JAPANESE_RE.search(text):
        reasons.append("japanese_residual")
    if not source.strip():
        reasons.append("empty_japanese_source")
    return text, reasons


def translate_units_with_terra(
    provider: StructuredProvider,
    *,
    title_id: str,
    units: list[dict[str, Any]],
    resume: bool = True,
    batch_size: int = 40,
    max_source_characters: int = 12_000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Produce a complete machine draft and verifiable receipts.

    Evidence quality is copied into each decision but never used to suppress a
    translation call. Candidate promotion is deliberately handled by the packaging
    layer so a suspect source can still yield a complete, clearly labelled draft.
    """

    unit_ids = [str(unit.get("unit_id") or "") for unit in units]
    if not units or any(not unit_id for unit_id in unit_ids):
        raise ValueError("Every translation unit requires a non-empty unit_id")
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("translation unit_id values must be unique")

    decisions: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    source_profile = _source_profile(units)
    for batch_index, batch in enumerate(
        _chunks(units, batch_size=batch_size, max_source_characters=max_source_characters), 1
    ):
        compact_units: list[dict[str, Any]] = []
        for index, unit in enumerate(batch):
            compact_units.append(
                {
                    "unit_id": unit["unit_id"],
                    "start": unit.get("start"),
                    "end": unit.get("end"),
                    "source_japanese": unit.get("source_japanese", unit.get("text_raw", "")),
                    "previous_source_japanese": (
                        batch[index - 1].get("source_japanese", batch[index - 1].get("text_raw", ""))
                        if index > 0
                        else ""
                    ),
                    "next_source_japanese": (
                        batch[index + 1].get("source_japanese", batch[index + 1].get("text_raw", ""))
                        if index + 1 < len(batch)
                        else ""
                    ),
                    "source_quality_status": unit.get("quality_status", "suspect"),
                    "asr_warnings": unit.get("asr_warnings", unit.get("warnings", [])),
                }
            )
        call_id = f"batch-{batch_index:04d}.translation.terra"
        response, receipt = provider.run_structured(
            role="translation-terra",
            title_id=title_id,
            call_id=call_id,
            prompt=TERRA_TRANSLATION_PROMPT,
            payload={
                "title_id": title_id,
                "contract": "complete_machine_draft_not_final",
                "source_profile": source_profile,
                "units": compact_units,
            },
            schema=TRANSLATION_BATCH_SCHEMA,
            resume=resume,
        )
        rows = response.get("translations")
        if not isinstance(rows, list):
            raise ValueError(f"Terra response lacks translations array: {call_id}")
        response_ids = [
            str(row.get("unit_id") or "")
            for row in rows
            if isinstance(row, dict) and str(row.get("unit_id") or "")
        ]
        by_id = {
            str(row.get("unit_id") or ""): row
            for row in rows
            if isinstance(row, dict) and str(row.get("unit_id") or "")
        }
        expected = [str(unit["unit_id"]) for unit in batch]
        missing = sorted(set(expected) - set(by_id))
        extra = sorted(set(by_id) - set(expected))
        duplicates = sorted(
            unit_id for unit_id in set(expected) if response_ids.count(unit_id) > 1
        )
        if missing or duplicates:
            raise ValueError(
                f"Terra batch coverage mismatch for {call_id}: "
                f"missing={missing}, extra={extra}, duplicates={duplicates}"
            )
        if extra:
            receipt = {
                **receipt,
                "response_contract_warnings": [
                    *receipt.get("response_contract_warnings", []),
                    "unexpected_unit_ids_dropped:" + ",".join(extra),
                ],
            }
        for unit in batch:
            unit_id = str(unit["unit_id"])
            translated = by_id[unit_id]
            source = str(unit.get("source_japanese", unit.get("text_raw", "")))
            source_quality_status = str(unit.get("quality_status", "suspect"))
            classification = _recovery_classification(
                translated.get("recovery_classification"),
                source_quality_status=source_quality_status,
            )
            unresolved_marker_allowed = (
                source_quality_status == "unusable" and classification == "UNRESOLVED"
            )
            source_ko, source_reasons = _complete_text(
                translated.get("source_faithful_korean"),
                source=source,
                unresolved_marker_allowed=unresolved_marker_allowed,
            )
            viewer_ko, viewer_reasons = _complete_text(
                translated.get("viewer_natural_korean"),
                source=source,
                unresolved_marker_allowed=unresolved_marker_allowed,
            )
            reasons = list(
                dict.fromkeys(
                    [
                        *[str(value) for value in translated.get("review_required_reasons", [])],
                        *source_reasons,
                        *viewer_reasons,
                    ]
                )
            )
            decisions.append(
                {
                    "unit_id": unit_id,
                    "start": unit.get("start"),
                    "end": unit.get("end"),
                    "source_japanese": source,
                    "source_faithful_korean": source_ko,
                    "viewer_natural_korean": viewer_ko,
                    "confidence": translated.get("confidence", "low"),
                    "uncertain_slots": [str(value) for value in translated.get("uncertain_slots", [])],
                    "review_required_reasons": reasons,
                    "source_quality_status": source_quality_status,
                    "source_profile": source_profile,
                    "recovery_classification": classification,
                    "recovery_basis": [str(value) for value in translated.get("recovery_basis", [])],
                    "evidence_ids": unit.get("evidence_ids", unit.get("evidence_refs", [])),
                    "translation_status": "machine_draft",
                    "terra_call_id": call_id,
                    "sol_call_id": None,
                    "visual_critical": False,
                }
            )
        receipts.append(receipt)
    if len(decisions) != len(units):
        raise AssertionError("Terra translation coverage invariant failed")
    return decisions, receipts


def audit_translations_with_sol(
    provider: StructuredProvider,
    *,
    title_id: str,
    decisions: list[dict[str, Any]],
    batch_size: int = 20,
    resume: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run an independent Sol back-translation audit over completed decisions."""

    if not decisions:
        raise ValueError("automated translation audit requires decisions")
    updated = [dict(row) for row in decisions]
    by_id = {str(row.get("unit_id") or ""): row for row in updated}
    if len(by_id) != len(updated) or "" in by_id:
        raise ValueError("automated translation audit requires unique unit IDs")
    receipts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(
        _chunks(updated, batch_size=batch_size, max_source_characters=12_000), 1
    ):
        payload_units = [
            {
                "unit_id": row["unit_id"],
                "source_japanese": row.get("source_japanese", ""),
                "source_faithful_korean": row.get("source_faithful_korean", ""),
                "viewer_natural_korean": row.get("viewer_natural_korean", ""),
                "source_quality_status": row.get("source_quality_status", "suspect"),
                "source_profile": row.get("source_profile", "NOT_VERIFIED"),
                "recovery_classification": row.get("recovery_classification", "RELIABLE"),
                "recovery_basis": row.get("recovery_basis", []),
            }
            for row in batch
        ]
        call_id = f"batch-{batch_index:04d}.translation-audit.sol"
        response, receipt = provider.run_structured(
            role="translation-audit-sol",
            title_id=title_id,
            call_id=call_id,
            prompt=AUTOMATED_AUDIT_PROMPT,
            payload={"title_id": title_id, "units": payload_units},
            schema=AUTOMATED_AUDIT_SCHEMA,
            resume=resume,
        )
        rows = response.get("audits")
        if not isinstance(rows, list):
            raise ValueError(f"Sol automated audit lacks audits array: {call_id}")
        response_ids = [
            str(row.get("unit_id") or "")
            for row in rows
            if isinstance(row, dict) and str(row.get("unit_id") or "")
        ]
        by_response_id = {
            str(row.get("unit_id") or ""): row
            for row in rows
            if isinstance(row, dict) and str(row.get("unit_id") or "")
        }
        expected = [str(row["unit_id"]) for row in batch]
        missing = sorted(set(expected) - set(by_response_id))
        duplicates = sorted(
            unit_id for unit_id in set(expected) if response_ids.count(unit_id) > 1
        )
        if missing or duplicates:
            raise ValueError(
                f"Sol automated audit coverage mismatch for {call_id}: "
                f"missing={missing}, duplicates={duplicates}"
            )
        extra = sorted(set(by_response_id) - set(expected))
        if extra:
            receipt = {
                **receipt,
                "response_contract_warnings": [
                    *receipt.get("response_contract_warnings", []),
                    "unexpected_unit_ids_dropped:" + ",".join(extra),
                ],
            }
        for row in batch:
            unit_id = str(row["unit_id"])
            audit = by_response_id[unit_id]
            flags = {
                "question_preserved": bool(audit.get("question_preserved")),
                "negation_preserved": bool(audit.get("negation_preserved")),
                "numeric_tokens_preserved": bool(audit.get("numeric_tokens_preserved")),
            }
            model_verdict = str(audit.get("verdict") or "unknown")
            if model_verdict not in {"pass", "fail", "unknown"}:
                model_verdict = "unknown"
            reasons = [
                *[str(value) for value in audit.get("reasons", [])],
                *[str(value) for value in audit.get("uncertainty_codes", [])],
            ]
            if not all(flags.values()):
                model_verdict = "fail"
                reasons.append("sol_invariant_check_failed")
            if bool(audit.get("meaning_flip")):
                model_verdict = "fail"
                reasons.append("sol_meaning_flip")
            existing_status = str(row.get("automated_quality_status") or "fallback")
            final_status = "passed" if existing_status == "passed" and model_verdict == "pass" else "fallback"
            decision = by_id[unit_id]
            decision["automated_quality_status"] = final_status
            decision["automated_quality_call_id"] = call_id
            decision["automated_quality_backtranslation_japanese"] = str(
                audit.get("backtranslation_japanese") or ""
            )
            decision["automated_quality_model_score"] = float(
                audit.get("source_fidelity_score") or 0.0
            )
            decision["automated_quality_reasons"] = list(
                dict.fromkeys([*decision.get("automated_quality_reasons", []), *reasons])
            )
            records.append(
                {
                    "unit_id": unit_id,
                    "status": final_status,
                    "model_verdict": model_verdict,
                    "source_fidelity_score": decision["automated_quality_model_score"],
                    "backtranslation_japanese": decision["automated_quality_backtranslation_japanese"],
                    **flags,
                    "reasons": decision["automated_quality_reasons"],
                    "audit_call_id": call_id,
                    "sol_audit_status": "completed",
                    "meaning_flip": bool(audit.get("meaning_flip")),
                }
            )
        receipts.append(receipt)
    return [by_id[str(row["unit_id"])] for row in decisions], receipts, records


def review_translation_with_visuals(
    provider: StructuredProvider,
    *,
    title_id: str,
    decisions: list[dict[str, Any]],
    units_by_id: dict[str, dict[str, Any]],
    frames_by_unit: dict[str, list[Path]],
    resume: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply bounded Sol reviews and return updated decisions, receipts, and records."""

    updated = {str(row["unit_id"]): dict(row) for row in decisions}
    receipts: list[dict[str, Any]] = []
    visual_records: list[dict[str, Any]] = []
    for unit_id, frame_paths in frames_by_unit.items():
        if unit_id not in updated or unit_id not in units_by_id:
            raise ValueError(f"Unknown visual-review unit_id: {unit_id}")
        frames = [Path(path).expanduser().resolve() for path in frame_paths]
        if not frames:
            continue
        if len(frames) > 3:
            raise ValueError(f"Visual review for {unit_id} exceeds the 3-frame limit")
        decision = updated[unit_id]
        unit = units_by_id[unit_id]
        call_id = f"{unit_id}.visual.critique.sol"
        try:
            response, receipt = provider.run_structured(
                role="critique-sol",
                title_id=title_id,
                call_id=call_id,
                prompt=SOL_VISUAL_REVIEW_PROMPT,
                payload={
                    "title_id": title_id,
                    "unit": {
                        "unit_id": unit_id,
                        "start": unit.get("start"),
                        "end": unit.get("end"),
                        "source_japanese": unit.get("source_japanese", unit.get("text_raw", "")),
                        "asr_warnings": unit.get("asr_warnings", unit.get("warnings", [])),
                    },
                    "draft": {
                        "source_faithful_korean": decision["source_faithful_korean"],
                        "viewer_natural_korean": decision["viewer_natural_korean"],
                        "confidence": decision["confidence"],
                        "uncertain_slots": decision["uncertain_slots"],
                    },
                    "allowed_visual_slots": [
                        "speaker",
                        "addressee",
                        "deictic_location",
                        "on_screen_text",
                        "scene_continuity",
                    ],
                },
                schema=VISUAL_REVIEW_SCHEMA,
                resume=resume,
                image_paths=frames,
                allow_image_transfer=True,
            )
        except CodexUsageLimitError as exc:
            decision["review_required_reasons"] = list(
                dict.fromkeys(
                    [
                        *decision.get("review_required_reasons", []),
                        "visual_review_blocked_usage_limit",
                    ]
                )
            )
            decision["sol_call_id"] = call_id
            visual_records.append(
                {
                    "unit_id": unit_id,
                    "selection_reason": "translation_ambiguity_or_low_confidence",
                    "frame_paths": [str(path) for path in frames],
                    "frame_sha256": [],
                    "allowed_visual_slots": {},
                    "critical_visual_impact": False,
                    "verdict": "escalate",
                    "model_call_receipt": {
                        "status": "blocked",
                        "external_transfer": False,
                        "pixel_external_transfer_count": 0,
                        "provider": "codex-cli",
                        "role": exc.role,
                        "call_id": exc.call_id,
                        "reason": "usage-limit",
                        "retry_after": exc.retry_after,
                    },
                }
            )
            break
        if str(response.get("unit_id") or "") != unit_id:
            raise ValueError(f"Sol visual response unit mismatch for {unit_id}")
        verdict = str(response.get("verdict") or "escalate")
        critical = bool(response.get("critical_visual_impact"))
        reasons = [str(value) for value in response.get("review_required_reasons", [])]
        response_reason_text = " ".join(reasons).casefold()
        repair_requires_asr_inference = any(
            marker in response_reason_text for marker in _ASR_INFERENCE_MARKERS
        )
        visual_repair_allowed = (
            decision.get("source_quality_status") != "unusable"
            and not repair_requires_asr_inference
        )
        effective_verdict = verdict
        if verdict == "repair" and not critical and visual_repair_allowed:
            source_ko, source_reasons = _complete_text(
                response.get("source_faithful_korean"),
                source=decision["source_japanese"],
                unresolved_marker_allowed=False,
            )
            viewer_ko, viewer_reasons = _complete_text(
                response.get("viewer_natural_korean"),
                source=decision["source_japanese"],
                unresolved_marker_allowed=False,
            )
            decision["source_faithful_korean"] = source_ko
            decision["viewer_natural_korean"] = viewer_ko
            reasons.extend(source_reasons)
            reasons.extend(viewer_reasons)
        elif verdict == "repair":
            reasons.append("visual_text_repair_not_applied_asr_inference_or_unusable_source")
            effective_verdict = "escalate"
        if critical or verdict == "escalate":
            reasons.append("critical_visual_context_requires_human_review")
        if verdict != "repair" or visual_repair_allowed:
            decision["confidence"] = response.get("confidence", decision["confidence"])
        decision["review_required_reasons"] = list(
            dict.fromkeys([*decision.get("review_required_reasons", []), *reasons])
        )
        decision["sol_call_id"] = call_id
        decision["visual_critical"] = critical
        receipts.append(receipt)
        visual_records.append(
            {
                "unit_id": unit_id,
                "selection_reason": "translation_ambiguity_or_low_confidence",
                "frame_paths": [str(path) for path in frames],
                "frame_sha256": [item["sha256"] for item in receipt.get("image_attachments", [])],
                "allowed_visual_slots": response.get("visual_slots", {}),
                "critical_visual_impact": critical,
                "model_verdict": verdict,
                "verdict": effective_verdict,
                "model_call_receipt": receipt,
            }
        )
    return [updated[str(row["unit_id"])] for row in decisions], receipts, visual_records
