from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any, Iterable, Mapping, Protocol

from jsonschema import Draft202012Validator

from .codex_exec_provider import CodexTimeoutError, CodexUsageLimitError


_JAPANESE_RE = re.compile(r"[\u3040-\u30ff]")
_ONLY_HOLD_RE = re.compile(r"^[\s.…・･ー\-~～!?！？]+$")
_UNRESOLVED_MARKER = "[불명]"
_HOLD_MARKERS = frozenset({_UNRESOLVED_MARKER, "[원문 불명확]", "[검수 보류]"})
_RECOVERY_CLASSIFICATIONS = frozenset({"RELIABLE", "FUNCTIONAL_RECOVERY", "UNRESOLVED"})
_SEMANTIC_SLOT_KEYS = (
    "speech_act",
    "question",
    "polarity",
    "refusal_permission",
    "stop_continue",
    "command_strength",
    "speaker",
    "addressee",
    "actor",
    "action",
    "target",
    "location",
    "direction",
    "tense_aspect",
    "completion",
    "intensity",
    "numeric_tokens",
    "register",
)
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
The payload contains scene_packets. Return one decision for every packet, in payload
order, with its exact unit_id. Each scene_packet has one focus_unit and its preceding
and following turns. Use the context to resolve discourse, omitted participants, and
register; do not use it to overwrite a clear current utterance or invent a missing
action, target, location, relationship, consent state, result, or intensity.

The viewer-natural field is the primary subtitle. Draft it first in idiomatic Korean
with scene-appropriate spoken rhythm, directness, and adult-genre register when the
source supports that register. Then derive source-faithful as a semantic baseline
with the identical supported meaning. Do not make the display line literal or neutral
merely to resemble the baseline. Do not replace lexical content with a generic reaction
or add a body part, action, relationship, consent state, location, or result.

Every result must include all semantic_slots with strings or null. Use
recovery_classification as the source state: RELIABLE, FUNCTIONAL_RECOVERY, or
UNRESOLVED. FUNCTIONAL_RECOVERY requires a concrete recovery_basis and may recover
only the broad function supported by the supplied evidence. UNRESOLVED requires both
Korean fields to be exactly [불명]. A natural Korean line must stay within the
source-faithful meaning. Do not mark any result approved or final.

Use risk_codes only for a concrete candidate-translation conflict affecting a
critical meaning slot, and name it SEMANTIC_CONFLICT_<SLOT>. Do not put source
provenance, ASR quality, low confidence, damaged-source, or style diagnostics in
risk_codes. Set critic_required=true only when such a SEMANTIC_CONFLICT code is
present; ordinary uncertainty belongs in the metadata and does not by itself call
for a second model.
"""


_LEXICAL_COMPLETION_RETRY_ADDENDUM = """\

## Retry correction

The prior draft used an unresolved or hold marker for at least one focus unit that
contains lexical Japanese. Correct the entire supplied batch. For every lexical
Japanese focus source, set recovery_classification to RELIABLE or
FUNCTIONAL_RECOVERY, supply a concrete recovery_basis for FUNCTIONAL_RECOVERY, and
write complete source-faithful and viewer-natural Korean. Never output [불명],
[원문 불명확], or [검수 보류] for a lexical Japanese line. Keep uncertainty in
metadata rather than replacing the dialogue.
"""


_LEXICAL_COMPLETION_ESCALATION_ADDENDUM = """\

## Mandatory lexical completion

This is a single focus unit whose Japanese source contains lexical content. An
unresolved classification or any hold marker is invalid for this response. Select
the most plausible source-bounded meaning, even if the evidence is weak, and return
complete natural Korean in both fields. Record the uncertainty in confidence,
uncertain_slots, competing_interpretations, and critic_required instead of omitting
the line. Do not use [불명], [원문 불명확], or [검수 보류].
"""


_BATCH_COVERAGE_RETRY_ADDENDUM = """\

## Mandatory batch coverage retry

The prior response omitted one or more requested focus units. Return exactly one
translation object for every supplied scene_packet, in payload order, using each
focus_unit's exact unit_id. An empty translations array, a partial array, duplicate
unit_id values, or substitute IDs is invalid. Complete every Korean field and every
required semantic field for each requested unit; do not skip a line because it is
short, noisy, repetitive, adult, or uncertain.
"""


def _terra_translation_prompt() -> str:
    """Load the versioned process-title contract and append the runtime constraints."""

    contract_path = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "integrated-noisy-asr-recovery-v1.md"
    )
    if not contract_path.is_file():
        raise FileNotFoundError(f"missing Terra prompt contract: {contract_path}")
    return contract_path.read_text(encoding="utf-8") + "\n\n## Runtime addendum\n\n" + TERRA_TRANSLATION_PROMPT

SOL_VISUAL_REVIEW_PROMPT = """\
Independently review one Japanese-to-Korean subtitle decision using only the given
Japanese text, neighboring transcript context, the draft translations, and up to
three attached timestamp frames. Pixels may resolve only these slots: speaker,
addressee, deictic_location, on_screen_text, and scene_continuity. Never add an
action, body part, relationship, or spoken proposition merely because it is visible.
If a visual observation would materially change a critical semantic slot, set
critical_visual_impact=true and mark the unit machine-uncertain without replacing
the source-faithful fallback. Keep or repair the Korean text only when it remains
licensed by the Japanese source.
"""

VISUAL_BOUND_REPAIR_PROMPT = """\
Create one revised Japanese-to-Korean subtitle decision from the Japanese source,
neighboring turns, the existing machine draft, and a bounded visual observation
record produced by an independent critic. The visual record may resolve only
speaker, addressee, deictic_location, on_screen_text, or scene_continuity. It is not
evidence that a visible action, body part, relationship, emotion, consent state, or
result was spoken. Preserve source force, polarity, target, tense, direction, and
intensity. Return the complete Terra translation schema, including every semantic
slot, preserved_meaning, competing_interpretations, risk_codes, and critic_required.
For a lexical Japanese focus source, use RELIABLE or FUNCTIONAL_RECOVERY and write
complete Korean in both fields; do not use [불명], [원문 불명확], or [검수 보류].
This is a machine candidate that will be checked again; never claim final status.
"""

AUTOMATED_AUDIT_PROMPT = """\
You are an independent semantic-error critic for Japanese-to-Korean subtitle
decisions. This is not a naturalness, tone, censorship, or literalness review.
Do not rewrite, neutralize, or prefer the source-faithful wording over the
viewer-natural candidate.

Compare the Japanese source, local context, candidate Korean, and the translator's
semantic slots. Report only a concrete meaning error: lost or reversed question,
polarity, refusal/permission, stop/continue, command force, participant, target,
direction, tense/completion, numeric token, or intensity; or an addition unsupported
by the supplied evidence. Natural Korean omission, subtitle compression, direct
adult-genre register, and a different but equivalent reaction are not errors by
themselves.

Return pass when no concrete semantic error is found, unknown only when evidence is
genuinely insufficient to determine a semantic issue, and fail only for a meaning
flip, critical-slot loss, or unsupported addition. Do not promote a machine candidate
to human-final status. FUNCTIONAL_RECOVERY and UNRESOLVED remain machine-uncertain
even if their local structural checks pass.
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
                    "semantic_slots",
                    "preserved_meaning",
                    "competing_interpretations",
                    "risk_codes",
                    "critic_required",
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
                    "semantic_slots": {
                        "type": "object",
                        "required": list(_SEMANTIC_SLOT_KEYS),
                        "properties": {
                            key: {"type": ["string", "null"]}
                            for key in _SEMANTIC_SLOT_KEYS
                        },
                        "additionalProperties": False,
                    },
                    "preserved_meaning": {"type": "array", "items": {"type": "string"}},
                    "competing_interpretations": {
                        "type": "array", "items": {"type": "string"}
                    },
                    "risk_codes": {"type": "array", "items": {"type": "string"}},
                    "critic_required": {"type": "boolean"},
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
                    "critical_slot_issues",
                    "unsupported_addition",
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
                    "critical_slot_issues": {
                        "type": "array", "items": {"type": "string"}
                    },
                    "unsupported_addition": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}


class StructuredProvider(Protocol):
    def run_structured(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]: ...


def _validate_structured_response(
    response: object, schema: dict[str, Any], *, call_id: str
) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError(f"structured response must be an object: {call_id}")
    error = next(iter(Draft202012Validator(schema).iter_errors(response)), None)
    if error is not None:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ValueError(
            f"structured response schema error for {call_id} at {location}: {error.message}"
        )
    return response


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


def _source_evidence_summary(value: object) -> dict[str, Any]:
    """Keep source diagnostics useful without repeating raw ASR records per neighbor."""

    evidence = dict(value) if isinstance(value, dict) else {}
    fusion = dict(evidence.get("asr_fusion") or {})
    route = dict(evidence.get("utterance_route") or {})
    hypotheses = []
    for item in fusion.get("family_hypotheses", []):
        if not isinstance(item, dict):
            continue
        hypotheses.append(
            {
                "source_family": item.get("source_family"),
                "text": item.get("text"),
            }
        )
    return {
        "bridge_state": evidence.get("bridge_state"),
        "source_quality_status": evidence.get("source_quality_status"),
        "critical_risk_slots": list(evidence.get("critical_risk_slots") or []),
        "asr_fusion": {
            "state": fusion.get("state"),
            "risk_codes": list(fusion.get("risk_codes") or []),
            "family_hypotheses": hypotheses,
        },
        "utterance_route": {
            "utterance_kind": route.get("utterance_kind"),
            "reason_codes": list(route.get("reason_codes") or []),
        },
    }


def _scene_packet_unit(
    unit: dict[str, Any], *, sequence_number: int, include_evidence_summary: bool
) -> dict[str, Any]:
    """Return only source-bound context the translation model may inspect."""

    result = {
        "unit_id": str(unit["unit_id"]),
        "sequence_number": sequence_number,
        "start": unit.get("start"),
        "end": unit.get("end"),
        "source_japanese": unit.get("source_japanese", unit.get("text_raw", "")),
        "speaker": unit.get("speaker", "speaker_unknown"),
        "source_quality_status": unit.get("quality_status", "suspect"),
        "asr_warnings": unit.get("asr_warnings", unit.get("warnings", [])),
        "evidence_refs": unit.get("evidence_ids", unit.get("evidence_refs", [])),
        "consistency_context": dict(unit.get("consistency_context") or {}),
    }
    if include_evidence_summary:
        result["source_evidence"] = _source_evidence_summary(unit.get("source_evidence"))
    return result


def _scene_packet(
    units: list[dict[str, Any]], *, focus_index: int, context_radius: int = 3
) -> dict[str, Any]:
    """Build one bounded translation packet without promoting context to evidence."""

    if context_radius < 1:
        raise ValueError("context_radius must be positive")
    focus = _scene_packet_unit(
        units[focus_index], sequence_number=focus_index + 1, include_evidence_summary=True
    )
    previous = [
        _scene_packet_unit(
            units[index], sequence_number=index + 1, include_evidence_summary=False
        )
        for index in range(max(0, focus_index - context_radius), focus_index)
    ]
    following = [
        _scene_packet_unit(
            units[index], sequence_number=index + 1, include_evidence_summary=False
        )
        for index in range(focus_index + 1, min(len(units), focus_index + context_radius + 1))
    ]
    return {
        "scene_id": f"translation-window-{focus_index + 1:06d}",
        "focus_unit": focus,
        "previous_units": previous,
        "next_units": following,
        "screen_context_policy": "may_resolve only supplied visual observations; never invent spoken content",
    }


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


def _has_lexical_source(value: object) -> bool:
    """Return whether the focus source has text that should be translated."""

    return any(character.isalnum() for character in str(value or ""))


def _recovery_classification(
    value: object, *, source_quality_status: str, source: str
) -> str:
    classification = str(value or "").strip().upper()
    if classification not in _RECOVERY_CLASSIFICATIONS:
        raise ValueError(f"unsupported recovery classification: {value!r}")
    if classification == "UNRESOLVED" and _has_lexical_source(source):
        raise ValueError(
            "unresolved_marker classification cannot replace a lexical Japanese source"
        )
    if classification == "FUNCTIONAL_RECOVERY" and source_quality_status == "trusted":
        raise ValueError("FUNCTIONAL_RECOVERY cannot replace a trusted source")
    if classification == "RELIABLE" and source_quality_status == "unusable":
        raise ValueError("RELIABLE cannot be claimed for an unusable source")
    return classification


def _lexical_completion_retry_unit_ids(
    batch: Iterable[dict[str, Any]], response: object
) -> list[str]:
    """Identify cached model rows that erased a translatable Japanese line.

    A retry uses a distinct request identity, so a bad cached batch cannot repeatedly
    abort a resumed title run after the lexical-completion contract was tightened.
    """

    if not isinstance(response, dict):
        return []
    rows = response.get("translations")
    if not isinstance(rows, list):
        return []
    rows_by_id = {
        str(row.get("unit_id") or ""): row
        for row in rows
        if isinstance(row, dict) and str(row.get("unit_id") or "")
    }
    retry_ids: list[str] = []
    for unit in batch:
        source = str(unit.get("source_japanese", unit.get("text_raw", "")))
        if not _has_lexical_source(source):
            continue
        unit_id = str(unit.get("unit_id") or "")
        translated = rows_by_id.get(unit_id)
        if not isinstance(translated, dict):
            continue
        if str(translated.get("recovery_classification") or "").strip().upper() == "UNRESOLVED":
            retry_ids.append(unit_id)
            continue
        if any(
            str(translated.get(field) or "").strip() in _HOLD_MARKERS
            for field in ("source_faithful_korean", "viewer_natural_korean")
        ):
            retry_ids.append(unit_id)
    return retry_ids


def _requires_lexical_completion_retry(
    batch: Iterable[dict[str, Any]], response: object
) -> bool:
    return bool(_lexical_completion_retry_unit_ids(batch, response))


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
    if text in _HOLD_MARKERS and text != _UNRESOLVED_MARKER:
        raise ValueError("human_hold_marker_cannot_replace_translation")
    if _UNRESOLVED_MARKER in text:
        if text != _UNRESOLVED_MARKER:
            raise ValueError("unresolved_marker_must_be_the_entire_translation")
        if not unresolved_marker_allowed:
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
    max_batch_workers: int = 1,
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

    if max_batch_workers < 1:
        raise ValueError("max_batch_workers must be positive")

    decisions: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    source_profile = _source_profile(units)
    global_index = {str(unit["unit_id"]): index for index, unit in enumerate(units)}
    call_semaphore = BoundedSemaphore(max_batch_workers)
    batches = list(
        enumerate(
            _chunks(units, batch_size=batch_size, max_source_characters=max_source_characters),
            1,
        )
    )

    def validated_batch_rows(
        response: dict[str, Any],
        *,
        batch: list[dict[str, Any]],
        call_id: str,
    ) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
        response = _validate_structured_response(
            response, TRANSLATION_BATCH_SCHEMA, call_id=call_id
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
        return by_id, missing, extra, duplicates

    def translate_batch(batch_index: int, batch: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        scene_packets: list[dict[str, Any]] = []
        for unit in batch:
            index = global_index[str(unit["unit_id"])]
            scene_packets.append(_scene_packet(units, focus_index=index))
        call_id = f"batch-{batch_index:04d}.translation.terra"
        with call_semaphore:
            return provider.run_structured(
                role="translation-terra",
                title_id=title_id,
                call_id=call_id,
                prompt=_terra_translation_prompt(),
                payload={
                    "title_id": title_id,
                    "contract": "complete_machine_draft_not_final",
                    "source_profile": source_profile,
                    "scene_packets": scene_packets,
                },
                schema=TRANSLATION_BATCH_SCHEMA,
                resume=resume,
            )

    with ThreadPoolExecutor(max_workers=max_batch_workers) as executor:
        results = executor.map(lambda item: translate_batch(*item), batches)
        for (batch_index, batch), (response, receipt) in zip(batches, results, strict=True):
            base_call_id = f"batch-{batch_index:04d}.translation.terra"
            call_id = base_call_id
            receipts_for_batch = [receipt]
            by_id, missing, extra, duplicates = validated_batch_rows(
                response,
                batch=batch,
                call_id=call_id,
            )
            if missing or duplicates:
                call_id = f"{base_call_id}.coverage-retry"
                with call_semaphore:
                    response, receipt = provider.run_structured(
                        role="translation-terra",
                        title_id=title_id,
                        call_id=call_id,
                        prompt=_terra_translation_prompt() + _BATCH_COVERAGE_RETRY_ADDENDUM,
                        payload={
                            "title_id": title_id,
                            "contract": "complete_machine_draft_not_final",
                            "source_profile": source_profile,
                            "scene_packets": [
                                _scene_packet(
                                    units,
                                    focus_index=global_index[str(unit["unit_id"])],
                                )
                                for unit in batch
                            ],
                        },
                        schema=TRANSLATION_BATCH_SCHEMA,
                        resume=resume,
                    )
                receipts_for_batch.append(receipt)
                by_id, missing, extra, duplicates = validated_batch_rows(
                    response,
                    batch=batch,
                    call_id=call_id,
                )
            if missing or duplicates:
                raise ValueError(
                    f"Terra batch coverage mismatch for {call_id}: "
                    f"missing={missing}, extra={extra}, duplicates={duplicates}"
                )
            call_ids_by_unit = {str(unit["unit_id"]): call_id for unit in batch}
            if extra:
                receipt = {
                    **receipt,
                    "response_contract_warnings": [
                        *receipt.get("response_contract_warnings", []),
                        "unexpected_unit_ids_dropped:" + ",".join(extra),
                    ],
                }
                receipts_for_batch[-1] = receipt
            remaining_retry_ids = _lexical_completion_retry_unit_ids(batch, response)
            retry_units = [
                unit for unit in batch if str(unit["unit_id"]) in remaining_retry_ids
            ]

            def retry_lexical_unit(
                unit: dict[str, Any],
            ) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
                unit_id = str(unit["unit_id"])
                unit_call_id = f"{base_call_id}.{unit_id}.lexical-completion-retry"
                with call_semaphore:
                    unit_response, unit_receipt = provider.run_structured(
                        role="translation-terra",
                        title_id=title_id,
                        call_id=unit_call_id,
                        prompt=(
                            _terra_translation_prompt()
                            + _LEXICAL_COMPLETION_RETRY_ADDENDUM
                        ),
                        payload={
                            "title_id": title_id,
                            "contract": "complete_machine_draft_not_final",
                            "source_profile": source_profile,
                            "scene_packets": [
                                _scene_packet(
                                    units,
                                    focus_index=global_index[unit_id],
                                )
                            ],
                        },
                        schema=TRANSLATION_BATCH_SCHEMA,
                        resume=resume,
                    )
                unit_response = _validate_structured_response(
                    unit_response,
                    TRANSLATION_BATCH_SCHEMA,
                    call_id=unit_call_id,
                )
                unit_rows = unit_response.get("translations")
                if (
                    not isinstance(unit_rows, list)
                    or len(unit_rows) != 1
                    or not isinstance(unit_rows[0], dict)
                    or str(unit_rows[0].get("unit_id") or "") != unit_id
                ):
                    raise ValueError(
                        f"lexical completion retry coverage mismatch for {unit_call_id}"
                    )
                if _requires_lexical_completion_retry([unit], unit_response):
                    escalation_call_id = f"{unit_call_id}.escalation"
                    with call_semaphore:
                        unit_response, unit_receipt = provider.run_structured(
                            role="translation-terra",
                            title_id=title_id,
                            call_id=escalation_call_id,
                            prompt=(
                                _terra_translation_prompt()
                                + _LEXICAL_COMPLETION_RETRY_ADDENDUM
                                + _LEXICAL_COMPLETION_ESCALATION_ADDENDUM
                            ),
                            payload={
                                "title_id": title_id,
                                "contract": "complete_machine_draft_not_final",
                                "source_profile": source_profile,
                                "scene_packets": [
                                    _scene_packet(
                                        units,
                                        focus_index=global_index[unit_id],
                                    )
                                ],
                            },
                            schema=TRANSLATION_BATCH_SCHEMA,
                            resume=resume,
                        )
                    unit_response = _validate_structured_response(
                        unit_response,
                        TRANSLATION_BATCH_SCHEMA,
                        call_id=escalation_call_id,
                    )
                    unit_rows = unit_response.get("translations")
                    if (
                        not isinstance(unit_rows, list)
                        or len(unit_rows) != 1
                        or not isinstance(unit_rows[0], dict)
                        or str(unit_rows[0].get("unit_id") or "") != unit_id
                    ):
                        raise ValueError(
                            f"lexical completion escalation coverage mismatch for {escalation_call_id}"
                        )
                    if _requires_lexical_completion_retry([unit], unit_response):
                        raise ValueError(
                            f"lexical completion escalation failed for lexical Japanese unit: {unit_id}"
                        )
                    unit_call_id = escalation_call_id
                return unit_id, unit_rows[0], unit_call_id, unit_receipt

            if retry_units:
                with ThreadPoolExecutor(
                    max_workers=min(max_batch_workers, len(retry_units))
                ) as retry_executor:
                    retry_results = list(retry_executor.map(retry_lexical_unit, retry_units))
                for unit_id, translated, unit_call_id, unit_receipt in retry_results:
                    by_id[unit_id] = translated
                    call_ids_by_unit[unit_id] = unit_call_id
                    receipts_for_batch.append(unit_receipt)
            for unit in batch:
                unit_id = str(unit["unit_id"])
                translated = by_id[unit_id]
                source = str(unit.get("source_japanese", unit.get("text_raw", "")))
                source_quality_status = str(unit.get("quality_status", "suspect"))
                classification = _recovery_classification(
                    translated.get("recovery_classification"),
                    source_quality_status=source_quality_status,
                    source=source,
                )
                recovery_basis = [
                    str(value).strip()
                    for value in translated.get("recovery_basis", [])
                    if str(value).strip()
                ]
                if classification == "FUNCTIONAL_RECOVERY" and not recovery_basis:
                    raise ValueError(
                        "structured response schema error: "
                        "FUNCTIONAL_RECOVERY requires a concrete recovery_basis"
                    )
                if classification == "UNRESOLVED" and any(
                    str(translated.get(field) or "").strip() != _UNRESOLVED_MARKER
                    for field in ("source_faithful_korean", "viewer_natural_korean")
                ):
                    raise ValueError("UNRESOLVED requires both translations to be exactly [불명]")
                unresolved_marker_allowed = (
                    not _has_lexical_source(source) and classification == "UNRESOLVED"
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
                        "recovery_basis": recovery_basis,
                        "semantic_slots": dict(translated["semantic_slots"]),
                        "preserved_meaning": [
                            str(value) for value in translated["preserved_meaning"]
                        ],
                        "competing_interpretations": [
                            str(value)
                            for value in translated["competing_interpretations"]
                        ],
                        "risk_codes": [str(value) for value in translated["risk_codes"]],
                        "critic_required": bool(translated["critic_required"]),
                        "evidence_ids": unit.get("evidence_ids", unit.get("evidence_refs", [])),
                        "source_evidence": dict(unit.get("source_evidence") or {}),
                        "translation_status": "machine_draft",
                        "terra_call_id": call_ids_by_unit[unit_id],
                        "sol_call_id": None,
                        "visual_critical": False,
                    }
                )
            receipts.extend(receipts_for_batch)
    if len(decisions) != len(units):
        raise AssertionError("Terra translation coverage invariant failed")
    return decisions, receipts


_TRANSLATOR_SEMANTIC_CONFLICT_PREFIX = "SEMANTIC_CONFLICT_"


def _has_translator_semantic_conflict(decision: Mapping[str, Any]) -> bool:
    """Return whether the translator declared a concrete critical-slot conflict.

    ``risk_codes`` historically included source provenance such as
    ``SUSPECT_SOURCE``.  Those diagnostics describe input confidence, not a
    discovered error in the Korean candidate, and must not fan out into a
    full-title Sol audit.
    """

    return any(
        str(code).strip().upper().startswith(_TRANSLATOR_SEMANTIC_CONFLICT_PREFIX)
        for code in decision.get("risk_codes", [])
    )


def _semantic_audit_selection_reasons(decision: Mapping[str, Any]) -> list[str]:
    """Return concrete reasons to spend an independent semantic-audit call.

    The translation stage is responsible for dialogue rhythm and genre register.
    This second stage is intentionally narrow: it samples only decisions with a
    declared semantic ambiguity or a deterministic/visual signal that can affect
    meaning. Source provenance warnings alone must not trigger a full-title rewrite
    or penalize natural Korean wording.
    """

    reasons = [
        str(value)
        for value in decision.get("semantic_audit_selection_reasons", [])
        if str(value).strip()
    ]
    if _has_translator_semantic_conflict(decision):
        reasons.append("translator_declared_semantic_conflict")
        if bool(decision.get("critic_required")):
            reasons.append("translator_requested_semantic_critic")
    if bool(decision.get("visual_critical")):
        reasons.append("critical_visual_semantic_context")
    evidence = decision.get("source_evidence")
    if isinstance(evidence, Mapping) and evidence.get("critical_risk_slots"):
        reasons.append("source_evidence_critical_slot_conflict")
    return list(dict.fromkeys(reasons))


def audit_translations_with_sol(
    provider: StructuredProvider,
    *,
    title_id: str,
    decisions: list[dict[str, Any]],
    batch_size: int = 20,
    resume: bool = True,
    max_batch_workers: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit selected semantic risks without scoring or rewriting subtitle style."""

    if not decisions:
        raise ValueError("automated translation audit requires decisions")
    if max_batch_workers < 1:
        raise ValueError("max_batch_workers must be positive")
    updated = [dict(row) for row in decisions]
    by_id = {str(row.get("unit_id") or ""): row for row in updated}
    if len(by_id) != len(updated) or "" in by_id:
        raise ValueError("automated translation audit requires unique unit IDs")
    receipts: list[dict[str, Any]] = []
    selection_by_id = {
        str(row["unit_id"]): _semantic_audit_selection_reasons(row)
        for row in updated
    }
    for decision in updated:
        unit_id = str(decision["unit_id"])
        decision["semantic_audit_status"] = "not-selected"
        decision["semantic_audit_selection_reasons"] = selection_by_id[unit_id]
        decision["semantic_audit_issue_codes"] = []
    records_by_id: dict[str, dict[str, Any]] = {
        unit_id: {
            "unit_id": unit_id,
            "semantic_audit_status": "not-selected",
            "semantic_audit_selection_reasons": reasons,
            "semantic_audit_scope": "targeted-semantic-only",
        }
        for unit_id, reasons in selection_by_id.items()
    }
    selected = [
        row for row in updated if selection_by_id[str(row["unit_id"])]
    ]
    decision_index = {
        str(row["unit_id"]): index for index, row in enumerate(updated)
    }
    batches = list(
        enumerate(
            _chunks(selected, batch_size=batch_size, max_source_characters=12_000),
            1,
        )
    )

    def audit_batch(batch_index: int, batch: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        payload_units = []
        for row in batch:
            index = decision_index[str(row["unit_id"])]
            payload_units.append({
                "unit_id": row["unit_id"],
                "source_japanese": row.get("source_japanese", ""),
                "source_faithful_korean": row.get("source_faithful_korean", ""),
                "viewer_natural_korean": row.get("viewer_natural_korean", ""),
                "source_quality_status": row.get("source_quality_status", "suspect"),
                "source_profile": row.get("source_profile", "NOT_VERIFIED"),
                "recovery_classification": row.get("recovery_classification", "RELIABLE"),
                "recovery_basis": row.get("recovery_basis", []),
                "semantic_slots": dict(row.get("semantic_slots") or {}),
                "preserved_meaning": row.get("preserved_meaning", []),
                "competing_interpretations": row.get("competing_interpretations", []),
                "risk_codes": row.get("risk_codes", []),
                "critic_required": bool(row.get("critic_required")),
                "source_evidence": dict(row.get("source_evidence") or {}),
                "previous_source_japanese": (
                    updated[index - 1].get("source_japanese", "") if index > 0 else ""
                ),
                "next_source_japanese": (
                    updated[index + 1].get("source_japanese", "")
                    if index + 1 < len(updated)
                    else ""
                ),
            })
        call_id = f"batch-{batch_index:04d}.translation-audit.sol"
        return provider.run_structured(
            role="translation-audit-sol",
            title_id=title_id,
            call_id=call_id,
            prompt=AUTOMATED_AUDIT_PROMPT,
            payload={"title_id": title_id, "units": payload_units},
            schema=AUTOMATED_AUDIT_SCHEMA,
            resume=resume,
        )

    with ThreadPoolExecutor(max_workers=max_batch_workers) as executor:
        results = executor.map(lambda item: audit_batch(*item), batches)
        for (batch_index, batch), (response, receipt) in zip(batches, results, strict=True):
            call_id = f"batch-{batch_index:04d}.translation-audit.sol"
            response = _validate_structured_response(
                response, AUTOMATED_AUDIT_SCHEMA, call_id=call_id
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
                semantic_issue_codes: list[str] = []
                if not all(flags.values()):
                    semantic_issue_codes.append("sol_invariant_check_failed")
                if bool(audit.get("meaning_flip")):
                    semantic_issue_codes.append("sol_meaning_flip")
                critical_slot_issues = [
                    str(value) for value in audit.get("critical_slot_issues", [])
                ]
                if critical_slot_issues:
                    semantic_issue_codes.extend(
                        f"sol_critical_slot:{value}" for value in critical_slot_issues
                    )
                if bool(audit.get("unsupported_addition")):
                    semantic_issue_codes.append("sol_unsupported_addition")
                backtranslation = str(audit.get("backtranslation_japanese") or "").strip()
                fidelity_score = float(audit.get("source_fidelity_score") or 0.0)
                if not backtranslation:
                    reasons.append("sol_backtranslation_empty")
                if model_verdict == "pass" and fidelity_score < 0.8:
                    reasons.append("sol_source_fidelity_advisory_low")
                if semantic_issue_codes:
                    semantic_audit_status = "issue"
                elif model_verdict == "pass" and backtranslation:
                    semantic_audit_status = "pass"
                else:
                    # An incomplete critic response is inconclusive evidence, not
                    # evidence that a natural subtitle should be flattened or
                    # replaced with a literal source-faithful wording.
                    semantic_audit_status = "inconclusive"
                existing_status = str(row.get("automated_quality_status") or "passed")
                final_status = (
                    "fallback"
                    if existing_status == "fallback" or semantic_audit_status == "issue"
                    else "passed"
                )
                decision = by_id[unit_id]
                decision["automated_quality_status"] = final_status
                decision["automated_quality_call_id"] = call_id
                decision["automated_quality_backtranslation_japanese"] = backtranslation
                decision["automated_quality_model_score"] = fidelity_score
                decision["automated_quality_reasons"] = list(
                    dict.fromkeys(
                        [
                            *decision.get("automated_quality_reasons", []),
                            *reasons,
                            *semantic_issue_codes,
                        ]
                    )
                )
                decision["semantic_audit_status"] = semantic_audit_status
                decision["semantic_audit_selection_reasons"] = selection_by_id[unit_id]
                decision["semantic_audit_issue_codes"] = semantic_issue_codes
                if semantic_audit_status == "issue":
                    decision["translation_status"] = "machine_uncertain"
                records_by_id[unit_id] = {
                    "unit_id": unit_id,
                    "status": final_status,
                    "model_verdict": model_verdict,
                    "semantic_audit_status": semantic_audit_status,
                    "semantic_audit_scope": "targeted-semantic-only",
                    "semantic_audit_selection_reasons": selection_by_id[unit_id],
                    "semantic_audit_issue_codes": semantic_issue_codes,
                    "source_fidelity_score": decision["automated_quality_model_score"],
                    "backtranslation_japanese": decision["automated_quality_backtranslation_japanese"],
                    **flags,
                    "reasons": decision["automated_quality_reasons"],
                    "audit_call_id": call_id,
                    "sol_audit_status": "completed",
                    "meaning_flip": bool(audit.get("meaning_flip")),
                    "critical_slot_issues": critical_slot_issues,
                    "unsupported_addition": bool(audit.get("unsupported_addition")),
                }
            receipts.append(receipt)
    return (
        [by_id[str(row["unit_id"])] for row in decisions],
        receipts,
        [records_by_id[str(row["unit_id"])] for row in decisions],
    )


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
    ordered_units = sorted(
        units_by_id.values(), key=lambda row: (float(row.get("start", 0.0)), str(row.get("unit_id", "")))
    )
    unit_index = {str(row["unit_id"]): index for index, row in enumerate(ordered_units)}
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
        index = unit_index[unit_id]
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
                        "previous_source_japanese": (
                            ordered_units[index - 1].get("source_japanese", "")
                            if index > 0
                            else ""
                        ),
                        "next_source_japanese": (
                            ordered_units[index + 1].get("source_japanese", "")
                            if index + 1 < len(ordered_units)
                            else ""
                        ),
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
        except (CodexUsageLimitError, CodexTimeoutError) as exc:
            block_reason = (
                "usage-limit" if isinstance(exc, CodexUsageLimitError) else "timeout"
            )
            block_code = "usage_limit" if block_reason == "usage-limit" else block_reason
            decision["review_required_reasons"] = list(
                dict.fromkeys(
                    [
                        *decision.get("review_required_reasons", []),
                        f"visual_review_blocked_{block_code}",
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
                    "verdict": "machine-uncertain-not-sent",
                    "model_call_receipt": {
                        "status": "blocked",
                        "external_transfer": False,
                        "pixel_external_transfer_count": 0,
                        "provider": "codex-cli",
                        "role": exc.role,
                        "call_id": exc.call_id,
                        "reason": block_reason,
                        "retry_after": getattr(exc, "retry_after", None),
                    },
                }
            )
            break
        response = _validate_structured_response(
            response, VISUAL_REVIEW_SCHEMA, call_id=call_id
        )
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
            and not critical
        )
        effective_verdict = verdict
        repair_receipt: dict[str, Any] | None = None
        if verdict == "repair" and visual_repair_allowed:
            repair_call_id = f"{unit_id}.visual-repair.terra"
            repair_response, repair_receipt = provider.run_structured(
                role="translation-terra",
                title_id=title_id,
                call_id=repair_call_id,
                prompt=VISUAL_BOUND_REPAIR_PROMPT,
                payload={
                    "title_id": title_id,
                    "contract": "source_bound_visual_slot_repair_not_final",
                    "units": [
                        {
                            "unit_id": unit_id,
                            "source_japanese": decision["source_japanese"],
                            "previous_source_japanese": unit.get(
                                "previous_source_japanese", ""
                            ),
                            "next_source_japanese": unit.get(
                                "next_source_japanese", ""
                            ),
                            "source_quality_status": decision.get(
                                "source_quality_status", "suspect"
                            ),
                            "draft": {
                                "source_faithful_korean": decision[
                                    "source_faithful_korean"
                                ],
                                "viewer_natural_korean": decision[
                                    "viewer_natural_korean"
                                ],
                                "semantic_slots": decision.get("semantic_slots", {}),
                                "preserved_meaning": decision.get("preserved_meaning", []),
                                "competing_interpretations": decision.get(
                                    "competing_interpretations", []
                                ),
                                "risk_codes": decision.get("risk_codes", []),
                                "critic_required": bool(decision.get("critic_required")),
                            },
                            "approved_visual_slots": response.get("visual_slots", {}),
                        }
                    ],
                },
                schema=TRANSLATION_BATCH_SCHEMA,
                resume=resume,
            )
            repair_response = _validate_structured_response(
                repair_response, TRANSLATION_BATCH_SCHEMA, call_id=repair_call_id
            )
            repair_rows = repair_response["translations"]
            if len(repair_rows) != 1 or str(repair_rows[0].get("unit_id")) != unit_id:
                raise ValueError(f"visual Terra repair coverage mismatch for {unit_id}")
            repaired = repair_rows[0]
            classification = _recovery_classification(
                repaired.get("recovery_classification"),
                source_quality_status=str(decision.get("source_quality_status", "suspect")),
                source=str(decision["source_japanese"]),
            )
            recovery_basis = [
                str(value).strip()
                for value in repaired.get("recovery_basis", [])
                if str(value).strip()
            ]
            if classification == "FUNCTIONAL_RECOVERY" and not recovery_basis:
                raise ValueError("FUNCTIONAL_RECOVERY requires a concrete recovery_basis")
            source_ko, source_reasons = _complete_text(
                repaired.get("source_faithful_korean"),
                source=decision["source_japanese"],
                unresolved_marker_allowed=False,
            )
            viewer_ko, viewer_reasons = _complete_text(
                repaired.get("viewer_natural_korean"),
                source=decision["source_japanese"],
                unresolved_marker_allowed=False,
            )
            from .automated_quality import audit_translation_decision

            repair_audit = audit_translation_decision(
                unit_id=unit_id,
                source_japanese=str(decision["source_japanese"]),
                source_faithful_korean=source_ko,
                viewer_natural_korean=viewer_ko,
                source_quality_status=str(decision.get("source_quality_status", "suspect")),
                confidence=str(repaired.get("confidence") or "low"),
                existing_reasons=[*source_reasons, *viewer_reasons],
                recovery_classification=classification,
                recovery_basis=recovery_basis,
            )
            if repair_audit["hard_failures"]:
                reasons.append("visual_repair_failed_deterministic_gate")
                reasons.extend(repair_audit["hard_failures"])
                effective_verdict = "machine-uncertain"
            else:
                decision["source_faithful_korean"] = source_ko
                decision["viewer_natural_korean"] = viewer_ko
                decision["confidence"] = repaired.get("confidence", decision["confidence"])
                decision["recovery_classification"] = classification
                decision["recovery_basis"] = recovery_basis
                decision["semantic_slots"] = dict(repaired["semantic_slots"])
                decision["preserved_meaning"] = [
                    str(value) for value in repaired["preserved_meaning"]
                ]
                decision["competing_interpretations"] = [
                    str(value) for value in repaired["competing_interpretations"]
                ]
                decision["risk_codes"] = [str(value) for value in repaired["risk_codes"]]
                decision["critic_required"] = bool(repaired["critic_required"])
                decision["visual_repair_terra_call_id"] = repair_call_id
                effective_verdict = "repair-applied-pending-independent-audit"
        elif verdict == "repair":
            reasons.append(
                "visual_text_repair_not_applied_without_source_bound_translation_pass"
            )
            if repair_requires_asr_inference or decision.get("source_quality_status") == "unusable":
                reasons.append("visual_text_repair_not_applied_asr_inference_or_unusable_source")
            effective_verdict = "machine-uncertain"
        if critical or effective_verdict in {"escalate", "machine-uncertain"}:
            reasons.append("critical_visual_context_requires_machine_uncertain")
        decision["review_required_reasons"] = list(
            dict.fromkeys([*decision.get("review_required_reasons", []), *reasons])
        )
        decision["sol_call_id"] = call_id
        decision["visual_critical"] = critical
        receipts.append(receipt)
        if repair_receipt is not None:
            receipts.append(repair_receipt)
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
                "repair_call_id": decision.get("visual_repair_terra_call_id"),
                "model_call_receipt": receipt,
            }
        )
    return [updated[str(row["unit_id"])] for row in decisions], receipts, visual_records
