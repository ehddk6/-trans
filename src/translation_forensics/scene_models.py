from __future__ import annotations

"""Strict scene-v2 domain records and cross-record invariants.

The records in this module are deliberately smaller than the model prompts that
produce them.  A record can be accepted only after its own shape is valid and
the corresponding coverage validator has checked it against canonical source
units.  This keeps schema-valid but misaligned model output out of packaging.
"""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping, Sequence, TypeVar

from .srt import has_japanese


SCENE_SCHEMA_VERSION = "scene-v2"
SEMANTIC_FRAME_SCHEMA_VERSION = "1"

EXPECTED_TRANSLATION_DELTAS = frozenset({"none", "minor", "major", "critical"})
UTTERANCE_TYPES = frozenset({"dialogue", "narration", "reaction", "song", "on_screen_text", "unknown"})
SPEECH_ACTS = frozenset({"statement", "question", "request", "command", "response", "refusal", "permission", "reaction", "unknown"})
VISUAL_RESOLVABLE_SLOTS = frozenset(
    {
        "speaker",
        "addressee",
        "deictic_location",
        "deictic_referent",
        "on_screen_text",
        "scene_continuity",
    }
)
SEMANTIC_SLOT_NAMES = frozenset(
    {
        "utterance_type",
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
    }
)
STYLE_PROHIBITED_SEMANTIC_SLOTS = frozenset(
    {"actor", "target", "action", "relation", "consent", "location"}
)
SEMANTIC_DRIFT_CATEGORIES = frozenset(
    {
        "addition",
        "omission",
        "polarity",
        "speech_act",
        "question_statement",
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
        "numeric_token",
        "unsupported_relation_or_result",
    }
)
SEMANTIC_DRIFT_SEVERITIES = frozenset({"minor", "major", "critical", "unknown"})
KOREAN_DIALOGUE_CATEGORIES = frozenset(
    {
        "translationese_word_order",
        "redundant_subject_or_pronoun",
        "awkward_particle_usage",
        "unnatural_omission",
        "over_explained_dialogue",
        "response_mismatch",
        "awkward_interjection",
        "unnatural_ending",
        "register_inconsistency",
        "address_term_inconsistency",
        "speaker_voice_inconsistency",
        "scene_coherence",
        "subtitle_rhythm",
        "verbosity",
        "bad_cue_segmentation",
    }
)
KOREAN_DIALOGUE_SEVERITIES = frozenset({"minor", "major"})
REPAIR_SCOPES = frozenset({"dialogue_text", "segmentation", "register", "span", "none"})


class SceneModelError(ValueError):
    """Raised when a scene-v2 artifact violates a local or coverage contract."""


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SceneModelError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _strings(value: object, field_name: str, *, non_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise SceneModelError(f"{field_name} must be an array of strings")
    result = tuple(_required_text(item, f"{field_name}[]") for item in value)
    if non_empty and not result:
        raise SceneModelError(f"{field_name} must not be empty")
    if len(set(result)) != len(result):
        raise SceneModelError(f"{field_name} must not contain duplicates")
    return result


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SceneModelError(f"{field_name} must be an object")
    return dict(value)


def _strict_mapping(
    row: Mapping[str, Any], *, required: frozenset[str], optional: frozenset[str] = frozenset()
) -> None:
    missing = required - set(row)
    unknown = set(row) - required - optional
    if missing:
        raise SceneModelError(f"missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise SceneModelError(f"unknown fields: {', '.join(sorted(unknown))}")


def _unit_id(unit: object) -> str:
    value = unit.get("unit_id") if isinstance(unit, Mapping) else getattr(unit, "unit_id", None)
    return _required_text(value, "unit_id")


def _unit_timing(unit: object) -> tuple[float, float]:
    if isinstance(unit, Mapping):
        start, end = unit.get("start"), unit.get("end")
    else:
        start, end = getattr(unit, "start", None), getattr(unit, "end", None)
    try:
        result = float(start), float(end)
    except (TypeError, ValueError) as exc:
        raise SceneModelError("unit timing must be numeric") from exc
    if not all(math.isfinite(value) for value in result) or result[0] < 0 or result[1] <= result[0]:
        raise SceneModelError("unit timing is invalid")
    return result


def _finite_timing(start: object, end: object) -> tuple[float, float]:
    try:
        values = float(start), float(end)
    except (TypeError, ValueError) as exc:
        raise SceneModelError("scene timing must be numeric") from exc
    if not all(math.isfinite(value) for value in values) or values[0] < 0 or values[1] <= values[0]:
        raise SceneModelError("scene timing is invalid")
    return values


@dataclass(frozen=True, slots=True)
class DialogueScene:
    schema_version: str
    scene_id: str
    unit_ids: tuple[str, ...]
    start: float
    end: float
    boundary_before: str
    boundary_after: str
    boundary_evidence: tuple[str, ...]
    speaker_candidates: tuple[str, ...]
    source_evidence_refs: tuple[str, ...]

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "scene_id",
            "unit_ids",
            "start",
            "end",
            "boundary_before",
            "boundary_after",
            "boundary_evidence",
            "speaker_candidates",
            "source_evidence_refs",
        }
    )

    def __post_init__(self) -> None:
        if self.schema_version != SCENE_SCHEMA_VERSION:
            raise SceneModelError(f"unsupported dialogue scene schema: {self.schema_version}")
        _required_text(self.scene_id, "scene_id")
        _strings(self.unit_ids, "unit_ids", non_empty=True)
        _finite_timing(self.start, self.end)
        _required_text(self.boundary_before, "boundary_before")
        _required_text(self.boundary_after, "boundary_after")
        _strings(self.boundary_evidence, "boundary_evidence")
        _strings(self.speaker_candidates, "speaker_candidates")
        _strings(self.source_evidence_refs, "source_evidence_refs")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "DialogueScene":
        _strict_mapping(row, required=cls._REQUIRED)
        return cls(
            schema_version=_required_text(row["schema_version"], "schema_version"),
            scene_id=_required_text(row["scene_id"], "scene_id"),
            unit_ids=_strings(row["unit_ids"], "unit_ids", non_empty=True),
            start=float(row["start"]),
            end=float(row["end"]),
            boundary_before=_required_text(row["boundary_before"], "boundary_before"),
            boundary_after=_required_text(row["boundary_after"], "boundary_after"),
            boundary_evidence=_strings(row["boundary_evidence"], "boundary_evidence"),
            speaker_candidates=_strings(row["speaker_candidates"], "speaker_candidates"),
            source_evidence_refs=_strings(row["source_evidence_refs"], "source_evidence_refs"),
        )

    def json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "unit_ids": list(self.unit_ids),
            "start": self.start,
            "end": self.end,
            "boundary_before": self.boundary_before,
            "boundary_after": self.boundary_after,
            "boundary_evidence": list(self.boundary_evidence),
            "speaker_candidates": list(self.speaker_candidates),
            "source_evidence_refs": list(self.source_evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class SemanticFrame:
    schema_version: str
    scene_id: str
    unit_id: str
    utterance_type: str
    semantic_summary: str
    speech_act: str
    question: str
    polarity: str
    refusal_permission: str
    stop_continue: str
    command_strength: str
    speaker: str | None
    addressee: str | None
    actor: str | None
    action: str | None
    target: str | None
    location: str | None
    direction: str | None
    tense_aspect: str | None
    completion: str
    intensity: str | None
    numeric_tokens: tuple[str, ...]
    register: str
    response_to_unit_id: str | None
    continues_from_unit_id: str | None
    continues_to_unit_id: str | None
    must_preserve: tuple[str, ...]
    uncertain_slots: tuple[str, ...]
    competing_interpretations: tuple[str, ...]
    visual_resolvable_slots: tuple[str, ...]
    expected_translation_delta: str
    audio_only_uncertainty: bool
    confidence: float | str
    evidence_refs: tuple[str, ...]

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version", "scene_id", "unit_id", "utterance_type", "semantic_summary",
            "speech_act", "question", "polarity", "refusal_permission", "stop_continue",
            "command_strength", "speaker", "addressee", "actor", "action", "target",
            "location", "direction", "tense_aspect", "completion", "intensity",
            "numeric_tokens", "register", "response_to_unit_id", "continues_from_unit_id",
            "continues_to_unit_id", "must_preserve", "uncertain_slots",
            "competing_interpretations", "visual_resolvable_slots",
            "expected_translation_delta", "audio_only_uncertainty", "confidence", "evidence_refs",
        }
    )

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_FRAME_SCHEMA_VERSION:
            raise SceneModelError(f"unsupported semantic frame schema: {self.schema_version}")
        _required_text(self.scene_id, "scene_id")
        _required_text(self.unit_id, "unit_id")
        _required_text(self.utterance_type, "utterance_type")
        _required_text(self.semantic_summary, "semantic_summary")
        if self.utterance_type not in UTTERANCE_TYPES:
            raise SceneModelError("utterance_type is invalid")
        for name in (
            "speaker", "addressee", "actor", "action", "target", "location", "direction",
            "tense_aspect", "intensity", "response_to_unit_id",
            "continues_from_unit_id", "continues_to_unit_id",
        ):
            _optional_text(getattr(self, name), name)
        if self.speech_act not in SPEECH_ACTS:
            raise SceneModelError("speech_act is invalid")
        if self.polarity not in {"positive", "negative", "mixed", "unknown"}:
            raise SceneModelError("polarity is invalid")
        if self.refusal_permission not in {"refusal", "permission", "neither", "unknown"}:
            raise SceneModelError("refusal_permission is invalid")
        if self.stop_continue not in {"stop", "continue", "neither", "unknown"}:
            raise SceneModelError("stop_continue is invalid")
        if self.command_strength not in {"none", "suggestion", "request", "command", "unknown"}:
            raise SceneModelError("command_strength is invalid")
        if self.question not in {"yes", "no", "unknown"}:
            raise SceneModelError("question must be yes, no, or unknown")
        if self.completion not in {"complete", "incomplete", "ongoing", "unknown"}:
            raise SceneModelError("completion is invalid")
        if self.register not in {"formal", "informal", "neutral", "unknown"}:
            raise SceneModelError("register is invalid")
        if not isinstance(self.audio_only_uncertainty, bool):
            raise SceneModelError("audio_only_uncertainty must be boolean")
        for name in (
            "numeric_tokens", "must_preserve", "uncertain_slots", "competing_interpretations",
            "visual_resolvable_slots", "evidence_refs",
        ):
            _strings(getattr(self, name), name)
        if not self.evidence_refs:
            raise SceneModelError("evidence_refs must not be empty")
        if not self.must_preserve:
            raise SceneModelError("must_preserve must not be empty")
        if not set(self.visual_resolvable_slots) <= VISUAL_RESOLVABLE_SLOTS:
            raise SceneModelError("visual_resolvable_slots contains a non-visual slot")
        if self.expected_translation_delta not in EXPECTED_TRANSLATION_DELTAS:
            raise SceneModelError("expected_translation_delta is invalid")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float, str)):
            raise SceneModelError("confidence must be 0..1 or high/medium/low/unknown")
        if isinstance(self.confidence, (int, float)):
            if not math.isfinite(float(self.confidence)) or not 0 <= float(self.confidence) <= 1:
                raise SceneModelError("numeric confidence must be between 0 and 1")
        elif self.confidence not in {"high", "medium", "low", "unknown"}:
            raise SceneModelError("string confidence must be high, medium, low, or unknown")
        for slot in set(self.must_preserve) & SEMANTIC_SLOT_NAMES:
            value = getattr(self, slot)
            if value is None or value == () or value == "":
                raise SceneModelError(f"must_preserve slot {slot} has no value")
        if set(self.must_preserve) & set(self.uncertain_slots):
            raise SceneModelError("must_preserve and uncertain_slots contradict each other")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SemanticFrame":
        _strict_mapping(row, required=cls._REQUIRED)
        confidence = row["confidence"]
        if isinstance(confidence, int) and not isinstance(confidence, bool):
            confidence = float(confidence)
        return cls(
            schema_version=_required_text(row["schema_version"], "schema_version"),
            scene_id=_required_text(row["scene_id"], "scene_id"),
            unit_id=_required_text(row["unit_id"], "unit_id"),
            utterance_type=_required_text(row["utterance_type"], "utterance_type"),
            semantic_summary=_required_text(row["semantic_summary"], "semantic_summary"),
            speech_act=_required_text(row["speech_act"], "speech_act"),
            question=_required_text(row["question"], "question"),
            polarity=_required_text(row["polarity"], "polarity"),
            refusal_permission=_required_text(row["refusal_permission"], "refusal_permission"),
            stop_continue=_required_text(row["stop_continue"], "stop_continue"),
            command_strength=_required_text(row["command_strength"], "command_strength"),
            speaker=_optional_text(row["speaker"], "speaker"),
            addressee=_optional_text(row["addressee"], "addressee"),
            actor=_optional_text(row["actor"], "actor"),
            action=_optional_text(row["action"], "action"),
            target=_optional_text(row["target"], "target"),
            location=_optional_text(row["location"], "location"),
            direction=_optional_text(row["direction"], "direction"),
            tense_aspect=_optional_text(row["tense_aspect"], "tense_aspect"),
            completion=_required_text(row["completion"], "completion"),
            intensity=_optional_text(row["intensity"], "intensity"),
            numeric_tokens=_strings(row["numeric_tokens"], "numeric_tokens"),
            register=_required_text(row["register"], "register"),
            response_to_unit_id=_optional_text(row["response_to_unit_id"], "response_to_unit_id"),
            continues_from_unit_id=_optional_text(row["continues_from_unit_id"], "continues_from_unit_id"),
            continues_to_unit_id=_optional_text(row["continues_to_unit_id"], "continues_to_unit_id"),
            must_preserve=_strings(row["must_preserve"], "must_preserve"),
            uncertain_slots=_strings(row["uncertain_slots"], "uncertain_slots"),
            competing_interpretations=_strings(row["competing_interpretations"], "competing_interpretations"),
            visual_resolvable_slots=_strings(row["visual_resolvable_slots"], "visual_resolvable_slots"),
            expected_translation_delta=_required_text(row["expected_translation_delta"], "expected_translation_delta"),
            audio_only_uncertainty=row["audio_only_uncertainty"],
            confidence=confidence,
            evidence_refs=_strings(row["evidence_refs"], "evidence_refs"),
        )

    def json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "unit_id": self.unit_id,
            "utterance_type": self.utterance_type,
            "semantic_summary": self.semantic_summary,
            "speech_act": self.speech_act,
            "question": self.question,
            "polarity": self.polarity,
            "refusal_permission": self.refusal_permission,
            "stop_continue": self.stop_continue,
            "command_strength": self.command_strength,
            "speaker": self.speaker,
            "addressee": self.addressee,
            "actor": self.actor,
            "action": self.action,
            "target": self.target,
            "location": self.location,
            "direction": self.direction,
            "tense_aspect": self.tense_aspect,
            "completion": self.completion,
            "intensity": self.intensity,
            "numeric_tokens": list(self.numeric_tokens),
            "register": self.register,
            "response_to_unit_id": self.response_to_unit_id,
            "continues_from_unit_id": self.continues_from_unit_id,
            "continues_to_unit_id": self.continues_to_unit_id,
            "must_preserve": list(self.must_preserve),
            "uncertain_slots": list(self.uncertain_slots),
            "competing_interpretations": list(self.competing_interpretations),
            "visual_resolvable_slots": list(self.visual_resolvable_slots),
            "expected_translation_delta": self.expected_translation_delta,
            "audio_only_uncertainty": self.audio_only_uncertainty,
            "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class SceneDialogueTurn:
    scene_id: str
    turn_id: str
    source_unit_ids: tuple[str, ...]
    speaker_id: str | None
    korean: str
    relation_to_previous: str | None = None
    segmentation_hint: str | None = None

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {"scene_id", "turn_id", "source_unit_ids", "speaker_id", "korean", "relation_to_previous", "segmentation_hint"}
    )

    def __post_init__(self) -> None:
        _required_text(self.scene_id, "scene_id")
        _required_text(self.turn_id, "turn_id")
        _strings(self.source_unit_ids, "source_unit_ids", non_empty=True)
        _optional_text(self.speaker_id, "speaker_id")
        korean = _required_text(self.korean, "korean")
        if has_japanese(korean):
            raise SceneModelError("korean dialogue turn contains Japanese text")
        if not re.search(r"[가-힣]", korean):
            raise SceneModelError("korean dialogue turn must contain Korean text")
        _optional_text(self.relation_to_previous, "relation_to_previous")
        _optional_text(self.segmentation_hint, "segmentation_hint")
        if self.relation_to_previous not in {"new", "response", "continuation", "overlap", "unknown"}:
            raise SceneModelError("relation_to_previous is invalid")
        if self.segmentation_hint not in {"single_cue", "split_across_cues", "merge_with_next", "model_decide"}:
            raise SceneModelError("segmentation_hint is invalid")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SceneDialogueTurn":
        _strict_mapping(row, required=cls._REQUIRED)
        return cls(
            scene_id=_required_text(row["scene_id"], "scene_id"),
            turn_id=_required_text(row["turn_id"], "turn_id"),
            source_unit_ids=_strings(row["source_unit_ids"], "source_unit_ids", non_empty=True),
            speaker_id=_optional_text(row["speaker_id"], "speaker_id"),
            korean=_required_text(row["korean"], "korean"),
            relation_to_previous=_optional_text(row["relation_to_previous"], "relation_to_previous"),
            segmentation_hint=_optional_text(row["segmentation_hint"], "segmentation_hint"),
        )

    def json(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "turn_id": self.turn_id,
            "source_unit_ids": list(self.source_unit_ids),
            "speaker_id": self.speaker_id,
            "korean": self.korean,
            "relation_to_previous": self.relation_to_previous,
            "segmentation_hint": self.segmentation_hint,
        }


@dataclass(frozen=True, slots=True)
class SceneCueProjection:
    scene_id: str
    unit_id: str
    text: str
    source_turn_ids: tuple[str, ...]
    projection_notes: tuple[str, ...] = ()

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {"scene_id", "unit_id", "text", "source_turn_ids", "projection_notes"}
    )

    def __post_init__(self) -> None:
        _required_text(self.scene_id, "scene_id")
        _required_text(self.unit_id, "unit_id")
        text = _required_text(self.text, "text")
        if has_japanese(text):
            raise SceneModelError("scene cue projection contains Japanese text")
        if not re.search(r"[가-힣]", text):
            raise SceneModelError("scene cue projection must contain Korean text")
        _strings(self.source_turn_ids, "source_turn_ids", non_empty=True)
        _strings(self.projection_notes, "projection_notes")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SceneCueProjection":
        _strict_mapping(row, required=cls._REQUIRED)
        return cls(
            scene_id=_required_text(row["scene_id"], "scene_id"),
            unit_id=_required_text(row["unit_id"], "unit_id"),
            text=_required_text(row["text"], "text"),
            source_turn_ids=_strings(row["source_turn_ids"], "source_turn_ids", non_empty=True),
            projection_notes=_strings(row["projection_notes"], "projection_notes"),
        )

    def json(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "unit_id": self.unit_id,
            "text": self.text,
            "source_turn_ids": list(self.source_turn_ids),
            "projection_notes": list(self.projection_notes),
        }


@dataclass(frozen=True, slots=True)
class VisualSemanticObservation:
    scene_id: str
    unit_id: str
    observed_slots: tuple[Mapping[str, str], ...]
    unresolved_slots: tuple[str, ...]
    confidence: str
    frame_evidence_refs: tuple[str, ...]
    unsupported_inference_warnings: tuple[str, ...]

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {"scene_id", "unit_id", "observed_slots", "unresolved_slots", "confidence", "frame_evidence_refs", "unsupported_inference_warnings"}
    )

    def __post_init__(self) -> None:
        _required_text(self.scene_id, "scene_id")
        _required_text(self.unit_id, "unit_id")
        if not isinstance(self.observed_slots, tuple):
            raise SceneModelError("observed_slots must be an array")
        observed_pairs: set[tuple[str, str]] = set()
        for item in self.observed_slots:
            if not isinstance(item, Mapping) or set(item) != {"slot", "observation"}:
                raise SceneModelError("observed_slots entries require slot and observation")
            slot = _required_text(item.get("slot"), "observed_slots[].slot")
            observation = _required_text(
                item.get("observation"), "observed_slots[].observation"
            )
            if slot not in VISUAL_RESOLVABLE_SLOTS:
                raise SceneModelError("visual observation contains a non-visual slot")
            if (slot, observation) in observed_pairs:
                raise SceneModelError("observed_slots must not contain duplicates")
            observed_pairs.add((slot, observation))
        _strings(self.unresolved_slots, "unresolved_slots")
        if not set(self.unresolved_slots) <= VISUAL_RESOLVABLE_SLOTS:
            raise SceneModelError("unresolved_slots contains a non-visual slot")
        if self.confidence not in {"high", "medium", "low", "unknown"}:
            raise SceneModelError("visual confidence is invalid")
        _strings(self.frame_evidence_refs, "frame_evidence_refs")
        if not self.frame_evidence_refs:
            raise SceneModelError("frame_evidence_refs must not be empty")
        _strings(self.unsupported_inference_warnings, "unsupported_inference_warnings")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "VisualSemanticObservation":
        _strict_mapping(row, required=cls._REQUIRED)
        observed_slots = row["observed_slots"]
        if not isinstance(observed_slots, (list, tuple)):
            raise SceneModelError("observed_slots must be an array")
        return cls(
            scene_id=_required_text(row["scene_id"], "scene_id"),
            unit_id=_required_text(row["unit_id"], "unit_id"),
            observed_slots=tuple(
                dict(_mapping(item, "observed_slots[]")) for item in observed_slots
            ),
            unresolved_slots=_strings(row["unresolved_slots"], "unresolved_slots"),
            confidence=_required_text(row["confidence"], "confidence"),
            frame_evidence_refs=_strings(row["frame_evidence_refs"], "frame_evidence_refs"),
            unsupported_inference_warnings=_strings(row["unsupported_inference_warnings"], "unsupported_inference_warnings"),
        )

    def json(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "unit_id": self.unit_id,
            "observed_slots": [dict(item) for item in self.observed_slots],
            "unresolved_slots": list(self.unresolved_slots),
            "confidence": self.confidence,
            "frame_evidence_refs": list(self.frame_evidence_refs),
            "unsupported_inference_warnings": list(self.unsupported_inference_warnings),
        }


@dataclass(frozen=True, slots=True)
class SemanticDriftIssue:
    scene_id: str
    unit_ids: tuple[str, ...]
    target_span: str
    category: str
    severity: str
    expected: str
    observed: str
    evidence_refs: tuple[str, ...]
    verdict: str

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {"scene_id", "unit_ids", "target_span", "category", "severity", "expected", "observed", "evidence_refs", "verdict"}
    )

    def __post_init__(self) -> None:
        _required_text(self.scene_id, "scene_id")
        _strings(self.unit_ids, "unit_ids", non_empty=True)
        _required_text(self.target_span, "target_span")
        if self.category not in SEMANTIC_DRIFT_CATEGORIES:
            raise SceneModelError("semantic critic returned a non-semantic category")
        if self.severity not in SEMANTIC_DRIFT_SEVERITIES:
            raise SceneModelError("semantic issue severity is invalid")
        _required_text(self.expected, "expected")
        _required_text(self.observed, "observed")
        _strings(self.evidence_refs, "evidence_refs")
        if not self.evidence_refs:
            raise SceneModelError("semantic issue evidence_refs must not be empty")
        _required_text(self.verdict, "verdict")
        if self.verdict not in {"pass", "issue", "uncertain"}:
            raise SceneModelError("semantic issue verdict is invalid")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SemanticDriftIssue":
        _strict_mapping(row, required=cls._REQUIRED)
        return cls(
            scene_id=_required_text(row["scene_id"], "scene_id"),
            unit_ids=_strings(row["unit_ids"], "unit_ids", non_empty=True),
            target_span=_required_text(row["target_span"], "target_span"),
            category=_required_text(row["category"], "category"),
            severity=_required_text(row["severity"], "severity"),
            expected=_required_text(row["expected"], "expected"),
            observed=_required_text(row["observed"], "observed"),
            evidence_refs=_strings(row["evidence_refs"], "evidence_refs"),
            verdict=_required_text(row["verdict"], "verdict"),
        )

    def json(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "unit_ids": list(self.unit_ids),
            "target_span": self.target_span,
            "category": self.category,
            "severity": self.severity,
            "expected": self.expected,
            "observed": self.observed,
            "evidence_refs": list(self.evidence_refs),
            "verdict": self.verdict,
        }


@dataclass(frozen=True, slots=True)
class KoreanDialogueIssue:
    scene_id: str
    unit_ids: tuple[str, ...]
    target_span: str
    category: str
    severity: str
    reason: str
    repair_scope: str

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {"scene_id", "unit_ids", "target_span", "category", "severity", "reason", "repair_scope"}
    )

    def __post_init__(self) -> None:
        _required_text(self.scene_id, "scene_id")
        _strings(self.unit_ids, "unit_ids", non_empty=True)
        _required_text(self.target_span, "target_span")
        if self.category not in KOREAN_DIALOGUE_CATEGORIES:
            raise SceneModelError("Korean dialogue issue category is invalid")
        if self.severity not in KOREAN_DIALOGUE_SEVERITIES:
            raise SceneModelError("Korean dialogue issue severity is invalid")
        _required_text(self.reason, "reason")
        if self.repair_scope not in REPAIR_SCOPES:
            raise SceneModelError("repair_scope is invalid")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "KoreanDialogueIssue":
        _strict_mapping(row, required=cls._REQUIRED)
        return cls(
            scene_id=_required_text(row["scene_id"], "scene_id"),
            unit_ids=_strings(row["unit_ids"], "unit_ids", non_empty=True),
            target_span=_required_text(row["target_span"], "target_span"),
            category=_required_text(row["category"], "category"),
            severity=_required_text(row["severity"], "severity"),
            reason=_required_text(row["reason"], "reason"),
            repair_scope=_required_text(row["repair_scope"], "repair_scope"),
        )

    def json(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "unit_ids": list(self.unit_ids),
            "target_span": self.target_span,
            "category": self.category,
            "severity": self.severity,
            "reason": self.reason,
            "repair_scope": self.repair_scope,
        }


def validate_scene_partition(units: Sequence[object], scenes: Sequence[DialogueScene]) -> None:
    """Validate exact, ordered, non-overlapping scene coverage of source units."""

    if not units:
        raise SceneModelError("scene partition requires source units")
    if not scenes:
        raise SceneModelError("scene partition must not be empty")
    expected = tuple(_unit_id(unit) for unit in units)
    if len(set(expected)) != len(expected):
        raise SceneModelError("source units contain duplicate unit IDs")
    actual = tuple(unit_id for scene in scenes for unit_id in scene.unit_ids)
    if actual != expected:
        raise SceneModelError("scenes must cover every source unit exactly once and in order")
    scene_ids = tuple(scene.scene_id for scene in scenes)
    if len(set(scene_ids)) != len(scene_ids):
        raise SceneModelError("scene IDs must be unique")
    unit_by_id = {_unit_id(unit): unit for unit in units}
    for scene in scenes:
        first_start = _unit_timing(unit_by_id[scene.unit_ids[0]])[0]
        last_end = _unit_timing(unit_by_id[scene.unit_ids[-1]])[1]
        if not math.isclose(scene.start, first_start, abs_tol=1e-9):
            raise SceneModelError(f"{scene.scene_id}: start does not match its first unit")
        if not math.isclose(scene.end, last_end, abs_tol=1e-9):
            raise SceneModelError(f"{scene.scene_id}: end does not match its last unit")


def validate_semantic_frame_coverage(
    scene: DialogueScene,
    frames: Sequence[SemanticFrame],
    *,
    valid_evidence_refs: Iterable[str],
) -> None:
    expected = scene.unit_ids
    actual = tuple(frame.unit_id for frame in frames)
    if actual != expected:
        raise SceneModelError("semantic frames must cover the scene exactly once and in order")
    allowed_refs = set(valid_evidence_refs)
    for frame in frames:
        if frame.scene_id != scene.scene_id:
            raise SceneModelError("semantic frame references another scene")
        invalid_refs = set(frame.evidence_refs) - allowed_refs
        if invalid_refs:
            raise SceneModelError(f"semantic frame contains unknown evidence refs: {sorted(invalid_refs)}")
        if any(ref.startswith("style-memory:") for ref in frame.evidence_refs):
            raise SceneModelError("style memory is not semantic evidence")
        for linked_id in (
            frame.response_to_unit_id,
            frame.continues_from_unit_id,
            frame.continues_to_unit_id,
        ):
            if linked_id is not None and linked_id not in expected:
                raise SceneModelError("semantic frame links to a unit outside its scene")


def validate_scene_dialogue_coverage(
    scene: DialogueScene, turns: Sequence[SceneDialogueTurn]
) -> None:
    if not turns:
        raise SceneModelError("scene realization must contain dialogue turns")
    actual = tuple(unit_id for turn in turns for unit_id in turn.source_unit_ids)
    if actual != scene.unit_ids:
        raise SceneModelError("dialogue turns must consume scene units exactly once and in order")
    turn_ids: set[str] = set()
    for turn in turns:
        if turn.scene_id != scene.scene_id:
            raise SceneModelError("dialogue turn references another scene")
        if turn.turn_id in turn_ids:
            raise SceneModelError("dialogue turn IDs must be unique within a scene")
        turn_ids.add(turn.turn_id)


def validate_scene_cue_projection(
    scene: DialogueScene,
    projections: Sequence[SceneCueProjection],
    *,
    turns: Sequence[SceneDialogueTurn] | None = None,
) -> None:
    actual = tuple(projection.unit_id for projection in projections)
    if actual != scene.unit_ids:
        raise SceneModelError("cue projections must cover scene units exactly once and in order")
    valid_turn_ids = {turn.turn_id for turn in turns or ()}
    seen_by_text: dict[str, set[str]] = {}
    for projection in projections:
        if projection.scene_id != scene.scene_id:
            raise SceneModelError("cue projection references another scene")
        if valid_turn_ids and not set(projection.source_turn_ids) <= valid_turn_ids:
            raise SceneModelError("cue projection references an unknown source turn")
        normalized = " ".join(projection.text.split())
        prior_turn_ids = seen_by_text.get(normalized, set())
        if prior_turn_ids & set(projection.source_turn_ids):
            raise SceneModelError("the same complete turn text was duplicated across cues")
        seen_by_text.setdefault(normalized, set()).update(projection.source_turn_ids)


ModelT = TypeVar("ModelT")


def write_model_jsonl(path: Path, records: Iterable[object]) -> Path:
    """Write dataclass artifacts as deterministic UTF-8/LF JSONL."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for record in records:
        serializer = getattr(record, "json", None)
        if not callable(serializer):
            raise SceneModelError("JSONL record does not provide json()")
        rows.append(json.dumps(serializer(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    destination.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8", newline="\n")
    return destination


def load_model_jsonl(path: Path, model: type[ModelT]) -> tuple[ModelT, ...]:
    """Read strict object-per-line artifacts using a model's from_mapping()."""

    records: list[ModelT] = []
    parser = getattr(model, "from_mapping", None)
    if not callable(parser):
        raise SceneModelError("JSONL model does not provide from_mapping()")
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise SceneModelError("JSONL row must be an object")
            records.append(parser(row))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SceneModelError(f"{path}:{line_number}: invalid scene artifact") from exc
    return tuple(records)


__all__ = [
    "DialogueScene",
    "EXPECTED_TRANSLATION_DELTAS",
    "KOREAN_DIALOGUE_CATEGORIES",
    "KoreanDialogueIssue",
    "REPAIR_SCOPES",
    "SCENE_SCHEMA_VERSION",
    "SEMANTIC_DRIFT_CATEGORIES",
    "SEMANTIC_FRAME_SCHEMA_VERSION",
    "SEMANTIC_SLOT_NAMES",
    "STYLE_PROHIBITED_SEMANTIC_SLOTS",
    "SceneCueProjection",
    "SceneDialogueTurn",
    "SceneModelError",
    "SemanticDriftIssue",
    "SemanticFrame",
    "VISUAL_RESOLVABLE_SLOTS",
    "VisualSemanticObservation",
    "load_model_jsonl",
    "validate_scene_cue_projection",
    "validate_scene_dialogue_coverage",
    "validate_scene_partition",
    "validate_semantic_frame_coverage",
    "write_model_jsonl",
]
