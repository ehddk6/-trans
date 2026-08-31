from __future__ import annotations

"""Confirmed-only speaker style and bounded dialogue-state memory."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .scene_models import (
    STYLE_PROHIBITED_SEMANTIC_SLOTS,
    SceneModelError,
    load_model_jsonl,
    write_model_jsonl,
)


CONFIRMED_STYLE_STATUSES = frozenset(
    {"confirmed", "human-approved", "human_approved", "approved"}
)
DIALOGUE_MEMORY_STATUSES = frozenset(
    {*CONFIRMED_STYLE_STATUSES, "provisional-style-only", "machine-draft", "rejected"}
)
_ALLOWED_TURN_KEYS = frozenset({"turn_id", "speaker_id", "korean"})


def _text(value: object, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SceneModelError(f"{name} must be a non-empty string")
    return value.strip()


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or isinstance(value, (str, bytes)):
        raise SceneModelError(f"{name} must be an array")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise SceneModelError(f"{name} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise SceneModelError(f"{name} must not contain duplicates")
    return result


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SceneModelError(f"{name} must be an object")
    return dict(value)


@dataclass(frozen=True, slots=True)
class SpeakerStyleProfile:
    title_id: str
    speaker_id: str
    status: str
    register: str | None
    first_person: str | None
    second_person: str | None
    address_terms: tuple[str, ...]
    common_endings: tuple[str, ...]
    average_utterance_length: float | None
    interjection_tendencies: tuple[str, ...]
    prohibited_inferences: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        _text(self.title_id, "title_id")
        _text(self.speaker_id, "speaker_id")
        _text(self.status, "status")
        _text(self.register, "register", optional=True)
        _text(self.first_person, "first_person", optional=True)
        _text(self.second_person, "second_person", optional=True)
        for name in (
            "address_terms", "common_endings", "interjection_tendencies",
            "prohibited_inferences", "evidence_refs",
        ):
            _strings(getattr(self, name), name)
        if self.average_utterance_length is not None:
            if isinstance(self.average_utterance_length, bool) or not isinstance(
                self.average_utterance_length, (int, float)
            ):
                raise SceneModelError("average_utterance_length must be numeric or null")
            if self.average_utterance_length < 0:
                raise SceneModelError("average_utterance_length must not be negative")
        _object(self.provenance, "provenance")
        missing_prohibitions = STYLE_PROHIBITED_SEMANTIC_SLOTS - set(self.prohibited_inferences)
        if missing_prohibitions:
            raise SceneModelError(
                "style profile must explicitly prohibit semantic inference for: "
                + ", ".join(sorted(missing_prohibitions))
            )

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SpeakerStyleProfile":
        required = {
            "title_id", "speaker_id", "status", "register", "first_person", "second_person",
            "address_terms", "common_endings", "average_utterance_length",
            "interjection_tendencies", "prohibited_inferences", "evidence_refs", "provenance",
        }
        missing, unknown = required - set(row), set(row) - required
        if missing or unknown:
            raise SceneModelError(
                f"invalid style profile fields; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        average = row["average_utterance_length"]
        return cls(
            title_id=str(_text(row["title_id"], "title_id")),
            speaker_id=str(_text(row["speaker_id"], "speaker_id")),
            status=str(_text(row["status"], "status")),
            register=_text(row["register"], "register", optional=True),
            first_person=_text(row["first_person"], "first_person", optional=True),
            second_person=_text(row["second_person"], "second_person", optional=True),
            address_terms=_strings(row["address_terms"], "address_terms"),
            common_endings=_strings(row["common_endings"], "common_endings"),
            average_utterance_length=None if average is None else float(average),
            interjection_tendencies=_strings(row["interjection_tendencies"], "interjection_tendencies"),
            prohibited_inferences=_strings(row["prohibited_inferences"], "prohibited_inferences"),
            evidence_refs=_strings(row["evidence_refs"], "evidence_refs"),
            provenance=_object(row["provenance"], "provenance"),
        )

    def json(self) -> dict[str, Any]:
        return {
            "title_id": self.title_id,
            "speaker_id": self.speaker_id,
            "status": self.status,
            "register": self.register,
            "first_person": self.first_person,
            "second_person": self.second_person,
            "address_terms": list(self.address_terms),
            "common_endings": list(self.common_endings),
            "average_utterance_length": self.average_utterance_length,
            "interjection_tendencies": list(self.interjection_tendencies),
            "prohibited_inferences": list(self.prohibited_inferences),
            "evidence_refs": list(self.evidence_refs),
            "provenance": dict(self.provenance),
        }

    @property
    def confirmed_for_use(self) -> bool:
        if self.status not in CONFIRMED_STYLE_STATUSES:
            return False
        machine_generated = self.provenance.get("machine_generated") is True
        human_approved = self.provenance.get("human_approved") is True
        return not machine_generated or human_approved


@dataclass(frozen=True, slots=True)
class SceneDialogueMemory:
    title_id: str
    scene_id: str
    speaker_turns: tuple[Mapping[str, str], ...]
    pending_question: str | None
    open_topic: str | None
    status: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        _text(self.title_id, "title_id")
        _text(self.scene_id, "scene_id")
        if self.status not in DIALOGUE_MEMORY_STATUSES:
            raise SceneModelError("dialogue memory status is invalid")
        _text(self.pending_question, "pending_question", optional=True)
        _text(self.open_topic, "open_topic", optional=True)
        _object(self.provenance, "provenance")
        if not isinstance(self.speaker_turns, tuple):
            raise SceneModelError("speaker_turns must be a tuple")
        for turn in self.speaker_turns:
            if not isinstance(turn, Mapping):
                raise SceneModelError("speaker_turns must contain objects")
            if set(turn) != _ALLOWED_TURN_KEYS:
                raise SceneModelError("speaker turn contains unsupported or semantic fields")
            for key in _ALLOWED_TURN_KEYS:
                _text(turn.get(key), f"speaker_turns[].{key}")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SceneDialogueMemory":
        required = {
            "title_id", "scene_id", "speaker_turns", "pending_question", "open_topic",
            "status", "provenance",
        }
        missing, unknown = required - set(row), set(row) - required
        if missing or unknown:
            raise SceneModelError(
                f"invalid dialogue memory fields; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        turns = row["speaker_turns"]
        if not isinstance(turns, (tuple, list)):
            raise SceneModelError("speaker_turns must be an array")
        return cls(
            title_id=str(_text(row["title_id"], "title_id")),
            scene_id=str(_text(row["scene_id"], "scene_id")),
            speaker_turns=tuple(dict(_object(turn, "speaker_turns[]")) for turn in turns),
            pending_question=_text(row["pending_question"], "pending_question", optional=True),
            open_topic=_text(row["open_topic"], "open_topic", optional=True),
            status=str(_text(row["status"], "status")),
            provenance=_object(row["provenance"], "provenance"),
        )

    def json(self) -> dict[str, Any]:
        return {
            "title_id": self.title_id,
            "scene_id": self.scene_id,
            "speaker_turns": [dict(turn) for turn in self.speaker_turns],
            "pending_question": self.pending_question,
            "open_topic": self.open_topic,
            "status": self.status,
            "provenance": dict(self.provenance),
        }


def confirmed_style_profiles(
    profiles: Iterable[SpeakerStyleProfile | Mapping[str, Any]],
    *,
    title_id: str | None = None,
    speaker_ids: Iterable[str] | None = None,
) -> tuple[SpeakerStyleProfile, ...]:
    """Select usable profiles without treating machine status as confirmation."""

    allowed_speakers = set(speaker_ids) if speaker_ids is not None else None
    result: list[SpeakerStyleProfile] = []
    seen: set[tuple[str, str]] = set()
    for raw in profiles:
        profile = raw if isinstance(raw, SpeakerStyleProfile) else SpeakerStyleProfile.from_mapping(raw)
        if not profile.confirmed_for_use:
            continue
        if title_id is not None and profile.title_id != title_id:
            continue
        if allowed_speakers is not None and profile.speaker_id not in allowed_speakers:
            continue
        key = profile.title_id, profile.speaker_id
        if key in seen:
            raise SceneModelError("multiple confirmed style profiles exist for one speaker")
        seen.add(key)
        result.append(profile)
    return tuple(result)


def assert_style_context_semantic_safe(context: Mapping[str, Any]) -> None:
    """Reject semantic facts smuggled through a style-only context."""

    prohibited = set(context) & STYLE_PROHIBITED_SEMANTIC_SLOTS
    for value in context.values():
        if isinstance(value, Mapping):
            prohibited.update(set(value) & STYLE_PROHIBITED_SEMANTIC_SLOTS)
    if prohibited:
        raise SceneModelError(
            "style context must not contain semantic slots: " + ", ".join(sorted(prohibited))
        )


def style_context_for_scene(
    profiles: Iterable[SpeakerStyleProfile | Mapping[str, Any]],
    *,
    title_id: str,
    speaker_ids: Iterable[str],
) -> tuple[dict[str, Any], ...]:
    """Return only confirmed expression preferences, never source meaning."""

    selected = confirmed_style_profiles(
        profiles, title_id=title_id, speaker_ids=speaker_ids
    )
    contexts: list[dict[str, Any]] = []
    for profile in selected:
        context = {
            "speaker_id": profile.speaker_id,
            "status": profile.status,
            "register": profile.register,
            "first_person": profile.first_person,
            "second_person": profile.second_person,
            "address_terms": list(profile.address_terms),
            "common_endings": list(profile.common_endings),
            "average_utterance_length": profile.average_utterance_length,
            "interjection_tendencies": list(profile.interjection_tendencies),
            "evidence_refs": list(profile.evidence_refs),
            "provenance": dict(profile.provenance),
        }
        assert_style_context_semantic_safe(context)
        contexts.append(context)
    return tuple(contexts)


def dialogue_memory_context(
    memory: SceneDialogueMemory | Mapping[str, Any], *, max_previous_turns: int = 4
) -> dict[str, Any]:
    """Return at most four prior Korean turns for style continuity only."""

    if not 1 <= max_previous_turns <= 4:
        raise SceneModelError("max_previous_turns must be between 1 and 4")
    record = memory if isinstance(memory, SceneDialogueMemory) else SceneDialogueMemory.from_mapping(memory)
    usable = record.status in CONFIRMED_STYLE_STATUSES | {"provisional-style-only"}
    if not usable:
        return {"status": "ignored", "previous_turns": []}
    context = {
        "status": record.status,
        "scene_id": record.scene_id,
        "usage": "style-only",
        "previous_turns": [dict(turn) for turn in record.speaker_turns[-max_previous_turns:]],
        "provenance": dict(record.provenance),
    }
    assert_style_context_semantic_safe(context)
    return context


def write_style_profiles_jsonl(path: Path, profiles: Iterable[SpeakerStyleProfile]) -> Path:
    return write_model_jsonl(path, profiles)


def load_style_profiles_jsonl(path: Path) -> tuple[SpeakerStyleProfile, ...]:
    return load_model_jsonl(path, SpeakerStyleProfile)


def write_scene_dialogue_memory_jsonl(path: Path, records: Iterable[SceneDialogueMemory]) -> Path:
    return write_model_jsonl(path, records)


def load_scene_dialogue_memory_jsonl(path: Path) -> tuple[SceneDialogueMemory, ...]:
    return load_model_jsonl(path, SceneDialogueMemory)


__all__ = [
    "CONFIRMED_STYLE_STATUSES",
    "DIALOGUE_MEMORY_STATUSES",
    "SceneDialogueMemory",
    "SpeakerStyleProfile",
    "assert_style_context_semantic_safe",
    "confirmed_style_profiles",
    "dialogue_memory_context",
    "load_scene_dialogue_memory_jsonl",
    "load_style_profiles_jsonl",
    "style_context_for_scene",
    "write_scene_dialogue_memory_jsonl",
    "write_style_profiles_jsonl",
]
