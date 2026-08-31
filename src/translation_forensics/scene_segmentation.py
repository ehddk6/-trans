from __future__ import annotations

"""Deterministic scene construction and targeted-visual selection for scene-v2."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .scene_models import (
    SCENE_SCHEMA_VERSION,
    VISUAL_RESOLVABLE_SLOTS,
    DialogueScene,
    SceneModelError,
    SemanticFrame,
    load_model_jsonl,
    validate_scene_partition,
    write_model_jsonl,
)


@dataclass(frozen=True, slots=True)
class SceneSegmentationConfig:
    gap_threshold_seconds: float = 4.0
    max_units: int = 24
    max_source_characters: int = 12_000

    def __post_init__(self) -> None:
        if self.gap_threshold_seconds <= 0:
            raise SceneModelError("gap_threshold_seconds must be positive")
        if self.max_units < 1:
            raise SceneModelError("max_units must be at least 1")
        if self.max_source_characters < 1:
            raise SceneModelError("max_source_characters must be at least 1")


@dataclass(frozen=True, slots=True)
class SceneBoundaryHint:
    reason: str
    evidence_refs: tuple[str, ...] = ()
    hard_boundary: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise SceneModelError("scene boundary hint reason must not be empty")
        if not isinstance(self.hard_boundary, bool):
            raise SceneModelError("hard_boundary must be boolean")
        if not isinstance(self.evidence_refs, tuple) or any(
            not isinstance(value, str) or not value.strip() for value in self.evidence_refs
        ):
            raise SceneModelError("scene boundary evidence refs must be non-empty strings")


@dataclass(slots=True)
class _SceneDraft:
    units: list[object]
    boundary_before: str
    boundary_after: str = "end_of_title"
    boundary_evidence: list[str] | None = None

    def __post_init__(self) -> None:
        if self.boundary_evidence is None:
            self.boundary_evidence = []


def _value(unit: object, name: str, default: Any = None) -> Any:
    return unit.get(name, default) if isinstance(unit, Mapping) else getattr(unit, name, default)


def _unit_id(unit: object) -> str:
    value = _value(unit, "unit_id")
    if not isinstance(value, str) or not value.strip():
        raise SceneModelError("every source unit needs a non-empty unit_id")
    return value.strip()


def _unit_text(unit: object) -> str:
    value = _value(unit, "text_raw", _value(unit, "source_japanese", ""))
    if not isinstance(value, str) or not value.strip():
        raise SceneModelError(f"{_unit_id(unit)}: source text must not be empty")
    return value.strip()


def _unit_start(unit: object) -> float:
    try:
        return float(_value(unit, "start"))
    except (TypeError, ValueError) as exc:
        raise SceneModelError(f"{_unit_id(unit)}: start must be numeric") from exc


def _unit_end(unit: object) -> float:
    try:
        return float(_value(unit, "end"))
    except (TypeError, ValueError) as exc:
        raise SceneModelError(f"{_unit_id(unit)}: end must be numeric") from exc


def _unit_speaker(unit: object) -> str:
    value = _value(unit, "speaker", "speaker_unknown")
    return str(value).strip() or "speaker_unknown"


def _unit_evidence_refs(unit: object) -> tuple[str, ...]:
    value = _value(unit, "evidence_ids", ())
    if not isinstance(value, (tuple, list)):
        raise SceneModelError(f"{_unit_id(unit)}: evidence_ids must be an array")
    return tuple(str(item).strip() for item in value if str(item).strip())


def _coerce_hint(value: object) -> SceneBoundaryHint | None:
    if value is None:
        return None
    if isinstance(value, SceneBoundaryHint):
        return value
    if isinstance(value, str):
        return SceneBoundaryHint(reason=value)
    if isinstance(value, Mapping):
        refs = value.get("evidence_refs", ())
        if not isinstance(refs, (tuple, list)):
            raise SceneModelError("boundary hint evidence_refs must be an array")
        return SceneBoundaryHint(
            reason=str(value.get("reason") or ""),
            evidence_refs=tuple(str(item) for item in refs),
            hard_boundary=value.get("hard_boundary", True),
        )
    if isinstance(value, (tuple, list)):
        reasons = tuple(str(item).strip() for item in value if str(item).strip())
        if not reasons:
            return None
        return SceneBoundaryHint(reason="+".join(reasons))
    raise SceneModelError("unsupported scene boundary hint")


def _fallback_scene_id(index: int, unit_ids: Sequence[str]) -> str:
    digest = hashlib.sha256("\x1f".join(unit_ids).encode("utf-8")).hexdigest()[:12]
    return f"scene-{index:04d}-{digest}"


def _validate_explicit_scene_ids(
    units: Sequence[object], explicit_scene_ids: Mapping[str, str]
) -> dict[str, str]:
    known = {_unit_id(unit) for unit in units}
    unknown = set(explicit_scene_ids) - known
    if unknown:
        raise SceneModelError(f"explicit scene map contains unknown unit IDs: {sorted(unknown)}")
    normalized: dict[str, str] = {}
    for unit_id, raw_scene_id in explicit_scene_ids.items():
        scene_id = str(raw_scene_id).strip()
        if not scene_id:
            raise SceneModelError(f"explicit scene ID is empty for {unit_id}")
        normalized[unit_id] = scene_id
    closed: set[str] = set()
    previous: str | None = None
    for unit in units:
        current = normalized.get(_unit_id(unit))
        if current == previous:
            continue
        if previous is not None:
            closed.add(previous)
        if current is not None and current in closed:
            raise SceneModelError("an explicit scene ID must form one contiguous unit range")
        previous = current
    return normalized


def build_dialogue_scenes(
    units: Sequence[object],
    *,
    config: SceneSegmentationConfig | None = None,
    explicit_scene_ids: Mapping[str, str] | None = None,
    boundary_hints: Mapping[str, object] | None = None,
) -> tuple[DialogueScene, ...]:
    """Build reproducible scenes, honoring explicit IDs before fallback signals.

    ``boundary_hints`` is keyed by the unit that starts the new scene. A plain
    speaker change or ASR-confidence change should be passed as a soft hint and
    is intentionally not treated as a hard boundary.
    """

    source_units = tuple(units)
    if not source_units:
        raise SceneModelError("cannot construct scenes from an empty unit list")
    cfg = config or SceneSegmentationConfig()
    ids = tuple(_unit_id(unit) for unit in source_units)
    if len(set(ids)) != len(ids):
        raise SceneModelError("source unit IDs must be unique")
    for unit in source_units:
        if _unit_start(unit) < 0 or _unit_end(unit) <= _unit_start(unit):
            raise SceneModelError(f"{_unit_id(unit)}: invalid source timing")
        _unit_text(unit)
    explicit = _validate_explicit_scene_ids(source_units, explicit_scene_ids or {})
    hints = boundary_hints or {}
    unknown_hint_ids = set(hints) - set(ids)
    if unknown_hint_ids:
        raise SceneModelError(f"boundary hints contain unknown unit IDs: {sorted(unknown_hint_ids)}")

    drafts: list[_SceneDraft] = [_SceneDraft([source_units[0]], "start_of_title")]
    current_chars = len(_unit_text(source_units[0]))
    for previous, current in zip(source_units, source_units[1:]):
        current_id = _unit_id(current)
        previous_id = _unit_id(previous)
        reason: str | None = None
        evidence: list[str] = []

        previous_explicit = explicit.get(previous_id)
        current_explicit = explicit.get(current_id)
        explicit_continuity = (
            previous_explicit == current_explicit and current_explicit is not None
        )
        if previous_explicit != current_explicit and (
            previous_explicit is not None or current_explicit is not None
        ):
            reason = "explicit_scene_id"
            evidence.append(
                f"explicit_scene_id:{current_explicit or '<fallback>'}"
            )
        elif not explicit_continuity:
            hint = _coerce_hint(hints.get(current_id))
            if hint is not None and hint.hard_boundary:
                reason = f"scene_hint:{hint.reason}"
                evidence.extend(hint.evidence_refs)
            elif _unit_start(current) - _unit_end(previous) > cfg.gap_threshold_seconds:
                gap = _unit_start(current) - _unit_end(previous)
                reason = "silence_gap"
                evidence.append(f"silence_gap_seconds:{gap:.3f}")

        current_draft = drafts[-1]
        cap_reason: str | None = None
        if len(current_draft.units) >= cfg.max_units:
            cap_reason = "forced_cap:max_units"
        elif current_chars + len(_unit_text(current)) > cfg.max_source_characters:
            cap_reason = "forced_cap:max_source_characters"
        if cap_reason is not None:
            reason = cap_reason
            evidence = [cap_reason]

        if reason is None:
            current_draft.units.append(current)
            current_chars += len(_unit_text(current))
            continue
        current_draft.boundary_after = reason
        current_draft.boundary_evidence.extend(evidence)
        drafts.append(_SceneDraft([current], reason, boundary_evidence=list(evidence)))
        current_chars = len(_unit_text(current))

    explicit_occurrences: dict[str, int] = {}
    explicit_totals: dict[str, int] = {}
    for draft in drafts:
        values = {explicit.get(_unit_id(unit)) for unit in draft.units}
        values.discard(None)
        if len(values) == 1:
            explicit_id = next(iter(values))
            explicit_totals[explicit_id] = explicit_totals.get(explicit_id, 0) + 1

    scenes: list[DialogueScene] = []
    for index, draft in enumerate(drafts, 1):
        unit_ids = tuple(_unit_id(unit) for unit in draft.units)
        values = {explicit.get(unit_id) for unit_id in unit_ids}
        values.discard(None)
        if len(values) == 1:
            explicit_id = next(iter(values))
            explicit_occurrences[explicit_id] = explicit_occurrences.get(explicit_id, 0) + 1
            scene_id = (
                explicit_id
                if explicit_totals[explicit_id] == 1
                else f"{explicit_id}--part-{explicit_occurrences[explicit_id]:02d}"
            )
        else:
            scene_id = _fallback_scene_id(index, unit_ids)
        speakers = tuple(dict.fromkeys(_unit_speaker(unit) for unit in draft.units))
        source_refs = tuple(
            dict.fromkeys(
                ref for unit in draft.units for ref in _unit_evidence_refs(unit)
            )
        )
        scenes.append(
            DialogueScene(
                schema_version=SCENE_SCHEMA_VERSION,
                scene_id=scene_id,
                unit_ids=unit_ids,
                start=_unit_start(draft.units[0]),
                end=_unit_end(draft.units[-1]),
                boundary_before=draft.boundary_before,
                boundary_after=draft.boundary_after,
                boundary_evidence=tuple(dict.fromkeys(draft.boundary_evidence)),
                speaker_candidates=speakers,
                source_evidence_refs=source_refs,
            )
        )
    validate_scene_partition(source_units, scenes)
    return tuple(scenes)


def should_use_targeted_visual(frame: SemanticFrame | Mapping[str, Any]) -> bool:
    """Apply the exact scene-v2 visual trigger; confidence alone is irrelevant."""

    if not isinstance(frame, SemanticFrame):
        frame = SemanticFrame.from_mapping(frame)
    return (
        bool(set(frame.uncertain_slots) & VISUAL_RESOLVABLE_SLOTS)
        and len(frame.competing_interpretations) >= 2
        and frame.expected_translation_delta in {"major", "critical"}
        and not frame.audio_only_uncertainty
    )


def targeted_visual_candidates(
    frames: Iterable[SemanticFrame],
) -> tuple[SemanticFrame, ...]:
    return tuple(frame for frame in frames if should_use_targeted_visual(frame))


def write_dialogue_scenes_jsonl(path: Path, scenes: Iterable[DialogueScene]) -> Path:
    return write_model_jsonl(path, scenes)


def load_dialogue_scenes_jsonl(path: Path) -> tuple[DialogueScene, ...]:
    return load_model_jsonl(path, DialogueScene)


__all__ = [
    "SceneBoundaryHint",
    "SceneSegmentationConfig",
    "build_dialogue_scenes",
    "load_dialogue_scenes_jsonl",
    "should_use_targeted_visual",
    "targeted_visual_candidates",
    "write_dialogue_scenes_jsonl",
]
