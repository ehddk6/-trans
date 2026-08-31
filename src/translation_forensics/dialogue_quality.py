from __future__ import annotations

"""Separated critic contracts and deterministic scene-aware quality checks."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .automated_quality import audit_translation_decision
from .scene_models import (
    DialogueScene,
    KoreanDialogueIssue,
    SceneCueProjection,
    SceneDialogueTurn,
    SceneModelError,
    SemanticDriftIssue,
    SemanticFrame,
    validate_scene_cue_projection,
    validate_scene_dialogue_coverage,
    validate_scene_partition,
    validate_semantic_frame_coverage,
)


_CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SEMANTIC_FAILURE_CATEGORIES = {
    "question_force_not_preserved": "question",
    "negation_not_preserved": "polarity",
    "numeric_token_not_preserved": "numeric_tokens",
    "stop_command_not_preserved": "stop_continue",
    "continue_command_not_preserved": "stop_continue",
    "refusal_not_preserved": "refusal_permission",
    "permission_not_preserved": "refusal_permission",
    "request_command_not_preserved": "command_strength",
    "deictic_location_not_preserved": "direction",
    "momentary_aspect_not_preserved": "completion",
}


@dataclass(frozen=True, slots=True)
class SceneQAIssue:
    scene_id: str
    unit_ids: tuple[str, ...]
    category: str
    severity: str
    reason: str
    deterministic: bool = True

    def __post_init__(self) -> None:
        if not self.scene_id.strip():
            raise SceneModelError("scene QA issue needs a scene_id")
        if not self.unit_ids or any(not unit_id.strip() for unit_id in self.unit_ids):
            raise SceneModelError("scene QA issue needs non-empty unit_ids")
        if self.severity not in {"error", "review", "warning"}:
            raise SceneModelError("scene QA issue severity is invalid")
        if not self.category.strip() or not self.reason.strip():
            raise SceneModelError("scene QA issue category and reason must not be empty")

    def json(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "unit_ids": list(self.unit_ids),
            "category": self.category,
            "severity": self.severity,
            "reason": self.reason,
            "deterministic": self.deterministic,
        }


@dataclass(frozen=True, slots=True)
class SceneQAReport:
    status: str
    machine_status: str
    scenes_checked: int
    units_checked: int
    issues: tuple[SceneQAIssue, ...]

    def __post_init__(self) -> None:
        if self.status not in {"passed", "review", "failed"}:
            raise SceneModelError("scene QA report status is invalid")
        if self.machine_status not in {"machine-verified", "machine-uncertain"}:
            raise SceneModelError("scene QA machine status is invalid")
        if self.scenes_checked < 0 or self.units_checked < 0:
            raise SceneModelError("scene QA counts must not be negative")
        if self.status == "passed" and self.issues:
            raise SceneModelError("a passed scene QA report cannot contain issues")

    def json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "machine_status": self.machine_status,
            "scenes_checked": self.scenes_checked,
            "units_checked": self.units_checked,
            "issues": [issue.json() for issue in self.issues],
        }


def _coerce_semantic_issue(
    issue: SemanticDriftIssue | Mapping[str, Any],
) -> SemanticDriftIssue:
    return issue if isinstance(issue, SemanticDriftIssue) else SemanticDriftIssue.from_mapping(issue)


def _coerce_dialogue_issue(
    issue: KoreanDialogueIssue | Mapping[str, Any],
) -> KoreanDialogueIssue:
    return issue if isinstance(issue, KoreanDialogueIssue) else KoreanDialogueIssue.from_mapping(issue)


def validate_semantic_drift_issues(
    scene: DialogueScene,
    issues: Iterable[SemanticDriftIssue | Mapping[str, Any]],
    *,
    valid_evidence_refs: Iterable[str] | None = None,
) -> tuple[SemanticDriftIssue, ...]:
    """Validate semantic-only issue rows; style categories cannot enter this pass."""

    allowed_units = set(scene.unit_ids)
    allowed_refs = None if valid_evidence_refs is None else set(valid_evidence_refs)
    result: list[SemanticDriftIssue] = []
    for raw in issues:
        issue = _coerce_semantic_issue(raw)
        if issue.scene_id != scene.scene_id:
            raise SceneModelError("semantic critic issue references another scene")
        if not set(issue.unit_ids) <= allowed_units:
            raise SceneModelError("semantic critic issue references a unit outside its scene")
        if allowed_refs is not None and not set(issue.evidence_refs) <= allowed_refs:
            raise SceneModelError("semantic critic issue references unknown evidence")
        result.append(issue)
    return tuple(result)


def validate_korean_dialogue_issues(
    scene: DialogueScene,
    issues: Iterable[KoreanDialogueIssue | Mapping[str, Any]],
) -> tuple[KoreanDialogueIssue, ...]:
    """Validate naturalness-only issue rows; revised text is an unknown field."""

    allowed_units = set(scene.unit_ids)
    result: list[KoreanDialogueIssue] = []
    for raw in issues:
        issue = _coerce_dialogue_issue(raw)
        if issue.scene_id != scene.scene_id:
            raise SceneModelError("Korean dialogue issue references another scene")
        if not set(issue.unit_ids) <= allowed_units:
            raise SceneModelError("Korean dialogue issue references a unit outside its scene")
        result.append(issue)
    return tuple(result)


def semantic_critic_disposition(
    issues: Iterable[SemanticDriftIssue | Mapping[str, Any]],
) -> dict[str, Any]:
    values = tuple(_coerce_semantic_issue(issue) for issue in issues)
    active = tuple(issue for issue in values if issue.verdict == "issue")
    critical = tuple(issue for issue in active if issue.severity == "critical")
    repairable = tuple(issue for issue in active if issue.severity in {"major", "critical"})
    return {
        "status": (
            "machine-uncertain" if critical else "repair-required" if repairable
            else "review" if active else "pass"
        ),
        "critical_issue_count": len(critical),
        "repair_required": bool(repairable),
        "affected_scene_ids": sorted({issue.scene_id for issue in repairable}),
    }


def korean_dialogue_critic_disposition(
    issues: Iterable[KoreanDialogueIssue | Mapping[str, Any]],
) -> dict[str, Any]:
    values = tuple(_coerce_dialogue_issue(issue) for issue in issues)
    repairable = tuple(issue for issue in values if issue.severity == "major")
    return {
        "status": "repair-required" if repairable else "review" if values else "pass",
        "major_issue_count": len(repairable),
        "repair_required": bool(repairable),
        "affected_scene_ids": sorted({issue.scene_id for issue in repairable}),
    }


def select_repair_scene_ids(
    semantic_issues: Iterable[SemanticDriftIssue | Mapping[str, Any]],
    dialogue_issues: Iterable[KoreanDialogueIssue | Mapping[str, Any]],
    *,
    deterministic_failed_scene_ids: Iterable[str] = (),
) -> tuple[str, ...]:
    """Select only scenes that meet the explicit targeted-repair policy."""

    selected = {str(value) for value in deterministic_failed_scene_ids if str(value)}
    selected.update(
        issue.scene_id
        for issue in (_coerce_semantic_issue(value) for value in semantic_issues)
        if issue.verdict == "issue" and issue.severity in {"major", "critical"}
    )
    selected.update(
        issue.scene_id
        for issue in (_coerce_dialogue_issue(value) for value in dialogue_issues)
        if issue.severity == "major"
    )
    return tuple(sorted(selected))


def _value(record: object, name: str, default: Any = None) -> Any:
    return record.get(name, default) if isinstance(record, Mapping) else getattr(record, name, default)


def _unit_id(record: object) -> str:
    return str(_value(record, "unit_id") or "")


def _source_text(record: object) -> str:
    return str(_value(record, "text_raw", _value(record, "source_japanese", "")) or "")


def _timing(record: object) -> tuple[float, float]:
    return float(_value(record, "start")), float(_value(record, "end"))


def _scene_records(
    records: Iterable[object], scene_id: str
) -> tuple[object, ...]:
    return tuple(record for record in records if str(_value(record, "scene_id")) == scene_id)


def _semantic_groups(
    scene: DialogueScene,
    projections: Sequence[SceneCueProjection],
    turns: Sequence[SceneDialogueTurn],
) -> tuple[tuple[tuple[str, ...], str], ...]:
    turn_units = {turn.turn_id: turn.source_unit_ids for turn in turns}
    groups: dict[tuple[str, ...], list[SceneCueProjection]] = {}
    order: list[tuple[str, ...]] = []
    for projection in projections:
        key = projection.source_turn_ids
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(projection)
    result: list[tuple[tuple[str, ...], str]] = []
    for turn_ids in order:
        if turn_units and all(turn_id in turn_units for turn_id in turn_ids):
            unit_ids = tuple(
                dict.fromkeys(
                    unit_id for turn_id in turn_ids for unit_id in turn_units[turn_id]
                )
            )
        else:
            unit_ids = tuple(projection.unit_id for projection in groups[turn_ids])
        if not set(unit_ids) <= set(scene.unit_ids):
            raise SceneModelError("alignment group references a unit outside its scene")
        target = " ".join(projection.text for projection in groups[turn_ids])
        result.append((unit_ids, target))
    return tuple(result)


def audit_scene_translation_decisions(
    units: Sequence[object],
    scenes: Sequence[DialogueScene],
    projections: Sequence[SceneCueProjection],
    *,
    semantic_frames: Sequence[SemanticFrame] = (),
    dialogue_turns: Sequence[SceneDialogueTurn] = (),
    max_characters_per_second: float = 20.0,
    max_line_characters: int = 42,
) -> SceneQAReport:
    """Run scene-aware structural checks and review-labelled semantic heuristics.

    Structural failures are deterministic errors. Cross-language surface checks
    are deliberately labelled ``review`` because a regex cannot prove semantic
    drift; a model critic cannot silently override either record.
    """

    if max_characters_per_second <= 0 or max_line_characters < 1:
        raise SceneModelError("scene QA readability limits must be positive")
    validate_scene_partition(units, scenes)
    unit_map = {_unit_id(unit): unit for unit in units}
    all_projections = tuple(projections)
    all_frames = tuple(semantic_frames)
    all_turns = tuple(dialogue_turns)
    issues: list[SceneQAIssue] = []

    for scene in scenes:
        scene_projections = tuple(
            projection for projection in all_projections if projection.scene_id == scene.scene_id
        )
        scene_turns = tuple(turn for turn in all_turns if turn.scene_id == scene.scene_id)
        scene_frames = tuple(frame for frame in all_frames if frame.scene_id == scene.scene_id)
        try:
            if scene_turns:
                validate_scene_dialogue_coverage(scene, scene_turns)
            validate_scene_cue_projection(
                scene, scene_projections, turns=scene_turns or None
            )
        except SceneModelError as exc:
            issues.append(
                SceneQAIssue(
                    scene.scene_id, scene.unit_ids, "unit_coverage_or_projection", "error", str(exc)
                )
            )
            continue
        if scene_frames:
            valid_refs = {
                str(ref)
                for unit_id in scene.unit_ids
                for ref in (_value(unit_map[unit_id], "evidence_ids", ()) or ())
            }
            try:
                validate_semantic_frame_coverage(
                    scene, scene_frames, valid_evidence_refs=valid_refs
                )
            except SceneModelError as exc:
                issues.append(
                    SceneQAIssue(
                        scene.scene_id, scene.unit_ids, "semantic_frame_coverage", "error", str(exc)
                    )
                )

        for projection in scene_projections:
            if _CONTROL_CHARACTER_RE.search(projection.text):
                issues.append(
                    SceneQAIssue(
                        scene.scene_id, (projection.unit_id,), "invalid_character", "error",
                        "cue contains a control character",
                    )
                )
            start, end = _timing(unit_map[projection.unit_id])
            visible_characters = len("".join(projection.text.split()))
            cps = visible_characters / max(end - start, 0.001)
            if cps > max_characters_per_second:
                issues.append(
                    SceneQAIssue(
                        scene.scene_id, (projection.unit_id,), "reading_speed", "warning",
                        f"cue reading speed {cps:.2f} exceeds {max_characters_per_second:.2f} cps",
                    )
                )
            if any(len(line) > max_line_characters for line in projection.text.splitlines()):
                issues.append(
                    SceneQAIssue(
                        scene.scene_id, (projection.unit_id,), "line_length", "warning",
                        f"cue line exceeds {max_line_characters} characters",
                    )
                )

        for group_unit_ids, target in _semantic_groups(scene, scene_projections, scene_turns):
            source = " ".join(_source_text(unit_map[unit_id]) for unit_id in group_unit_ids)
            audit = audit_translation_decision(
                unit_id=f"{scene.scene_id}:{'+'.join(group_unit_ids)}",
                source_japanese=source,
                source_faithful_korean=target,
                viewer_natural_korean=target,
                source_quality_status="trusted",
                confidence="high",
            )
            for failure in audit["hard_failures"]:
                category = _SEMANTIC_FAILURE_CATEGORIES.get(str(failure))
                if category is None:
                    continue
                issues.append(
                    SceneQAIssue(
                        scene.scene_id,
                        group_unit_ids,
                        category,
                        "review",
                        f"alignment-group surface check: {failure}",
                    )
                )

    unknown_projection_scenes = {
        projection.scene_id for projection in all_projections
    } - {scene.scene_id for scene in scenes}
    if unknown_projection_scenes:
        raise SceneModelError(
            f"cue projections reference unknown scenes: {sorted(unknown_projection_scenes)}"
        )
    unique_issues = tuple(
        dict.fromkeys(
            (
                issue.scene_id,
                issue.unit_ids,
                issue.category,
                issue.severity,
                issue.reason,
                issue.deterministic,
            )
            for issue in issues
        )
    )
    normalized_issues = tuple(
        SceneQAIssue(scene_id, unit_ids, category, severity, reason, deterministic)
        for scene_id, unit_ids, category, severity, reason, deterministic in unique_issues
    )
    if any(issue.severity == "error" for issue in normalized_issues):
        status = "failed"
    elif normalized_issues:
        status = "review"
    else:
        status = "passed"
    return SceneQAReport(
        status=status,
        machine_status="machine-verified" if status == "passed" else "machine-uncertain",
        scenes_checked=len(scenes),
        units_checked=len(units),
        issues=normalized_issues,
    )


def write_semantic_drift_audit_jsonl(
    path: Path, issues: Iterable[SemanticDriftIssue]
) -> Path:
    return _write_jsonl(path, (issue.json() for issue in issues))


def write_korean_dialogue_audit_jsonl(
    path: Path, issues: Iterable[KoreanDialogueIssue]
) -> Path:
    return _write_jsonl(path, (issue.json() for issue in issues))


def write_scene_qa_jsonl(path: Path, reports: Iterable[SceneQAReport]) -> Path:
    return _write_jsonl(path, (report.json() for report in reports))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    destination.write_text(payload, encoding="utf-8", newline="\n")
    return destination


__all__ = [
    "SceneQAIssue",
    "SceneQAReport",
    "audit_scene_translation_decisions",
    "korean_dialogue_critic_disposition",
    "select_repair_scene_ids",
    "semantic_critic_disposition",
    "validate_korean_dialogue_issues",
    "validate_semantic_drift_issues",
    "write_korean_dialogue_audit_jsonl",
    "write_scene_qa_jsonl",
    "write_semantic_drift_audit_jsonl",
]
