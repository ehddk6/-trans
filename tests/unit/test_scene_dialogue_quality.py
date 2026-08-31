from pathlib import Path

import json
import pytest
from jsonschema import Draft202012Validator

from translation_forensics.dialogue_quality import (
    audit_scene_translation_decisions,
    korean_dialogue_critic_disposition,
    select_repair_scene_ids,
    semantic_critic_disposition,
    validate_korean_dialogue_issues,
    validate_semantic_drift_issues,
)
from translation_forensics.scene_models import (
    DialogueScene,
    KoreanDialogueIssue,
    SceneCueProjection,
    SceneDialogueTurn,
    SceneModelError,
    SemanticDriftIssue,
)


ROOT = Path(__file__).resolve().parents[2]


def _scene() -> DialogueScene:
    return DialogueScene("scene-v2", "S1", ("u1", "u2"), 0.0, 4.0, "start_of_title", "end_of_title", (), ("A", "B"), ("e1", "e2"))


def _units():
    return (
        {"unit_id": "u1", "start": 0.0, "end": 2.0, "text_raw": "いい？", "evidence_ids": ["e1"]},
        {"unit_id": "u2", "start": 2.1, "end": 4.0, "text_raw": "うん", "evidence_ids": ["e2"]},
    )


def _turns():
    return (
        SceneDialogueTurn("S1", "t1", ("u1",), "A", "괜찮아?", "new", "single_cue"),
        SceneDialogueTurn("S1", "t2", ("u2",), "B", "응.", "response", "single_cue"),
    )


def _projections():
    return (
        SceneCueProjection("S1", "u1", "괜찮아?", ("t1",), ()),
        SceneCueProjection("S1", "u2", "응.", ("t2",), ()),
    )


def _semantic(severity: str = "major") -> SemanticDriftIssue:
    return SemanticDriftIssue(
        "S1", ("u1",), "괜찮아", "question_statement", severity,
        "질문", "평서", ("e1",), "issue",
    )


def _dialogue(severity: str = "major") -> KoreanDialogueIssue:
    return KoreanDialogueIssue(
        "S1", ("u2",), "응.", "response_mismatch", severity,
        "앞 질문에 맞지 않는 반응이다", "dialogue_text",
    )


def test_critic_models_match_repository_schemas_and_remain_separated():
    semantic_schema = json.loads((ROOT / "schemas" / "scene-semantic-drift-critic-v1.schema.json").read_text(encoding="utf-8"))
    dialogue_schema = json.loads((ROOT / "schemas" / "korean-dialogue-critic-v1.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(semantic_schema).validate({"audits": [_semantic().json()]})
    Draft202012Validator(dialogue_schema).validate({"audits": [_dialogue().json()]})
    assert validate_semantic_drift_issues(_scene(), (_semantic(),), valid_evidence_refs={"e1"})
    assert validate_korean_dialogue_issues(_scene(), (_dialogue(),))

    bad_semantic = _semantic().json() | {"category": "unnatural_ending"}
    with pytest.raises(SceneModelError, match="non-semantic"):
        validate_semantic_drift_issues(_scene(), (bad_semantic,))
    modified_dialogue = _dialogue().json() | {"revised_korean": "응"}
    with pytest.raises(SceneModelError, match="unknown fields"):
        validate_korean_dialogue_issues(_scene(), (modified_dialogue,))


def test_issue_dispositions_and_targeted_repair_selection():
    assert semantic_critic_disposition((_semantic("critical"),))["status"] == "machine-uncertain"
    assert korean_dialogue_critic_disposition((_dialogue("major"),))["repair_required"] is True
    other = KoreanDialogueIssue(
        "S2", ("x",), "그래", "verbosity", "minor", "조금 길다", "none"
    )
    assert select_repair_scene_ids((_semantic(),), (_dialogue(), other)) == ("S1",)


def test_deterministic_scene_qa_passes_ordered_alignment_groups():
    report = audit_scene_translation_decisions(
        _units(), (_scene(),), _projections(), dialogue_turns=_turns(),
        max_characters_per_second=100,
    )
    assert report.status == "passed"
    assert report.machine_status == "machine-verified"


def test_deterministic_scene_qa_records_coverage_failure_instead_of_hiding_it():
    report = audit_scene_translation_decisions(
        _units(), (_scene(),), _projections()[:1], dialogue_turns=_turns(),
        max_characters_per_second=100,
    )
    assert report.status == "failed"
    assert report.machine_status == "machine-uncertain"
    assert report.issues[0].category == "unit_coverage_or_projection"
