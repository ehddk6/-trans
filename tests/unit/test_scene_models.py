import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from translation_forensics.scene_models import (
    DialogueScene,
    SceneCueProjection,
    SceneDialogueTurn,
    SceneModelError,
    SemanticFrame,
    VisualSemanticObservation,
    load_model_jsonl,
    validate_scene_cue_projection,
    validate_scene_dialogue_coverage,
    validate_semantic_frame_coverage,
    write_model_jsonl,
)
from translation_forensics.scene_segmentation import should_use_targeted_visual


ROOT = Path(__file__).resolve().parents[2]


def _scene() -> DialogueScene:
    return DialogueScene(
        schema_version="scene-v2",
        scene_id="S1",
        unit_ids=("u1", "u2"),
        start=0.0,
        end=3.0,
        boundary_before="start_of_title",
        boundary_after="end_of_title",
        boundary_evidence=(),
        speaker_candidates=("A", "B"),
        source_evidence_refs=("e1", "e2"),
    )


def _frame(unit_id: str, **changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "1",
        "scene_id": "S1",
        "unit_id": unit_id,
        "utterance_type": "dialogue",
        "semantic_summary": "질문 화행을 유지한다",
        "speech_act": "question",
        "question": "yes",
        "polarity": "positive",
        "refusal_permission": "neither",
        "stop_continue": "neither",
        "command_strength": "none",
        "speaker": None,
        "addressee": None,
        "actor": None,
        "action": None,
        "target": None,
        "location": None,
        "direction": None,
        "tense_aspect": None,
        "completion": "unknown",
        "intensity": None,
        "numeric_tokens": [],
        "register": "unknown",
        "response_to_unit_id": None,
        "continues_from_unit_id": None,
        "continues_to_unit_id": None,
        "must_preserve": ["question"],
        "uncertain_slots": ["speaker"],
        "competing_interpretations": ["A가 말함", "B가 말함"],
        "visual_resolvable_slots": ["speaker"],
        "expected_translation_delta": "major",
        "audio_only_uncertainty": False,
        "confidence": "low",
        "evidence_refs": [f"e{unit_id[-1]}"],
    }
    value.update(changes)
    return value


def test_actual_semantic_schema_mapping_is_accepted_by_visual_trigger():
    frame = _frame("u1")
    schema = json.loads(
        (ROOT / "schemas" / "scene-semantic-reconstruction-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema).validate({"frames": [frame]})

    assert SemanticFrame.from_mapping(frame).schema_version == "1"
    assert should_use_targeted_visual(frame) is True


def test_visual_observation_model_matches_prompt_schema():
    row = {
        "scene_id": "S1",
        "unit_id": "u1",
        "observed_slots": [{"slot": "speaker", "observation": "왼쪽 인물이 발화 중이다"}],
        "unresolved_slots": ["addressee"],
        "confidence": "medium",
        "frame_evidence_refs": ["frame:u1"],
        "unsupported_inference_warnings": [],
    }
    schema = json.loads(
        (ROOT / "schemas" / "scene-visual-semantic-observation-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema).validate({"observations": [row]})
    assert VisualSemanticObservation.from_mapping(row).json() == row


def test_semantic_coverage_rejects_duplicate_or_unknown_evidence():
    scene = _scene()
    first = SemanticFrame.from_mapping(_frame("u1"))
    duplicate = SemanticFrame.from_mapping(_frame("u1", evidence_refs=["e1"]))
    with pytest.raises(SceneModelError, match="exactly once"):
        validate_semantic_frame_coverage(
            scene, (first, duplicate), valid_evidence_refs={"e1", "e2"}
        )

    second = SemanticFrame.from_mapping(_frame("u2", evidence_refs=["missing"]))
    with pytest.raises(SceneModelError, match="unknown evidence"):
        validate_semantic_frame_coverage(
            scene, (first, second), valid_evidence_refs={"e1", "e2"}
        )


def test_semantic_frame_rejects_style_memory_as_evidence():
    first = SemanticFrame.from_mapping(_frame("u1", evidence_refs=["style-memory:A"]))
    second = SemanticFrame.from_mapping(_frame("u2"))
    with pytest.raises(SceneModelError, match="not semantic evidence"):
        validate_semantic_frame_coverage(
            _scene(), (first, second),
            valid_evidence_refs={"style-memory:A", "e2"},
        )


def test_dialogue_and_projection_require_exact_ordered_coverage():
    scene = _scene()
    turn = SceneDialogueTurn(
        scene_id="S1",
        turn_id="t1",
        source_unit_ids=("u1", "u2"),
        speaker_id=None,
        korean="괜찮아? 좋아.",
        relation_to_previous="new",
        segmentation_hint="split_across_cues",
    )
    validate_scene_dialogue_coverage(scene, (turn,))
    projections = (
        SceneCueProjection("S1", "u1", "괜찮아?", ("t1",), ()),
        SceneCueProjection("S1", "u2", "좋아.", ("t1",), ()),
    )
    validate_scene_cue_projection(scene, projections, turns=(turn,))

    with pytest.raises(SceneModelError, match="exactly once"):
        validate_scene_cue_projection(scene, projections[:1], turns=(turn,))


def test_projection_rejects_duplicate_complete_turn_and_japanese():
    scene = _scene()
    with pytest.raises(SceneModelError, match="Japanese"):
        SceneCueProjection("S1", "u1", "いいよ", ("t1",), ())
    duplicated = (
        SceneCueProjection("S1", "u1", "좋아.", ("t1",), ()),
        SceneCueProjection("S1", "u2", "좋아.", ("t1",), ()),
    )
    with pytest.raises(SceneModelError, match="duplicated"):
        validate_scene_cue_projection(scene, duplicated)


def test_scene_jsonl_round_trip_is_strict(tmp_path: Path):
    path = write_model_jsonl(tmp_path / "dialogue_scenes.jsonl", [_scene()])
    assert load_model_jsonl(path, DialogueScene) == (_scene(),)
    path.write_text(path.read_text(encoding="utf-8").replace('"scene_id":"S1"', '"extra":1,"scene_id":"S1"'), encoding="utf-8")
    with pytest.raises(SceneModelError, match="invalid scene artifact"):
        load_model_jsonl(path, DialogueScene)
