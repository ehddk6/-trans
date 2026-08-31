from translation_forensics.scene_models import SemanticFrame, validate_scene_partition
from translation_forensics.scene_segmentation import (
    SceneBoundaryHint,
    SceneSegmentationConfig,
    build_dialogue_scenes,
    should_use_targeted_visual,
)


def _unit(unit_id: str, start: float, end: float, text: str, speaker: str = "A"):
    return {
        "unit_id": unit_id,
        "start": start,
        "end": end,
        "text_raw": text,
        "speaker": speaker,
        "evidence_ids": [f"e:{unit_id}"],
    }


def _frame(**changes: object) -> SemanticFrame:
    value = {
        "schema_version": "1", "scene_id": "S1", "unit_id": "u1",
        "utterance_type": "dialogue", "semantic_summary": "지시 대상을 판별해야 한다",
        "speech_act": "statement", "question": "no", "polarity": "positive",
        "refusal_permission": "neither", "stop_continue": "neither",
        "command_strength": "none", "speaker": None, "addressee": None,
        "actor": None, "action": None, "target": None, "location": None,
        "direction": None, "tense_aspect": None, "completion": "unknown",
        "intensity": None, "numeric_tokens": [], "register": "unknown",
        "response_to_unit_id": None, "continues_from_unit_id": None,
        "continues_to_unit_id": None, "must_preserve": ["speech_act"],
        "uncertain_slots": ["deictic_referent"],
        "competing_interpretations": ["손", "물건"],
        "visual_resolvable_slots": ["deictic_referent"],
        "expected_translation_delta": "major", "audio_only_uncertainty": False,
        "confidence": "low", "evidence_refs": ["e:u1"],
    }
    value.update(changes)
    return SemanticFrame.from_mapping(value)


def test_segmentation_is_deterministic_and_covers_units_once():
    units = (
        _unit("u1", 0.0, 1.0, "いい？"),
        _unit("u2", 1.2, 2.0, "うん", "B"),
        _unit("u3", 7.0, 8.0, "次だ"),
    )
    first = build_dialogue_scenes(units)
    second = build_dialogue_scenes(units)
    assert first == second
    assert [scene.unit_ids for scene in first] == [("u1", "u2"), ("u3",)]
    validate_scene_partition(units, first)


def test_question_response_is_not_split_by_speaker_change_or_soft_hint():
    units = (
        _unit("q", 0.0, 1.0, "いい？", "A"),
        _unit("a", 1.1, 1.5, "うん", "B"),
    )
    scenes = build_dialogue_scenes(
        units,
        boundary_hints={"a": SceneBoundaryHint("speaker_change", hard_boundary=False)},
    )
    assert len(scenes) == 1


def test_explicit_scene_ids_have_priority_and_remain_reproducible():
    units = (
        _unit("u1", 0.0, 1.0, "一"),
        _unit("u2", 10.0, 11.0, "二"),
        _unit("u3", 11.2, 12.0, "三"),
    )
    scenes = build_dialogue_scenes(
        units, explicit_scene_ids={"u1": "A", "u2": "A", "u3": "B"}
    )
    assert [(scene.scene_id, scene.unit_ids) for scene in scenes] == [
        ("A", ("u1", "u2")), ("B", ("u3",))
    ]
    assert scenes[1].boundary_before == "explicit_scene_id"


def test_forced_cap_reason_is_recorded_on_both_boundary_sides():
    units = tuple(_unit(f"u{index}", index * 1.1, index * 1.1 + 1, str(index)) for index in range(3))
    scenes = build_dialogue_scenes(
        units, config=SceneSegmentationConfig(max_units=2)
    )
    assert scenes[0].boundary_after == "forced_cap:max_units"
    assert scenes[1].boundary_before == "forced_cap:max_units"
    assert "forced_cap:max_units" in scenes[0].boundary_evidence
    assert "forced_cap:max_units" in scenes[1].boundary_evidence


def test_visual_trigger_requires_every_v2_condition():
    assert should_use_targeted_visual(_frame()) is True
    assert should_use_targeted_visual(_frame(uncertain_slots=[])) is False
    assert should_use_targeted_visual(_frame(competing_interpretations=["하나"])) is False
    assert should_use_targeted_visual(_frame(expected_translation_delta="minor")) is False
    assert should_use_targeted_visual(_frame(audio_only_uncertainty=True)) is False
    assert should_use_targeted_visual(_frame(uncertain_slots=["register"], visual_resolvable_slots=[])) is False
