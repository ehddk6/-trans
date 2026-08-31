from pathlib import Path

import pytest

from translation_forensics.scene_models import SceneModelError
from translation_forensics.style_memory import (
    SceneDialogueMemory,
    SpeakerStyleProfile,
    confirmed_style_profiles,
    dialogue_memory_context,
    load_style_profiles_jsonl,
    style_context_for_scene,
    write_style_profiles_jsonl,
)


PROHIBITED = ("actor", "target", "action", "relation", "consent", "location")


def _profile(status: str, **changes: object) -> SpeakerStyleProfile:
    values = {
        "title_id": "TITLE", "speaker_id": "A", "status": status,
        "register": "informal", "first_person": "나", "second_person": None,
        "address_terms": ("선생님",), "common_endings": ("-어",),
        "average_utterance_length": 8.0, "interjection_tendencies": ("응",),
        "prohibited_inferences": PROHIBITED, "evidence_refs": ("human:1",),
        "provenance": {"source": "human-style-review"},
    }
    values.update(changes)
    return SpeakerStyleProfile(**values)


def test_only_confirmed_or_human_approved_style_is_used():
    profiles = (_profile("machine-draft"), _profile("confirmed"))
    assert confirmed_style_profiles(profiles) == (profiles[1],)

    machine_claim = _profile("confirmed", provenance={"machine_generated": True})
    assert confirmed_style_profiles((machine_claim,)) == ()
    approved = _profile(
        "confirmed", provenance={"machine_generated": True, "human_approved": True}
    )
    assert confirmed_style_profiles((approved,)) == (approved,)


def test_style_context_cannot_carry_semantic_slots():
    context = style_context_for_scene(
        (_profile("confirmed"),), title_id="TITLE", speaker_ids=("A",)
    )
    assert len(context) == 1
    for forbidden in PROHIBITED:
        assert forbidden not in context[0]

    with pytest.raises(SceneModelError, match="explicitly prohibit"):
        _profile("confirmed", prohibited_inferences=("actor",))


def test_provisional_dialogue_memory_is_bounded_to_four_style_turns():
    memory = SceneDialogueMemory(
        title_id="TITLE", scene_id="S1",
        speaker_turns=tuple(
            {"turn_id": f"t{i}", "speaker_id": "A", "korean": f"대사 {i}"}
            for i in range(6)
        ),
        pending_question="question-open", open_topic="topic-open",
        status="provisional-style-only", provenance={"run_id": "run-1"},
    )
    context = dialogue_memory_context(memory)
    assert [turn["turn_id"] for turn in context["previous_turns"]] == ["t2", "t3", "t4", "t5"]
    assert "pending_question" not in context
    assert "open_topic" not in context


def test_style_profile_jsonl_round_trip(tmp_path: Path):
    profile = _profile("confirmed")
    path = write_style_profiles_jsonl(tmp_path / "style.jsonl", (profile,))
    assert load_style_profiles_jsonl(path) == (profile,)
