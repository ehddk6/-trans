from __future__ import annotations

import json

import pytest

from translation_forensics.scene_translation import (
    SceneTranslationError,
    run_scene_translation_v2,
)


def _contracts():
    schema = {"type": "object"}
    return {
        key: {"role": key, "model": "fake-scene-model", "prompt": f"prompt:{key}", "schema": schema}
        for key in ("semantic", "visual_observation", "dialogue", "segmentation", "source_faithful", "semantic_critic", "dialogue_critic", "repair")
    }


def _units():
    return [
        {"unit_id": "u1", "start": 0.0, "end": 1.0, "text_raw": "行く？", "evidence_ids": ["e1"]},
        {"unit_id": "u2", "start": 1.0, "end": 2.0, "text_raw": "うん", "evidence_ids": ["e2"]},
        {"unit_id": "u3", "start": 8.0, "end": 9.0, "text_raw": "待って", "evidence_ids": ["e3"]},
    ]


def _scenes():
    return [
        {"scene_id": "scene-1", "unit_ids": ["u1", "u2"]},
        {"scene_id": "scene-2", "unit_ids": ["u3"]},
    ]


def _frame(unit_id):
    return {
        "schema_version": "1", "scene_id": "scene-1" if unit_id in {"u1", "u2"} else "scene-2", "unit_id": unit_id,
        "utterance_type": "dialogue", "semantic_summary": "발화", "speech_act": "statement", "question": "no",
        "polarity": "positive", "refusal_permission": "neither", "stop_continue": "neither", "command_strength": "none",
        "speaker": None, "addressee": None, "actor": None, "action": None, "target": None, "location": None,
        "direction": None, "tense_aspect": None, "completion": "unknown", "intensity": None, "numeric_tokens": [],
        "register": "neutral", "response_to_unit_id": None, "continues_from_unit_id": None, "continues_to_unit_id": None,
        "must_preserve": ["polarity"], "uncertain_slots": ["speaker"] if unit_id == "u1" else [],
        "competing_interpretations": ["A", "B"] if unit_id == "u1" else [],
        "visual_resolvable_slots": ["speaker"] if unit_id == "u1" else [], "expected_translation_delta": "major" if unit_id == "u1" else "none",
        "audio_only_uncertainty": False, "confidence": "low", "evidence_refs": ["e1"],
    }


class FakeSceneProvider:
    def __init__(self, *, semantic_regression: bool = False) -> None:
        self.calls = []
        self.semantic_regression = semantic_regression

    def run_structured(self, **kwargs):
        self.calls.append(kwargs)
        role = kwargs["role"]
        if role == "visual_observation":
            unit_id = kwargs["payload"]["unit_id"]
            return {"observations": [{"unit_id": unit_id, "observed_slots": [{"slot": "speaker", "observation": "left"}], "unresolved_slots": [], "confidence": "high", "frame_evidence_refs": ["frame-u1"], "unsupported_inference_warnings": []}]}, {"pixel_external_transfer_count": 1}
        ids = [unit["unit_id"] for unit in kwargs["payload"]["scene"]["units"]]
        if role == "semantic":
            return {"semantic_frames": [_frame(unit_id) for unit_id in ids]}, {"cache_hit": False}
        if role == "dialogue":
            return {"realizations": [{"turn_id": f"t-{unit_id}", "source_unit_ids": [unit_id], "korean": f"자연 {unit_id}"} for unit_id in ids]}, {}
        if role == "segmentation":
            turns = kwargs["payload"]["realizations"]
            return {"projections": [{"unit_id": turn["source_unit_ids"][0], "text": turn["korean"]} for turn in turns]}, {}
        if role == "source_faithful":
            return {"translations": [{"unit_id": unit_id, "source_faithful_korean": f"충실 {unit_id}"} for unit_id in ids]}, {}
        if role == "semantic_critic":
            repair = ".repair-" in kwargs["call_id"]
            issue = repair and self.semantic_regression and "u2" in ids
            return {"audits": [{"unit_id": unit_id, "verdict": "issue" if issue and unit_id == "u2" else "pass"} for unit_id in ids]}, {}
        if role == "dialogue_critic":
            repair = ".repair-" in kwargs["call_id"]
            return {
                "audits": [] if repair or "u2" not in ids else [{
                    "unit_ids": ["u2"], "category": "unnatural_ending", "severity": "major",
                    "target_span": "자연 u2", "reason": "synthetic", "repair_scope": "dialogue_text",
                }]
            }, {}
        if role == "repair":
            affected = kwargs["payload"]["affected_unit_ids"]
            return {"repairs": [{"unit_id": unit_id, "viewer_natural_korean": f"수정 {unit_id}"} for unit_id in affected]}, {}
        raise AssertionError(role)


def test_scene_pipeline_runs_separated_passes_and_preserves_input_boundaries():
    provider = FakeSceneProvider()
    result = run_scene_translation_v2(
        provider, title_id="TEST-001", units=_units(), scenes=_scenes(), contracts=_contracts(),
        repair_attempts=1, visual_policy="metadata",
    )

    assert [row["unit_id"] for row in result.decisions] == ["u1", "u2", "u3"]
    assert result.decisions[1]["viewer_natural_korean"] == "수정 u2"
    assert result.decisions[1]["source_faithful_korean"] == "충실 u2"
    assert result.decisions[1]["repair_attempts"] == 1
    assert result.architecture_status == "experimental-unbenchmarked"
    assert result.benchmark_status == "not-run"
    assert set(result.artifacts) == {
        "translation-input/dialogue_scenes.jsonl", "semantic_frames.jsonl", "scene_realizations.jsonl",
        "scene_cue_projection.jsonl", "scene_source_faithful.jsonl", "semantic_drift_audit.jsonl",
        "korean_dialogue_audit.jsonl", "repair_history.jsonl",
    }
    assert {receipt["role"] for receipt in result.receipts} == {
        "semantic", "dialogue", "segmentation", "source_faithful", "semantic_critic", "dialogue_critic", "repair",
    }
    assert all(receipt["model"] == "fake-scene-model" for receipt in result.receipts)
    assert all(len(receipt["prompt_sha256"]) == 64 for receipt in result.receipts)
    assert all(receipt["external_transfer"] is False for receipt in result.receipts)
    assert len(result.cache_identity) == 64

    for call in provider.calls:
        if call["role"] in {"dialogue", "dialogue_critic", "repair"}:
            assert "source_faithful" not in json.dumps(call["payload"], ensure_ascii=False).lower()
        if call["role"] != "semantic":
            assert "image_paths" not in call
    assert provider.calls[0]["image_paths"] == []


def test_semantic_regression_rolls_back_repaired_scene_and_reaudits():
    provider = FakeSceneProvider(semantic_regression=True)
    result = run_scene_translation_v2(
        provider, title_id="TEST-001", units=_units(), scenes=_scenes(), contracts=_contracts(), repair_attempts=1,
    )

    assert result.decisions[1]["viewer_natural_korean"] == "자연 u2"
    assert result.artifacts["repair_history.jsonl"][0]["status"] == "rolled_back_semantic_regression"
    assert any("rollback" in call["call_id"] for call in provider.calls if call["role"] == "semantic_critic")


def test_scene_pipeline_rejects_missing_or_duplicate_coverage_before_provider_call():
    provider = FakeSceneProvider()
    with pytest.raises(SceneTranslationError, match="exactly once"):
        run_scene_translation_v2(
            provider, title_id="TEST-001", units=_units(),
            scenes=[{"scene_id": "x", "unit_ids": ["u1", "u1", "u2", "u3"]}], contracts=_contracts(),
        )
    assert provider.calls == []


def test_targeted_visual_observer_is_source_bound_and_metadata_never_invokes_it(tmp_path):
    seen = []

    def observer(**kwargs):
        seen.append(kwargs)
        return {"observed_slots": {"speaker": "left"}, "unresolved_slots": [], "confidence": 0.8, "frame_evidence_refs": ["frame-u1"], "unsupported_inference_warnings": []}

    provider = FakeSceneProvider()
    image = tmp_path / "frame.png"
    image.write_bytes(b"synthetic image")
    metadata = {"u1": {"image_paths": [image]}}
    result = run_scene_translation_v2(
        provider, title_id="TEST-001", units=_units(), scenes=_scenes(), contracts=_contracts(),
        visual_policy="targeted", visual_context_by_unit=metadata, visual_observer=observer,
    )
    assert [record["unit_id"] for record in result.artifacts["visual_semantic_observations.jsonl"]] == ["u1"]
    assert seen and "viewer_natural_korean" not in seen[0]["semantic_frame"]

    run_scene_translation_v2(
        FakeSceneProvider(), title_id="TEST-001", units=_units(), scenes=_scenes(), contracts=_contracts(),
        visual_policy="metadata", visual_context_by_unit=metadata, visual_observer=observer,
    )
    assert len(seen) == 1


def test_targeted_visual_uses_structured_provider_and_low_confidence_alone_does_not(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"synthetic image")
    provider = FakeSceneProvider()
    result = run_scene_translation_v2(
        provider, title_id="TEST-001", units=_units(), scenes=_scenes(), contracts=_contracts(),
        visual_policy="targeted", visual_context_by_unit={"u1": {"image_paths": [image]}},
    )
    visual_calls = [call for call in provider.calls if call["role"] == "visual_observation"]
    assert len(visual_calls) == 1
    assert visual_calls[0]["image_paths"] == [image]
    assert any(receipt["role"] == "visual_observation" and receipt["external_transfer"] for receipt in result.receipts)
    assert result.artifacts["visual_semantic_observations.jsonl"][0]["observation_provenance"].endswith("visual-semantic-observation")

    provider = FakeSceneProvider()
    # u2 is low confidence but has no ambiguity / major visual delta, so it is ineligible.
    run_scene_translation_v2(
        provider, title_id="TEST-001", units=_units(), scenes=_scenes(), contracts=_contracts(),
        visual_policy="targeted", visual_context_by_unit={"u2": {"image_paths": [image]}},
    )
    assert not [call for call in provider.calls if call["role"] == "visual_observation"]
