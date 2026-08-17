from __future__ import annotations

from translation_forensics.utterance_routing import (
    apply_utterance_route,
    classify_utterance_kind,
    render_nonlexical,
)


def agreement(speech_act: str | None) -> dict:
    return {"selected_frame": {"speech_act": speech_act}}


def acoustic(state: str, *texts: str) -> dict:
    return {
        "asr_fusion": {
            "state": state,
            "family_hypotheses": [
                {
                    "source_family": f"family-{index}",
                    "text": text,
                    "evidence_refs": [f"asr:{index}"],
                }
                for index, text in enumerate(texts, 1)
            ],
        }
    }


def test_meaningful_short_refusal_is_lexical_speech() -> None:
    result = classify_utterance_kind(
        source_text="いや",
        agreement=agreement("vocalization"),
        acoustic_record=acoustic("dual_agreement", "いや", "いや"),
    )
    assert result["utterance_kind"] == "lexical_speech"


def test_meaningful_request_is_not_vocalization() -> None:
    result = classify_utterance_kind(
        source_text="お願い",
        agreement=agreement(None),
        acoustic_record=acoustic("dual_conflict", "お願い", "ああ"),
    )
    assert result["utterance_kind"] == "lexical_speech"


def test_dual_asr_plus_frame_supports_vocalization() -> None:
    result = classify_utterance_kind(
        source_text="",
        agreement=agreement("vocalization"),
        acoustic_record=acoustic("dual_compatible", "ああ", "あ"),
    )
    assert result["utterance_kind"] == "vocalization"
    assert result["confidence"] == "high"
    assert result["evidence_refs"] == ["asr:1", "asr:2"]


def test_source_plus_frame_can_support_vocalization_conservatively() -> None:
    result = classify_utterance_kind(
        source_text="はぁ…",
        agreement=agreement("vocalization"),
        acoustic_record=acoustic("single_family", "はぁ"),
    )
    assert result["utterance_kind"] == "vocalization"
    assert result["confidence"] == "medium"


def test_conflicting_asr_with_lexical_content_stays_speech() -> None:
    result = classify_utterance_kind(
        source_text="",
        agreement=agreement("vocalization"),
        acoustic_record=acoustic("dual_conflict", "ああ", "行かない"),
    )
    assert result["utterance_kind"] == "lexical_speech"


def test_corrupt_source_can_be_recovered_only_from_dual_asr_and_frame() -> None:
    result = classify_utterance_kind(
        source_text="���",
        agreement=agreement("vocalization"),
        acoustic_record=acoustic("dual_agreement", "あ", "あ"),
        source_quality_record={"source_quality_status": "unusable"},
    )
    assert result["utterance_kind"] == "vocalization"
    assert "source-text-corruption" in result["reason_codes"]


def test_corrupt_source_alone_is_never_a_vocalization() -> None:
    result = classify_utterance_kind(
        source_text="���",
        agreement=agreement(None),
        acoustic_record=acoustic("single_family", "あ"),
        source_quality_record={"source_quality_status": "unusable"},
    )
    assert result["utterance_kind"] == "unknown"
    assert "source-text-corruption" in result["reason_codes"]


def test_nonverbal_needs_two_signal_classes() -> None:
    result = classify_utterance_kind(
        source_text="[music]",
        agreement=agreement("nonverbal"),
        acoustic_record=acoustic("empty"),
    )
    assert result["utterance_kind"] == "nonverbal"


def test_renderer_uses_controlled_surface_only() -> None:
    assert render_nonlexical("vocalization", "あっ") == "앗!"
    assert render_nonlexical("nonverbal", "music") == "[비언어음]"
    assert render_nonlexical("vocalization", "unknown") == "…"


def test_apply_route_records_controlled_vocalization() -> None:
    routed = apply_utterance_route(
        agreement("vocalization"),
        source_text="あっ",
        acoustic_record=acoustic("dual_agreement", "あっ", "あっ"),
    )
    assert routed["recovery_state"] == "vocalization"
    assert routed["controlled_nonlexical_korean"] == "앗!"
    assert routed["rendered_slots"] == ["speech_act"]


def test_lexical_content_overrides_nonlexical_model_consensus_to_abstain() -> None:
    base = agreement("vocalization")
    base["recovery_state"] = "vocalization"
    routed = apply_utterance_route(
        base,
        source_text="いや",
        acoustic_record=acoustic("dual_agreement", "いや", "いや"),
    )
    assert routed["utterance_kind"] == "lexical_speech"
    assert routed["recovery_state"] == "abstained"
    assert "lexical-nonlexical-route-mismatch" in routed["uncertainty_codes"]


def test_insufficient_vocalization_support_does_not_keep_vocalization_state() -> None:
    base = agreement("vocalization")
    base["recovery_state"] = "vocalization"
    routed = apply_utterance_route(
        base,
        source_text="",
        acoustic_record=acoustic("single_family", "あ"),
    )
    assert routed["utterance_kind"] == "unknown"
    assert routed["recovery_state"] == "abstained"


def test_untrusted_lexical_source_does_not_override_dual_vocalization() -> None:
    result = classify_utterance_kind(
        source_text="行かない",
        agreement=agreement("vocalization"),
        acoustic_record=acoustic("dual_agreement", "あ", "あ"),
        source_quality_record={"source_quality_status": "unusable"},
    )

    assert result["utterance_kind"] == "vocalization"
    assert "untrusted-source-kind-ignored" in result["reason_codes"]


def test_apply_route_uses_acoustic_surface_when_source_is_corrupt() -> None:
    base = agreement("vocalization")
    base["recovery_state"] = "vocalization"
    routed = apply_utterance_route(
        base,
        source_text="���",
        acoustic_record=acoustic("dual_agreement", "あっ", "あっ"),
        source_quality_record={"source_quality_status": "unusable"},
    )

    assert routed["recovery_state"] == "vocalization"
    assert routed["controlled_nonlexical_korean"] == "앗!"
    assert "���" not in routed["controlled_nonlexical_korean"]
