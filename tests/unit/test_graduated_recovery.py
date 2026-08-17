from __future__ import annotations

from translation_forensics.graduated_recovery import (
    apply_graduated_evidence_gate,
    apply_review_outcomes,
    build_frame_agreement,
    derive_slot_corroboration,
    needs_context_rerun,
)


def frame(**overrides):
    base = {
        "speech_act": "command",
        "question": False,
        "polarity": "negative",
        "refusal_permission": None,
        "command_strength": "command",
        "speaker": None,
        "actor": "speaker-a",
        "action": "stop",
        "target": "speaker-b",
        "location": None,
        "tense_aspect": "present",
        "direction": "stop",
        "intensity": "high",
    }
    return base | overrides


def test_fully_agreed_frame_remains_consensus():
    agreement = build_frame_agreement(frame(), frame())

    assert agreement["recovery_state"] == "accepted_consensus"
    assert agreement["render_blocking_conflicts"] == []
    assert agreement["selected_frame"]["action"] == "stop"
    assert agreement["slot_provenance"]["action"]["provenance"] == "dual_agreement"


def test_descriptive_conflict_is_omitted_without_blocking_core():
    agreement = build_frame_agreement(
        frame(location="inside"),
        frame(location="outside"),
    )

    assert agreement["recovery_state"] == "accepted_partial"
    assert agreement["selected_frame"]["location"] is None
    assert agreement["slot_provenance"]["location"]["provenance"] == "conflict_omitted"
    assert agreement["render_blocking_conflicts"] == []
    assert needs_context_rerun(agreement) is False


def test_polarity_conflict_never_selects_one_model_silently():
    agreement = build_frame_agreement(
        frame(polarity="positive"),
        frame(polarity="negative"),
    )

    assert agreement["recovery_state"] == "abstained"
    assert agreement["selected_frame"]["polarity"] is None
    assert agreement["slot_provenance"]["polarity"]["provenance"] == "conflict_blocking"
    assert needs_context_rerun(agreement) is True


def test_independent_evidence_can_resolve_single_model_polarity():
    agreement = build_frame_agreement(
        frame(polarity="positive"),
        frame(polarity="negative"),
        corroboration={
            "polarity": {
                "model": "sol",
                "independent": True,
                "evidence_refs": ["asr-fusion:block-1"],
            }
        },
    )

    assert agreement["recovery_state"] == "recovered_single_model"
    assert agreement["selected_frame"]["polarity"] == "negative"
    assert agreement["resolved_conflicts"] == ["polarity"]
    assert agreement["slot_provenance"]["polarity"]["provenance"] == "sol_only_supported"


def test_actor_conflict_allows_impersonal_partial_frame():
    agreement = build_frame_agreement(
        frame(actor="speaker-a"),
        frame(actor="speaker-b"),
    )

    assert agreement["recovery_state"] == "accepted_partial"
    assert agreement["selected_frame"]["actor"] is None
    assert agreement["selected_frame"]["action"] == "stop"
    assert agreement["render_blocking_conflicts"] == []


def test_action_conflict_can_reduce_supported_refusal_to_minimal_speech_act():
    agreement = build_frame_agreement(
        frame(
            speech_act="refusal",
            question=False,
            refusal_permission="refusal",
            command_strength=None,
            action="continue",
        ),
        frame(
            speech_act="refusal",
            question=False,
            refusal_permission="refusal",
            command_strength=None,
            action="touch",
        ),
    )

    assert agreement["recovery_state"] == "minimal_speech_act"
    assert agreement["selected_frame"]["speech_act"] == "refusal"
    assert agreement["selected_frame"]["action"] is None
    assert agreement["render_blocking_conflicts"] == ["action"]


def test_question_statement_mismatch_fails_coherence():
    agreement = build_frame_agreement(
        frame(speech_act="statement", question=True),
        frame(speech_act="statement", question=True),
    )

    assert agreement["recovery_state"] == "abstained"
    assert any(
        finding["code"] == "question-speech-act-mismatch"
        for finding in agreement["coherence_findings"]
    )


def test_uncorroborated_coverage_gap_stays_absent():
    agreement = build_frame_agreement(
        frame(location="inside"),
        frame(location=None),
    )

    assert agreement["slot_coverage_gaps"] == ["location"]
    assert agreement["selected_frame"]["location"] is None
    assert agreement["slot_provenance"]["location"]["provenance"] == "absent"


def test_context_retry_marks_safe_result_as_recovered_context():
    agreement = build_frame_agreement(frame(), frame(), context_retried=True)

    assert agreement["recovery_state"] == "recovered_context"


def test_damaged_source_without_dual_asr_uses_precomputed_minimal_fallback():
    agreement = build_frame_agreement(
        frame(
            speech_act="refusal",
            refusal_permission="refusal",
            command_strength=None,
            action=None,
            actor=None,
            target=None,
            direction=None,
        ),
        frame(
            speech_act="refusal",
            refusal_permission="refusal",
            command_strength=None,
            action=None,
            actor=None,
            target=None,
            direction=None,
        ),
    )
    rows = [
        {
            "block_number": 1,
            "source_faithful_korean": "구체적인 행동을 거절해",
            "viewer_natural_korean": "그건 하지 마",
            "conservative_source_faithful_korean": "싫어",
            "conservative_viewer_natural_korean": "안 돼",
            "rendered_slots": ["speech_act", "refusal_permission", "action"],
            "fallback_rendered_slots": ["speech_act", "refusal_permission"],
            "fallback_recovery_state": "minimal_speech_act",
        }
    ]

    gated = apply_graduated_evidence_gate(
        rows,
        {1: agreement},
        {1: {"source_quality_status": "unusable"}},
        {1: {"independent_source_families": []}},
    )

    assert gated[0]["recovery_state"] == "minimal_speech_act"
    assert gated[0]["source_status"] == "unresolved"
    assert gated[0]["viewer_natural_korean"] == "안 돼"


def test_review_prefers_safe_fallback_over_whole_block_deletion():
    rows = [
        {
            "block_number": 1,
            "source_faithful_korean": "침대 안에서 멈춰",
            "viewer_natural_korean": "침대 안에서 그만해",
            "conservative_source_faithful_korean": "멈춰",
            "conservative_viewer_natural_korean": "그만해",
            "recovery_state": "accepted_partial",
            "fallback_recovery_state": "minimal_speech_act",
            "rendered_slots": ["speech_act", "action", "location"],
            "fallback_rendered_slots": ["speech_act", "action"],
            "fallback_evidence_safe": True,
        }
    ]
    reviews = [
        {
            "block_number": 1,
            "verdict": "repair",
            "critical_slot_conflicts": [],
            "unsupported_additions": ["location: 침대 안"],
            "reason": "Location is not supported.",
        }
    ]

    reviewed = apply_review_outcomes(rows, reviews)

    assert reviewed[0]["viewer_natural_korean"] == "그만해"
    assert reviewed[0]["source_status"] == "unresolved"
    assert reviewed[0]["recovery_state"] == "minimal_speech_act"


def test_review_abstains_when_no_safe_candidate_survives():
    rows = [
        {
            "block_number": 1,
            "source_faithful_korean": "해도 돼",
            "viewer_natural_korean": "해도 돼",
            "recovery_state": "recovered_single_model",
        }
    ]
    reviews = [
        {
            "block_number": 1,
            "verdict": "quarantine",
            "critical_slot_conflicts": ["refusal_permission"],
            "unsupported_additions": [],
            "reason": "Permission conflicts with refusal.",
        }
    ]

    reviewed = apply_review_outcomes(rows, reviews)

    assert reviewed[0]["viewer_natural_korean"] == "…"
    assert reviewed[0]["recovery_state"] == "abstained"


def test_vocalization_gate_uses_controlled_renderer_not_model_text():
    agreement = build_frame_agreement(
        frame(
            speech_act="vocalization",
            question=None,
            polarity=None,
            command_strength=None,
            actor=None,
            action=None,
            target=None,
            tense_aspect=None,
            direction=None,
            intensity=None,
        ),
        frame(
            speech_act="vocalization",
            question=None,
            polarity=None,
            command_strength=None,
            actor=None,
            action=None,
            target=None,
            tense_aspect=None,
            direction=None,
            intensity=None,
        ),
    )
    agreement["controlled_nonlexical_korean"] = "앗!"
    rows = [
        {
            "block_number": 1,
            "source_faithful_korean": "모델이 추측한 문장",
            "viewer_natural_korean": "모델이 추측한 문장",
            "conservative_source_faithful_korean": "모델 추측",
            "conservative_viewer_natural_korean": "모델 추측",
        }
    ]
    gated = apply_graduated_evidence_gate(
        rows,
        {1: agreement},
        {1: {"source_quality_status": "trusted"}},
        {1: {"independent_source_families": ["a", "b"]}},
    )
    assert gated[0]["recovery_state"] == "vocalization"
    assert gated[0]["source_faithful_korean"] == "앗!"
    assert gated[0]["viewer_natural_korean"] == "앗!"


def test_overclaiming_fallback_is_sanitized_even_when_primary_is_safe():
    agreement = build_frame_agreement(frame(), frame())
    row = {
        "block_number": 1,
        "source_faithful_korean": "멈춰",
        "viewer_natural_korean": "멈춰",
        "conservative_source_faithful_korean": "침대 안에서 멈춰",
        "conservative_viewer_natural_korean": "침대 안에서 멈춰",
        "recovery_state": "accepted_consensus",
        "fallback_recovery_state": "accepted_partial",
        "rendered_slots": ["speech_act", "action"],
        "fallback_rendered_slots": ["speech_act", "action", "location"],
    }
    [gated] = apply_graduated_evidence_gate(
        [row],
        {1: agreement},
        {1: {"source_quality_status": "trusted"}},
        {1: {"independent_source_families": []}},
    )
    assert gated["fallback_evidence_safe"] is False
    assert gated["fallback_recovery_state"] == "abstained"
    assert gated["conservative_viewer_natural_korean"] == "…"
    [reviewed] = apply_review_outcomes(
        [gated],
        [
            {
                "block_number": 1,
                "verdict": "repair",
                "critical_slot_conflicts": [],
                "unsupported_additions": [],
                "claim_findings": [],
                "fallback_blocking": False,
                "reason": "primary rejected",
            }
        ],
    )
    assert reviewed["recovery_state"] == "abstained"
    assert reviewed["viewer_natural_korean"] == "…"


def test_single_model_fallback_cannot_claim_consensus():
    agreement = build_frame_agreement(
        frame(polarity="positive"),
        frame(polarity="negative"),
        corroboration={
            "polarity": {
                "model": "sol",
                "independent": True,
                "evidence_refs": ["asr-fusion:block-1"],
            }
        },
    )
    row = {
        "block_number": 1,
        "source_faithful_korean": "주장을 번역",
        "viewer_natural_korean": "자연스러운 번역",
        "conservative_source_faithful_korean": "보수적 번역",
        "conservative_viewer_natural_korean": "보수적 자연어",
        "rendered_slots": list(agreement["rendered_slots"]),
        "fallback_rendered_slots": ["speech_act", "action", "polarity"],
        "fallback_recovery_state": "accepted_consensus",
    }

    [gated] = apply_graduated_evidence_gate(
        [row],
        {1: agreement},
        {1: {"source_quality_status": "trusted"}},
        {1: {"independent_source_families": ["terra", "sol"]}},
    )

    assert agreement["recovery_state"] == "recovered_single_model"
    assert gated["fallback_recovery_state"] == "abstained"
    assert gated["fallback_evidence_safe"] is False

    [reviewed] = apply_review_outcomes(
        [gated],
        [
            {
                "block_number": 1,
                "verdict": "repair",
                "critical_slot_conflicts": [],
                "unsupported_additions": [],
                "claim_findings": [],
                "fallback_blocking": False,
                "reason": "primary rejected",
            }
        ],
    )

    assert reviewed["recovery_state"] == "abstained"
    assert reviewed["viewer_natural_korean"] == "…"


def test_controlled_vocalization_survives_review_fallback_selection():
    agreement = build_frame_agreement(
        frame(
            speech_act="vocalization",
            question=None,
            polarity=None,
            command_strength=None,
            actor=None,
            action=None,
            target=None,
            tense_aspect=None,
            direction=None,
            intensity=None,
        ),
        frame(
            speech_act="vocalization",
            question=None,
            polarity=None,
            command_strength=None,
            actor=None,
            action=None,
            target=None,
            tense_aspect=None,
            direction=None,
            intensity=None,
        ),
    )
    agreement["controlled_nonlexical_korean"] = "앗!"
    [gated] = apply_graduated_evidence_gate(
        [
            {
                "block_number": 1,
                "source_faithful_korean": "모델 문장",
                "viewer_natural_korean": "모델 문장",
                "conservative_source_faithful_korean": "모델 fallback",
                "conservative_viewer_natural_korean": "모델 fallback",
            }
        ],
        {1: agreement},
        {1: {"source_quality_status": "trusted"}},
        {1: {"independent_source_families": ["a", "b"]}},
    )
    [reviewed] = apply_review_outcomes(
        [gated],
        [
            {
                "block_number": 1,
                "verdict": "repair",
                "critical_slot_conflicts": [],
                "unsupported_additions": [],
                "claim_findings": [],
                "fallback_blocking": False,
                "reason": "primary rejected",
            }
        ],
    )
    assert reviewed["recovery_state"] == "vocalization"
    assert reviewed["viewer_natural_korean"] == "앗!"


def fusion_record(*, state="dual_agreement", alignment="block-aligned"):
    return {
        "asr_fusion": {
            "state": state,
            "alignment_strength": alignment,
            "family_hypotheses": [
                {"evidence_refs": ["asr:w:faster-whisper-large-v3"]},
                {"evidence_refs": ["asr:w:reazonspeech-k2-v2"]},
            ],
        }
    }


def test_trusted_source_can_corroborate_one_model_slot():
    terra = frame(
        polarity="negative",
        slot_support={
            "polarity": {
                "basis": "source-text",
                "evidence_refs": ["source-srt:block-9"],
            }
        },
    )
    sol = frame(polarity="positive")
    corroboration = derive_slot_corroboration(
        terra,
        sol,
        acoustic_record={},
        source_quality_record={"source_quality_status": "trusted"},
        block_number=9,
    )
    agreement = build_frame_agreement(terra, sol, corroboration=corroboration)

    assert agreement["recovery_state"] == "recovered_single_model"
    assert agreement["selected_frame"]["polarity"] == "negative"
    assert agreement["slot_provenance"]["polarity"]["provenance"] == "terra_only_supported"
    assert agreement["slot_provenance"]["polarity"]["evidence_refs"] == [
        "source-srt:block-9"
    ]


def test_suspect_source_cannot_resolve_meaning_flipping_conflict():
    terra = frame(
        polarity="negative",
        slot_support={
            "polarity": {
                "basis": "source-text",
                "evidence_refs": [],
            }
        },
    )
    sol = frame(polarity="positive")
    corroboration = derive_slot_corroboration(
        terra,
        sol,
        acoustic_record={},
        source_quality_record={"source_quality_status": "suspect"},
        block_number=9,
    )
    agreement = build_frame_agreement(terra, sol, corroboration=corroboration)

    assert corroboration == {}
    assert agreement["recovery_state"] == "abstained"
    assert agreement["render_blocking_conflicts"] == ["polarity"]


def test_block_aligned_dual_asr_can_corroborate_one_model():
    refs = [
        "asr:w:faster-whisper-large-v3",
        "asr:w:reazonspeech-k2-v2",
    ]
    terra = frame(
        action="stop",
        slot_support={
            "action": {
                "basis": "dual-asr",
                "evidence_refs": refs,
            }
        },
    )
    sol = frame(action="continue")
    corroboration = derive_slot_corroboration(
        terra,
        sol,
        acoustic_record=fusion_record(),
        source_quality_record={"source_quality_status": "unusable"},
        block_number=3,
    )
    agreement = build_frame_agreement(terra, sol, corroboration=corroboration)

    assert corroboration["action"]["model"] == "terra"
    assert agreement["selected_frame"]["action"] == "stop"
    assert agreement["recovery_state"] == "recovered_single_model"


def test_expanded_dual_asr_does_not_resolve_flipping_slot():
    refs = [
        "asr:w:faster-whisper-large-v3",
        "asr:w:reazonspeech-k2-v2",
    ]
    terra = frame(
        polarity="negative",
        slot_support={
            "polarity": {
                "basis": "dual-asr",
                "evidence_refs": refs,
            }
        },
    )
    sol = frame(polarity="positive")
    corroboration = derive_slot_corroboration(
        terra,
        sol,
        acoustic_record=fusion_record(alignment="expanded"),
        source_quality_record={"source_quality_status": "unusable"},
        block_number=3,
    )

    assert corroboration == {}


def test_expanded_dual_asr_may_corroborate_descriptive_slot():
    refs = [
        "asr:w:faster-whisper-large-v3",
        "asr:w:reazonspeech-k2-v2",
    ]
    terra = frame(
        location="inside",
        slot_support={
            "location": {
                "basis": "dual-asr",
                "evidence_refs": refs,
            }
        },
    )
    sol = frame(location="outside")
    corroboration = derive_slot_corroboration(
        terra,
        sol,
        acoustic_record=fusion_record(alignment="expanded"),
        source_quality_record={"source_quality_status": "unusable"},
        block_number=3,
    )
    agreement = build_frame_agreement(terra, sol, corroboration=corroboration)

    assert corroboration["location"]["model"] == "terra"
    assert agreement["selected_frame"]["location"] == "inside"
    assert agreement["recovery_state"] == "recovered_single_model"


def test_stop_continue_direction_conflict_is_promoted_to_blocking():
    agreement = build_frame_agreement(
        frame(direction="stop"),
        frame(direction="continue"),
    )
    assert agreement["slot_provenance"]["direction"]["provenance"] == "conflict_blocking"
    assert agreement["render_blocking_conflicts"] == ["direction"]
    assert agreement["recovery_state"] == "minimal_speech_act"


def test_completed_progressive_conflict_is_not_silently_omitted():
    agreement = build_frame_agreement(
        frame(tense_aspect="completed"),
        frame(tense_aspect="progressive"),
    )
    assert agreement["render_blocking_conflicts"] == ["tense_aspect"]
    assert agreement["recovery_state"] == "minimal_speech_act"


def test_intensity_remains_omittable_when_command_strength_is_explicit():
    agreement = build_frame_agreement(
        frame(intensity="low", command_strength="command"),
        frame(intensity="high", command_strength="command"),
    )
    assert agreement["slot_provenance"]["intensity"]["provenance"] == "conflict_omitted"
    assert agreement["recovery_state"] == "accepted_partial"


def test_intensity_becomes_blocking_when_it_is_only_force_signal():
    agreement = build_frame_agreement(
        frame(intensity="low", command_strength=None),
        frame(intensity="high", command_strength=None),
    )
    assert agreement["slot_provenance"]["intensity"]["provenance"] == "conflict_blocking"
    assert agreement["recovery_state"] == "minimal_speech_act"


def test_interrogative_request_is_not_inherently_incoherent():
    agreement = build_frame_agreement(
        frame(
            speech_act="request",
            question=True,
            command_strength="request",
            polarity="positive",
        ),
        frame(
            speech_act="request",
            question=True,
            command_strength="request",
            polarity="positive",
        ),
    )
    assert agreement["coherence_findings"] == []
    assert agreement["recovery_state"] == "accepted_consensus"


def test_damaged_source_dual_asr_conflict_is_not_adequate_support():
    agreement = build_frame_agreement(
        frame(
            speech_act="refusal",
            refusal_permission="refusal",
            command_strength=None,
            action=None,
            actor=None,
            target=None,
            direction=None,
        ),
        frame(
            speech_act="refusal",
            refusal_permission="refusal",
            command_strength=None,
            action=None,
            actor=None,
            target=None,
            direction=None,
        ),
    )
    rows = [
        {
            "block_number": 1,
            "source_faithful_korean": "싫어",
            "viewer_natural_korean": "안 돼",
            "conservative_source_faithful_korean": "싫어",
            "conservative_viewer_natural_korean": "안 돼",
            "rendered_slots": ["speech_act", "refusal_permission"],
            "fallback_rendered_slots": ["speech_act", "refusal_permission"],
            "fallback_recovery_state": "minimal_speech_act",
        }
    ]
    gated = apply_graduated_evidence_gate(
        rows,
        {1: agreement},
        {1: {"source_quality_status": "unusable"}},
        {
            1: {
                "independent_source_families": ["a", "b"],
                "asr_fusion": {
                    "state": "dual_conflict",
                    "alignment_strength": "block-aligned",
                    "risk_codes": ["polarity-marker-divergence"],
                },
            }
        },
    )
    assert gated[0]["recovery_state"] == "minimal_speech_act"
    assert "damaged-source-without-dual-acoustic-evidence" in gated[0]["uncertainty_codes"]


def test_extreme_intensity_conflict_in_forceful_command_is_blocking():
    agreement = build_frame_agreement(
        frame(
            speech_act="prohibition",
            refusal_permission="prohibition",
            command_strength=None,
            intensity="high",
        ),
        frame(
            speech_act="prohibition",
            refusal_permission="prohibition",
            command_strength=None,
            intensity="low",
        ),
    )

    assert agreement["slot_provenance"]["intensity"]["provenance"] == "conflict_blocking"
    assert agreement["recovery_state"] == "minimal_speech_act"


def test_opposite_direction_conflict_is_not_silently_omitted():
    agreement = build_frame_agreement(
        frame(direction="in"),
        frame(direction="out"),
    )

    assert agreement["slot_provenance"]["direction"]["provenance"] == "conflict_blocking"
    assert agreement["recovery_state"] == "minimal_speech_act"


def test_non_opposite_direction_conflict_remains_omittable():
    agreement = build_frame_agreement(
        frame(direction="up"),
        frame(direction="toward"),
    )

    assert agreement["slot_provenance"]["direction"]["provenance"] == "conflict_omitted"
    assert agreement["recovery_state"] == "accepted_partial"


def test_source_text_corroboration_requires_exact_block_reference():
    terra = frame(polarity="positive") | {
        "slot_support": {
            "polarity": {
                "basis": "source-text",
                "evidence_refs": ["source-srt:block-7"],
            }
        }
    }
    sol = frame(polarity="negative")
    unsupported = derive_slot_corroboration(
        terra,
        sol,
        acoustic_record={},
        source_quality_record={"source_quality_status": "trusted"},
        block_number=8,
    )
    supported = derive_slot_corroboration(
        terra,
        sol,
        acoustic_record={},
        source_quality_record={"source_quality_status": "trusted"},
        block_number=7,
    )

    assert "polarity" not in unsupported
    assert supported["polarity"]["model"] == "terra"


def test_v2_raw_conflicts_do_not_recreate_block_level_quarantine():
    row = {
        "block_number": 1,
        "source_faithful_korean": "멈춰",
        "viewer_natural_korean": "멈춰",
        "conservative_source_faithful_korean": "그만",
        "conservative_viewer_natural_korean": "그만",
        "source_status": "accepted",
        "viewer_status": "supported",
        "recovery_state": "accepted_partial",
        "fallback_recovery_state": "minimal_speech_act",
        "rendered_slots": ["speech_act", "action"],
        "fallback_rendered_slots": ["speech_act"],
        "uncertainty_codes": [],
    }
    review = {
        "block_number": 1,
        "verdict": "accept",
        "critical_slot_conflicts": ["location"],
        "unsupported_additions": [],
        "claim_findings": [{"slot": "location", "disposition": "omit", "reason": "omitted"}],
        "fallback_blocking": False,
        "reason": "safe partial",
    }

    [result] = apply_review_outcomes([row], [review])

    assert result["viewer_natural_korean"] == "멈춰"
    assert result["recovery_state"] == "accepted_partial"


def test_review_fallback_replaces_actual_rendered_slots_and_status():
    row = {
        "block_number": 1,
        "source_faithful_korean": "문 안에서 멈춰",
        "viewer_natural_korean": "안에서 멈춰",
        "conservative_source_faithful_korean": "멈춰",
        "conservative_viewer_natural_korean": "멈춰",
        "source_status": "accepted",
        "viewer_status": "supported",
        "recovery_state": "accepted_partial",
        "fallback_recovery_state": "accepted_partial",
        "rendered_slots": ["speech_act", "action", "location"],
        "fallback_rendered_slots": ["speech_act", "action"],
        "fallback_evidence_safe": True,
        "uncertainty_codes": [],
    }
    review = {
        "block_number": 1,
        "verdict": "repair",
        "critical_slot_conflicts": ["location"],
        "unsupported_additions": ["location"],
        "claim_findings": [{"slot": "location", "disposition": "omit", "reason": "unsupported"}],
        "fallback_blocking": False,
        "reason": "use fallback",
    }

    [result] = apply_review_outcomes([row], [review])

    assert result["viewer_natural_korean"] == "멈춰"
    assert result["rendered_slots"] == ["action", "speech_act"]
    assert result["source_status"] == "accepted"
    assert result["viewer_status"] == "supported"
    assert result["recovery_state"] == "accepted_partial"


def test_completed_vs_present_aspect_is_blocking():
    agreement = build_frame_agreement(
        frame(tense_aspect="completed"),
        frame(tense_aspect="present"),
    )

    assert agreement["slot_provenance"]["tense_aspect"]["provenance"] == "conflict_blocking"
    assert agreement["recovery_state"] == "minimal_speech_act"


def test_review_does_not_accept_unscoped_fallback_text():
    row = {
        "block_number": 1,
        "source_faithful_korean": "해도 돼",
        "viewer_natural_korean": "해도 돼",
        "conservative_source_faithful_korean": "아마 안 돼",
        "conservative_viewer_natural_korean": "안 될지도 몰라",
        "recovery_state": "recovered_single_model",
        "fallback_recovery_state": "minimal_speech_act",
        "rendered_slots": ["speech_act", "refusal_permission"],
        "fallback_rendered_slots": [],
    }
    review = {
        "block_number": 1,
        "verdict": "quarantine",
        "critical_slot_conflicts": ["refusal_permission"],
        "unsupported_additions": [],
        "claim_findings": [
            {
                "slot": "refusal_permission",
                "disposition": "blocking",
                "reason": "meaning flips",
            }
        ],
        "fallback_blocking": False,
        "reason": "fallback has no declared claims",
    }

    [reviewed] = apply_review_outcomes([row], [review])

    assert reviewed["recovery_state"] == "abstained"
    assert reviewed["viewer_natural_korean"] == "…"


def test_gate_rejects_nonvocal_primary_without_rendered_slots():
    agreement = build_frame_agreement(frame(), frame())
    row = {
        "block_number": 1,
        "source_faithful_korean": "멈춰",
        "viewer_natural_korean": "그만해",
        "conservative_source_faithful_korean": "…",
        "conservative_viewer_natural_korean": "…",
        "recovery_state": "accepted_consensus",
        "fallback_recovery_state": "abstained",
        "rendered_slots": [],
        "fallback_rendered_slots": [],
        "uncertainty_codes": [],
    }

    [gated] = apply_graduated_evidence_gate(
        [row],
        {1: agreement},
        {1: {"source_quality_status": "trusted"}},
        {1: {"independent_source_families": []}},
    )

    assert gated["recovery_state"] == "abstained"
    assert "translation-missing-rendered-slots" in gated["reason"]
