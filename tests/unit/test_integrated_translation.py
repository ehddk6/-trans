from __future__ import annotations

from pathlib import Path

import pytest

from translation_forensics.codex_exec_provider import CodexUsageLimitError
from translation_forensics.integrated_translation import (
    AUTOMATED_AUDIT_PROMPT,
    _terra_translation_prompt,
    audit_translations_with_sol,
    review_translation_with_visuals,
    translate_units_with_terra,
)


def _semantic_slots(**values):
    keys = (
        "speech_act", "question", "polarity", "refusal_permission", "stop_continue",
        "command_strength", "speaker", "addressee", "actor", "action", "target",
        "location", "direction", "tense_aspect", "completion", "intensity",
        "numeric_tokens", "register",
    )
    return {key: values.get(key) for key in keys}


class FakeProvider:
    def __init__(self) -> None:
        self.calls = []

    def run_structured(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["role"] == "translation-terra":
            visual_repair = (
                kwargs["payload"].get("contract")
                == "source_bound_visual_slot_repair_not_final"
            )
            units = kwargs["payload"].get("units") or [
                packet["focus_unit"] for packet in kwargs["payload"]["scene_packets"]
            ]
            return {
                "translations": [
                    {
                        "unit_id": unit["unit_id"],
                        "source_faithful_korean": (
                            "거기예요." if visual_repair else "직역 " + unit["unit_id"]
                        ),
                        "viewer_natural_korean": (
                            "거기 있어요." if visual_repair else "자연 " + unit["unit_id"]
                        ),
                        "confidence": "low" if unit["unit_id"] == "u2" else "high",
                        "uncertain_slots": ["addressee"] if unit["unit_id"] == "u2" else [],
                        "review_required_reasons": [],
                        "recovery_classification": (
                            "FUNCTIONAL_RECOVERY" if unit["unit_id"] == "u2" else "RELIABLE"
                        ),
                        "recovery_basis": ["neighboring_turns"] if unit["unit_id"] == "u2" else [],
                        "semantic_slots": _semantic_slots(
                            speech_act="statement", question="no", polarity="positive"
                        ),
                        "preserved_meaning": ["polarity"],
                        "competing_interpretations": [],
                        "risk_codes": (
                            ["SEMANTIC_CONFLICT_ADDRESSEE"]
                            if unit["unit_id"] == "u2"
                            else []
                        ),
                        "critic_required": unit["unit_id"] == "u2",
                    }
                    for unit in units
                ]
            }, {"call_id": kwargs["call_id"], "image_attachments": []}
        if kwargs["role"] == "translation-audit-sol":
            return {
                "audits": [
                    {
                        "unit_id": unit["unit_id"],
                        "verdict": "pass",
                        "backtranslation_japanese": unit["source_japanese"],
                        "source_fidelity_score": 0.95,
                        "question_preserved": True,
                        "negation_preserved": True,
                        "numeric_tokens_preserved": True,
                        "reasons": [],
                        "meaning_flip": False,
                        "uncertainty_codes": [],
                        "critical_slot_issues": [],
                        "unsupported_addition": False,
                    }
                    for unit in kwargs["payload"]["units"]
                ]
            }, {"call_id": kwargs["call_id"], "image_attachments": []}
        image_paths = kwargs["image_paths"]
        return {
            "unit_id": kwargs["payload"]["unit"]["unit_id"],
            "verdict": "repair",
            "source_faithful_korean": "수정 직역",
            "viewer_natural_korean": "수정 자연",
            "confidence": "medium",
            "critical_visual_impact": False,
            "visual_slots": {
                "speaker": [],
                "addressee": ["한 명"],
                "deictic_location": [],
                "on_screen_text": [],
                "scene_continuity": [],
            },
            "review_required_reasons": [],
        }, {
            "call_id": kwargs["call_id"],
            "image_attachments": [{"sha256": f"hash-{path.name}"} for path in image_paths],
        }


def _units():
    return [
        {"unit_id": "u1", "start": 0.0, "end": 1.0, "source_japanese": "行く？", "quality_status": "trusted"},
        {"unit_id": "u2", "start": 1.0, "end": 2.0, "source_japanese": "そこ。", "quality_status": "suspect"},
    ]


def test_terra_translates_every_unit_even_when_source_is_suspect():
    provider = FakeProvider()
    decisions, receipts = translate_units_with_terra(provider, title_id="TEST-001", units=_units(), batch_size=1)
    assert [row["unit_id"] for row in decisions] == ["u1", "u2"]
    assert all(row["viewer_natural_korean"] for row in decisions)
    assert decisions[1]["source_quality_status"] == "suspect"
    assert decisions[0]["recovery_classification"] == "RELIABLE"
    assert decisions[1]["recovery_classification"] == "FUNCTIONAL_RECOVERY"
    assert decisions[0]["source_profile"] == "MIXED"
    assert len(receipts) == 2
    first_packet = provider.calls[0]["payload"]["scene_packets"][0]
    second_packet = provider.calls[1]["payload"]["scene_packets"][0]
    assert first_packet["next_units"][0]["source_japanese"] == "そこ。"
    assert second_packet["previous_units"][0]["source_japanese"] == "行く？"
    assert set(decisions[0]["semantic_slots"]) == {
        "speech_act", "question", "polarity", "refusal_permission", "stop_continue",
        "command_strength", "speaker", "addressee", "actor", "action", "target",
        "location", "direction", "tense_aspect", "completion", "intensity",
        "numeric_tokens", "register",
    }


def test_parallel_terra_batches_preserve_source_order():
    provider = FakeProvider()
    units = [
        {
            "unit_id": f"u{index}",
            "start": float(index),
            "end": float(index + 1),
            "source_japanese": f"台詞{index}",
            "quality_status": "suspect" if index == 2 else "trusted",
        }
        for index in range(1, 5)
    ]

    decisions, receipts = translate_units_with_terra(
        provider,
        title_id="TEST-001",
        units=units,
        batch_size=1,
        max_batch_workers=2,
    )

    assert [row["unit_id"] for row in decisions] == ["u1", "u2", "u3", "u4"]
    assert [receipt["call_id"] for receipt in receipts] == [
        "batch-0001.translation.terra",
        "batch-0002.translation.terra",
        "batch-0003.translation.terra",
        "batch-0004.translation.terra",
    ]


def test_terra_permits_unresolved_marker_only_for_missing_source_text():
    class UnresolvedProvider(FakeProvider):
        def run_structured(self, **kwargs):
            return {
                "translations": [
                    {
                        "unit_id": "u1",
                        "source_faithful_korean": "[불명]",
                        "viewer_natural_korean": "[불명]",
                        "confidence": "low",
                        "uncertain_slots": ["utterance_function"],
                        "review_required_reasons": ["source_unusable"],
                        "recovery_classification": "UNRESOLVED",
                        "recovery_basis": [],
                        "semantic_slots": _semantic_slots(),
                        "preserved_meaning": [],
                        "competing_interpretations": ["unknown"],
                        "risk_codes": ["source_unusable"],
                        "critic_required": True,
                    }
                ]
            }, {"call_id": kwargs["call_id"], "image_attachments": []}

    units = [{
        "unit_id": "u1", "start": 0.0, "end": 1.0,
        "source_japanese": "",
        "quality_status": "unusable",
    }]
    decisions, _ = translate_units_with_terra(UnresolvedProvider(), title_id="TEST-001", units=units)
    assert decisions[0]["source_faithful_korean"] == "[불명]"
    assert decisions[0]["recovery_classification"] == "UNRESOLVED"


def test_terra_rejects_unresolved_marker_for_suspect_lexical_source():
    class SuspectUnresolvedProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            response["translations"][0].update(
                {
                    "source_faithful_korean": "[불명]",
                    "viewer_natural_korean": "[불명]",
                    "recovery_classification": "UNRESOLVED",
                    "recovery_basis": [],
                    "uncertain_slots": ["utterance_function"],
                    "critic_required": True,
                }
            )
            return response, receipt

    units = [{
        "unit_id": "u1", "start": 0.0, "end": 1.0,
        "source_japanese": "聞き取れない", "quality_status": "suspect",
    }]
    with pytest.raises(ValueError, match="lexical Japanese"):
        translate_units_with_terra(
            SuspectUnresolvedProvider(), title_id="TEST-001", units=units
        )


def test_terra_retries_cached_unresolved_marker_for_lexical_source():
    class RetryProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            if (
                kwargs["role"] == "translation-terra"
                and not kwargs["call_id"].endswith(".lexical-completion-retry")
            ):
                response["translations"][0].update(
                    {
                        "source_faithful_korean": "[불명]",
                        "viewer_natural_korean": "[불명]",
                        "recovery_classification": "UNRESOLVED",
                        "recovery_basis": [],
                    }
                )
            return response, receipt

    provider = RetryProvider()
    decisions, receipts = translate_units_with_terra(
        provider,
        title_id="TEST-001",
        units=[{
            "unit_id": "u1", "start": 0.0, "end": 1.0,
            "source_japanese": "聞き取れない", "quality_status": "suspect",
        }],
    )

    assert decisions[0]["viewer_natural_korean"] == "자연 u1"
    assert decisions[0]["recovery_classification"] == "RELIABLE"
    assert decisions[0]["terra_call_id"] == (
        "batch-0001.translation.terra.u1.lexical-completion-retry"
    )
    assert [call["call_id"] for call in provider.calls] == [
        "batch-0001.translation.terra",
        "batch-0001.translation.terra.u1.lexical-completion-retry",
    ]
    assert [receipt["call_id"] for receipt in receipts] == [
        "batch-0001.translation.terra",
        "batch-0001.translation.terra.u1.lexical-completion-retry",
    ]


def test_terra_retries_multiple_lexical_units_without_changing_receipt_order():
    class MultipleRetryProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            if (
                kwargs["role"] == "translation-terra"
                and kwargs["call_id"].endswith(".translation.terra")
            ):
                for row in response["translations"]:
                    row.update(
                        {
                            "source_faithful_korean": "[불명]",
                            "viewer_natural_korean": "[불명]",
                            "recovery_classification": "UNRESOLVED",
                            "recovery_basis": [],
                        }
                    )
            return response, receipt

    decisions, receipts = translate_units_with_terra(
        MultipleRetryProvider(),
        title_id="TEST-001",
        units=_units(),
        batch_size=2,
        max_batch_workers=2,
    )

    assert [row["unit_id"] for row in decisions] == ["u1", "u2"]
    assert [row["terra_call_id"] for row in decisions] == [
        "batch-0001.translation.terra.u1.lexical-completion-retry",
        "batch-0001.translation.terra.u2.lexical-completion-retry",
    ]
    assert [receipt["call_id"] for receipt in receipts] == [
        "batch-0001.translation.terra",
        "batch-0001.translation.terra.u1.lexical-completion-retry",
        "batch-0001.translation.terra.u2.lexical-completion-retry",
    ]


def test_terra_escalates_a_single_unit_that_repeats_unresolved_marker():
    class EscalationProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            if (
                kwargs["role"] == "translation-terra"
                and not kwargs["call_id"].endswith(".escalation")
            ):
                response["translations"][0].update(
                    {
                        "source_faithful_korean": "[불명]",
                        "viewer_natural_korean": "[불명]",
                        "recovery_classification": "UNRESOLVED",
                        "recovery_basis": [],
                    }
                )
            return response, receipt

    provider = EscalationProvider()
    decisions, receipts = translate_units_with_terra(
        provider,
        title_id="TEST-001",
        units=[{
            "unit_id": "u1", "start": 0.0, "end": 1.0,
            "source_japanese": "聞き取れない", "quality_status": "suspect",
        }],
    )

    assert decisions[0]["viewer_natural_korean"] == "자연 u1"
    assert decisions[0]["terra_call_id"] == (
        "batch-0001.translation.terra.u1.lexical-completion-retry.escalation"
    )
    assert [call["call_id"] for call in provider.calls] == [
        "batch-0001.translation.terra",
        "batch-0001.translation.terra.u1.lexical-completion-retry",
        "batch-0001.translation.terra.u1.lexical-completion-retry.escalation",
    ]
    assert [receipt["call_id"] for receipt in receipts] == [
        "batch-0001.translation.terra",
        "batch-0001.translation.terra.u1.lexical-completion-retry.escalation",
    ]


def test_terra_rejects_human_hold_marker_for_lexical_source():
    class HoldMarkerProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            response["translations"][0]["source_faithful_korean"] = "[원문 불명확]"
            response["translations"][0]["viewer_natural_korean"] = "[원문 불명확]"
            return response, receipt

    with pytest.raises(ValueError, match="lexical completion escalation failed"):
        translate_units_with_terra(
            HoldMarkerProvider(), title_id="TEST-001", units=_units()[:1]
        )


def test_terra_rejects_unresolved_classification_with_invented_text():
    class InventedUnresolvedProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            response["translations"][0].update(
                {
                    "source_faithful_korean": "좋아",
                    "viewer_natural_korean": "좋아",
                    "recovery_classification": "UNRESOLVED",
                    "recovery_basis": [],
                }
            )
            return response, receipt

    units = [{
        "unit_id": "u1", "start": 0.0, "end": 1.0,
        "source_japanese": "ご視聴ありがとうございました", "quality_status": "unusable",
    }]
    with pytest.raises(ValueError, match="lexical Japanese"):
        translate_units_with_terra(
            InventedUnresolvedProvider(), title_id="TEST-001", units=units
        )


def test_terra_rejects_unresolved_marker_for_lexical_source():
    class InvalidMarkerProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            response["translations"][0]["source_faithful_korean"] = "[불명]"
            response["translations"][0]["viewer_natural_korean"] = "[불명]"
            response["translations"][0]["recovery_classification"] = "UNRESOLVED"
            return response, receipt

    with pytest.raises(ValueError, match="lexical Japanese"):
        translate_units_with_terra(InvalidMarkerProvider(), title_id="TEST-001", units=_units()[:1])


def test_terra_rejects_unresolved_marker_embedded_in_other_text():
    class MixedMarkerProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            response["translations"][0]["source_faithful_korean"] = "잘 [불명]"
            return response, receipt

    with pytest.raises(ValueError, match="entire_translation"):
        translate_units_with_terra(
            MixedMarkerProvider(), title_id="TEST-001", units=_units()[:1]
        )


def test_terra_rejects_missing_recovery_contract_fields_from_any_provider():
    class MissingRecoveryProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            response["translations"][0].pop("recovery_classification")
            return response, receipt

    with pytest.raises(ValueError, match="schema error"):
        translate_units_with_terra(
            MissingRecoveryProvider(), title_id="TEST-001", units=_units()[:1]
        )


def test_terra_rejects_missing_semantic_slots_from_any_provider():
    class MissingSlotsProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            response["translations"][0].pop("semantic_slots")
            return response, receipt

    with pytest.raises(ValueError, match="semantic_slots"):
        translate_units_with_terra(
            MissingSlotsProvider(), title_id="TEST-001", units=_units()[:1]
        )


def test_terra_scene_packet_carries_three_turns_of_context():
    provider = FakeProvider()
    units = [
        {
            "unit_id": f"u{index}", "start": float(index), "end": float(index + 1),
            "source_japanese": f"台詞{index}",
            "quality_status": "suspect" if index == 2 else "trusted",
            "speaker": f"speaker-{index % 2}",
        }
        for index in range(1, 8)
    ]
    translate_units_with_terra(provider, title_id="TEST-001", units=units, batch_size=7)
    packet = provider.calls[0]["payload"]["scene_packets"][3]
    assert packet["focus_unit"]["unit_id"] == "u4"
    assert [row["unit_id"] for row in packet["previous_units"]] == ["u1", "u2", "u3"]
    assert [row["unit_id"] for row in packet["next_units"]] == ["u5", "u6", "u7"]
    assert packet["focus_unit"]["speaker"] == "speaker-0"


def test_terra_requires_basis_for_functional_recovery():
    class MissingBasisProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            response["translations"][0]["recovery_classification"] = "FUNCTIONAL_RECOVERY"
            response["translations"][0]["recovery_basis"] = []
            return response, receipt

    suspect = [{**_units()[1], "unit_id": "u1"}]
    with pytest.raises(ValueError, match="schema error"):
        translate_units_with_terra(
            MissingBasisProvider(), title_id="TEST-001", units=suspect
        )


def test_terra_retries_missing_batch_coverage_with_a_fresh_call():
    class CoverageRetryProvider(FakeProvider):
        def run_structured(self, **kwargs):
            if kwargs["call_id"] == "batch-0001.translation.terra":
                self.calls.append(kwargs)
                return {"translations": []}, {"call_id": kwargs["call_id"]}
            return super().run_structured(**kwargs)

    provider = CoverageRetryProvider()
    decisions, receipts = translate_units_with_terra(
        provider, title_id="TEST-001", units=_units()
    )

    assert [row["unit_id"] for row in decisions] == ["u1", "u2"]
    assert [row["terra_call_id"] for row in decisions] == [
        "batch-0001.translation.terra.coverage-retry",
        "batch-0001.translation.terra.coverage-retry",
    ]
    assert [receipt["call_id"] for receipt in receipts] == [
        "batch-0001.translation.terra",
        "batch-0001.translation.terra.coverage-retry",
    ]
    assert "Mandatory batch coverage retry" in provider.calls[1]["prompt"]


def test_terra_rejects_missing_batch_coverage_after_retry():
    class MissingProvider(FakeProvider):
        def __init__(self):
            super().__init__()

        def run_structured(self, **kwargs):
            self.calls.append(kwargs)
            return {"translations": []}, {"call_id": kwargs["call_id"]}

    provider = MissingProvider()
    with pytest.raises(ValueError, match="coverage mismatch"):
        translate_units_with_terra(provider, title_id="TEST-001", units=_units())
    assert [call["call_id"] for call in provider.calls] == [
        "batch-0001.translation.terra",
        "batch-0001.translation.terra.coverage-retry",
    ]


def test_terra_drops_only_unrequested_extra_rows_and_records_warning():
    class ExtraProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            response["translations"].append(
                {
                    "unit_id": "not-requested",
                    "source_faithful_korean": "범위 밖",
                    "viewer_natural_korean": "범위 밖",
                        "confidence": "high",
                        "uncertain_slots": [],
                        "review_required_reasons": [],
                        "recovery_classification": "RELIABLE",
                        "recovery_basis": [],
                        "semantic_slots": _semantic_slots(),
                        "preserved_meaning": [],
                        "competing_interpretations": [],
                        "risk_codes": [],
                        "critic_required": False,
                }
            )
            return response, receipt

    decisions, receipts = translate_units_with_terra(
        ExtraProvider(), title_id="TEST-001", units=_units()
    )
    assert [row["unit_id"] for row in decisions] == ["u1", "u2"]
    assert receipts[0]["response_contract_warnings"] == [
        "unexpected_unit_ids_dropped:not-requested"
    ]


def test_sol_audit_returns_item_level_evidence_without_rewriting_translation():
    provider = FakeProvider()
    decisions, _ = translate_units_with_terra(provider, title_id="TEST-001", units=_units())
    original = [row["viewer_natural_korean"] for row in decisions]
    updated, receipts, records = audit_translations_with_sol(
        provider,
        title_id="TEST-001",
        decisions=decisions,
        batch_size=1,
        max_batch_workers=2,
    )
    assert [row["viewer_natural_korean"] for row in updated] == original
    assert len(receipts) == 1
    assert len(records) == 2
    assert records[0]["semantic_audit_status"] == "not-selected"
    assert records[0]["semantic_audit_selection_reasons"] == []
    assert records[1]["semantic_audit_status"] == "pass"
    assert records[1]["audit_call_id"].endswith("translation-audit.sol")
    audit_call = next(
        call for call in provider.calls if call["role"] == "translation-audit-sol"
    )
    assert [row["unit_id"] for row in audit_call["payload"]["units"]] == ["u2"]
    assert audit_call["payload"]["units"][0]["previous_source_japanese"] == "行く？"


def test_sol_skips_generic_source_and_uncertainty_metadata():
    provider = FakeProvider()
    decisions, _ = translate_units_with_terra(
        provider, title_id="TEST-001", units=_units()[:1]
    )
    decisions[0].update(
        {
            "critic_required": True,
            "uncertain_slots": ["addressee"],
            "competing_interpretations": ["A or B"],
            "risk_codes": [
                "SUSPECT_SOURCE",
                "SINGLE_FAMILY_ASR",
                "GARBLED_SOURCE",
            ],
        }
    )

    updated, receipts, records = audit_translations_with_sol(
        provider, title_id="TEST-001", decisions=decisions
    )

    assert receipts == []
    assert updated[0]["semantic_audit_status"] == "not-selected"
    assert records[0]["semantic_audit_selection_reasons"] == []
    assert not [call for call in provider.calls if call["role"] == "translation-audit-sol"]


def test_sol_skips_heuristic_fallback_without_an_explicit_semantic_conflict():
    provider = FakeProvider()
    decisions, _ = translate_units_with_terra(
        provider, title_id="TEST-001", units=_units()[:1]
    )
    decisions[0]["automated_quality_status"] = "fallback"

    updated, receipts, records = audit_translations_with_sol(
        provider, title_id="TEST-001", decisions=decisions
    )

    assert receipts == []
    assert updated[0]["semantic_audit_status"] == "not-selected"
    assert records[0]["semantic_audit_selection_reasons"] == []


def test_sol_missing_backtranslation_is_inconclusive_without_flattening_display():
    class WeakAuditProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            if kwargs["role"] == "translation-audit-sol":
                response["audits"][0]["backtranslation_japanese"] = ""
                response["audits"][0]["source_fidelity_score"] = 0.0
            return response, receipt

    provider = WeakAuditProvider()
    decisions, _ = translate_units_with_terra(
        provider, title_id="TEST-001", units=_units()[:1]
    )
    decisions[0]["automated_quality_status"] = "passed"
    decisions[0]["critic_required"] = True
    decisions[0]["risk_codes"] = ["SEMANTIC_CONFLICT_MEANING"]
    updated, _, records = audit_translations_with_sol(
        provider, title_id="TEST-001", decisions=decisions
    )
    assert updated[0]["automated_quality_status"] == "passed"
    assert records[0]["model_verdict"] == "pass"
    assert records[0]["semantic_audit_status"] == "inconclusive"
    assert "sol_backtranslation_empty" in records[0]["reasons"]


def test_sol_critical_slot_issue_forces_fallback():
    class CriticalSlotProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            if kwargs["role"] == "translation-audit-sol":
                response["audits"][0]["critical_slot_issues"] = ["polarity"]
            return response, receipt

    provider = CriticalSlotProvider()
    decisions, _ = translate_units_with_terra(
        provider, title_id="TEST-001", units=_units()[:1]
    )
    decisions[0]["automated_quality_status"] = "passed"
    decisions[0]["critic_required"] = True
    decisions[0]["risk_codes"] = ["SEMANTIC_CONFLICT_POLARITY"]
    updated, _, records = audit_translations_with_sol(
        provider, title_id="TEST-001", decisions=decisions
    )
    assert updated[0]["automated_quality_status"] == "fallback"
    assert records[0]["model_verdict"] == "pass"
    assert records[0]["semantic_audit_status"] == "issue"
    assert "sol_critical_slot:polarity" in records[0]["reasons"]


def test_translation_and_audit_prompts_keep_style_and_semantics_separate():
    translation_prompt = _terra_translation_prompt()
    assert "viewer-natural field is the primary subtitle" in translation_prompt
    assert "Draft it first" in translation_prompt
    assert "not a naturalness, tone, censorship, or literalness review" in AUTOMATED_AUDIT_PROMPT
    assert "adult-genre register" in AUTOMATED_AUDIT_PROMPT


def test_visual_review_is_bounded_and_records_transfer(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"frame")
    provider = FakeProvider()
    decisions, _ = translate_units_with_terra(provider, title_id="TEST-001", units=_units())
    updated, receipts, records = review_translation_with_visuals(
        provider,
        title_id="TEST-001",
        decisions=decisions,
        units_by_id={row["unit_id"]: row for row in _units()},
        frames_by_unit={"u2": [image]},
    )
    assert updated[1]["viewer_natural_korean"] == "거기 있어요."
    assert updated[1]["visual_repair_terra_call_id"] == "u2.visual-repair.terra"
    assert records[0]["verdict"] == "repair-applied-pending-independent-audit"
    assert [call["role"] for call in provider.calls[-2:]] == ["critique-sol", "translation-terra"]
    assert len(receipts) == 2
    assert receipts[0]["image_attachments"][0]["sha256"] == "hash-frame.jpg"
    assert records[0]["allowed_visual_slots"]["addressee"] == ["한 명"]
    visual_call = provider.calls[-2]
    assert visual_call["allow_image_transfer"] is True
    assert visual_call["image_paths"] == [image.resolve()]


def test_visual_review_does_not_repair_suspect_asr_lexical_content(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"frame")
    class ASRInferenceProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            if kwargs["role"] == "critique-sol":
                response["review_required_reasons"] = ["ASR 오인식으로 보여 원음 확인 필요"]
            return response, receipt

    provider = ASRInferenceProvider()
    decisions, _ = translate_units_with_terra(provider, title_id="TEST-001", units=_units())
    original = decisions[1]["viewer_natural_korean"]
    updated, _, records = review_translation_with_visuals(
        provider,
        title_id="TEST-001",
        decisions=decisions,
        units_by_id={row["unit_id"]: row for row in _units()},
        frames_by_unit={"u2": [image]},
    )
    assert updated[1]["viewer_natural_korean"] == original
    assert "visual_text_repair_not_applied_asr_inference_or_unusable_source" in updated[1]["review_required_reasons"]
    assert records[0]["model_verdict"] == "repair"
    assert records[0]["verdict"] == "machine-uncertain"
    assert "critical_visual_context_requires_machine_uncertain" in updated[1]["review_required_reasons"]
    assert "critical_visual_context_requires_human_review" not in updated[1]["review_required_reasons"]


def test_visual_review_rejects_more_than_three_frames(tmp_path):
    frames = []
    for index in range(4):
        frame = tmp_path / f"{index}.jpg"
        frame.write_bytes(b"x")
        frames.append(frame)
    provider = FakeProvider()
    decisions, _ = translate_units_with_terra(provider, title_id="TEST-001", units=_units())
    with pytest.raises(ValueError, match="3-frame"):
        review_translation_with_visuals(
            provider,
            title_id="TEST-001",
            decisions=decisions,
            units_by_id={row["unit_id"]: row for row in _units()},
            frames_by_unit={"u2": frames},
        )


def test_visual_usage_limit_escalates_without_losing_complete_draft(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"frame")

    class LimitedProvider(FakeProvider):
        def run_structured(self, **kwargs):
            if kwargs["role"] == "critique-sol":
                raise CodexUsageLimitError(
                    "limited",
                    role="critique-sol",
                    call_id=kwargs["call_id"],
                    retry_after="later",
                )
            return super().run_structured(**kwargs)

    provider = LimitedProvider()
    decisions, _ = translate_units_with_terra(provider, title_id="TEST-001", units=_units())
    updated, receipts, records = review_translation_with_visuals(
        provider,
        title_id="TEST-001",
        decisions=decisions,
        units_by_id={row["unit_id"]: row for row in _units()},
        frames_by_unit={"u2": [image]},
    )
    assert updated[1]["viewer_natural_korean"] == "자연 u2"
    assert "visual_review_blocked_usage_limit" in updated[1]["review_required_reasons"]
    assert receipts == []
    assert records[0]["model_call_receipt"]["status"] == "blocked"
    assert records[0]["model_call_receipt"]["pixel_external_transfer_count"] == 0
