from __future__ import annotations

from pathlib import Path

import pytest

from translation_forensics.codex_exec_provider import CodexUsageLimitError
from translation_forensics.integrated_translation import (
    audit_translations_with_sol,
    review_translation_with_visuals,
    translate_units_with_terra,
)


class FakeProvider:
    def __init__(self) -> None:
        self.calls = []

    def run_structured(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["role"] == "translation-terra":
            return {
                "translations": [
                    {
                        "unit_id": unit["unit_id"],
                        "source_faithful_korean": "직역 " + unit["unit_id"],
                        "viewer_natural_korean": "자연 " + unit["unit_id"],
                        "confidence": "low" if unit["unit_id"] == "u2" else "high",
                        "uncertain_slots": ["addressee"] if unit["unit_id"] == "u2" else [],
                        "review_required_reasons": [],
                        "recovery_classification": (
                            "FUNCTIONAL_RECOVERY" if unit["unit_id"] == "u2" else "RELIABLE"
                        ),
                        "recovery_basis": ["neighboring_turns"] if unit["unit_id"] == "u2" else [],
                    }
                    for unit in kwargs["payload"]["units"]
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


def test_terra_permits_unresolved_marker_only_for_unusable_unresolved_unit():
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
                    }
                ]
            }, {"call_id": kwargs["call_id"], "image_attachments": []}

    units = [{
        "unit_id": "u1", "start": 0.0, "end": 1.0,
        "source_japanese": "ご視聴ありがとうございました",
        "quality_status": "unusable",
    }]
    decisions, _ = translate_units_with_terra(UnresolvedProvider(), title_id="TEST-001", units=units)
    assert decisions[0]["source_faithful_korean"] == "[불명]"
    assert decisions[0]["recovery_classification"] == "UNRESOLVED"


def test_terra_rejects_unresolved_marker_for_recoverable_source():
    class InvalidMarkerProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            response["translations"][0]["source_faithful_korean"] = "[불명]"
            response["translations"][0]["viewer_natural_korean"] = "[불명]"
            response["translations"][0]["recovery_classification"] = "UNRESOLVED"
            return response, receipt

    with pytest.raises(ValueError, match="unresolved_marker"):
        translate_units_with_terra(InvalidMarkerProvider(), title_id="TEST-001", units=_units()[:1])


def test_terra_rejects_missing_batch_coverage():
    class MissingProvider(FakeProvider):
        def run_structured(self, **kwargs):
            return {"translations": []}, {}

    with pytest.raises(ValueError, match="coverage mismatch"):
        translate_units_with_terra(MissingProvider(), title_id="TEST-001", units=_units())


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
        provider, title_id="TEST-001", decisions=decisions, batch_size=1
    )
    assert [row["viewer_natural_korean"] for row in updated] == original
    assert len(receipts) == len(records) == 2
    assert all(row["sol_audit_status"] == "completed" for row in records)
    assert all(row["audit_call_id"].endswith("translation-audit.sol") for row in records)


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
    assert updated[1]["viewer_natural_korean"] == "수정 자연"
    assert receipts[0]["image_attachments"][0]["sha256"] == "hash-frame.jpg"
    assert records[0]["allowed_visual_slots"]["addressee"] == ["한 명"]
    visual_call = provider.calls[-1]
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
    assert records[0]["verdict"] == "escalate"


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
