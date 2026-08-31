from __future__ import annotations

import json
from pathlib import Path

import pytest

from translation_forensics.automated_quality import audit_translation_decision
from translation_forensics.evidence_bridge import (
    build_unit_source_evidence,
    canonical_asr_family,
    prepare_acoustic_evidence,
)
from translation_forensics.evidence_bridge_evaluation import (
    evaluate_source_evidence_contrasts,
)
from translation_forensics.integrated_pipeline import (
    ResumeRejected,
    build_translation_inputs,
    load_translation_units,
)


ROOT = Path(__file__).resolve().parents[2]


def test_family_aliases_collapse_to_one_vote() -> None:
    prepared = prepare_acoustic_evidence(
        {
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "transcripts": [
                {
                    "source_family": "whisper",
                    "text": "行く",
                    "utterance_id": "u1",
                    "alignment_scope": "utterance-timestamp",
                },
                {
                    "source_family": "faster-whisper-large-v3",
                    "text": "行く",
                    "utterance_id": "u1-retry",
                    "alignment_scope": "utterance-timestamp",
                },
            ],
        }
    )
    assert prepared["independent_source_families"] == ["whisper"]
    assert prepared["asr_fusion"]["state"] == "single_family"
    assert len(prepared["asr_fusion"]["family_hypotheses"]) == 1


def test_unknown_family_labels_cannot_manufacture_independence() -> None:
    prepared = prepare_acoustic_evidence(
        {
            "transcripts": [
                {"source_family": "made-up-a", "text": "はい"},
                {"source_family": "made-up-b", "text": "はい"},
            ]
        }
    )
    assert canonical_asr_family("made-up-a") == "unverified"
    assert prepared["independent_source_families"] == []
    assert prepared["asr_fusion"]["state"] == "single_family"
    assert prepared["family_lineage_status"] == "contains-unverified-family"


def test_meaning_flip_conflict_blocks_previously_passing_translation() -> None:
    evidence = build_unit_source_evidence(
        unit_id="u1",
        start=0.0,
        end=1.0,
        source_text="もう行くよ",
        asr_metrics={
            "backend": "faster-whisper-large-v3",
            "qwen_alternatives": ["もう行かないよ"],
        },
        source_quality_status="trusted",
        evidence_ids=["transcript:u1"],
    )
    common = {
        "unit_id": "u1",
        "source_japanese": "もう行くよ",
        "source_faithful_korean": "이제 갈게.",
        "viewer_natural_korean": "이제 갈게.",
        "source_quality_status": "trusted",
        "confidence": "high",
    }
    baseline = audit_translation_decision(**common)
    bridged = audit_translation_decision(**common, source_evidence=evidence)
    assert baseline["status"] == "passed"
    assert bridged["status"] == "passed"
    assert "asr_polarity_marker_divergence" in bridged["warnings"]
    assert bridged["source_evidence_critical_slots"] == ["polarity"]
    assert bridged["semantic_audit_recommended"] is True
    assert bridged["semantic_audit_selection_reasons"] == [
        "source_evidence_critical_slot_conflict"
    ]
    assert evidence["raw_asr_metrics"]["qwen_alternatives"] == ["もう行かないよ"]


def test_absent_evidence_is_unavailable_and_does_not_change_status() -> None:
    evidence = build_unit_source_evidence(
        unit_id="u1",
        start=0.0,
        end=1.0,
        source_text="こんにちは",
        asr_metrics=None,
        source_quality_status="trusted",
    )
    result = audit_translation_decision(
        unit_id="u1",
        source_japanese="こんにちは",
        source_faithful_korean="안녕하세요.",
        viewer_natural_korean="안녕하세요.",
        source_quality_status="trusted",
        confidence="high",
        source_evidence=evidence,
    )
    assert evidence["bridge_state"] == "unavailable"
    assert result["status"] == "passed"
    assert not [reason for reason in result["warnings"] if reason.startswith("asr_")]


def test_translation_unit_round_trip_preserves_bridge_record(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript_ja.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "utterance_id": "u1",
                "start": 0.0,
                "end": 1.0,
                "text_raw": "行く",
                "source_segment_ids": [1],
                "words": [],
                "warnings": [],
                "asr_metrics": {
                    "backend": "whisper",
                    "qwen_alternatives": ["行く？"],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    artifacts = build_translation_inputs(transcript, tmp_path / "inputs")
    loaded = load_translation_units(artifacts.translation_units_ja_jsonl)
    assert artifacts.units[0].source_evidence == loaded[0].source_evidence
    assert loaded[0].source_evidence["asr_fusion"]["state"] == "dual_conflict"
    assert loaded[0].source_evidence["critical_risk_slots"] == ["question", "speech_act"]


def test_v1_units_remain_readable_but_v2_cannot_silently_drop_bridge(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript_ja.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "utterance_id": "u1",
                "start": 0.0,
                "end": 1.0,
                "text_raw": "はい",
                "source_segment_ids": [1],
                "words": [],
                "warnings": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    artifacts = build_translation_inputs(transcript, tmp_path / "inputs")
    path = artifacts.translation_units_ja_jsonl
    row = json.loads(path.read_text(encoding="utf-8"))
    row.pop("source_evidence")
    row["schema_version"] = "1"
    legacy = tmp_path / "legacy-v1.jsonl"
    legacy.write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    assert load_translation_units(legacy)[0].source_evidence == {}

    row["schema_version"] = "2"
    broken = tmp_path / "broken-v2.jsonl"
    broken.write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ResumeRejected, match="lacks source_evidence"):
        load_translation_units(broken)


def test_contrast_suite_queues_hazards_without_clean_regressions() -> None:
    report = evaluate_source_evidence_contrasts(
        ROOT / "tests" / "fixtures" / "source-evidence-contrast-cases.json"
    )
    assert report["status"] == "pass", report["errors"]
    assert report["hazard_count"] == 6
    assert report["newly_queued_hazards"] == 6
    assert report["missed_hazards"] == 0
    assert report["clean_control_count"] == 3
    assert report["clean_control_regressions"] == 0
    assert report["human_quality_proven"] is False
    frozen = json.loads(
        (
            ROOT
            / "evaluation"
            / "engineering"
            / "canonical-evidence-bridge-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert frozen == report
