from __future__ import annotations

from translation_forensics.asr_fusion import add_asr_fusion, fuse_asr_transcripts


WHISPER = "faster-whisper-large-v3"
REAZON = "reazonspeech-k2-v2"


def row(
    family: str,
    text: str,
    *,
    scope: str = "utterance-timestamp",
    utterance: str = "u-1",
    window: str = "w-1",
    start: float = 1.0,
    end: float = 2.0,
) -> dict:
    return {
        "source_family": family,
        "text": text,
        "alignment_scope": scope,
        "utterance_id": utterance,
        "window_id": window,
        "start_seconds": start,
        "end_seconds": end,
        "clip_start_seconds": 0.0,
        "clip_end_seconds": 28.0,
    }


def test_same_family_windows_are_one_vote() -> None:
    fusion = fuse_asr_transcripts(
        [
            row(WHISPER, "やめて", window="w-1"),
            row(WHISPER, "やめて", window="w-2"),
        ]
    )
    assert fusion["state"] == "single_family"
    assert len(fusion["family_hypotheses"]) == 1


def test_timestamp_evidence_outranks_wide_context() -> None:
    fusion = fuse_asr_transcripts(
        [
            row(
                WHISPER,
                "scene wide unrelated text",
                scope="overlapping-window",
                utterance="",
                window="wide",
            ),
            row(WHISPER, "やめて", utterance="local"),
            row(REAZON, "やめて", utterance="local-r"),
        ]
    )
    assert fusion["state"] == "dual_agreement"
    whisper = next(
        item for item in fusion["family_hypotheses"] if item["source_family"] == WHISPER
    )
    assert whisper["text"] == "やめて"
    assert whisper["alignment_scope"] == "utterance-timestamp"


def test_exact_dual_family_agreement() -> None:
    fusion = fuse_asr_transcripts(
        [row(WHISPER, "行かない"), row(REAZON, "行かない")]
    )
    assert fusion["state"] == "dual_agreement"
    assert fusion["alignment_strength"] == "block-aligned"


def test_surface_variants_can_be_compatible() -> None:
    fusion = fuse_asr_transcripts(
        [row(WHISPER, "もうやめて"), row(REAZON, "やめて")]
    )
    assert fusion["state"] == "dual_compatible"
    assert "やめて" in fusion["shared_spans"]


def test_semantically_unrelated_surfaces_remain_conflict() -> None:
    fusion = fuse_asr_transcripts(
        [row(WHISPER, "行かない"), row(REAZON, "ここに来て")]
    )
    assert fusion["state"] == "dual_conflict"


def test_context_windows_are_alternatives_not_concatenated() -> None:
    fusion = fuse_asr_transcripts(
        [
            row(
                WHISPER,
                "first context",
                scope="overlapping-window",
                utterance="",
                window="w-1",
            ),
            row(
                WHISPER,
                "second context",
                scope="overlapping-window",
                utterance="",
                window="w-2",
            ),
        ],
        block_start=1.0,
        block_end=2.0,
    )
    hypothesis = fusion["family_hypotheses"][0]
    assert hypothesis["text"] in {"first context", "second context"}
    assert hypothesis["text"] != "first context second context"


def test_add_asr_fusion_is_additive() -> None:
    original = {
        "block_number": 7,
        "transcripts": [row(WHISPER, "はい"), row(REAZON, "はい")],
        "evidence_refs": ["existing"],
    }
    updated = add_asr_fusion(original)
    assert "asr_fusion" not in original
    assert updated["evidence_refs"] == ["existing"]
    assert updated["asr_fusion"]["state"] == "dual_agreement"


def test_polarity_marker_divergence_is_never_surface_compatible() -> None:
    fusion = fuse_asr_transcripts(
        [row(WHISPER, "もう行く"), row(REAZON, "もう行かない")]
    )
    assert fusion["state"] == "dual_conflict"
    assert fusion["risk_codes"] == ["polarity-marker-divergence"]


def test_question_marker_divergence_is_explicit() -> None:
    fusion = fuse_asr_transcripts(
        [row(WHISPER, "行く？"), row(REAZON, "行く")]
    )
    assert fusion["state"] == "dual_conflict"
    assert fusion["risk_codes"] == ["question-marker-divergence"]


def test_stop_continue_markers_are_conflict() -> None:
    fusion = fuse_asr_transcripts(
        [row(WHISPER, "やめて"), row(REAZON, "続けて")]
    )
    assert fusion["state"] == "dual_conflict"
    assert fusion["risk_codes"] == ["stop-continue-marker-divergence"]


def test_srt_timestamps_drive_wide_window_overlap_selection():
    record = {
        "schema_name": "translation-forensics/block-acoustic-evidence",
        "schema_version": "1",
        "block_number": 1,
        "start": "00:00:10,000",
        "end": "00:00:12,000",
        "transcripts": [
            {
                "window_id": "far",
                "source_family": "whisper",
                "text": "遠い候補",
                "clip_start_seconds": 0.0,
                "clip_end_seconds": 8.0,
                "alignment_scope": "overlapping-window",
            },
            {
                "window_id": "near",
                "source_family": "whisper",
                "text": "近い候補",
                "clip_start_seconds": 9.0,
                "clip_end_seconds": 13.0,
                "alignment_scope": "overlapping-window",
            },
        ],
        "evidence_refs": [],
        "independent_source_families": ["whisper"],
        "block_alignment_status": "window-context-only",
    }

    fused = add_asr_fusion(record)

    assert fused["asr_fusion"]["family_hypotheses"][0]["text"] == "近い候補"


def test_sentence_final_particle_does_not_hide_polarity_risk():
    fusion = fuse_asr_transcripts(
        [
            row(WHISPER, "行かないよ", scope="utterance-timestamp"),
            row(REAZON, "行くよ", scope="utterance-timestamp"),
        ]
    )

    assert fusion["state"] == "dual_conflict"
    assert "polarity-marker-divergence" in fusion["risk_codes"]
