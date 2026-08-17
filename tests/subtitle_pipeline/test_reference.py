from pathlib import Path
import subprocess
import sys

import pytest

from subtitle_pipeline.optimizer import segments_to_utterances, source_faithful_cues, viewer_cues
from subtitle_pipeline.pipeline import run_pipeline
from subtitle_pipeline.pipeline import rebuild_presentation_from_transcript
from subtitle_pipeline.profiles import get_profile
from subtitle_pipeline.reference import segments_from_reference_rows
from subtitle_pipeline.srt import validate_cues


def test_reference_primary_preserves_text_and_repairs_only_overlapping_timing():
    rows = [
        {"id": "1", "start": 1.0, "end": 2.0, "text": "first source"},
        {"id": "2", "start": 1.8, "end": 3.0, "text": "second source"},
    ]

    segments = segments_from_reference_rows(rows)
    utterances = segments_to_utterances(segments, "strict")
    source = source_faithful_cues(utterances)
    viewer = viewer_cues(utterances, get_profile("viewer_ja"), "strict")

    assert [cue.text for cue in source] == ["first source", "second source"]
    assert [cue.text.replace("\n", "") for cue in viewer] == ["first source", "second source"]
    assert "reference_timing_adjusted" in segments[1].warnings
    assert validate_cues(source) == []
    assert validate_cues(viewer) == []


def test_reference_backend_needs_no_video_and_preserves_internal_spaces(tmp_path):
    reference = tmp_path / "reference.srt"
    reference.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n原文  の  空白\n",
        encoding="utf-8",
    )

    report = run_pipeline(None, tmp_path / "output", backend="reference", reference_srt=reference)
    result = tmp_path / "output" / "reference_primary"

    assert report["backend"] == "reference"
    assert "原文  の  空白" in (result / "source_faithful_ja.srt").read_text(encoding="utf-8")
    assert (result / "viewer_ja.srt").is_file()


def test_reference_source_is_not_misclassified_by_asr_heuristics(tmp_path):
    reference = tmp_path / "reference.srt"
    reference.write_text("1\n00:00:01,000 --> 00:00:02,000\nababababab\n", encoding="utf-8")

    report = run_pipeline(None, tmp_path / "output", backend="reference", reference_srt=reference)

    assert report["runaway_repetition_suspicions"] == 0
    assert {item["code"] for item in report["content_quality_gate"]["reasons"]} == {
        "remaining_viewer_runaway_repetition"
    }


def test_reference_backend_rejects_an_srt_without_cues(tmp_path):
    reference = tmp_path / "empty.srt"
    reference.write_text("not an srt", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one valid"):
        run_pipeline(None, Path(tmp_path / "output"), backend="reference", reference_srt=reference)


def test_reference_backend_rejects_hangul_dominant_translation(tmp_path):
    reference = tmp_path / "translated.srt"
    reference.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n이 자막은 일본어 원문이 아니라 한국어 번역입니다.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Hangul-dominant"):
        run_pipeline(None, tmp_path / "output", backend="reference", reference_srt=reference)


def test_reference_backend_rejects_a_short_all_hangul_translation(tmp_path):
    reference = tmp_path / "short-translated.srt"
    reference.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n한국어 번역입니다\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Hangul-dominant"):
        run_pipeline(None, tmp_path / "output", backend="reference", reference_srt=reference)


def test_reference_language_checker_supports_batch_fallback(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    script = project_root / "scripts" / "check_reference_language.py"
    japanese = tmp_path / "japanese.srt"
    korean = tmp_path / "korean.srt"
    japanese.write_text("1\n00:00:00,000 --> 00:00:01,000\n日本語の原文です。\n", encoding="utf-8")
    korean.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n이 파일은 일본어가 아니라 한국어 번역 자막입니다.\n",
        encoding="utf-8",
    )

    accepted = subprocess.run([sys.executable, str(script), "--reference-srt", str(japanese)], check=False)
    rejected = subprocess.run([sys.executable, str(script), "--reference-srt", str(korean)], check=False)

    assert accepted.returncode == 0
    assert rejected.returncode == 1


def test_rebuild_keeps_source_raw_and_applies_viewer_compaction_only(tmp_path):
    transcript = tmp_path / "transcript_ja.jsonl"
    transcript.write_text(
        '{"utterance_id":"utt_000001","source_segment_ids":[1],"start":0.0,"end":2.0,'
        '"text_raw":"あ、あ、あ、あ、あ、","text_normalized":"ignored","words":[],'
        '"speaker":"speaker_unknown","asr_metrics":{},"warnings":[]}\n',
        encoding="utf-8",
    )

    report = rebuild_presentation_from_transcript(transcript, tmp_path / "rebuilt", normalization="viewer")
    result = tmp_path / "rebuilt"

    assert "あ、あ、あ、あ、あ、" in (result / "source_faithful_ja.srt").read_text(encoding="utf-8")
    assert "あ…" in (result / "viewer_ja.srt").read_text(encoding="utf-8")
    assert (result / "transcript_ja.jsonl").read_text(encoding="utf-8") == transcript.read_text(encoding="utf-8")
    assert (result / "review_manifest.json").is_file()
    assert (result / "review_report.html").is_file()
    assert report["viewer_runaway_compactions"] == 1


def test_rebuild_with_reference_preserves_source_diagnostics_and_review_rows(tmp_path):
    transcript = tmp_path / "transcript_ja.jsonl"
    transcript.write_text(
        '{"utterance_id":"utt_000001","source_segment_ids":[1],"start":0.0,"end":2.0,'
        '"text_raw":"canonical  source text","text_normalized":"canonical  source text","words":[],'
        '"speaker":"speaker_unknown","asr_metrics":{},"warnings":[]}\n',
        encoding="utf-8",
    )
    source = tmp_path / "source_faithful_ja.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:02,000\ncanonical  source text\n", encoding="utf-8")
    reference = tmp_path / "reference.srt"
    reference.write_text("1\n00:00:00,000 --> 00:00:02,000\ncanonical  source text\n", encoding="utf-8")

    report = rebuild_presentation_from_transcript(
        transcript, tmp_path / "rebuilt", normalization="viewer", reference_srt=reference, review_samples=4,
    )
    manifest = (tmp_path / "rebuilt" / "review_manifest.json").read_text(encoding="utf-8")

    assert report["backend"] == "rebuild"
    assert report["reference_comparison"]["text_diagnostics"]["source_faithful_to_reference"]["exact_text_match"] is True
    assert "external_reference" in manifest
    assert '"target_windows": 4' in manifest
    assert (tmp_path / "rebuilt" / "source_faithful_ja.srt").read_bytes() == source.read_bytes()
