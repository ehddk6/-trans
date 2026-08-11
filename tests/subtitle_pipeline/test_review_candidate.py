import json
from pathlib import Path

from subtitle_pipeline.models import SourceSegment, Word
from subtitle_pipeline.review_candidate import build_source_review_candidate
from subtitle_pipeline.srt import read_srt_rows


def _write_inputs(tmp_path: Path, *, repeated: bool = True) -> tuple[Path, Path, Path]:
    source = tmp_path / "source_faithful_ja.srt"
    source.write_text(
        "1\n00:00:10,000 --> 00:00:12,000\nold first\n\n"
        "2\n00:00:30,000 --> 00:00:32,000\nold second\n",
        encoding="utf-8",
    )
    text_second = "old second" if not repeated else "same phrase"
    transcript = tmp_path / "transcript_ja.jsonl"
    transcript.write_text(
        "\n".join([
            json.dumps({"utterance_id": "utt_1", "source_segment_ids": [1], "start": 10, "end": 12, "text_raw": "same phrase" if repeated else "old first", "text_normalized": "same phrase" if repeated else "old first", "warnings": []}),
            json.dumps({"utterance_id": "utt_2", "source_segment_ids": [2], "start": 30, "end": 32, "text_raw": text_second, "text_normalized": text_second, "warnings": []}),
        ]) + "\n",
        encoding="utf-8",
    )
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    return source, transcript, video


def test_candidate_changes_only_agreed_periodic_target_and_preserves_source(tmp_path):
    source, transcript, video = _write_inputs(tmp_path)
    source_before = source.read_bytes()
    calls = []

    def fake_qwen(_input, _runtime, *, targeted_intervals, **_kwargs):
        calls.append(targeted_intervals[0])
        return [
            SourceSegment(1, 30.0, 32.0, "new proposal", [Word("new proposal", 30.0, 32.0, source_interval=0)]),
            SourceSegment(2, 30.0, 32.0, "new proposal", [Word("new proposal", 30.0, 32.0, source_interval=1)]),
        ], 40.0

    result = build_source_review_candidate(
        transcript, video, tmp_path, object(), max_targets=4, duration=40.0, qwen_runner=fake_qwen,
    )

    candidate = read_srt_rows(tmp_path / "source_review_candidate_ja.srt")
    evidence = json.loads((tmp_path / "source_review_evidence.json").read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert result["proposal_count"] == 1
    assert [row["text"] for row in candidate] == ["old first", "new proposal"]
    assert [(row["start"], row["end"]) for row in candidate] == [(10.0, 12.0), (30.0, 32.0)]
    assert source.read_bytes() == source_before
    assert evidence["targets"][0]["decision"] == "qwen_proposal"


def test_candidate_keeps_whisper_text_when_qwen_fails(tmp_path):
    source, transcript, video = _write_inputs(tmp_path)

    def failing_qwen(*_args, **_kwargs):
        raise RuntimeError("pilot failure")

    result = build_source_review_candidate(
        transcript, video, tmp_path, object(), max_targets=4, duration=40.0, qwen_runner=failing_qwen,
    )

    candidate = read_srt_rows(tmp_path / "source_review_candidate_ja.srt")
    evidence = json.loads((tmp_path / "source_review_evidence.json").read_text(encoding="utf-8"))
    assert result["proposal_count"] == 0
    assert candidate[1]["text"] == "old second"
    assert evidence["targets"][0]["decision"] == "qwen_failed_source_preserved"


def test_candidate_does_not_call_qwen_without_periodic_anomaly(tmp_path):
    source, transcript, video = _write_inputs(tmp_path, repeated=False)
    calls = []

    def fake_qwen(*_args, **_kwargs):
        calls.append(True)
        return [], 40.0

    result = build_source_review_candidate(
        transcript, video, tmp_path, object(), max_targets=4, duration=40.0, qwen_runner=fake_qwen,
    )

    assert calls == []
    assert result["target_count"] == 0
    assert (tmp_path / "source_review_candidate_ja.srt").read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
