from subtitle_pipeline.ensemble import build_qwen_verification_intervals, reconcile_whisper_and_qwen
from subtitle_pipeline.models import SourceSegment, Word
from subtitle_pipeline.qwen import qwen_words_to_segments


def whisper(identifier, start, end, text, **metrics):
    return SourceSegment(
        identifier, start, end, text, [Word(text, start, end, backend="whisper")],
        no_speech_prob=metrics.get("no_speech_prob", 0.1),
        avg_logprob=metrics.get("avg_logprob", -0.2),
        compression_ratio=metrics.get("compression_ratio", 1.1),
        warnings=metrics.get("warnings", []),
    )


def qwen(identifier, start, end, text):
    return SourceSegment(
        identifier, start, end, text, [Word(text, start, end, backend="qwen")],
        metadata={"backend": "qwen", "decision": "qwen_verification"},
    )


def test_qwen_plan_is_bounded_and_targets_warning_reference_and_audit_windows():
    whisper_segments = [whisper(0, 10, 12, "first", warnings=["low_asr_confidence"]), whisper(1, 30, 31, "second")]
    intervals = build_qwen_verification_intervals(
        whisper_segments, 100, reference_rows=[{"start": 55.0, "end": 56.0, "text": "reference only"}], audit_samples=2,
    )

    reasons = {reason for interval in intervals for reason in interval["reasons"]}
    assert {"whisper_warning", "reference_uncovered", "distributed_audit"} <= reasons
    assert all(0 <= interval["start"] < interval["end"] <= 100 for interval in intervals)
    assert len(intervals) <= 16


def test_qwen_plan_targets_non_adjacent_repetition_as_review_evidence():
    whisper_segments = [
        whisper(0, 20, 22, "same phrase", warnings=["possible_periodic_repetition"]),
        whisper(1, 50, 52, "same phrase"),
    ]

    intervals = build_qwen_verification_intervals(whisper_segments, 100, audit_samples=0)

    assert any("whisper_warning" in interval["reasons"] for interval in intervals)


def test_qwen_recovers_only_a_whisper_gap_without_replacing_primary_text():
    whisper_segments = [whisper(0, 0, 1, "left"), whisper(1, 3, 4, "right")]
    qwen_segments = [qwen(0, 1.3, 2.4, "recovered")]

    result = reconcile_whisper_and_qwen(whisper_segments, qwen_segments, duration=5.0)

    assert [segment.text for segment in result] == ["left", "recovered", "right"]
    assert qwen_segments[0].metadata["decision"] == "qwen_recovery"
    assert qwen_segments[0].metadata["recovery_decision"] == "accepted"
    assert whisper_segments[0].metadata["decision"] == "whisper_primary"


def test_qwen_zero_duration_alignment_is_review_evidence_not_automatic_recovery():
    whisper_segments = [whisper(0, 0, 1, "left"), whisper(1, 3, 4, "right")]
    raw_word = Word("missing", 1.5, 1.5, backend="qwen")
    qwen_segments = qwen_words_to_segments([raw_word], decision="qwen_verification")

    result = reconcile_whisper_and_qwen(whisper_segments, qwen_segments, duration=5.0)

    assert [segment.text for segment in result] == ["left", "right"]
    assert (raw_word.start, raw_word.end) == (1.5, 1.5)
    assert qwen_segments[0].metadata["recovery_decision"] == "rejected"
    assert qwen_segments[0].metadata["recovery_rejection_reasons"] == ["no_positive_alignment_span"]


def test_qwen_overlong_or_runaway_alignment_is_not_automatic_recovery():
    whisper_segments = [whisper(0, 0, 1, "left"), whisper(1, 12, 13, "right")]
    overlong = qwen(0, 2.0, 10.0, "plausible words")
    runaway = qwen(1, 10.2, 11.0, "ああああああ")

    result = reconcile_whisper_and_qwen(whisper_segments, [overlong, runaway], duration=14.0)

    assert [segment.text for segment in result] == ["left", "right"]
    assert "overlong_alignment_span" in overlong.metadata["recovery_rejection_reasons"]
    assert "runaway_repetition" in runaway.metadata["recovery_rejection_reasons"]


def test_engine_disagreement_is_logged_without_auto_replacing_whisper():
    whisper_segments = [whisper(0, 0, 2, "あいうえお")]
    qwen_segments = [qwen(0, 0.2, 1.8, "かきくけこ")]

    result = reconcile_whisper_and_qwen(whisper_segments, qwen_segments, duration=2.0)

    assert result == whisper_segments
    assert "engine_disagreement" in whisper_segments[0].warnings
    assert whisper_segments[0].metadata["qwen_alternatives"] == ["かきくけこ"]


def test_targeted_qwen_groups_never_cross_source_audit_windows():
    words = [
        Word("first", 1.0, 1.5, backend="qwen", source_interval=0),
        Word("second", 40.0, 40.5, backend="qwen", source_interval=1),
    ]

    segments = qwen_words_to_segments(words, decision="qwen_verification")

    assert [segment.text for segment in segments] == ["first", "second"]
    assert all(segment.metadata["decision"] == "qwen_verification" for segment in segments)


def test_ensemble_forwards_original_media_as_qwen_cache_identity(tmp_path, monkeypatch):
    from subtitle_pipeline import ensemble

    source = tmp_path / "movie.mp4"
    normalized = tmp_path / "temporary-audio.wav"
    source.write_bytes(b"source")
    normalized.write_bytes(b"normalized")
    monkeypatch.setattr(ensemble, "_prepare_whisper_audio", lambda _path: (normalized, None))
    monkeypatch.setattr(
        ensemble,
        "transcribe_whisper",
        lambda *_args, **_kwargs: ([whisper(0, 0.0, 1.0, "primary")], 5.0),
    )
    captured = {}

    def fake_qwen(input_path, _runtime, **kwargs):
        captured["input_path"] = input_path
        captured["cache_identity_path"] = kwargs.get("cache_identity_path")
        return [], 5.0

    monkeypatch.setattr(ensemble, "transcribe_qwen", fake_qwen)

    ensemble.transcribe_ensemble(source, object(), "model", audit_samples=1)

    assert captured["input_path"] == normalized
    assert captured["cache_identity_path"] == source
