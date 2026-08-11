import inspect
import json

from subtitle_pipeline.checks import annotate_asr_warnings
from subtitle_pipeline.artifacts import qc_report, reference_comparison
from subtitle_pipeline.models import Cue, SourceSegment, Utterance, Word
from subtitle_pipeline.optimizer import _collapse_runaway_vocalisation_cues, _extend_short_display_cues, source_faithful_cues, viewer_cues
from subtitle_pipeline.profiles import get_profile
from subtitle_pipeline.srt import format_timestamp, parse_timestamp, read_srt_rows, validate_cues
from subtitle_pipeline.text import compact_high_confidence_runaway_repetition, compact_runaway_repetition, display_width, is_bad_boundary, wrap_two_lines


def utterance(text, start, end, words=None, speaker="speaker_unknown"):
    return Utterance("utt_000001", [1], start, end, text, text, words or [], speaker)


def test_srt_round_trip_and_ordering():
    assert parse_timestamp(format_timestamp(3723.456)) == 3723.456
    assert parse_timestamp("01:23,360") == 83.36
    assert parse_timestamp("0:06:35,830") == 395.83
    assert parse_timestamp("02:15:1,160") == 8101.16
    cues = source_faithful_cues([utterance("こんにちは", 0, 1), utterance("はい", 1, 2)])
    assert validate_cues(cues) == []


def test_overlap_and_empty_validation():
    cues = [
        Cue(1, 0, 2, "あ", "あ", [1]),
        Cue(2, 1, 3, "", "", [2]),
    ]
    assert any(error.startswith("overlap") for error in validate_cues(cues))
    assert any(error.startswith("empty_text") for error in validate_cues(cues))


def test_tiny_cues_remain_positive_after_srt_millisecond_rounding():
    from subtitle_pipeline.optimizer import _remove_overlaps

    cue = Cue(1, 1.2345, 1.2345, "text", "text", [1])
    result = _remove_overlaps([cue], get_profile("viewer_ja"))

    assert parse_timestamp(format_timestamp(result[0].end)) > parse_timestamp(format_timestamp(result[0].start))


def test_japanese_display_width_and_two_line_wrap():
    assert display_width("日本語ABC") == 4.5
    lines = wrap_two_lines("これは、とても長い日本語の字幕テストです。", 10, 12)
    assert len(lines) == 2 and all(line for line in lines)


def test_two_line_wrap_prefers_a_split_that_respects_both_line_limits():
    lines = wrap_two_lines("あいうえおかきくけこさしすせそたちつてと", 10, 10)
    assert len(lines) == 2
    assert all(display_width(line) <= 10 for line in lines)


def test_protected_japanese_boundaries():
    assert is_bad_boundary("私", "は")
    assert is_bad_boundary("行っ", "ます")
    assert is_bad_boundary("3", "人")
    assert not is_bad_boundary("行きます。", "次です")


def test_long_word_timed_segment_is_resplit_without_content_loss():
    tokens = ["これは", "とても", "長い", "日本語", "の", "文章です。", "次の", "文です。"]
    words = [Word(token, i * 1.5, i * 1.5 + 1.1) for i, token in enumerate(tokens)]
    original = "".join(tokens)
    cues = viewer_cues([utterance(original, 0, 11.6, words)], get_profile("viewer_ja"), "conservative")
    assert len(cues) >= 2
    assert "".join(c.text.replace("\n", "") for c in cues) == original
    assert validate_cues(cues) == []
    assert all(cue.duration <= get_profile("viewer_ja").max_duration for cue in cues)


def test_viewer_restores_authoritative_punctuation_omitted_from_word_text():
    source = utterance("こんにちは。", 0, 2, [Word("こんにちは", 0, 2)])

    cues = viewer_cues([source], get_profile("viewer_ja"), "strict")

    assert "".join(cue.text.replace("\n", "") for cue in cues) == "こんにちは。"
    assert cues[0].start == 0 and cues[0].end == 2
    assert "word_text_reconciled" in cues[0].warnings
    assert "timestamp_precision_limited" not in cues[0].warnings


def test_irreconcilable_word_text_falls_back_to_one_authoritative_atom_for_review():
    original = "こんにちは。今日は元気です。"
    source = utterance(original, 0, 4, [
        Word("こんばんは。", 0, 2),
        Word("今日は元気です。", 2, 4),
    ])

    cues = viewer_cues([source], get_profile("viewer_ja"), "strict")

    assert len(cues) == 1
    assert cues[0].text == original
    assert cues[0].start == 0 and cues[0].end == 4
    assert {"timestamp_precision_limited", "word_text_mismatch", "review_required"} <= set(cues[0].warnings)


def test_wordless_segment_is_not_uniformly_time_split():
    source = utterance("句読点のない長い日本語文です", 0, 12)
    cues = viewer_cues([source], get_profile("viewer_ja"), "conservative")
    assert len(cues) == 1
    assert "timestamp_precision_limited" in cues[0].warnings
    assert "duration_limit_unavoidable" in cues[0].warnings


def test_silence_and_speaker_changes_are_preferred_boundaries():
    words = [Word("こんにちは。", 0, 1), Word("どうしたの？", 2.5, 3.5)]
    first = utterance("こんにちは。どうしたの？", 0, 3.5, words)
    cues = viewer_cues([first], get_profile("anime_ja"), "conservative")
    assert len(cues) >= 2
    assert "long_silence" in cues[0].split_reasons


def test_tiny_alignment_gap_does_not_force_an_unreadable_split():
    words = [
        Word("まとまった前半", 0.0, 2.0),
        Word("まとまった後半", 2.01, 4.01),
    ]
    original = "".join(word.text for word in words)

    cues = viewer_cues(
        [utterance(original, 0.0, 4.01, words)], get_profile("viewer_ja"), "strict",
    )

    assert len(cues) == 1
    assert cues[0].text.replace("\n", "") == original


def test_compactness_profile_changes_cue_tradeoff_without_changing_text():
    words = [
        Word("最初の文です。", 0.0, 2.0),
        Word("次の文です。", 2.05, 4.05),
    ]
    original = "".join(word.text for word in words)

    anime = viewer_cues([utterance(original, 0.0, 4.05, words)], get_profile("anime_ja"), "strict")
    compact = viewer_cues([utterance(original, 0.0, 4.05, words)], get_profile("compact_ja"), "strict")

    assert len(anime) == 2
    assert len(compact) == 1
    assert "".join(cue.text.replace("\n", "") for cue in anime) == original
    assert "".join(cue.text.replace("\n", "") for cue in compact) == original


def test_speaker_change_is_not_merged_into_one_cue():
    first = utterance("はい。", 0, 1, [Word("はい。", 0, 1)], speaker="speaker_a")
    second = utterance("いいえ。", 1.05, 2.05, [Word("いいえ。", 1.05, 2.05)], speaker="speaker_b")
    cues = viewer_cues([first, second], get_profile("viewer_ja"), "strict")
    assert len(cues) == 2
    assert cues[0].speaker == "speaker_a" and cues[1].speaker == "speaker_b"
    assert "speaker_change" in cues[0].split_reasons


def test_scene_change_is_a_soft_audited_signal():
    cues = viewer_cues([utterance("短い発話です。", 0, 2, [Word("短い発話です。", 0, 2)])], get_profile("viewer_ja"), "strict", [0.5])
    assert cues[0].scene_distance == 0.5


def test_zero_duration_word_timing_is_repaired_without_invalid_cue():
    cues = viewer_cues([
        utterance("はい。いいえ。", 1, 1, [Word("はい。", 1, 1), Word("いいえ。", 1, 1)])
    ], get_profile("anime_ja"), "strict")
    assert validate_cues(cues) == []
    assert all(cue.end > cue.start for cue in cues)


def test_short_viewer_cue_extends_only_into_measured_silence():
    cues = [
        Cue(1, 0.0, 0.4, "first", "first", [1]),
        Cue(2, 1.0, 2.0, "second", "second", [2]),
    ]

    result = _extend_short_display_cues(cues, get_profile("viewer_ja"))

    assert result[0].start == 0.0 and result[0].end == 0.8
    assert "viewer_duration_extended_into_silence" in result[0].warnings
    assert "short_duration_review_required" not in result[0].warnings
    assert result[0].end <= result[1].start


def test_short_viewer_cue_without_silence_remains_for_review():
    cues = [
        Cue(1, 0.0, 0.4, "first", "first", [1]),
        Cue(2, 0.4, 1.0, "second", "second", [2]),
    ]

    result = _extend_short_display_cues(cues, get_profile("viewer_ja"))

    assert result[0].end == 0.4
    assert "short_duration_review_required" in result[0].warnings


def test_repetition_empty_and_hallucination_warnings():
    segments = [
        SourceSegment(0, 0, 1, "同じ文です。", no_speech_prob=.1),
        SourceSegment(1, 1, 2, "同じ文です。", no_speech_prob=.9, compression_ratio=3, avg_logprob=-1.5),
        SourceSegment(2, 2, 3, ""),
    ]
    annotate_asr_warnings(segments)
    assert "possible_repetition" in segments[1].warnings
    assert "possible_silence_hallucination" in segments[1].warnings
    assert "empty_asr_text" in segments[2].warnings


def test_non_adjacent_repetition_is_review_evidence_not_deletion_authority():
    segments = [
        SourceSegment(0, 0, 1, "今日は晴れです"),
        SourceSegment(1, 20, 21, "今日は晴れです"),
    ]

    annotate_asr_warnings(segments)

    assert "possible_periodic_repetition" in segments[1].warnings
    assert segments[1].text == "今日は晴れです"


def test_runaway_repetition_is_flagged_for_review_without_changing_text():
    segment = SourceSegment(0, 0, 2, "ababababab")
    annotate_asr_warnings([segment])
    assert "possible_runaway_repetition" in segment.warnings
    assert segment.text == "ababababab"


def test_viewer_compacts_only_runaway_repetition():
    assert compact_runaway_repetition("はぁはぁはぁはぁはぁ") == "はぁ…"
    assert compact_runaway_repetition("あ、あ、あ、あ、あ、") == "あ…"
    assert compact_runaway_repetition("ん?ん?ん?ん?ん?") == "ん…"
    assert compact_runaway_repetition("これは通常の台詞です") == "これは通常の台詞です"
    assert compact_runaway_repetition("はいはいはいはいはい") == "はいはいはいはいはい"


def test_high_confidence_compactor_handles_any_repeated_asr_unit():
    assert compact_high_confidence_runaway_repetition("今、今、今、今、今、") == "今…"
    assert compact_high_confidence_runaway_repetition("腹が腹が腹が腹が腹が") == "腹が…"
    assert compact_high_confidence_runaway_repetition("チョッ チョッ チョッ チョッ チョッ") == "チョッ…"


def test_viewer_uses_general_repeat_compaction_only_with_high_compression_evidence():
    repeated = "今今今今今"
    warned = Utterance("warned", [1], 0, 2, repeated, repeated, [], warnings=["high_compression_ratio"])
    ordinary = Utterance("ordinary", [2], 2, 4, repeated, repeated, [], warnings=[])

    cues = viewer_cues([warned, ordinary], get_profile("viewer_ja"), "viewer")

    assert cues[0].normalized_text == "今…"
    assert "viewer_high_confidence_runaway_compacted" in cues[0].warnings
    assert cues[1].normalized_text == repeated


def test_consecutive_compacted_vocalisation_cues_collapse_for_display_only():
    cues = [
        Cue(1, 0, 3, "〜…", "〜…", [7], warnings=["possible_runaway_repetition", "viewer_runaway_repetition_compacted"]),
        Cue(2, 3, 6, "〜…", "〜…", [7], warnings=["possible_runaway_repetition", "viewer_runaway_repetition_compacted"]),
        Cue(3, 6, 9, "〜…", "〜…", [7], warnings=["possible_runaway_repetition", "viewer_runaway_repetition_compacted"]),
    ]

    result = _collapse_runaway_vocalisation_cues(cues, get_profile("viewer_ja"))

    assert len(result) == 1
    assert result[0].start == 0 and result[0].end == 7
    assert "viewer_runaway_repetition_collapsed" in result[0].warnings


def test_short_or_substring_phrases_are_not_false_repetition_warnings():
    segments = [SourceSegment(0, 0, 1, "はい"), SourceSegment(1, 1, 2, "はいはい")]
    annotate_asr_warnings(segments)
    assert "possible_repetition" not in segments[1].warnings


def test_no_translation_is_introduced():
    original = "これは日本語だけの発話です。"
    cues = viewer_cues([utterance(original, 0, 2, [Word(original, 0, 2)])], get_profile("viewer_ja"), "strict")
    assert cues[0].text == original


def test_layout_normalization_preserves_repeated_source_text():
    original = "ああああああ"
    cues = viewer_cues(
        [utterance(original, 0, 2, [Word(original, 0, 2)])], get_profile("viewer_ja"), "layout",
    )

    assert "".join(cue.text.replace("\n", "") for cue in cues) == original
    assert cues[0].normalized_text == original


def test_numpy_like_asr_metrics_can_be_serialized_in_reports():
    class NumpyLike:
        def item(self):
            return 3

    from subtitle_pipeline.pipeline import _json_default
    assert json.dumps({"metric": NumpyLike()}, default=_json_default) == '{"metric": 3}'


def test_vad_is_opt_in_to_preserve_low_volume_dialogue():
    from subtitle_pipeline.recognition import transcribe
    assert inspect.signature(transcribe).parameters["vad"].default is False


def test_container_audio_is_normalized_before_whisper(monkeypatch, tmp_path):
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import subtitle_pipeline.recognition as recognition

    ffmpeg_calls = []
    whisper_inputs = []

    def fake_run(args, check):
        ffmpeg_calls.append(args)
        Path(args[-1]).write_bytes(b"wav")

    class FakeWhisperModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def transcribe(self, audio, **_kwargs):
            whisper_inputs.append(audio)
            segment = SimpleNamespace(
                id=0, start=0.0, end=1.0, text="こんにちは", words=[],
                no_speech_prob=0.0, avg_logprob=0.0, compression_ratio=1.0,
            )
            return [segment], SimpleNamespace(duration=12.0)

    monkeypatch.setattr(recognition.subprocess, "run", fake_run)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"container")

    segments, duration = recognition.transcribe(video, "fake")

    assert duration == 12.0
    assert segments[0].text == "こんにちは"
    assert ffmpeg_calls[0][0:4] == ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel"]
    assert Path(whisper_inputs[0]).suffix == ".wav"


def test_rebuild_requires_no_asr_input_argument():
    from subtitle_pipeline.pipeline import rebuild_presentation_from_transcript
    assert callable(rebuild_presentation_from_transcript)


def test_read_srt_rows_preserves_user_text(tmp_path):
    path = tmp_path / "reference.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:02,000\n原文そのまま\n", encoding="utf-8")
    assert read_srt_rows(path) == [{"id": "1", "start": 1.0, "end": 2.0, "text": "原文そのまま"}]


def test_reference_comparison_is_diagnostic_not_a_target():
    cues = [Cue(1, 0, 2, "最終字幕", "最終字幕", [1])]
    source = [Cue(1, 0, 2, "参照", "参照", [1])]
    result = reference_comparison(cues, [{"id": 1, "start": 0.0, "end": 1.0, "text": "参照"}], source)
    assert result["final"]["cue_count"] == 1
    assert result["external_reference"]["character_count"] == 2
    assert result["text_diagnostics"]["source_faithful_to_reference"]["exact_text_match"] is True
    assert result["text_diagnostics"]["final_to_reference"]["character_count_ratio"] == 2.0


def test_content_quality_gate_requires_review_for_content_exceptions():
    cues = [
        Cue(1, 0, 2, "ababababab", "ababababab", [1], warnings=["possible_runaway_repetition"]),
        Cue(2, 2, 4, "長い表示", "長い表示", [2], warnings=["line_limit_unavoidable"]),
        Cue(3, 4, 4.2, "短い表示", "短い表示", [3], warnings=["short_duration_review_required"]),
    ]

    report = qc_report(cues, [], 4)

    assert report["validation_errors"] == []
    assert report["content_quality_gate"]["status"] == "review_required"
    assert report["content_quality_gate"]["remaining_viewer_runaway_cues"] == 1
    assert report["content_quality_gate"]["line_limit_exceptions"] == 1
    assert report["content_quality_gate"]["short_duration_exceptions"] == 1


def test_content_quality_gate_passes_when_no_content_or_structural_exceptions():
    report = qc_report([Cue(1, 0, 2, "通常の台詞", "通常の台詞", [1])], [], 2)

    assert report["content_quality_gate"]["status"] == "passed"


def test_qc_duration_limits_ignore_floating_point_noise_at_exact_boundary():
    report = qc_report([Cue(1, 431.54, 432.34, "boundary", "boundary", [1])], [], 1)

    assert report["under_0_8_seconds"] == 0
