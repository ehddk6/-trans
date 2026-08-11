from subtitle_pipeline.artifacts import qc_report
from subtitle_pipeline.models import Cue, SourceSegment, Word


SOURCE_WARNINGS = [
    "possible_repetition",
    "possible_runaway_repetition",
    "possible_silence_hallucination",
    "low_asr_confidence",
]


def test_qc_counts_propagated_source_warnings_once_per_segment():
    segment = SourceSegment(
        7,
        0.0,
        4.0,
        "source",
        warnings=SOURCE_WARNINGS + ["possible_repetition"],
    )
    cues = [
        Cue(1, 0.0, 2.0, "first", "first", [7], warnings=list(SOURCE_WARNINGS)),
        Cue(2, 2.0, 4.0, "second", "second", [7], warnings=list(SOURCE_WARNINGS)),
    ]

    report = qc_report(cues, [segment], 4.0)

    assert report["repetition_suspicions"] == 1
    assert report["runaway_repetition_suspicions"] == 1
    assert report["hallucination_suspicions"] == 1
    assert report["low_confidence_asr_segments"] == 1
    reasons = {item["code"]: item["count"] for item in report["content_quality_gate"]["reasons"]}
    assert reasons["possible_repetition"] == 1
    assert reasons["runaway_repetition_source_evidence"] == 1
    assert reasons["possible_silence_hallucination"] == 1


def test_qc_keeps_viewer_only_exceptions_per_final_cue():
    segment = SourceSegment(1, 0.0, 2.0, "source", warnings=["possible_repetition"])
    viewer_warnings = [
        "possible_repetition",
        "line_limit_unavoidable",
        "duration_limit_unavoidable",
        "short_duration_review_required",
    ]
    cues = [
        Cue(1, 0.0, 1.0, "first", "first", [1], warnings=list(viewer_warnings)),
        Cue(2, 1.0, 2.0, "second", "second", [1], warnings=list(viewer_warnings)),
    ]

    report = qc_report(cues, [segment], 2.0)

    assert report["repetition_suspicions"] == 1
    gate = report["content_quality_gate"]
    assert gate["line_limit_exceptions"] == 2
    assert gate["duration_limit_exceptions"] == 2
    assert gate["short_duration_exceptions"] == 2


def test_qc_reports_viewer_content_preservation_separately():
    source = [Cue(1, 0.0, 2.0, "ああああ", "ああああ", [1])]
    viewer = [Cue(1, 0.0, 2.0, "あ\nあああ", "ああああ", [1])]

    report = qc_report(viewer, [], 2.0, source_cues=source)

    assert report["content_preservation"]["exact_after_whitespace_normalization"] is True


def test_qc_requires_review_when_word_alignment_text_does_not_match_source():
    segment = SourceSegment(9, 0.0, 2.0, "authoritative")
    cues = [Cue(
        1, 0.0, 2.0, "authoritative", "authoritative", [9],
        warnings=["word_text_mismatch", "timestamp_precision_limited", "review_required"],
    )]

    report = qc_report(cues, [segment], 2.0)

    assert report["content_quality_gate"]["status"] == "review_required"
    assert report["content_quality_gate"]["word_text_mismatch_segments"] == 1
    assert {item["code"] for item in report["content_quality_gate"]["reasons"]} == {
        "word_text_mismatch",
    }


def test_qc_reports_raw_word_alignment_health_without_rewriting_words():
    words = [
        Word("zero", 0.0, 0.0),
        Word("long", 0.0, 8.0),
        Word("outside", 8.1, 8.5),
        Word("reversed", 7.5, 7.8),
    ]
    segment = SourceSegment(3, 0.0, 8.0, "source", words=words)
    original = [(word.start, word.end) for word in words]

    report = qc_report([Cue(1, 0.0, 2.0, "source", "source", [3])], [segment], 8.0)

    health = report["word_alignment_health"]
    assert health == {
        "total_words": 4,
        "non_finite_words": 0,
        "non_positive_duration_words": 1,
        "over_7_second_words": 1,
        "out_of_segment_words": 1,
        "non_monotonic_words": 1,
        "review_required_segments": 1,
    }
    assert report["content_quality_gate"]["word_alignment_review_segments"] == 1
    assert ("word_alignment_review_required", 1) in {
        (item["code"], item["count"]) for item in report["content_quality_gate"]["reasons"]
    }
    assert [(word.start, word.end) for word in words] == original


def test_qc_requires_review_for_only_non_positive_word_timing():
    word = Word("はい", 1.0, 1.0)
    segment = SourceSegment(4, 0.0, 2.0, "はい", words=[word])

    report = qc_report([Cue(1, 0.0, 2.0, "はい", "はい", [4])], [segment], 2.0)

    assert report["word_alignment_health"]["non_positive_duration_words"] == 1
    assert report["word_alignment_health"]["review_required_segments"] == 1
    assert report["content_quality_gate"]["status"] == "review_required"
    assert ("word_alignment_review_required", 1) in {
        (item["code"], item["count"]) for item in report["content_quality_gate"]["reasons"]
    }
