import json

from subtitle_pipeline.models import Cue, SourceSegment
from subtitle_pipeline.review import (
    build_review_manifest,
    render_review_report,
    write_review_artifacts,
)


def cue(identifier, start, end, text, **kwargs):
    return Cue(identifier, start, end, text, text, [identifier], **kwargs)


def test_review_manifest_is_deterministic_and_stratified():
    final = [
        cue(1, 5, 13.5, "長い字幕です"),
        cue(2, 100, 102, "最終字幕"),
        cue(3, 300, 302, "静かな場面", warnings=["long_silence"]),
    ]
    source = [
        SourceSegment(1, 0, 2, "最初の台詞"),
        SourceSegment(2, 10, 12, "次の台詞"),
        SourceSegment(3, 300, 302, "静かな場面"),
    ]
    whisper = [{"event_id": "recovery", "start": 200, "end": 201, "text": "補完候補", "decision": "whisper_recovery"}]
    disagreements = [{"event_id": "conflict", "start": 250, "end": 251, "qwen_text": "基準文", "whisper_text": "別の文", "decision": "review_required"}]

    first = build_review_manifest(final, source, whisper, disagreements, video_duration=600)
    second = build_review_manifest(final, source, whisper, disagreements, video_duration=600)

    assert first == second
    assert len(first["windows"]) == 30
    categories = {window["category"] for window in first["windows"]}
    assert {"engine_disagreement", "whisper_recovery", "long_cue", "silence_boundary"} <= categories
    assert any(row["text"] == "長い字幕です" for window in first["windows"] for row in window["rows"]["final"])


def test_review_artifacts_are_json_and_self_contained_html(tmp_path):
    manifest = write_review_artifacts(
        tmp_path / "review_manifest.json",
        tmp_path / "review_report.html",
        [cue(1, 0, 2, "最終結果")],
        [SourceSegment(1, 0, 2, "Qwen原文")],
        [{"start": 3, "end": 4, "text": "Whisper候補"}],
        external_reference=[{"start": 0, "end": 2, "text": "参照字幕"}],
        video_duration=80,
    )

    on_disk = json.loads((tmp_path / "review_manifest.json").read_text(encoding="utf-8"))
    html = (tmp_path / "review_report.html").read_text(encoding="utf-8")
    assert on_disk == manifest
    assert "Qwen/원문 근거" in html and "Whisper 근거/주 인식" in html
    assert "Qwen原文" in html and "最終結果" in html and "参照字幕" in html
    assert render_review_report(manifest).startswith("<!doctype html>")


def test_review_manifest_selects_runaway_repetition_source_evidence():
    source = [SourceSegment(1, 20, 22, "ああああああああ", warnings=["possible_runaway_repetition"])]

    manifest = build_review_manifest([], source, video_duration=60, target_windows=4)

    assert any(window["category"] == "runaway_repetition" for window in manifest["windows"])


def test_presentation_limit_exceptions_are_listed_and_prioritised_for_review():
    final = [
        cue(1, 5, 13, "長い字幕です", warnings=["duration_limit_unavoidable"]),
        cue(2, 20, 22, "長い行です", warnings=["line_limit_unavoidable"]),
        cue(3, 30, 30.2, "短い字幕です", warnings=["short_duration_review_required"]),
    ]

    manifest = build_review_manifest(final, [], video_duration=60, target_windows=4)

    assert [item["codes"] for item in manifest["unresolved_presentation_exceptions"]] == [
        ["duration_limit_unavoidable"], ["line_limit_unavoidable"], ["short_duration_review_required"],
    ]
    assert any(window["category"] == "presentation_limit_exception" for window in manifest["windows"])
    assert "해결되지 않은 표시 예외" in render_review_report(manifest)


def test_review_report_links_the_source_video_at_each_window(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"not-a-real-video")

    manifest = build_review_manifest(
        [cue(1, 5, 7, "final")], [], video_duration=20, target_windows=1, video_path=video,
    )
    html = render_review_report(manifest)

    assert manifest["video_path"] == str(video.resolve())
    assert "이 검수 구간 영상 재생" in html
    assert "source.mp4" in html


def test_review_report_uses_the_backend_specific_source_label():
    manifest = build_review_manifest(
        [cue(1, 0, 2, "final")], [], video_duration=20, target_windows=1,
        source_evidence_label="참조 기준 원문",
    )

    assert manifest["source_evidence_label"] == "참조 기준 원문"
    assert "참조 기준 원문" in render_review_report(manifest)


def test_review_report_includes_the_quality_gate_and_reasons():
    manifest = build_review_manifest(
        [cue(1, 0, 2, "final")], [], video_duration=20, target_windows=1,
        quality_gate={"status": "review_required", "reasons": [{"code": "short_duration_review_required", "count": 2}]},
    )
    html = render_review_report(manifest)

    assert manifest["quality_gate"]["status"] == "review_required"
    assert "품질 게이트: 수동 검수 필요" in html
    assert "0.8초 미만 수동 검수 (2)" in html
