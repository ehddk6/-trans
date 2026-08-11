from __future__ import annotations

import html
import json
import math
import re
from collections import Counter
from pathlib import Path

from .models import Cue, SourceSegment, Utterance
from .srt import validate_cues
from .text import content_signature, display_width, has_runaway_repetition


_DISPLAY_CODE_LABELS = {
    "long_silence": "긴 무음",
    "word_alignment": "단어 정렬",
    "sentence_end": "문장 끝",
    "timestamp_precision_limited": "타임코드 정밀도 제한",
    "duration_limit_unavoidable": "7초 제한 적용 불가피",
    "line_limit_unavoidable": "줄 길이 제한 불가피",
    "viewer_duration_extended_into_silence": "무음 쪽으로 표시 시간 확장",
    "short_duration_review_required": "0.8초 미만 수동 검수",
    "word_text_reconciled": "단어 정렬 누락 문장부호 복원",
    "word_text_mismatch": "단어 정렬과 원문 불일치",
}

_SOURCE_WARNING_CODES = {
    "engine_disagreement",
    "high_compression_ratio",
    "low_asr_confidence",
    "possible_repetition",
    "possible_periodic_repetition",
    "possible_runaway_repetition",
    "possible_silence_hallucination",
}


def _display_codes(values) -> str:
    return ", ".join(
        _DISPLAY_CODE_LABELS.get(str(value), str(value))
        for value in sorted({str(value) for value in values})
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def cue_audit(cues: list[Cue]) -> list[dict]:
    rows = []
    for cue in cues:
        rows.append({
            "cue_id": cue.id, "start": cue.start, "end": cue.end, "duration": cue.duration,
            "text_raw": cue.text.replace("\n", ""), "text_normalized": cue.normalized_text,
            "line_lengths": [display_width(line) for line in cue.text.splitlines()],
            "split_reasons": cue.split_reasons, "merge_reasons": cue.merge_reasons,
            "source_segment_ids": cue.source_segment_ids, "speaker": cue.speaker,
            "asr_metrics": cue.metrics, "warnings": cue.warnings,
            "scene_distance": cue.scene_distance, "adjacent_silence": cue.adjacent_silence,
        })
    return rows


def _source_warning_counts(cues: list[Cue], segments: list[SourceSegment]) -> Counter[str]:
    """Count ASR evidence once per source segment, not once per viewer cue.

    Viewer construction propagates source warnings to every cue derived from a
    segment.  Counting both lists therefore multiplies one recognition event by
    the number of presentation cues.  Source segments are authoritative when
    available.  The cue fallback keeps direct ``qc_report`` callers useful while
    still deduplicating a warning split across cues from the same source.
    """
    counts: Counter[str] = Counter()
    if segments:
        events = {
            (segment.id, warning)
            for segment in segments
            for warning in set(segment.warnings)
            if warning in _SOURCE_WARNING_CODES
        }
    else:
        events = {
            (tuple(sorted(set(cue.source_segment_ids))) or ("cue", cue.id), warning)
            for cue in cues
            for warning in set(cue.warnings)
            if warning in _SOURCE_WARNING_CODES
        }
    counts.update(warning for _, warning in events)
    return counts


def _viewer_warning_source_count(cues: list[Cue], warning: str) -> int:
    """Count a viewer-detected source issue once across any derived cues."""
    return len({
        tuple(sorted(set(cue.source_segment_ids))) or ("cue", cue.id)
        for cue in cues
        if warning in cue.warnings
    })


def _word_alignment_health(segments: list[SourceSegment]) -> dict[str, int]:
    """Report raw alignment anomalies without rewriting authoritative timings."""
    totals = Counter[str]()
    affected_segments: set[int] = set()
    tolerance = 0.05
    for segment in segments:
        previous_start = -math.inf
        segment_affected = False
        for word in segment.words:
            totals["total_words"] += 1
            values = (word.start, word.end)
            if not all(math.isfinite(value) for value in values):
                totals["non_finite_words"] += 1
                segment_affected = True
                continue
            duration = word.end - word.start
            if duration <= 0:
                totals["non_positive_duration_words"] += 1
                segment_affected = True
            if duration > 7.0 + 1e-6:
                totals["over_7_second_words"] += 1
                segment_affected = True
            if word.start < segment.start - tolerance or word.end > segment.end + tolerance:
                totals["out_of_segment_words"] += 1
                segment_affected = True
            if word.start < previous_start - 1e-6:
                totals["non_monotonic_words"] += 1
                segment_affected = True
            previous_start = max(previous_start, word.start)
        if segment_affected:
            affected_segments.add(segment.id)
    return {
        "total_words": totals["total_words"],
        "non_finite_words": totals["non_finite_words"],
        "non_positive_duration_words": totals["non_positive_duration_words"],
        "over_7_second_words": totals["over_7_second_words"],
        "out_of_segment_words": totals["out_of_segment_words"],
        "non_monotonic_words": totals["non_monotonic_words"],
        "review_required_segments": len(affected_segments),
    }


def qc_report(
    cues: list[Cue], segments: list[SourceSegment], video_duration: float,
    source_cues: list[Cue] | None = None,
) -> dict:
    durations = [cue.duration for cue in cues]
    timing_epsilon = 1e-6
    line_counts = Counter(len(cue.text.splitlines()) for cue in cues)
    source_warning_counts = _source_warning_counts(cues, segments)
    alignment_health = _word_alignment_health(segments)
    validation_errors = validate_cues(cues)
    runaway_suspicions = source_warning_counts["possible_runaway_repetition"]
    hallucination_suspicions = source_warning_counts["possible_silence_hallucination"]
    repetition_suspicions = source_warning_counts["possible_repetition"]
    periodic_repetition_suspicions = source_warning_counts["possible_periodic_repetition"]
    high_compression_suspicions = source_warning_counts["high_compression_ratio"]
    engine_disagreements = source_warning_counts["engine_disagreement"]
    remaining_viewer_runaways = sum(has_runaway_repetition(cue.normalized_text) for cue in cues)
    line_limit_exceptions = sum("line_limit_unavoidable" in cue.warnings for cue in cues)
    duration_limit_exceptions = sum("duration_limit_unavoidable" in cue.warnings for cue in cues)
    short_duration_exceptions = sum("short_duration_review_required" in cue.warnings for cue in cues)
    word_text_mismatches = _viewer_warning_source_count(cues, "word_text_mismatch")
    source_text = content_signature("".join(cue.normalized_text for cue in source_cues or []))
    viewer_text = content_signature("".join(cue.normalized_text for cue in cues))
    content_preservation = {
        "checked": source_cues is not None,
        "exact_after_whitespace_normalization": source_cues is not None and source_text == viewer_text,
        "source_characters": len(source_text) if source_cues is not None else None,
        "viewer_characters": len(viewer_text) if source_cues is not None else None,
    }
    content_reasons = []
    for code, count in (
        ("remaining_viewer_runaway_repetition", remaining_viewer_runaways),
        ("runaway_repetition_source_evidence", runaway_suspicions),
        ("possible_silence_hallucination", hallucination_suspicions),
        ("possible_repetition", repetition_suspicions),
        ("possible_periodic_repetition", periodic_repetition_suspicions),
        ("high_compression_ratio", high_compression_suspicions),
        ("engine_disagreement", engine_disagreements),
        ("line_limit_unavoidable", line_limit_exceptions),
        ("duration_limit_unavoidable", duration_limit_exceptions),
        ("short_duration_review_required", short_duration_exceptions),
        ("word_text_mismatch", word_text_mismatches),
        ("word_alignment_review_required", alignment_health["review_required_segments"]),
        ("viewer_content_mismatch", int(source_cues is not None and source_text != viewer_text)),
    ):
        if count:
            content_reasons.append({"code": code, "count": count})
    structural_error_count = len(validation_errors)
    quality_status = "failed" if structural_error_count else ("review_required" if content_reasons else "passed")
    return {
        "total_cues": len(cues), "average_cues_per_minute": len(cues) / (video_duration / 60) if video_duration else 0,
        "average_duration": sum(durations) / len(durations) if durations else 0,
        "min_duration": min(durations, default=0), "max_duration": max(durations, default=0),
        "under_0_8_seconds": sum(value < 0.8 - timing_epsilon for value in durations), "over_7_seconds": sum(value > 7 + timing_epsilon for value in durations),
        "one_line_ratio": line_counts[1] / len(cues) if cues else 0, "two_line_ratio": line_counts[2] / len(cues) if cues else 0,
        "line_limit_exceeded": sum(any(display_width(line) > 26 for line in cue.text.splitlines()) for cue in cues),
        "timecode_overlap_count": sum(error.startswith("overlap:") for error in validation_errors),
        "timecode_reversal_count": sum(error.startswith("non_positive") for error in validation_errors),
        "empty_cue_count": sum(not cue.text.strip() for cue in cues),
        "repetition_suspicions": repetition_suspicions,
        "periodic_repetition_suspicions": periodic_repetition_suspicions,
        "runaway_repetition_suspicions": runaway_suspicions,
        "hallucination_suspicions": hallucination_suspicions,
        "low_confidence_asr_segments": source_warning_counts["low_asr_confidence"],
        "high_compression_asr_segments": high_compression_suspicions,
        "engine_disagreement_segments": engine_disagreements,
        "speaker_mixing_suspicions": 0,
        "word_alignment_health": alignment_health,
        "source_segment_to_final_cue_ratio": len(cues) / len(segments) if segments else 0,
        "content_preservation": content_preservation,
        "validation_errors": validation_errors,
        "content_quality_gate": {
            "status": quality_status,
            "structural_error_count": structural_error_count,
            "remaining_viewer_runaway_cues": remaining_viewer_runaways,
            "line_limit_exceptions": line_limit_exceptions,
            "duration_limit_exceptions": duration_limit_exceptions,
            "short_duration_exceptions": short_duration_exceptions,
            "word_text_mismatch_segments": word_text_mismatches,
            "word_alignment_review_segments": alignment_health["review_required_segments"],
            "reasons": content_reasons,
        },
    }


def reference_comparison(
    cues: list[Cue], reference_rows: list[dict[str, object]], source_cues: list[Cue] | None = None,
) -> dict:
    """Expose reference SRT differences as diagnostics, never as a cue-count target."""
    def field(item: object, name: str) -> object:
        return getattr(item, name) if hasattr(item, name) else item[name]  # type: ignore[index]

    def metrics(items: list[object]) -> dict[str, float | int]:
        if not items:
            return {"cue_count": 0, "character_count": 0, "shown_seconds": 0.0}
        starts = [float(field(item, "start")) for item in items]
        ends = [float(field(item, "end")) for item in items]
        texts = [str(field(item, "text")) for item in items]
        return {
            "cue_count": len(items), "character_count": sum(len(text.replace("\n", "")) for text in texts),
            "shown_seconds": round(sum(max(0.0, end - start) for start, end in zip(starts, ends)), 3),
        }

    def comparison_text(items: list[object]) -> str:
        return re.sub(r"\s+", "", "".join(str(field(item, "text")) for item in items))

    def bigrams(text: str) -> set[str]:
        return {text[index:index + 2] for index in range(len(text) - 1)} or ({text} if text else set())

    def text_diagnostics(candidate: list[object], reference: list[object]) -> dict[str, float | int | bool | None]:
        candidate_text, reference_text = comparison_text(candidate), comparison_text(reference)
        candidate_bigrams, reference_bigrams = bigrams(candidate_text), bigrams(reference_text)
        overlap = len(candidate_bigrams & reference_bigrams)
        union = len(candidate_bigrams | reference_bigrams)
        return {
            "exact_text_match": candidate_text == reference_text,
            "candidate_characters": len(candidate_text),
            "reference_characters": len(reference_text),
            "character_count_ratio": round(len(candidate_text) / len(reference_text), 4) if reference_text else None,
            "bigram_jaccard": round(overlap / union, 4) if union else 1.0,
            "reference_bigram_coverage": round(overlap / len(reference_bigrams), 4) if reference_bigrams else 1.0,
            "candidate_bigram_precision": round(overlap / len(candidate_bigrams), 4) if candidate_bigrams else 1.0,
        }

    source = source_cues or cues
    return {
        "final": metrics(cues),
        "source_faithful": metrics(source),
        "external_reference": metrics(reference_rows),
        "text_diagnostics": {
            "final_to_reference": text_diagnostics(cues, reference_rows),
            "source_faithful_to_reference": text_diagnostics(source, reference_rows),
        },
    }


def write_comparison_report(path: Path, segments: list[SourceSegment], source: list[Cue], viewer: list[Cue]) -> None:
    primary_label = "참조 기준 원문" if segments and all(
        segment.metadata.get("backend") == "reference" for segment in segments
    ) else "ASR 원문"
    rows = []
    for index, segment in enumerate(segments[:20]):
        matching_source = [c for c in source if segment.id in c.source_segment_ids]
        matching_viewer = [c for c in viewer if segment.id in c.source_segment_ids]
        rows.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in [
            str(segment.id), f"{segment.start:.2f}–{segment.end:.2f}", segment.text,
            "<br>".join(c.text for c in matching_source), "<br>".join(c.text for c in matching_viewer),
            _display_codes(reason for c in matching_viewer for reason in c.split_reasons),
            _display_codes(warning for c in matching_viewer for warning in c.warnings),
        ]) + "</tr>")
    path.write_text("""<!doctype html><meta charset='utf-8'><title>자막 비교</title>
<style>body{font-family:sans-serif}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:.4rem;vertical-align:top;white-space:pre-wrap}</style>
<h1>일본어 자막 표시 비교</h1><p>원문 앞부분 20개 대표 구간입니다.</p>
<table><tr><th>구간</th><th>시간</th><th>""" + html.escape(primary_label) + """</th><th>원문 보존</th><th>시청용</th><th>경계 처리</th><th>경고</th></tr>"""
        + "".join(rows) + "</table>", encoding="utf-8")
