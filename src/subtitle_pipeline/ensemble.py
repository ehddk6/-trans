from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path

from .checks import annotate_asr_warnings
from .models import SourceSegment
from .qwen import QwenRuntime, transcribe_qwen
from .recognition import _prepare_whisper_audio, transcribe as transcribe_whisper
from .text import has_runaway_repetition, normalize_japanese


_WARNING_CODES = {
    "empty_asr_text", "high_compression_ratio", "low_asr_confidence",
    "possible_silence_hallucination", "possible_repetition", "possible_periodic_repetition",
    "possible_runaway_repetition", "review_required",
}
_WINDOW_SECONDS = 20.0
_MAX_QWEN_WINDOWS = 16


@dataclass(slots=True)
class EnsembleResult:
    segments: list[SourceSegment]
    qwen_segments: list[SourceSegment]
    whisper_segments: list[SourceSegment]
    duration: float
    verification_intervals: list[dict[str, object]]


def _merge_intervals(intervals: list[dict[str, object]]) -> list[dict[str, object]]:
    """Merge overlapping model work while retaining each audit reason."""
    merged: list[dict[str, object]] = []
    for interval in sorted(intervals, key=lambda item: (float(item["start"]), float(item["end"]))):
        start, end = float(interval["start"]), float(interval["end"])
        reasons = set(interval.get("reasons", []))
        if merged and start <= float(merged[-1]["end"]) + 0.15:
            merged[-1]["end"] = max(float(merged[-1]["end"]), end)
            merged[-1]["reasons"] = sorted(set(merged[-1]["reasons"]) | reasons)
        else:
            merged.append({"start": start, "end": end, "reasons": sorted(reasons)})
    return merged


def _window(start: float, end: float, duration: float, window_seconds: float = _WINDOW_SECONDS) -> dict[str, float]:
    width = min(window_seconds, duration)
    midpoint = (start + end) / 2
    window_start = min(max(0.0, midpoint - width / 2), max(0.0, duration - width))
    return {"start": window_start, "end": window_start + width}


def _coverage_gaps(segments: list[SourceSegment], duration: float, minimum_gap: float = 1.0) -> list[tuple[float, float]]:
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        start = min(duration, max(0.0, segment.start))
        if start - cursor >= minimum_gap:
            gaps.append((cursor, start))
        cursor = max(cursor, min(duration, segment.end))
    if duration - cursor >= minimum_gap:
        gaps.append((cursor, duration))
    return gaps


def _distributed(items: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    """Pick a bounded chronological sample without clustering in one scene."""
    if limit <= 0 or not items:
        return []
    ordered = sorted(items, key=lambda item: (float(item["start"]), float(item["end"])))
    if len(ordered) <= limit:
        return ordered
    indexes = {round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)} if limit > 1 else {len(ordered) // 2}
    return [item for index, item in enumerate(ordered) if index in indexes]


def _reference_misses(whisper_segments: list[SourceSegment], reference_rows: list[dict[str, object]] | None) -> list[dict[str, object]]:
    if not reference_rows:
        return []
    result: list[dict[str, object]] = []
    for row in reference_rows:
        start, end = float(row["start"]), float(row["end"])
        if not any(_overlaps(start, end, segment.start, segment.end) >= 0.20 for segment in whisper_segments):
            result.append({"start": start, "end": end, "reasons": ["reference_uncovered"]})
    return result


def build_qwen_verification_intervals(
    whisper_segments: list[SourceSegment],
    duration: float,
    *,
    reference_rows: list[dict[str, object]] | None = None,
    audit_samples: int = 4,
) -> list[dict[str, object]]:
    """Build a hard-bounded Qwen audit plan from Whisper evidence.

    The full video is transcribed once by Whisper.  Qwen receives only short,
    high-value windows: Whisper quality warnings, reference-only speech,
    unusually long Whisper gaps, and deterministic coverage samples.  This
    avoids a second feature-length ASR pass while retaining independent checks.
    """
    if duration <= 0:
        return []
    candidates: list[dict[str, object]] = []
    warning_windows = [
        {**_window(segment.start, segment.end, duration), "reasons": ["whisper_warning"]}
        for segment in whisper_segments
        if _WARNING_CODES & set(segment.warnings)
    ]
    candidates.extend(_distributed(warning_windows, 4))

    reference_windows = [
        {**_window(float(item["start"]), float(item["end"]), duration), "reasons": item["reasons"]}
        for item in _reference_misses(whisper_segments, reference_rows)
    ]
    candidates.extend(_distributed(reference_windows, 4))

    gap_windows = [
        {**_window(start, end, duration), "reasons": ["whisper_gap"]}
        for start, end in _coverage_gaps(whisper_segments, duration, minimum_gap=2.0)
    ]
    candidates.extend(_distributed(gap_windows, 4))

    for index in range(max(0, audit_samples)):
        midpoint = duration * ((index + 0.5) / audit_samples)
        candidates.append({**_window(midpoint, midpoint, duration), "reasons": ["distributed_audit"]})
    return _merge_intervals(candidates[:_MAX_QWEN_WINDOWS])


def _bigrams(text: str) -> set[str]:
    clean = normalize_japanese(text, "conservative")
    return {clean[index:index + 2] for index in range(max(0, len(clean) - 1))} or {clean}


def text_similarity(first: str, second: str) -> float:
    left, right = _bigrams(first), _bigrams(second)
    return len(left & right) / len(left | right) if left or right else 1.0


def _overlaps(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _qwen_recovery_rejection_reasons(segment: SourceSegment) -> list[str]:
    """Return fail-closed reasons that make automatic gap recovery unsafe.

    Rejected Qwen evidence remains available in the review artifacts.  This
    gate only decides whether independent text may be inserted into the
    source-faithful transcript automatically.
    """
    reasons: list[str] = []
    cleaned = normalize_japanese(segment.text, "conservative")
    if len(cleaned) < 2:
        reasons.append("insufficient_text")
    values = [(word.start, word.end) for word in segment.words]
    if not values or not all(math.isfinite(value) for pair in values for value in pair):
        reasons.append("missing_or_non_finite_alignment")
    elif not any(end - start > 1e-3 for start, end in values):
        reasons.append("no_positive_alignment_span")
    if not math.isfinite(segment.start) or not math.isfinite(segment.end):
        reasons.append("non_finite_segment_span")
    elif segment.end - segment.start > 7.0 + 1e-6:
        reasons.append("overlong_alignment_span")
    if has_runaway_repetition(cleaned) or "possible_runaway_repetition" in segment.warnings:
        reasons.append("runaway_repetition")
    return sorted(set(reasons))


def _qwen_is_recoverable(segment: SourceSegment) -> bool:
    return not _qwen_recovery_rejection_reasons(segment)


def reconcile_whisper_and_qwen(
    whisper_segments: list[SourceSegment], qwen_segments: list[SourceSegment], duration: float,
) -> list[SourceSegment]:
    """Keep Whisper primary, audit disagreements, and add only Qwen-proven gaps."""
    whisper = sorted(whisper_segments, key=lambda segment: (segment.start, segment.end))
    qwen = sorted(qwen_segments, key=lambda segment: (segment.start, segment.end))
    for segment in whisper:
        segment.metadata.update({"backend": "whisper", "decision": "whisper_primary"})

    for whisper_segment in whisper:
        competitors = [
            segment for segment in qwen
            if _overlaps(whisper_segment.start, whisper_segment.end, segment.start, segment.end) >= 0.35
            and _qwen_is_recoverable(segment)
        ]
        if competitors and max(text_similarity(whisper_segment.text, item.text) for item in competitors) < 0.15:
            whisper_segment.warnings = sorted(set(whisper_segment.warnings) | {"engine_disagreement", "review_required"})
            whisper_segment.metadata["qwen_alternatives"] = [item.text for item in competitors[:3]]

    recovered: list[SourceSegment] = []
    for gap_start, gap_end in _coverage_gaps(whisper, duration, minimum_gap=1.0):
        for candidate in qwen:
            if candidate.start < gap_start - 0.08 or candidate.end > gap_end + 0.08:
                continue
            candidate.metadata["recovery_candidate"] = True
            rejection_reasons = _qwen_recovery_rejection_reasons(candidate)
            if rejection_reasons:
                candidate.metadata.update({
                    "recovery_decision": "rejected",
                    "recovery_rejection_reasons": rejection_reasons,
                })
                continue
            neighbours = "".join(
                segment.text for segment in whisper
                if abs(segment.end - gap_start) < 2.0 or abs(segment.start - gap_end) < 2.0
            )
            if text_similarity(candidate.text, neighbours) >= 0.45:
                candidate.metadata.update({
                    "recovery_decision": "rejected",
                    "recovery_rejection_reasons": ["nearby_whisper_duplicate"],
                })
                continue
            candidate.warnings = sorted(set(candidate.warnings) | {"qwen_recovery", "review_required"})
            candidate.metadata.update({
                "backend": "qwen", "decision": "qwen_recovery", "recovery_decision": "accepted",
                "whisper_gap": [gap_start, gap_end],
            })
            recovered.append(candidate)

    combined = whisper + recovered
    combined.sort(key=lambda segment: (segment.start, segment.end, segment.metadata.get("backend") != "whisper"))
    for index, segment in enumerate(combined):
        segment.id = index
    return combined


def transcribe_ensemble(
    input_path: Path,
    runtime: QwenRuntime,
    whisper_model: str,
    language: str = "ja",
    *,
    reference_rows: list[dict[str, object]] | None = None,
    alignment_cache_path: Path | None = None,
    reuse_alignment_cache: bool = True,
    audit_samples: int = 4,
    cache_identity_path: Path | None = None,
    cache_identity_sha256: str | None = None,
) -> EnsembleResult:
    stable_cache_identity = cache_identity_path or input_path
    audio_path, temporary_audio = _prepare_whisper_audio(input_path)
    try:
        whisper_segments, duration = transcribe_whisper(
            audio_path, whisper_model, language=language, word_timestamps=True, vad=False,
            beam_size=5, temperature=0.0, condition_on_previous_text=False,
        )
        annotate_asr_warnings(whisper_segments)
        verification_intervals = build_qwen_verification_intervals(
            whisper_segments, duration, reference_rows=reference_rows, audit_samples=audit_samples,
        )
        qwen_segments: list[SourceSegment] = []
        if verification_intervals:
            qwen_segments, _ = transcribe_qwen(
                audio_path, runtime, language=language, targeted_intervals=verification_intervals,
                alignment_cache_path=alignment_cache_path, reuse_alignment_cache=reuse_alignment_cache,
                cache_identity_path=stable_cache_identity,
                cache_identity_sha256=cache_identity_sha256,
            )
        annotate_asr_warnings(qwen_segments)
        for segment in qwen_segments:
            segment.metadata.update({"backend": "qwen", "decision": "qwen_verification"})
        combined = reconcile_whisper_and_qwen(whisper_segments, qwen_segments, duration)
        return EnsembleResult(combined, qwen_segments, whisper_segments, duration, verification_intervals)
    finally:
        if temporary_audio is not None:
            temporary_audio.cleanup()
