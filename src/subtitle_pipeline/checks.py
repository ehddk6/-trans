from __future__ import annotations

from .models import SourceSegment
from .text import has_runaway_repetition, normalize_japanese


def annotate_asr_warnings(segments: list[SourceSegment]) -> None:
    previous = ""
    last_seen: dict[str, float] = {}
    for segment in segments:
        cleaned = normalize_japanese(segment.text, "conservative")
        if not cleaned:
            segment.warnings.append("empty_asr_text")
        if segment.compression_ratio is not None and segment.compression_ratio > 2.6:
            segment.warnings.append("high_compression_ratio")
        if segment.avg_logprob is not None and segment.avg_logprob < -1.0:
            segment.warnings.append("low_asr_confidence")
        if segment.no_speech_prob is not None and segment.no_speech_prob > 0.65 and cleaned:
            segment.warnings.extend(["possible_silence_hallucination", "review_required"])
        if has_runaway_repetition(cleaned):
            segment.warnings.extend(["possible_runaway_repetition", "review_required"])
        # Only an exact, non-trivial adjacent repeat is useful evidence.  Short phrases
        # and substring matches are common in natural Japanese dialogue.
        if cleaned and previous and len(cleaned) >= 4 and cleaned == previous:
            segment.warnings.extend(["possible_repetition", "review_required"])
        if cleaned and len(cleaned) >= 4:
            prior_start = last_seen.get(cleaned)
            if prior_start is not None and segment.start - prior_start >= 10.0:
                # This may be a genuine recurring phrase (for example an
                # outro), so it is review evidence, never deletion authority.
                segment.warnings.extend(["possible_periodic_repetition", "review_required"])
            last_seen[cleaned] = segment.start
        if cleaned:
            previous = cleaned
