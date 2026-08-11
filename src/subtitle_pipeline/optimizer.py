from __future__ import annotations

from dataclasses import dataclass
from math import inf
import unicodedata

from .models import Cue, SourceSegment, Utterance, Word
from .profiles import Profile
from .text import (
    boundary_kind, compact_high_confidence_runaway_repetition, display_width,
    is_bad_boundary, is_compacted_vocalisation, normalize_japanese, wrap_two_lines,
)


@dataclass(slots=True)
class Atom:
    text: str
    start: float
    end: float
    source_id: int
    speaker: str
    word: Word | None


def _is_safe_authoritative_gap(text: str) -> bool:
    """Return whether missing aligned text is presentation-only punctuation or spacing."""
    return all(character.isspace() or unicodedata.category(character).startswith("P") for character in text)


def _reconcile_authoritative_word_text(utterance: Utterance) -> list[str] | None:
    """Map omitted punctuation/spacing onto ordered word timings without changing words.

    A lexical omission or substitution is not safe to time heuristically.  In that
    case the caller keeps the authoritative utterance whole instead.
    """
    word_texts = [word.text for word in utterance.words]
    authoritative = utterance.text_normalized
    if "".join(word_texts) == authoritative:
        return word_texts
    if not authoritative or not word_texts or any(not text for text in word_texts):
        return None

    reconciled = list(word_texts)
    cursor = 0
    for index, word_text in enumerate(word_texts):
        start = authoritative.find(word_text, cursor)
        if start < 0:
            return None
        gap = authoritative[cursor:start]
        if not _is_safe_authoritative_gap(gap):
            return None
        if gap:
            if index == 0:
                reconciled[index] = gap + reconciled[index]
            else:
                reconciled[index - 1] += gap
        cursor = start + len(word_text)

    trailing = authoritative[cursor:]
    if not _is_safe_authoritative_gap(trailing):
        return None
    reconciled[-1] += trailing
    if "".join(reconciled) != authoritative:
        return None
    return reconciled


def segments_to_utterances(segments: list[SourceSegment], normalization: str) -> list[Utterance]:
    utterances: list[Utterance] = []
    for index, segment in enumerate(segments, 1):
        warnings = list(segment.warnings)
        if not segment.words:
            warnings.append("timestamp_precision_limited")
        utterances.append(Utterance(
            id=f"utt_{index:06}", source_segment_ids=[segment.id], start=segment.start, end=segment.end,
            text_raw=segment.text, text_normalized=normalize_japanese(segment.text, normalization),
            words=segment.words, speaker=segment.speaker,
            metrics={"no_speech_prob": segment.no_speech_prob, "avg_logprob": segment.avg_logprob,
                     "compression_ratio": segment.compression_ratio, **segment.metadata}, warnings=warnings,
        ))
    return utterances


def _atoms(utterances: list[Utterance]) -> list[Atom]:
    result: list[Atom] = []
    for utterance in utterances:
        if utterance.words:
            reconciled = _reconcile_authoritative_word_text(utterance)
            if reconciled is not None:
                if reconciled != [word.text for word in utterance.words]:
                    utterance.warnings = sorted(set(utterance.warnings) | {"word_text_reconciled"})
                result.extend(
                    Atom(text, word.start, word.end, utterance.source_segment_ids[0], utterance.speaker, word)
                    for text, word in zip(reconciled, utterance.words)
                )
            else:
                # The transcript is authoritative.  Do not let incomplete or
                # changed alignment tokens silently remove or replace its text.
                utterance.warnings = sorted(set(utterance.warnings) | {
                    "timestamp_precision_limited", "word_text_mismatch", "review_required",
                })
                result.append(Atom(
                    utterance.text_normalized, utterance.start, utterance.end,
                    utterance.source_segment_ids[0], utterance.speaker, None,
                ))
        else:
            # No invented timing: this atom can only remain whole.
            result.append(Atom(utterance.text_normalized, utterance.start, utterance.end,
                               utterance.source_segment_ids[0], utterance.speaker, None))
    return result


def _edge_cost(atoms: list[Atom], i: int, j: int, profile: Profile, scene_changes: list[float]) -> tuple[float, list[str]]:
    group = atoms[i:j]
    text = "".join(a.text for a in group).strip()
    if not text:
        return inf, ["empty"]
    start, end = group[0].start, group[-1].end
    duration, length = max(0.01, end - start), display_width(text)
    lines = max(1, int((length + profile.target_line_length - 1) // profile.target_line_length))
    # Every extra cue has a small presentation cost.  Without it, even a
    # millisecond-scale alignment gap makes the dynamic program prefer a split.
    # ``compactness`` deliberately changes that trade-off between profiles.
    cost = 3.0 * profile.compactness
    if lines > profile.max_lines:
        cost += 150 * (lines - profile.max_lines)
    if length > profile.absolute_line_limit * profile.max_lines:
        cost += 80 + (length - profile.absolute_line_limit * profile.max_lines) * 4
    # Estimate balanced line width here; exact wrapping is performed only for
    # the selected DP path.  Calling the full wrapper for every candidate edge
    # makes feature-length transcript rebuilds unnecessarily quadratic in text.
    estimated_line_width = length / min(lines, profile.max_lines)
    cost += max(0.0, estimated_line_width - profile.recommended_line_limit) * 1.5
    if duration < profile.min_duration:
        cost += (profile.min_duration - duration) * 25
    if duration > profile.max_duration:
        # Duration is a hard viewer constraint whenever an aligned internal item
        # boundary exists. A single indivisible aligned item is retained and audited.
        if len(group) > 1:
            return inf, ["duration_hard_limit"]
        cost += 500 + (duration - profile.max_duration) * 100
    if duration < profile.target_min_duration:
        cost += (profile.target_min_duration - duration) * 4
    elif duration > profile.target_max_duration:
        cost += (duration - profile.target_max_duration) * 4
    if len(text) <= 2 and duration < 1.3:
        cost += 25
    # A cut inside a cue is mildly disfavoured, but never forces a boundary.
    interior_cuts = sum(start + 0.08 < cut < end - 0.08 for cut in scene_changes)
    cost += interior_cuts * 3.0
    if j < len(atoms):
        if group[-1].speaker != atoms[j].speaker:
            boundary_bonus = -12.0
            reason = "speaker_change"
        else:
            gap = max(0.0, atoms[j].start - group[-1].end)
            if gap >= 0.35:
                boundary_bonus = -8.0 - min(gap, 1.5) * 6.0
                reason = "long_silence"
            else:
                reason = boundary_kind(group[-1].text, atoms[j].text)
                boundary_bonus = {
                    "sentence_end": -4.0,
                    "clause_or_breath": -2.0,
                    "connective": -2.0,
                }.get(reason, 0.0)
        cost += boundary_bonus
        if is_bad_boundary(group[-1].text, atoms[j].text):
            cost += 45
    else:
        reason = "end_of_speech"
    # Splitting a word-timestamp-less atom is deliberately impossible (atoms themselves are indivisible).
    return cost, [reason]


def _viewer_cues_from_atoms(atoms: list[Atom], profile: Profile, normalization: str, scene_changes: list[float]) -> list[Cue]:
    if not atoms:
        return []
    n = len(atoms)
    best = [inf] * (n + 1)
    previous = [-1] * (n + 1)
    reasons: list[list[str]] = [[] for _ in range(n + 1)]
    best[0] = 0.0
    for i in range(n):
        if best[i] == inf:
            continue
        # Candidate window is bounded for practical O(n*k) DP; any longer group is strongly undesirable.
        for j in range(i + 1, min(n, i + 32) + 1):
            if i < j - 1 and atoms[j - 1].word is None:
                break
            cost, why = _edge_cost(atoms, i, j, profile, scene_changes)
            if best[i] + cost < best[j]:
                best[j], previous[j], reasons[j] = best[i] + cost, i, why
    ranges: list[tuple[int, int, list[str]]] = []
    cursor = n
    while cursor > 0:
        start = previous[cursor]
        if start < 0:  # defensive fallback preserves content.
            start = cursor - 1
        ranges.append((start, cursor, reasons[cursor]))
        cursor = start
    ranges.reverse()
    cues: list[Cue] = []
    for cue_id, (start, end, why) in enumerate(ranges, 1):
        group = atoms[start:end]
        raw = "".join(a.text for a in group).strip()
        normalized = normalize_japanese(raw, normalization)
        conservative = normalize_japanese(raw, "conservative")
        lines = wrap_two_lines(normalized, profile.target_line_length, profile.absolute_line_limit)
        source_ids = list(dict.fromkeys(a.source_id for a in group))
        warnings: list[str] = []
        if normalization == "viewer" and normalized != conservative:
            warnings.append("viewer_runaway_repetition_compacted")
        if any(a.word is None for a in group):
            warnings.append("timestamp_precision_limited")
        scene_distance = min((min(abs(group[0].start - cut), abs(group[-1].end - cut)) for cut in scene_changes), default=None)
        cue_start = group[0].start
        cue_end = max(group[-1].end, cue_start + 0.05)
        if cue_end - cue_start > profile.max_duration:
            warnings.append("duration_limit_unavoidable")
        if any(display_width(line) > profile.absolute_line_limit for line in lines):
            warnings.append("line_limit_unavoidable")
        cues.append(Cue(cue_id, cue_start, cue_end, "\n".join(lines), normalized,
                        source_ids, group[0].speaker, why, warnings=warnings,
                        adjacent_silence=max(0.0, (atoms[end].start - group[-1].end)) if end < n else None,
                        scene_distance=scene_distance))
    return _remove_overlaps(cues, profile)


def _remove_overlaps(cues: list[Cue], profile: Profile) -> list[Cue]:
    # SRT is serialized to milliseconds.  A 10 ms in-memory duration can
    # still round to the same timestamp when the source starts at a half-ms
    # boundary, so leave one extra millisecond of output-safe headroom.
    minimum = 0.011
    for cue in cues:
        if cue.end <= cue.start:
            cue.end = cue.start + minimum
            cue.warnings.append("zero_duration_word_timing_adjusted")
    for previous, current in zip(cues, cues[1:]):
        if current.start < previous.end:
            # Consecutive words can legitimately share an ASR timestamp.  Preserve
            # ordering by moving the later cue forward; midpoint adjustment leaves
            # chains of same-timestamp cues overlapping again.
            current.start = previous.end
            if current.end <= current.start:
                current.end = current.start + minimum
                current.warnings.append("zero_duration_word_timing_adjusted")
            previous.warnings.append("timing_adjusted_to_avoid_overlap")
            current.warnings.append("timing_adjusted_to_avoid_overlap")
    for index, cue in enumerate(cues, 1):
        cue.id = index
    return cues


def _extend_short_display_cues(cues: list[Cue], profile: Profile) -> list[Cue]:
    """Use only neighbouring measured silence to reach the viewer minimum.

    The source timings remain untouched.  A viewer cue can occupy a verified gap
    before the next cue or after the preceding cue, but never crosses another
    aligned cue boundary.  Cues without enough surrounding silence stay marked
    for review instead of receiving invented timing.
    """

    for index, cue in enumerate(cues):
        cue.warnings = [warning for warning in cue.warnings if warning != "short_duration_review_required"]
        remaining = profile.min_duration - cue.duration
        if remaining <= 0:
            continue
        extended = False
        if index + 1 < len(cues):
            forward_silence = max(0.0, cues[index + 1].start - cue.end)
            forward_extension = min(remaining, forward_silence)
            cue.end += forward_extension
            remaining -= forward_extension
            extended = forward_extension > 0
        if remaining > 0 and index > 0:
            backward_silence = max(0.0, cue.start - cues[index - 1].end)
            backward_extension = min(remaining, backward_silence)
            cue.start -= backward_extension
            remaining -= backward_extension
            extended = extended or backward_extension > 0
        if extended:
            cue.warnings.append("viewer_duration_extended_into_silence")
        if cue.duration < profile.min_duration - 1e-9:
            cue.warnings.append("short_duration_review_required")
    return _remove_overlaps(cues, profile)


def source_faithful_cues(utterances: list[Utterance]) -> list[Cue]:
    return [Cue(i, u.start, u.end, u.text_normalized, u.text_normalized, u.source_segment_ids,
                u.speaker, ["source_segment_boundary"], warnings=list(u.warnings), metrics=dict(u.metrics))
            for i, u in enumerate(utterances, 1) if u.text_normalized]


def _collapse_runaway_vocalisation_cues(cues: list[Cue], profile: Profile) -> list[Cue]:
    """Keep one bounded display cue for a continuous, already-compacted vocalisation.

    Source SRT and transcript data retain every original ASR item.  This only
    prevents a single long hallucinated or non-verbal run from producing many
    identical viewer cues in immediate succession.
    """
    collapsed: list[Cue] = []
    index = 0
    while index < len(cues):
        first = cues[index]
        original_vocalisation = (
            "viewer_runaway_repetition_compacted" in first.warnings
            and "possible_runaway_repetition" in first.warnings
            and is_compacted_vocalisation(first.normalized_text)
        )
        high_confidence = (
            "viewer_high_confidence_runaway_compacted" in first.warnings
            and "high_compression_ratio" in first.warnings
            and first.normalized_text.endswith("…")
        )
        eligible = original_vocalisation or high_confidence
        if not eligible:
            collapsed.append(first)
            index += 1
            continue
        group = [first]
        cursor = index + 1
        while cursor < len(cues):
            candidate = cues[cursor]
            same_source = candidate.source_segment_ids == first.source_segment_ids
            contiguous = candidate.start <= group[-1].end + 0.15
            same_vocalisation = candidate.normalized_text == first.normalized_text
            candidate_original_vocalisation = (
                "viewer_runaway_repetition_compacted" in candidate.warnings
                and "possible_runaway_repetition" in candidate.warnings
                and is_compacted_vocalisation(candidate.normalized_text)
            )
            candidate_high_confidence = (
                "viewer_high_confidence_runaway_compacted" in candidate.warnings
                and "high_compression_ratio" in candidate.warnings
                and candidate.normalized_text.endswith("…")
            )
            candidate_eligible = candidate_original_vocalisation or candidate_high_confidence
            if not (same_source and contiguous and same_vocalisation and candidate_eligible):
                break
            group.append(candidate)
            cursor += 1
        if len(group) == 1:
            collapsed.append(first)
            index += 1
            continue
        first.end = min(group[-1].end, first.start + profile.max_duration)
        collapse_warning = (
            "viewer_high_confidence_runaway_collapsed"
            if "viewer_high_confidence_runaway_compacted" in first.warnings
            else "viewer_runaway_repetition_collapsed"
        )
        first.warnings = sorted(set(first.warnings) | {collapse_warning})
        first.split_reasons = sorted(set(first.split_reasons) | {"runaway_repetition_display_collapse"})
        collapsed.append(first)
        index = cursor
    return _remove_overlaps(collapsed, profile)


def viewer_cues(utterances: list[Utterance], profile: Profile, normalization: str, scene_changes: list[float] | None = None) -> list[Cue]:
    cues = _viewer_cues_from_atoms(_atoms(utterances), profile, normalization, scene_changes or [])
    by_source = {utterance.source_segment_ids[0]: utterance for utterance in utterances}
    for cue in cues:
        sources = [by_source[source_id] for source_id in cue.source_segment_ids if source_id in by_source]
        source_warnings = {warning for source in sources for warning in source.warnings}
        cue.warnings = sorted(set(cue.warnings) | source_warnings)
        if normalization == "viewer" and "high_compression_ratio" in source_warnings:
            compacted = compact_high_confidence_runaway_repetition(cue.normalized_text)
            if compacted != cue.normalized_text:
                cue.normalized_text = compacted
                cue.text = "\n".join(wrap_two_lines(compacted, profile.target_line_length, profile.absolute_line_limit))
                cue.warnings = sorted(set(cue.warnings) | {"viewer_high_confidence_runaway_compacted"})
        cue.metrics = {
            "backends": sorted({str(source.metrics.get("backend", "whisper")) for source in sources}),
            "decisions": sorted({str(source.metrics.get("decision", "whisper_primary")) for source in sources}),
            "whisper_alternatives": [
                alternative for source in sources for alternative in source.metrics.get("whisper_alternatives", [])
            ],
        }
    return _extend_short_display_cues(_collapse_runaway_vocalisation_cues(cues, profile), profile)
