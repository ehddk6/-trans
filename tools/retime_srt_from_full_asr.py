from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from translation_forensics.srt import (
    JAPANESE_RE,
    SubtitleBlock,
    parse_srt_text,
    render_srt,
    seconds_to_timecode,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    return raw.decode("utf-8")


def _normalize_japanese(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = "".join(
        chr(ord(character) - 0x60)
        if "\u30a1" <= character <= "\u30f6"
        else character
        for character in value
    )
    return re.sub(r"[^0-9A-Za-z\u3040-\u309f\u4e00-\u9fff]", "", value)


def _reading_duration(text: str) -> float:
    characters = len(re.sub(r"\s+", "", text))
    return min(8.0, max(0.8, characters / 7.0 + 0.8))


def _load_asr_segments(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw = sorted(
        [segment for row in rows for segment in row.get("segments", [])],
        key=lambda segment: (
            float(segment["start_seconds"]),
            float(segment["end_seconds"]),
        ),
    )
    deduplicated: list[dict[str, Any]] = []
    for segment in raw:
        normalized = _normalize_japanese(str(segment.get("text") or ""))
        if not normalized:
            continue
        duplicate = any(
            abs(float(segment["start_seconds"]) - float(previous["start_seconds"]))
            < 2.5
            and SequenceMatcher(
                None,
                normalized,
                _normalize_japanese(str(previous["text"])),
                autojunk=False,
            ).ratio()
            > 0.72
            for previous in deduplicated[-5:]
        )
        if not duplicate:
            deduplicated.append(segment)
    return deduplicated, len(raw)


def _align(
    source_texts: list[str], asr_segments: list[dict[str, Any]]
) -> list[tuple[int, int, float]]:
    left = [_normalize_japanese(text) for text in source_texts]
    right = [_normalize_japanese(str(segment["text"])) for segment in asr_segments]
    similarities = [
        [
            SequenceMatcher(None, source, transcript, autojunk=False).ratio()
            for transcript in right
        ]
        for source in left
    ]
    source_count = len(left)
    asr_count = len(right)
    scores = [[0.0] * (asr_count + 1) for _ in range(source_count + 1)]
    choices = [[0] * (asr_count + 1) for _ in range(source_count + 1)]
    for source_index in range(1, source_count + 1):
        scores[source_index][0] = scores[source_index - 1][0] - 0.45
        choices[source_index][0] = 1
    for asr_index in range(1, asr_count + 1):
        scores[0][asr_index] = scores[0][asr_index - 1] - 0.25
        choices[0][asr_index] = 2
    for source_index in range(1, source_count + 1):
        for asr_index in range(1, asr_count + 1):
            options = (
                scores[source_index - 1][asr_index] - 0.45,
                scores[source_index][asr_index - 1] - 0.25,
                scores[source_index - 1][asr_index - 1]
                + 2.4 * similarities[source_index - 1][asr_index - 1]
                - 1.0,
            )
            choice = max(range(3), key=lambda index: options[index])
            scores[source_index][asr_index] = options[choice]
            choices[source_index][asr_index] = choice + 1
    pairs: list[tuple[int, int, float]] = []
    source_index = source_count
    asr_index = asr_count
    while source_index or asr_index:
        choice = choices[source_index][asr_index]
        if choice == 3:
            pairs.append(
                (
                    source_index - 1,
                    asr_index - 1,
                    similarities[source_index - 1][asr_index - 1],
                )
            )
            source_index -= 1
            asr_index -= 1
        elif choice == 1:
            source_index -= 1
        elif choice == 2:
            asr_index -= 1
        else:
            break
    pairs.reverse()
    return pairs


def _interpolated_starts(
    *,
    source_count: int,
    anchors: dict[int, float],
    lower_bound: float,
    upper_bound: float,
) -> list[float]:
    if not anchors:
        raise ValueError("ASR alignment produced no usable timing anchors")
    starts = [0.0] * source_count
    anchor_indices = sorted(anchors)
    for source_index, start in anchors.items():
        starts[source_index] = start

    first = anchor_indices[0]
    first_span = max(0.1, anchors[first] - lower_bound)
    for source_index in range(first):
        starts[source_index] = lower_bound + first_span * (source_index + 1) / (first + 1)

    for left_index, right_index in zip(anchor_indices, anchor_indices[1:]):
        gap = right_index - left_index
        if gap <= 1:
            continue
        left_time = anchors[left_index]
        right_time = anchors[right_index]
        for source_index in range(left_index + 1, right_index):
            fraction = (source_index - left_index) / gap
            starts[source_index] = left_time + (right_time - left_time) * fraction

    last = anchor_indices[-1]
    trailing = source_count - last - 1
    if trailing:
        last_span = max(0.1, upper_bound - anchors[last])
        for offset, source_index in enumerate(range(last + 1, source_count), 1):
            starts[source_index] = anchors[last] + last_span * offset / (trailing + 1)

    for index in range(1, len(starts)):
        starts[index] = max(starts[index], starts[index - 1] + 0.05)
    if starts[-1] >= upper_bound:
        raise ValueError("Retimed subtitles exceed the media duration")
    return starts


def _qa(blocks: list[SubtitleBlock], media_duration: float) -> dict[str, Any]:
    invalid = [block.number for block in blocks if block.end_seconds <= block.start_seconds]
    out_of_order = [
        right.number
        for left, right in zip(blocks, blocks[1:])
        if right.start_seconds < left.start_seconds
    ]
    overlaps = [
        right.number
        for left, right in zip(blocks, blocks[1:])
        if left.end_seconds > right.start_seconds + 0.001
    ]
    japanese = [block.number for block in blocks if JAPANESE_RE.search(block.text)]
    beyond_media = [block.number for block in blocks if block.end_seconds > media_duration + 0.001]
    return {
        "pass": not (invalid or out_of_order or overlaps or japanese or beyond_media),
        "block_count": len(blocks),
        "invalid_duration_blocks": invalid,
        "out_of_order_blocks": out_of_order,
        "overlap_blocks": overlaps,
        "japanese_blocks": japanese,
        "beyond_media_blocks": beyond_media,
        "first_start": blocks[0].start if blocks else None,
        "last_end": blocks[-1].end if blocks else None,
    }


def _clip_overlaps(blocks: list[SubtitleBlock]) -> tuple[list[SubtitleBlock], int]:
    clipped: list[SubtitleBlock] = []
    changes = 0
    for index, block in enumerate(blocks):
        if index + 1 >= len(blocks):
            clipped.append(block)
            continue
        next_start = blocks[index + 1].start_seconds
        if block.end_seconds <= next_start + 0.001:
            clipped.append(block)
            continue
        if next_start <= block.start_seconds:
            raise ValueError(
                f"Cannot clip overlapping blocks at source block {block.number}"
            )
        clipped.append(
            replace(
                block,
                end=seconds_to_timecode(next_start),
                end_seconds=next_start,
            )
        )
        changes += 1
    return clipped, changes


def main() -> None:
    parser = argparse.ArgumentParser(description="Retime a corrupt SRT tail from full-media ASR.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--source-ja", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--asr", required=True, type=Path)
    parser.add_argument("--expected-asr-sha256", required=True)
    parser.add_argument("--retime-from-block", required=True, type=int)
    parser.add_argument("--media-duration", required=True, type=float)
    parser.add_argument("--minimum-anchor-score", type=float, default=0.4)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    for path, expected, label in (
        (args.input, args.expected_input_sha256, "input"),
        (args.source_ja, args.expected_source_sha256, "source"),
        (args.asr, args.expected_asr_sha256, "ASR"),
    ):
        found = _sha256(path)
        if found != expected.casefold():
            raise ValueError(f"{label} SHA-256 drifted: expected {expected}, found {found}")

    active = parse_srt_text(_decode(args.input), source=str(args.input))
    source = parse_srt_text(_decode(args.source_ja), source=str(args.source_ja))
    if len(active) != len(source):
        raise ValueError("Active and Japanese source cue counts differ")
    if not 1 < args.retime_from_block <= len(active):
        raise ValueError("Invalid --retime-from-block")
    tail_source = source[args.retime_from_block - 1 :]
    asr_segments, raw_asr_count = _load_asr_segments(args.asr)
    pairs = _align([block.text for block in tail_source], asr_segments)
    anchors = {
        source_index: float(asr_segments[asr_index]["start_seconds"])
        for source_index, asr_index, score in pairs
        if score >= args.minimum_anchor_score
        and len(_normalize_japanese(tail_source[source_index].text)) >= 2
    }
    previous = active[args.retime_from_block - 2]
    starts = _interpolated_starts(
        source_count=len(tail_source),
        anchors=anchors,
        lower_bound=previous.end_seconds,
        upper_bound=args.media_duration,
    )
    pair_by_source = {
        source_index: (asr_index, score)
        for source_index, asr_index, score in pairs
    }
    retimed_tail: list[SubtitleBlock] = []
    for index, (block, start) in enumerate(zip(active[args.retime_from_block - 1 :], starts)):
        next_start = starts[index + 1] if index + 1 < len(starts) else args.media_duration
        desired_end = start + _reading_duration(block.text)
        match = pair_by_source.get(index)
        if match and match[1] >= args.minimum_anchor_score:
            desired_end = min(
                desired_end,
                float(asr_segments[match[0]]["end_seconds"]),
            )
        end = min(desired_end, next_start - 0.02, args.media_duration)
        if end <= start:
            end = min(start + 0.03, next_start, args.media_duration)
        retimed_tail.append(
            replace(
                block,
                start=seconds_to_timecode(start),
                end=seconds_to_timecode(end),
                start_seconds=start,
                end_seconds=end,
            )
        )

    candidate = [*active[: args.retime_from_block - 1], *retimed_tail]
    ordered = sorted(candidate, key=lambda block: (block.start_seconds, block.number))
    reordered = sum(left.number != right.number for left, right in zip(candidate, ordered))
    ordered, overlap_clips = _clip_overlaps(ordered)
    candidate = [replace(block, number=index) for index, block in enumerate(ordered, 1)]
    qa = _qa(candidate, args.media_duration)
    if not qa["pass"]:
        raise ValueError(f"Retimed candidate failed QA: {json.dumps(qa, ensure_ascii=False)}")

    score_values = sorted(score for _, _, score in pairs)
    high_values = [score for score in score_values if score >= args.minimum_anchor_score]
    anchor_indices = sorted(anchors)
    largest_anchor_gap = max(
        (right - left - 1 for left, right in zip(anchor_indices, anchor_indices[1:])),
        default=0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_srt(candidate), encoding="utf-8-sig", newline="\n")
    report = {
        "schema_name": "translation-forensics/asr-tail-retime",
        "schema_version": 1,
        "title": args.title.upper(),
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "source_ja": str(args.source_ja),
        "source_ja_sha256": _sha256(args.source_ja),
        "asr": str(args.asr),
        "asr_sha256": _sha256(args.asr),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "retime_from_block": args.retime_from_block,
        "retimed_block_count": len(retimed_tail),
        "raw_asr_segment_count": raw_asr_count,
        "deduplicated_asr_segment_count": len(asr_segments),
        "alignment_pair_count": len(pairs),
        "anchor_count": len(anchors),
        "minimum_anchor_score": args.minimum_anchor_score,
        "median_pair_score": score_values[len(score_values) // 2] if score_values else 0.0,
        "median_anchor_score": high_values[len(high_values) // 2] if high_values else 0.0,
        "largest_unanchored_source_run": largest_anchor_gap,
        "reordered_block_count": reordered,
        "overlap_clip_count": overlap_clips,
        **qa,
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
