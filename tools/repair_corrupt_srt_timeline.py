from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from translation_forensics.srt import SubtitleBlock, parse_srt_text, render_srt, seconds_to_timecode


_BLOCK_SPLIT_RE = re.compile(r"\r?\n\r?\n+")
_TIMECODE_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2}),(\d{3})$")
_NONLEXICAL_RE = re.compile(
    r"^[\s,.!?…·~〜\-]*(?:하아+|아+|으+|음+|우+|앙+|흣+|응+|어+|오+|후+)"
    r"(?:[\s,.!?…·~〜\-]+(?:하아+|아+|으+|음+|우+|앙+|흣+|응+|어+|오+|후+))*"
    r"[\s,.!?…·~〜\-]*$"
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


def _lenient_seconds(value: str) -> float:
    match = _TIMECODE_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Unsupported malformed timecode: {value!r}")
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def _parse_lenient(path: Path) -> list[SubtitleBlock]:
    blocks: list[SubtitleBlock] = []
    for chunk in _BLOCK_SPLIT_RE.split(_decode(path).strip()):
        lines = chunk.splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            raise ValueError(f"Malformed SRT block: {chunk[:120]!r}")
        start, end = (part.strip() for part in lines[1].split("-->", 1))
        start_seconds = _lenient_seconds(start)
        end_seconds = _lenient_seconds(end)
        blocks.append(
            SubtitleBlock(
                number=int(lines[0].strip()),
                start=start,
                end=end,
                text="\n".join(lines[2:]).strip(),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )
    return blocks


def _pred_shift(block_number: int, value: float) -> float:
    if block_number > 160:
        return value
    hours = int(value // 3600)
    remainder = value - hours * 3600
    minutes = int(remainder // 60)
    seconds = remainder - minutes * 60
    if 15 <= block_number <= 27 and hours == 0 and minutes == 11:
        minutes = 1
    elif hours == 1 and minutes <= 8:
        hours = 0
        minutes += 10
    elif hours == 2:
        hours = 0
        if minutes <= 12:
            minutes += 10
    return hours * 3600 + minutes * 60 + seconds


def _reading_duration(text: str) -> float:
    flattened = " ".join(text.split())
    if _NONLEXICAL_RE.fullmatch(flattened):
        return 3.0
    characters = len(re.sub(r"\s+", "", flattened))
    return min(8.0, max(1.0, characters / 7.0 + 1.2))


def _repair_layout(blocks: list[SubtitleBlock]) -> tuple[list[SubtitleBlock], int]:
    repaired: list[SubtitleBlock] = []
    changes = 0
    for block in blocks:
        flattened = " ".join(block.text.split())
        if len(block.text.splitlines()) > 2 or any(
            len(line) > 42 for line in block.text.splitlines()
        ):
            if len(flattened) > 84:
                raise ValueError(
                    f"Subtitle block {block.number} cannot fit a two-line layout"
                )
            preferred = [
                index
                for index, character in enumerate(flattened, 1)
                if character in " ,.?!…"
            ]
            fitting = [
                index
                for index in range(1, len(flattened))
                if len(flattened[:index].rstrip()) <= 42
                and len(flattened[index:].lstrip()) <= 42
            ]
            preferred_fitting = [index for index in preferred if index in fitting]
            pool = preferred_fitting or fitting
            if not pool:
                raise ValueError(
                    f"Subtitle block {block.number} has no valid two-line split"
                )
            split_at = min(
                pool,
                key=lambda index: (
                    max(
                        len(flattened[:index].rstrip()),
                        len(flattened[index:].lstrip()),
                    ),
                    abs(
                        len(flattened[:index].rstrip())
                        - len(flattened[index:].lstrip())
                    ),
                    index,
                ),
            )
            lines = [
                flattened[:split_at].rstrip(),
                flattened[split_at:].lstrip(),
            ]
            repaired.append(replace(block, text="\n".join(lines)))
            changes += 1
        else:
            repaired.append(block)
    return repaired, changes


def _repair_pred(blocks: list[SubtitleBlock]) -> tuple[list[SubtitleBlock], dict[str, Any]]:
    shifted: list[SubtitleBlock] = []
    field_repairs = 0
    duration_repairs = 0
    for block in blocks:
        start = _pred_shift(block.number, block.start_seconds)
        end = _pred_shift(block.number, block.end_seconds)
        if abs(start - block.start_seconds) > 0.0005 or abs(end - block.end_seconds) > 0.0005:
            field_repairs += 1
        maximum = _reading_duration(block.text)
        if end <= start or end - start > maximum:
            end = start + maximum
            duration_repairs += 1
        shifted.append(
            replace(
                block,
                start=seconds_to_timecode(start),
                end=seconds_to_timecode(end),
                start_seconds=start,
                end_seconds=end,
            )
        )
    ordered = sorted(shifted, key=lambda block: (block.start_seconds, block.end_seconds, block.number))
    moved = sum(left.number != right.number for left, right in zip(shifted, ordered))
    ordered = [replace(block, number=index) for index, block in enumerate(ordered, 1)]
    return ordered, {
        "field_repair_count": field_repairs,
        "duration_repair_count": duration_repairs,
        "reordered_block_count": moved,
    }


def _repair_local(title: str, text: str) -> tuple[list[SubtitleBlock], dict[str, Any]]:
    if title == "START-626":
        replacements = [
            ("01:48,480 --> 01:01:55,430", "01:01:48,480 --> 01:01:55,430"),
        ]
    elif title == "MOON-057":
        replacements = [
            ("00:09:99,990", "00:09:59,990"),
            ("01:23,360 --> 00:12:34,220", "00:12:31,360 --> 00:12:34,220"),
            ("02:15:1,160 --> 00:21:52,640", "00:21:51,160 --> 00:21:52,640"),
            ("02:23:9,010 --> 00:23:10,350", "00:23:09,010 --> 00:23:10,350"),
            ("01:14:18,170 --> 00:14:19,050", "00:14:18,170 --> 00:14:19,050"),
            ("01:16:00,810 --> 00:16:01,290", "00:16:00,810 --> 00:16:01,290"),
            ("00:20:11,040 --> 00:19:11,800", "00:19:11,040 --> 00:19:11,800"),
            ("00:25:05,050 --> 00:24:57,690", "00:24:55,050 --> 00:24:57,690"),
            ("02:25:23,750 --> 00:25:25,230", "00:25:23,750 --> 00:25:25,230"),
            ("00:28:03,840 --> 00:28:03,060", "00:28:00,840 --> 00:28:03,060"),
            ("00:29:09,650 --> 00:28:11,930", "00:28:09,650 --> 00:28:11,930"),
            ("00:30:07,320 --> 00:29:13,640", "00:29:07,320 --> 00:29:13,640"),
            ("00:31:15,700 --> 00:29:16,700", "00:29:15,700 --> 00:29:16,700"),
            ("01:31:03,380 --> 00:31:08,370", "00:31:03,380 --> 00:31:08,370"),
            ("01:31:11,990 --> 00:31:13,230", "00:31:11,990 --> 00:31:13,230"),
            ("01:31:14,570 --> 01:31:33,020", "00:31:14,570 --> 00:31:33,020"),
            ("00:44:04,040 --> 00:44:45,460", "00:40:44,040 --> 00:40:45,460"),
            ("00:51:28,820 --> 00:49:45,780", "00:49:28,820 --> 00:49:45,780"),
            ("00:51:16,940 --> 01:51:00,650", "00:50:16,940 --> 00:51:00,650"),
            ("01:51:00,650 --> 01:51:02,730", "00:51:00,650 --> 00:51:02,730"),
            ("01:51:02,730 --> 01:52:40,200", "00:51:02,730 --> 00:52:40,200"),
            ("01:53:10,480 --> 01:53:54,710", "00:53:10,480 --> 00:53:54,710"),
            ("01:23:02,420 --> 01:20:38,410", "01:20:32,420 --> 01:20:38,410"),
            ("02:02:12,270 --> 01:59:13,470", "01:59:12,270 --> 01:59:13,470"),
            ("02:02:15,630 --> 01:59:16,870", "01:59:15,630 --> 01:59:16,870"),
            ("02:54:35,730 --> 02:51:49,470", "02:51:45,730 --> 02:51:49,470"),
        ]
    elif title == "SNOS-167":
        replacements = [
            ("30:02,820 --> 00:30:04,240", "00:30:02,820 --> 00:30:04,240"),
            ("0:41:17,730", "00:41:17,730"),
            ("00:34:34,060 --> 00:35:35,780", "00:35:34,060 --> 00:35:35,780"),
            ("00:40:55,760 --> 00:39:56,560", "00:39:55,760 --> 00:39:56,560"),
            ("00:39:56,560 --> 00:41:17,730", "00:39:56,560 --> 00:40:17,730"),
            ("01:41:17,730 --> 01:41:19,830", "00:41:17,730 --> 00:41:19,830"),
            ("01:41:59,600 --> 01:42:02,000", "00:41:59,600 --> 00:42:02,000"),
            ("00:43:37,830 --> 00:43:39,310", "00:44:37,830 --> 00:44:39,310"),
            ("01:35:53,450 --> 01:35:56,970", "01:33:53,450 --> 01:33:56,970"),
            ("01:35:57,550 --> 01:35:59,310", "01:33:57,550 --> 01:33:59,310"),
            ("01:36:01,370 --> 01:34:02,430", "01:34:01,370 --> 01:34:02,430"),
        ]
    else:
        raise ValueError(f"Unsupported title: {title}")
    repaired = text
    for old, new in replacements:
        if repaired.count(old) != 1:
            raise ValueError(f"Expected exactly one local repair target for {title}: {old}")
        repaired = repaired.replace(old, new)
    parsed = parse_srt_text(repaired, source=title)
    normalized: list[SubtitleBlock] = []
    duration_repairs = 0
    for block in parsed:
        maximum = _reading_duration(block.text)
        end = block.end_seconds
        if end <= block.start_seconds or end - block.start_seconds > maximum:
            end = block.start_seconds + maximum
            duration_repairs += 1
        normalized.append(
            replace(
                block,
                end=seconds_to_timecode(end),
                end_seconds=end,
            )
        )
    ordered = sorted(
        normalized,
        key=lambda block: (block.start_seconds, block.end_seconds, block.number),
    )
    moved = sum(left.number != right.number for left, right in zip(normalized, ordered))
    ordered = [replace(block, number=index) for index, block in enumerate(ordered, 1)]
    return ordered, {
        "field_repair_count": len(replacements),
        "duration_repair_count": duration_repairs,
        "reordered_block_count": moved,
        "replacements": [{"old": old, "new": new} for old, new in replacements],
    }


def _qa(blocks: list[SubtitleBlock], *, enforce_eight_seconds: bool) -> dict[str, Any]:
    invalid = [block.number for block in blocks if block.end_seconds <= block.start_seconds]
    out_of_order = [
        right.number
        for left, right in zip(blocks, blocks[1:])
        if right.start_seconds < left.start_seconds
    ]
    long = [block.number for block in blocks if block.duration > 8.001]
    empty = [block.number for block in blocks if not block.text.strip()]
    long_lines = [
        block.number
        for block in blocks
        if len(block.text.splitlines()) > 2
        or any(len(line) > 42 for line in block.text.splitlines())
    ]
    overlaps = [
        right.number
        for left, right in zip(blocks, blocks[1:])
        if left.end_seconds > right.start_seconds + 0.001
    ]
    return {
        "pass": not (
            invalid
            or out_of_order
            or empty
            or long_lines
            or (enforce_eight_seconds and long)
        ),
        "block_count": len(blocks),
        "invalid_duration_blocks": invalid,
        "out_of_order_blocks": out_of_order,
        "over_eight_second_blocks": long,
        "eight_second_limit_enforced": enforce_eight_seconds,
        "empty_blocks": empty,
        "layout_violation_blocks": long_lines,
        "overlap_blocks": overlaps,
        "first_start": blocks[0].start if blocks else None,
        "last_end": blocks[-1].end if blocks else None,
        "max_duration_seconds": max((block.duration for block in blocks), default=0.0),
        "max_lines": max((len(block.text.splitlines()) for block in blocks), default=0),
        "max_line_characters": max(
            (len(line) for block in blocks for line in block.text.splitlines()),
            default=0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair known corrupt SRT timeline patterns.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    title = args.title.upper()
    digest = _sha256(args.input)
    if digest != args.expected_sha256.casefold():
        raise ValueError(f"Input SHA-256 drifted: expected {args.expected_sha256}, found {digest}")
    if title == "PRED-488":
        candidate, repairs = _repair_pred(_parse_lenient(args.input))
    else:
        candidate, repairs = _repair_local(title, _decode(args.input))
    candidate, layout_repairs = _repair_layout(candidate)
    repairs["layout_repair_count"] = layout_repairs
    qa = _qa(candidate, enforce_eight_seconds=True)
    if not qa["pass"]:
        raise ValueError(f"Timeline QA failed: {json.dumps(qa, ensure_ascii=False)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_srt(candidate), encoding="utf-8-sig", newline="\n")
    report = {
        "title": title,
        "input": str(args.input),
        "input_sha256": digest,
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        **repairs,
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
