from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from translation_forensics.srt import (
    JAPANESE_RE,
    SubtitleBlock,
    parse_srt_text,
    render_srt,
    seconds_to_timecode,
    timecode_to_seconds,
)


_NONLEXICAL_RE = re.compile(
    r"^[\s,.!?…·~〜\-]*(?:하아+|아+|으+|음+|우+|앙+|흣+|응+|어+|오+|후+)"
    r"(?:[\s,.!?…·~〜\-]+(?:하아+|아+|으+|음+|우+|앙+|흣+|응+|어+|오+|후+))*"
    r"[\s,.!?…·~〜\-]*$"
)

_FIELD_OVERRIDES: dict[str, dict[int, tuple[str, str]]] = {
    "START-126-UC": {
        699: ("01:59:06,590", "01:59:06,830"),
    },
    "SONE-107": {
        142: ("00:09:16,700", "00:09:19,190"),
    },
    "OFJE-620-A": {
        64: ("01:14:27,520", "01:14:34,470"),
    },
    "MIDA-642": {
        82: ("00:07:17,080", "00:07:19,400"),
        226: ("00:24:24,500", "00:24:25,000"),
        377: ("00:49:06,090", "00:49:06,770"),
        411: ("00:51:13,130", "00:51:13,450"),
        438: ("00:53:50,000", "00:53:51,270"),
        499: ("01:04:03,260", "01:04:05,520"),
        643: ("01:28:09,000", "01:28:09,720"),
        656: ("01:30:13,000", "01:30:13,600"),
    },
    "SONE-955": {
        160: ("00:11:00,470", "00:11:00,870"),
    },
    "SONE-968": {
        340: ("00:51:43,890", "00:51:46,390"),
    },
    "HMN-869": {
        750: ("02:09:16,430", "02:09:19,680"),
    },
}

_JUR_COLLAPSE_REFERENCE_SHA256 = (
    "5630ab893a9e489ff96828753ba6de3657b949189e506573bb1135161e9030c7"
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


def _reading_duration(text: str) -> float:
    flattened = " ".join(text.split())
    if _NONLEXICAL_RE.fullmatch(flattened):
        return 2.5
    characters = len(re.sub(r"\s+", "", flattened))
    return min(8.0, max(0.5, characters / 7.0 + 0.8))


def _with_times(block: SubtitleBlock, start: str, end: str) -> SubtitleBlock:
    return replace(
        block,
        start=start,
        end=end,
        start_seconds=timecode_to_seconds(start),
        end_seconds=timecode_to_seconds(end),
    )


def _apply_field_overrides(
    title: str, blocks: list[SubtitleBlock]
) -> tuple[list[SubtitleBlock], list[dict[str, Any]]]:
    overrides = _FIELD_OVERRIDES.get(title, {})
    by_number = {block.number: block for block in blocks}
    missing = sorted(set(overrides) - set(by_number))
    if missing:
        raise ValueError(f"Missing field-override blocks for {title}: {missing}")
    ledger: list[dict[str, Any]] = []
    repaired: list[SubtitleBlock] = []
    for block in blocks:
        override = overrides.get(block.number)
        if not override:
            repaired.append(block)
            continue
        start, end = override
        ledger.append(
            {
                "block": block.number,
                "old_start": block.start,
                "old_end": block.end,
                "new_start": start,
                "new_end": end,
                "reason": "verified time-field corruption",
            }
        )
        repaired.append(_with_times(block, start, end))
    return repaired, ledger


def _repair_invalid_ends(
    blocks: list[SubtitleBlock], *, skip: set[int]
) -> tuple[list[SubtitleBlock], list[dict[str, Any]]]:
    ledger: list[dict[str, Any]] = []
    repaired: list[SubtitleBlock] = []
    for index, block in enumerate(blocks):
        if block.number in skip or block.end_seconds > block.start_seconds:
            repaired.append(block)
            continue
        next_start = next(
            (
                later.start_seconds
                for later in blocks[index + 1 :]
                if later.start_seconds > block.start_seconds
            ),
            None,
        )
        end = block.start_seconds + _reading_duration(block.text)
        if next_start is not None:
            end = min(end, next_start)
        if end <= block.start_seconds:
            end = block.start_seconds + 0.08
        repaired_block = replace(
            block,
            end=seconds_to_timecode(end),
            end_seconds=end,
        )
        ledger.append(
            {
                "block": block.number,
                "old_start": block.start,
                "old_end": block.end,
                "new_start": repaired_block.start,
                "new_end": repaired_block.end,
                "reason": "invalid or zero duration repaired from reading time and next cue",
            }
        )
        repaired.append(repaired_block)
    return repaired, ledger


def _collapse_jur_070(
    blocks: list[SubtitleBlock], reference: Path | None
) -> tuple[list[SubtitleBlock], list[dict[str, Any]]]:
    if reference is None:
        raise ValueError("JUR-070 requires --reference for the vocalization collapse")
    reference_hash = _sha256(reference)
    if reference_hash != _JUR_COLLAPSE_REFERENCE_SHA256:
        raise ValueError(
            "JUR-070 reference SHA-256 drifted: "
            f"expected {_JUR_COLLAPSE_REFERENCE_SHA256}, found {reference_hash}"
        )
    template = next(block for block in blocks if block.number == 168)
    replacements = [
        ("00:31:31,340", "00:31:37,340"),
        ("00:31:47,340", "00:31:53,340"),
        ("00:32:00,160", "00:32:06,160"),
        ("00:32:20,000", "00:32:26,000"),
    ]
    collapsed = [
        _with_times(replace(template, number=168 + index, text="하아…"), start, end)
        for index, (start, end) in enumerate(replacements)
    ]
    output: list[SubtitleBlock] = []
    for block in blocks:
        if block.number == 166 or 168 <= block.number <= 193 or block.number in {551, 552}:
            if block.number == 168:
                output.extend(collapsed)
            continue
        output.append(block)
    return output, [
        {
            "blocks": [166, *range(168, 194)],
            "action": "collapse",
            "replacement_count": len(collapsed),
            "reference": str(reference),
            "reference_sha256": reference_hash,
            "reason": "26 zero-duration repeated vocalizations collapsed against two source intervals",
        },
        {
            "blocks": [551, 552],
            "action": "drop",
            "reason": "non-speech tilde noise",
        },
    ]


def _qa(blocks: list[SubtitleBlock]) -> dict[str, Any]:
    invalid = [block.number for block in blocks if block.end_seconds <= block.start_seconds]
    out_of_order = [
        right.number
        for left, right in zip(blocks, blocks[1:])
        if right.start_seconds < left.start_seconds
    ]
    empty = [block.number for block in blocks if not block.text.strip()]
    overlaps = [
        right.number
        for left, right in zip(blocks, blocks[1:])
        if left.end_seconds > right.start_seconds + 0.001
    ]
    japanese = [block.number for block in blocks if JAPANESE_RE.search(block.text)]
    return {
        "pass": not (invalid or out_of_order or empty or japanese),
        "block_count": len(blocks),
        "invalid_duration_blocks": invalid,
        "out_of_order_blocks": out_of_order,
        "empty_blocks": empty,
        "japanese_blocks": japanese,
        "overlap_blocks": overlaps,
        "first_start": blocks[0].start if blocks else None,
        "last_end": blocks[-1].end if blocks else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair invalid active SRT cue durations.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    title = args.title.upper()
    digest = _sha256(args.input)
    if digest != args.expected_sha256.casefold():
        raise ValueError(
            f"Input SHA-256 drifted: expected {args.expected_sha256}, found {digest}"
        )
    source = parse_srt_text(_decode(args.input), source=str(args.input))
    field_repaired, field_ledger = _apply_field_overrides(title, source)
    jur_skip = {166, *range(168, 194), 551, 552} if title == "JUR-070" else set()
    duration_repaired, duration_ledger = _repair_invalid_ends(
        field_repaired, skip=jur_skip
    )
    structural_ledger: list[dict[str, Any]] = []
    candidate = duration_repaired
    if title == "JUR-070":
        candidate, structural_ledger = _collapse_jur_070(candidate, args.reference)
    ordered = sorted(candidate, key=lambda block: (block.start_seconds, block.number))
    reordered = sum(left.number != right.number for left, right in zip(candidate, ordered))
    candidate = [replace(block, number=index) for index, block in enumerate(ordered, 1)]
    qa = _qa(candidate)
    if not qa["pass"]:
        raise ValueError(f"Timeline QA failed: {json.dumps(qa, ensure_ascii=False)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_srt(candidate), encoding="utf-8-sig", newline="\n")
    report = {
        "schema_name": "invalid-srt-duration-repair",
        "schema_version": 1,
        "title": title,
        "input": str(args.input),
        "input_sha256": digest,
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "source_block_count": len(source),
        "field_repair_count": len(field_ledger),
        "duration_repair_count": len(duration_ledger),
        "structural_repair_count": len(structural_ledger),
        "reordered_block_count": reordered,
        "field_ledger": field_ledger,
        "duration_ledger": duration_ledger,
        "structural_ledger": structural_ledger,
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
