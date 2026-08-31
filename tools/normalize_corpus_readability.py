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
    parse_srt,
    render_srt,
    seconds_to_timecode,
    timecode_to_seconds,
)


PLACEHOLDER = "[안전상 번역 불가]"
NONLEXICAL_RE = re.compile(
    r"^[\s,.!?…·~〜\-]*(?:하아+|아+|으+|음+|우+|앙+|흣+|응+|어+|오+|후+)"
    r"(?:[\s,.!?…·~〜\-]+(?:하아+|아+|으+|음+|우+|앙+|흣+|응+|어+|오+|후+))*"
    r"[\s,.!?…·~〜\-]*$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.split()).strip()


def reading_duration(text: str) -> float:
    value = normalized(text)
    if NONLEXICAL_RE.fullmatch(value):
        return 2.5
    characters = len(re.sub(r"\s+", "", value))
    return min(8.0, max(1.0, characters / 7.0 + 0.8))


def with_times(
    block: SubtitleBlock, start_seconds: float, end_seconds: float
) -> SubtitleBlock:
    return replace(
        block,
        start=seconds_to_timecode(start_seconds),
        end=seconds_to_timecode(end_seconds),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def repair_embedded_srt(
    title: str, blocks: list[SubtitleBlock]
) -> tuple[list[SubtitleBlock], list[dict[str, Any]]]:
    output: list[SubtitleBlock] = []
    ledger: list[dict[str, Any]] = []
    for block in blocks:
        value = normalized(block.text)
        if title == "IPZZ-544" and block.number == 317 and "00:15:41,980 -->" in block.text:
            output.append(replace(block, text="맞으시죠?"))
            output.append(
                with_times(
                    replace(block, text="네."),
                    timecode_to_seconds("00:15:41,980"),
                    timecode_to_seconds("00:15:42,500"),
                )
            )
            ledger.append(
                {
                    "block": block.number,
                    "action": "split embedded SRT block",
                    "new_text": ["맞으시죠?", "네."],
                }
            )
            continue
        replacements: dict[int, tuple[str, str, str]] = {
            410: ("01:05:38,200", "01:05:46,960", "대박, 대박, 장난 아니야, 대박…"),
            474: ("01:16:22,950", "01:17:25,840", "앗, 하아…"),
            564: ("01:30:50,420", "01:30:51,480", "그대로 누워 있어요."),
        }
        if title == "START-126-UC" and block.number in replacements and "-->" in value:
            start, end, text = replacements[block.number]
            output.append(
                with_times(
                    replace(block, text=text),
                    timecode_to_seconds(start),
                    timecode_to_seconds(end),
                )
            )
            ledger.append(
                {
                    "block": block.number,
                    "action": "remove embedded SRT metadata",
                    "new_start": start,
                    "new_end": end,
                    "new_text": text,
                }
            )
            continue
        output.append(block)
    return output, ledger


def cap_stale_durations(
    blocks: list[SubtitleBlock], threshold: float = 30.0
) -> tuple[list[SubtitleBlock], list[dict[str, Any]]]:
    output: list[SubtitleBlock] = []
    ledger: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if block.duration <= threshold:
            output.append(block)
            continue
        end = block.start_seconds + reading_duration(block.text)
        next_start = next(
            (
                later.start_seconds
                for later in blocks[index + 1 :]
                if later.start_seconds > block.start_seconds
            ),
            None,
        )
        if next_start is not None:
            end = min(end, next_start)
        end = max(end, block.start_seconds + 0.08)
        repaired = with_times(block, block.start_seconds, end)
        output.append(repaired)
        ledger.append(
            {
                "block": block.number,
                "old_duration": round(block.duration, 3),
                "new_duration": round(repaired.duration, 3),
                "old_end": block.end,
                "new_end": repaired.end,
            }
        )
    return output, ledger


def merge_adjacent_duplicates(
    blocks: list[SubtitleBlock], max_gap: float = 0.12
) -> tuple[list[SubtitleBlock], list[dict[str, Any]]]:
    output: list[SubtitleBlock] = []
    ledger: list[dict[str, Any]] = []
    for block in blocks:
        if output:
            previous = output[-1]
            gap = block.start_seconds - previous.end_seconds
            if normalized(previous.text) == normalized(block.text) and gap <= max_gap:
                merged = with_times(
                    previous,
                    previous.start_seconds,
                    max(previous.end_seconds, block.end_seconds),
                )
                output[-1] = merged
                ledger.append(
                    {
                        "left_block": previous.number,
                        "right_block": block.number,
                        "gap_seconds": round(gap, 3),
                        "text": normalized(block.text),
                    }
                )
                continue
        output.append(block)
    return output, ledger


def split_chunks(text: str, limit: int = 84) -> list[str]:
    value = normalized(text)
    if len(value) <= limit:
        return [value]
    words = value.split()
    if len(words) == 1:
        return [value[index : index + limit] for index in range(0, len(value), limit)]
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > limit:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def wrap_two_lines(text: str, limit: int = 42) -> str:
    value = normalized(text)
    if len(value) <= limit:
        return value
    fitting = [
        index
        for index in range(1, len(value))
        if len(value[:index].rstrip()) <= limit
        and len(value[index:].lstrip()) <= limit
    ]
    preferred = [index for index in fitting if value[index - 1] in " ,.?!…"]
    pool = preferred or fitting
    if not pool:
        raise ValueError(f"Text cannot fit a two-line layout: {value!r}")
    split_at = min(
        pool,
        key=lambda index: (
            max(len(value[:index].rstrip()), len(value[index:].lstrip())),
            abs(len(value[:index].rstrip()) - len(value[index:].lstrip())),
        ),
    )
    return f"{value[:split_at].rstrip()}\n{value[split_at:].lstrip()}"


def normalize_layout(
    blocks: list[SubtitleBlock], *, line_limit: int
) -> tuple[list[SubtitleBlock], list[dict[str, Any]]]:
    output: list[SubtitleBlock] = []
    ledger: list[dict[str, Any]] = []
    for block in blocks:
        if len(block.lines) <= 2 and max(len(line) for line in block.lines) <= line_limit:
            output.append(block)
            continue
        chunks = split_chunks(block.text, limit=line_limit * 2)
        rendered = [wrap_two_lines(chunk, limit=line_limit) for chunk in chunks]
        duration = block.duration
        if duration / len(rendered) < 0.04:
            raise ValueError(
                f"Block {block.number} too short for {len(rendered)} layout chunks"
            )
        for index, text in enumerate(rendered):
            start = block.start_seconds + duration * index / len(rendered)
            end = block.start_seconds + duration * (index + 1) / len(rendered)
            output.append(with_times(replace(block, text=text), start, end))
        ledger.append(
            {
                "block": block.number,
                "old_lines": len(block.lines),
                "old_max_line": max(len(line) for line in block.lines),
                "new_cue_count": len(rendered),
            }
        )
    return output, ledger


def extend_micro_cues(
    blocks: list[SubtitleBlock], minimum: float = 0.5
) -> tuple[list[SubtitleBlock], list[dict[str, Any]]]:
    output: list[SubtitleBlock] = []
    ledger: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if block.duration >= 0.25:
            output.append(block)
            continue
        next_start = blocks[index + 1].start_seconds if index + 1 < len(blocks) else None
        proposed = block.start_seconds + minimum
        if next_start is not None:
            proposed = min(proposed, next_start)
        if proposed <= block.end_seconds + 0.001:
            output.append(block)
            continue
        repaired = with_times(block, block.start_seconds, proposed)
        output.append(repaired)
        ledger.append(
            {
                "block": block.number,
                "old_duration": round(block.duration, 3),
                "new_duration": round(repaired.duration, 3),
            }
        )
    return output, ledger


def qa(blocks: list[SubtitleBlock], *, line_limit: int) -> dict[str, Any]:
    invalid = [block.number for block in blocks if block.end_seconds <= block.start_seconds]
    out_of_order = [
        right.number
        for left, right in zip(blocks, blocks[1:])
        if right.start_seconds < left.start_seconds
    ]
    japanese = [block.number for block in blocks if JAPANESE_RE.search(block.text)]
    placeholders = [block.number for block in blocks if PLACEHOLDER in block.text]
    embedded = [block.number for block in blocks if "-->" in block.text]
    layout = [
        block.number
        for block in blocks
        if len(block.lines) > 2 or max(len(line) for line in block.lines) > line_limit
    ]
    return {
        "pass": not (invalid or out_of_order or japanese or placeholders or embedded or layout),
        "block_count": len(blocks),
        "invalid_duration_blocks": invalid,
        "out_of_order_blocks": out_of_order,
        "japanese_blocks": japanese,
        "placeholder_blocks": placeholders,
        "embedded_srt_blocks": embedded,
        "layout_violation_blocks": layout,
        "over_30_second_blocks": [block.number for block in blocks if block.duration > 30.0],
        "micro_cue_blocks": [block.number for block in blocks if block.duration < 0.25],
        "first_start": blocks[0].start if blocks else None,
        "last_end": blocks[-1].end if blocks else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize conservative corpus-wide SRT readability issues.")
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--line-limit", type=int, default=42)
    parser.add_argument("--duplicate-max-gap", type=float, default=0.12)
    parser.add_argument("--skip-duration-cap", action="store_true")
    parser.add_argument("--skip-duplicate-merge", action="store_true")
    parser.add_argument("--skip-micro-extension", action="store_true")
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for audit_row in sorted(audit["titles"], key=lambda row: row["title"].casefold()):
        title = audit_row["title"]
        path = Path(audit_row["srt"])
        digest = sha256(path)
        if digest != audit_row["sha256"]:
            raise ValueError(
                f"Active SHA-256 drifted for {title}: expected {audit_row['sha256']}, found {digest}"
            )
        blocks, _, _ = parse_srt(path)
        source_count = len(blocks)
        blocks, embedded_ledger = repair_embedded_srt(title, blocks)
        if args.skip_duration_cap:
            duration_ledger: list[dict[str, Any]] = []
        else:
            blocks, duration_ledger = cap_stale_durations(blocks)
        if args.skip_duplicate_merge:
            duplicate_ledger: list[dict[str, Any]] = []
        else:
            blocks, duplicate_ledger = merge_adjacent_duplicates(
                blocks, max_gap=args.duplicate_max_gap
            )
        blocks, layout_ledger = normalize_layout(blocks, line_limit=args.line_limit)
        if args.skip_micro_extension:
            micro_ledger: list[dict[str, Any]] = []
        else:
            blocks, micro_ledger = extend_micro_cues(blocks)
        blocks = [replace(block, number=index) for index, block in enumerate(blocks, 1)]
        result = qa(blocks, line_limit=args.line_limit)
        if not result["pass"]:
            raise ValueError(f"QA failed for {title}: {json.dumps(result, ensure_ascii=False)}")
        changed = bool(
            embedded_ledger
            or duration_ledger
            or duplicate_ledger
            or layout_ledger
            or micro_ledger
        )
        output = args.output_root / f"{title}.srt"
        output_hash = digest
        if changed:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(render_srt(blocks), encoding="utf-8-sig", newline="\n")
            output_hash = sha256(output)
        rows.append(
            {
                "title": title,
                "input": str(path),
                "input_sha256": digest,
                "output": str(output) if changed else None,
                "output_sha256": output_hash,
                "changed": changed,
                "source_block_count": source_count,
                "embedded_repair_count": len(embedded_ledger),
                "duration_cap_count": len(duration_ledger),
                "duplicate_merge_count": len(duplicate_ledger),
                "layout_repair_count": len(layout_ledger),
                "micro_extension_count": len(micro_ledger),
                "embedded_ledger": embedded_ledger,
                **result,
            }
        )

    report = {
        "schema_name": "translation-forensics/corpus-readability-normalization",
        "schema_version": 1,
        "audit": str(args.audit),
        "output_root": str(args.output_root),
        "line_limit": args.line_limit,
        "duration_cap_enabled": not args.skip_duration_cap,
        "duplicate_merge_enabled": not args.skip_duplicate_merge,
        "duplicate_max_gap_seconds": args.duplicate_max_gap,
        "micro_extension_enabled": not args.skip_micro_extension,
        "title_count": len(rows),
        "changed_title_count": sum(row["changed"] for row in rows),
        "embedded_repair_count": sum(row["embedded_repair_count"] for row in rows),
        "duration_cap_count": sum(row["duration_cap_count"] for row in rows),
        "duplicate_merge_count": sum(row["duplicate_merge_count"] for row in rows),
        "layout_repair_count": sum(row["layout_repair_count"] for row in rows),
        "micro_extension_count": sum(row["micro_extension_count"] for row in rows),
        "remaining_over_30_second_count": sum(len(row["over_30_second_blocks"]) for row in rows),
        "remaining_micro_cue_count": sum(len(row["micro_cue_blocks"]) for row in rows),
        "titles": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "titles"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
