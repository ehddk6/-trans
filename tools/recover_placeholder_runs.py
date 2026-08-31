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
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def placeholder_runs(blocks: list[SubtitleBlock], marker: str) -> list[list[SubtitleBlock]]:
    selected = [block for block in blocks if marker in block.text]
    runs: list[list[SubtitleBlock]] = []
    for block in selected:
        if not runs or block.number != runs[-1][-1].number + 1:
            runs.append([block])
        else:
            runs[-1].append(block)
    return runs


def overlap_seconds(left_start: float, left_end: float, right: SubtitleBlock) -> float:
    return max(0.0, min(left_end, right.end_seconds) - max(left_start, right.start_seconds))


def recover(
    base: list[SubtitleBlock],
    source: list[SubtitleBlock],
    *,
    marker: str,
    unresolved_overrides: dict[int, str],
) -> tuple[list[SubtitleBlock], list[dict[str, Any]]]:
    runs = placeholder_runs(base, marker)
    removed = {block.number for run in runs for block in run}
    retained = [block for block in base if block.number not in removed]
    inserted: list[SubtitleBlock] = []
    ledger: list[dict[str, Any]] = []

    for run_index, run in enumerate(runs, 1):
        start_seconds = run[0].start_seconds
        end_seconds = run[-1].end_seconds
        source_rows = [
            block
            for block in source
            if overlap_seconds(start_seconds, end_seconds, block) > 0
            and (
                start_seconds <= (block.start_seconds + block.end_seconds) / 2 <= end_seconds
                or overlap_seconds(start_seconds, end_seconds, block) / max(block.duration, 0.001) >= 0.5
            )
        ]
        recovered_numbers: list[int] = []
        for source_block in source_rows:
            clipped_start = max(start_seconds, source_block.start_seconds)
            clipped_end = min(end_seconds, source_block.end_seconds)
            if clipped_end <= clipped_start or not source_block.text.strip():
                continue
            recovered_numbers.append(source_block.number)
            inserted.append(
                SubtitleBlock(
                    number=source_block.number,
                    start=seconds_to_timecode(clipped_start),
                    end=seconds_to_timecode(clipped_end),
                    text=source_block.text.strip(),
                    start_seconds=clipped_start,
                    end_seconds=clipped_end,
                )
            )
        fallback_text = ""
        if not recovered_numbers and run_index in unresolved_overrides:
            fallback_text = unresolved_overrides[run_index].strip()
            if not fallback_text:
                raise ValueError(f"Fallback text for run {run_index} is empty")
            inserted.append(
                SubtitleBlock(
                    number=run[0].number,
                    start=run[0].start,
                    end=run[-1].end,
                    text=fallback_text,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                )
            )
        ledger.append(
            {
                "run": run_index,
                "base_start_block": run[0].number,
                "base_end_block": run[-1].number,
                "start": run[0].start,
                "end": run[-1].end,
                "removed_placeholder_blocks": [block.number for block in run],
                "inserted_source_blocks": recovered_numbers,
                "fallback_text": fallback_text,
                "removed_count": len(run),
                "inserted_count": len(recovered_numbers) + bool(fallback_text),
            }
        )

    combined = retained + inserted
    combined.sort(key=lambda block: (block.start_seconds, block.end_seconds, block.number))
    renumbered = [replace(block, number=index) for index, block in enumerate(combined, 1)]
    return renumbered, ledger


def qa_report(
    base: list[SubtitleBlock],
    candidate: list[SubtitleBlock],
    ledger: list[dict[str, Any]],
    *,
    marker: str,
) -> dict[str, Any]:
    text = "\n".join(block.text for block in candidate)
    empty = [block.number for block in candidate if not block.text.strip()]
    invalid = [block.number for block in candidate if block.end_seconds <= block.start_seconds]
    out_of_order = [
        right.number
        for left, right in zip(candidate, candidate[1:])
        if right.start_seconds < left.start_seconds
    ]
    overlong = [
        block.number
        for block in candidate
        if max(len(line) for line in block.lines) > 42
    ]
    too_many_lines = [block.number for block in candidate if len(block.lines) > 2]
    adjacent_duplicates = [
        right.number
        for left, right in zip(candidate, candidate[1:])
        if left.text.strip() == right.text.strip()
    ]
    marker_count = text.count(marker)
    japanese_count = len(JAPANESE_RE.findall(text))
    passed = not (marker_count or japanese_count or empty or invalid or out_of_order)
    return {
        "pass": passed,
        "base_block_count": len(base),
        "output_block_count": len(candidate),
        "placeholder_run_count": len(ledger),
        "removed_placeholder_block_count": sum(row["removed_count"] for row in ledger),
        "inserted_source_block_count": sum(row["inserted_count"] for row in ledger),
        "remaining_placeholder_count": marker_count,
        "japanese_character_count": japanese_count,
        "empty_blocks": empty,
        "invalid_duration_blocks": invalid,
        "out_of_order_blocks": out_of_order,
        "overlong_line_count": len(overlong),
        "overlong_line_blocks": overlong[:100],
        "too_many_lines_count": len(too_many_lines),
        "too_many_lines_blocks": too_many_lines[:100],
        "adjacent_duplicate_count": len(adjacent_duplicates),
        "adjacent_duplicate_blocks": adjacent_duplicates[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace contiguous placeholder runs with aligned source Korean cues.")
    parser.add_argument("--base-srt", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--source-ko", required=True, type=Path)
    parser.add_argument("--marker", default="[안전상 번역 불가]")
    parser.add_argument(
        "--unresolved-overrides",
        type=Path,
        help="Optional JSON object mapping 1-based run numbers to fallback Korean text.",
    )
    parser.add_argument("--output-srt", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--qa-report", required=True, type=Path)
    args = parser.parse_args()

    base_sha = sha256(args.base_srt)
    if base_sha != args.base_sha256.casefold():
        raise ValueError(f"Base SRT SHA-256 drifted: expected {args.base_sha256}, found {base_sha}")
    base, base_encoding, base_newline = parse_srt(args.base_srt)
    source, _, _ = parse_srt(args.source_ko)
    unresolved_overrides: dict[int, str] = {}
    if args.unresolved_overrides:
        raw_overrides = json.loads(args.unresolved_overrides.read_text(encoding="utf-8"))
        unresolved_overrides = {int(key): str(value) for key, value in raw_overrides.items()}
    candidate, ledger = recover(
        base,
        source,
        marker=args.marker,
        unresolved_overrides=unresolved_overrides,
    )
    report = qa_report(base, candidate, ledger, marker=args.marker)
    if not report["pass"]:
        raise ValueError(f"QA failed: {json.dumps(report, ensure_ascii=False)}")

    args.output_srt.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.qa_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_srt.write_text(render_srt(candidate), encoding="utf-8-sig", newline="\n")
    args.ledger.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ledger),
        encoding="utf-8",
        newline="\n",
    )
    report.update(
        {
            "base_srt": str(args.base_srt),
            "base_sha256": base_sha,
            "base_encoding": base_encoding,
            "base_newline": base_newline,
            "source_ko": str(args.source_ko),
            "source_ko_sha256": sha256(args.source_ko),
            "output_srt": str(args.output_srt),
            "output_sha256": sha256(args.output_srt),
            "ledger": str(args.ledger),
        }
    )
    args.qa_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
