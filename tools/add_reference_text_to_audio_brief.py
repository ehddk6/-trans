from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from translation_forensics.srt import SubtitleBlock, parse_srt


def seconds(block: SubtitleBlock) -> tuple[float, float]:
    return block.start_seconds, block.end_seconds


def overlaps(start: float, end: float, block: SubtitleBlock) -> float:
    block_start, block_end = seconds(block)
    return max(0.0, min(end, block_end) - max(start, block_start))


def parse_srt_time(value: str) -> float:
    hours, minutes, seconds_and_millis = value.split(":")
    seconds, millis = seconds_and_millis.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def nearest_reference_blocks(
    blocks: list[SubtitleBlock], *, start: float, end: float
) -> list[SubtitleBlock]:
    matches = [block for block in blocks if overlaps(start, end, block) > 0]
    if matches:
        return matches
    midpoint = (start + end) / 2
    return [min(blocks, key=lambda block: abs(((block.start_seconds + block.end_seconds) / 2) - midpoint))]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach overlapping Videos Japanese reference subtitle text to compact source-MP3 ASR evidence."
    )
    parser.add_argument("--brief", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    brief_path = args.brief.expanduser().resolve()
    reference_root = args.reference_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report: dict[str, Any] = json.loads(brief_path.read_text(encoding="utf-8"))
    missing_reference_titles = []
    for title_row in report["titles"]:
        title = str(title_row["title"])
        reference_path = reference_root / title / f"{title}.viewer_ja.srt"
        if not reference_path.is_file():
            missing_reference_titles.append(title)
            title_row["reference_srt"] = None
            continue
        blocks, _, _ = parse_srt(reference_path)
        title_row["reference_srt"] = str(reference_path)
        for candidate in title_row["candidates"]:
            reference_blocks = nearest_reference_blocks(
                blocks,
                start=parse_srt_time(str(candidate["start"])),
                end=parse_srt_time(str(candidate["end"])),
            )
            candidate["reference_blocks"] = [
                {
                    "block": block.number,
                    "start": block.start,
                    "end": block.end,
                    "text": block.text,
                }
                for block in reference_blocks
            ]
            candidate["reference_text"] = " ".join(block.text.replace("\n", " ") for block in reference_blocks)
    report["reference_root"] = str(reference_root)
    report["missing_reference_titles"] = missing_reference_titles
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "title_count": report["title_count"],
                "reliable_candidate_count": report["reliable_candidate_count"],
                "missing_reference_titles": missing_reference_titles,
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
