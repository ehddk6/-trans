from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from translation_forensics.srt import SubtitleBlock, compare_structure, parse_srt


BANNED_SOURCE_JA = (
    "ご視聴ありがとうございました",
    "次の動画でお会いしましょう",
    "おやすみなさい",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_japanese(text: str) -> str:
    return "".join(character for character in text if character.isalnum())


def similarity(left: str, right: str) -> float:
    left = normalize_japanese(left)
    right = normalize_japanese(right)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def overlap_seconds(left: SubtitleBlock, right: SubtitleBlock) -> float:
    return max(0.0, min(left.end_seconds, right.end_seconds) - max(left.start_seconds, right.start_seconds))


def source_overlap(
    target: SubtitleBlock,
    source_ja: list[SubtitleBlock],
    source_ko: list[SubtitleBlock],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ja, ko in zip(source_ja, source_ko):
        overlap = overlap_seconds(target, ja)
        if overlap <= 0:
            continue
        rows.append(
            {
                "block": ja.number,
                "start": ja.start,
                "end": ja.end,
                "overlap_seconds": round(overlap, 3),
                "ja": ja.text,
                "ko": ko.text,
                "banned_boilerplate": any(pattern in ja.text for pattern in BANNED_SOURCE_JA),
            }
        )
    return rows


def frame_paths(
    block: SubtitleBlock,
    *,
    photos_dir: Path,
    frame_count: int,
    frame_interval: float,
) -> list[dict[str, Any]]:
    times = [block.start_seconds, (block.start_seconds + block.end_seconds) / 2.0, block.end_seconds]
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for timestamp in times:
        index = max(1, min(frame_count, int(timestamp // frame_interval) + 1))
        if index in seen:
            continue
        seen.add(index)
        path = photos_dir / f"frame_{index:05d}.jpg"
        rows.append(
            {
                "timestamp_seconds": round(timestamp, 3),
                "frame_index": index,
                "path": str(path),
                "exists": path.is_file(),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Join subtitle, source-pair, ASR, and frame evidence.")
    parser.add_argument("--original-srt", required=True, type=Path)
    parser.add_argument("--improved-srt", required=True, type=Path)
    parser.add_argument("--source-ja", required=True, type=Path)
    parser.add_argument("--source-ko", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--asr-evidence", required=True, type=Path)
    parser.add_argument("--photos-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frame-interval", type=float, default=5.0)
    args = parser.parse_args()

    original, _, _ = parse_srt(args.original_srt)
    improved, _, _ = parse_srt(args.improved_srt)
    source_ja, _, _ = parse_srt(args.source_ja)
    source_ko, _, _ = parse_srt(args.source_ko)
    if not compare_structure(original, improved)["pass"]:
        raise ValueError("Original and improved SRT structure differs")
    if not compare_structure(source_ja, source_ko)["pass"]:
        raise ValueError("Source JA/KO structure differs")

    ledger = {
        int(row["block"]): row
        for row in load_jsonl(args.ledger)
        if str(row.get("file")) == args.original_srt.name
    }
    asr = {int(row["block"]): row for row in load_jsonl(args.asr_evidence)}
    improved_by_number = {block.number: block for block in improved}
    photos = sorted(args.photos_dir.glob("frame_*.jpg"))
    frame_count = len(photos)
    if not frame_count:
        raise ValueError(f"No frames found in {args.photos_dir}")

    review_rows: list[dict[str, Any]] = []
    for block in original:
        if block.number not in ledger:
            continue
        source_rows = source_overlap(block, source_ja, source_ko)
        source_text = " ".join(row["ja"] for row in source_rows if not row["banned_boilerplate"])
        asr_row = asr.get(block.number, {})
        asr_text = str(asr_row.get("eligible_asr_text") or "")
        source_asr_similarity = similarity(source_text, asr_text)
        original_asr_similarity = similarity(block.text, asr_text)
        risk_flags: list[str] = []
        if not asr_text:
            risk_flags.append("asr-missing")
        if asr_text and max(source_asr_similarity, original_asr_similarity) < 0.45:
            risk_flags.append("asr-text-conflict")
        if any(row["banned_boilerplate"] for row in source_rows):
            risk_flags.append("source-boilerplate-overlap")
        if block.duration > 20:
            risk_flags.append("long-merged-cue")
        if str(ledger[block.number].get("method")) == "manual-text-override":
            risk_flags.append("previous-manual-override")
        review_rows.append(
            {
                "block": block.number,
                "start": block.start,
                "end": block.end,
                "duration_seconds": round(block.duration, 3),
                "original_ja": block.text,
                "previous_ko": improved_by_number[block.number].text,
                "previous_method": ledger[block.number].get("method"),
                "previous_reason": ledger[block.number].get("reason", ""),
                "source_overlap": source_rows,
                "source_ja_combined": source_text,
                "source_ko_combined": " ".join(
                    row["ko"] for row in source_rows if not row["banned_boilerplate"]
                ),
                "asr_ja": asr_text,
                "asr_segments": asr_row.get("asr_segments", []),
                "original_asr_similarity": round(original_asr_similarity, 4),
                "source_asr_similarity": round(source_asr_similarity, 4),
                "risk_flags": risk_flags,
                "frames": frame_paths(
                    block,
                    photos_dir=args.photos_dir,
                    frame_count=frame_count,
                    frame_interval=args.frame_interval,
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = args.output_dir / "multimodal-review.jsonl"
    packet_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review_rows),
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "schema_name": "translation-forensics/multimodal-review",
        "schema_version": "1",
        "review_block_count": len(review_rows),
        "frame_count": frame_count,
        "frame_interval_seconds": args.frame_interval,
        "frame_timeline_note": "frame_00001=approximately 0s; index advances at approximately 5s",
        "blocks_with_asr": sum(bool(row["asr_ja"]) for row in review_rows),
        "blocks_without_asr": sum(not bool(row["asr_ja"]) for row in review_rows),
        "asr_text_conflicts": sum("asr-text-conflict" in row["risk_flags"] for row in review_rows),
        "long_merged_cues": sum("long-merged-cue" in row["risk_flags"] for row in review_rows),
        "manual_override_rows": sum("previous-manual-override" in row["risk_flags"] for row in review_rows),
        "all_frame_paths_exist": all(frame["exists"] for row in review_rows for frame in row["frames"]),
        "output": str(packet_path),
    }
    (args.output_dir / "multimodal-review-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
