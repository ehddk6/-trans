from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .srt import SubtitleBlock, seconds_to_timecode, timecode_to_seconds


REVIEW_FIELDS = [
    "block_number", "timecode", "review_band", "source_japanese", "provisional_korean",
    "confidence", "uncertain_scope", "required_evidence", "forensics_risk", "priority_action",
]
SCENE_FIELDS = [
    "scene_id", "start_time", "end_time", "duration_sec", "block_numbers", "highest_band",
    "original_audio", "dialogue_audio", "context_japanese", "blocks_json",
]


@dataclass
class ReviewScene:
    scene_id: str
    start_sec: float
    end_sec: float
    blocks: list[dict[str, str]]
    highest_band: str
    original_audio: str = ""
    dialogue_audio: str = ""

    def row(self) -> dict[str, str]:
        return {
            "scene_id": self.scene_id,
            "start_time": seconds_to_timecode(self.start_sec),
            "end_time": seconds_to_timecode(self.end_sec),
            "duration_sec": f"{self.end_sec - self.start_sec:.3f}",
            "block_numbers": ",".join(str(block.get("block_number", "")) for block in self.blocks),
            "highest_band": self.highest_band,
            "original_audio": self.original_audio,
            "dialogue_audio": self.dialogue_audio,
            "context_japanese": " / ".join(block.get("source_japanese", "") for block in self.blocks if block.get("source_japanese", "").strip()),
            "blocks_json": json.dumps(self.blocks, ensure_ascii=False),
        }


def read_review_queue(path: Path, *, bands: set[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = {"block_number", "timecode", "review_band"} - fields
        if missing:
            raise ValueError(f"review queue 필수 열 누락: {', '.join(sorted(missing))}")
        rows = []
        for row in reader:
            if bands and row.get("review_band", "") not in bands:
                continue
            start, end = [part.strip() for part in row["timecode"].split("-->", 1)]
            timecode_to_seconds(start)
            timecode_to_seconds(end)
            if int(row["block_number"]) < 1:
                raise ValueError("block_number는 1 이상이어야 합니다.")
            rows.append(dict(row))
    return sorted(rows, key=lambda row: (timecode_to_seconds(row["timecode"].split("-->", 1)[0]), int(row["block_number"])))


def _band_rank(value: str) -> int:
    return int(value[1:]) if value.startswith("P") and value[1:].isdigit() else 99


def build_review_scenes(rows: list[dict[str, str]], *, padding: float = 2.5, merge_gap: float = 1.5, max_scene: float = 45.0) -> list[ReviewScene]:
    scenes: list[ReviewScene] = []
    for row in rows:
        start, end = [part.strip() for part in row["timecode"].split("-->", 1)]
        start_sec = max(0.0, timecode_to_seconds(start) - padding)
        end_sec = timecode_to_seconds(end) + padding
        if not scenes or start_sec - scenes[-1].end_sec > merge_gap or end_sec - scenes[-1].start_sec > max_scene:
            scenes.append(ReviewScene(f"S{len(scenes) + 1:04d}", start_sec, end_sec, [row], row.get("review_band", "P4")))
        else:
            current = scenes[-1]
            current.end_sec = max(current.end_sec, end_sec)
            current.blocks.append(row)
            current.highest_band = min(current.highest_band, row.get("review_band", "P4"), key=_band_rank)
    return scenes


def write_review_scenes(path: Path, scenes: list[ReviewScene]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCENE_FIELDS)
        writer.writeheader()
        writer.writerows(scene.row() for scene in scenes)


def scenes_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def scene_rows_for_blocks(blocks: list[SubtitleBlock], rows: list[dict[str, str]]) -> list[ReviewScene]:
    by_number = {block.number: block for block in blocks}
    scenes: list[ReviewScene] = []
    for row in rows:
        block_numbers = [int(x) for x in row.get("block_numbers", "").split(",") if x.strip().isdigit()]
        selected = [by_number[number] for number in block_numbers if number in by_number]
        if not selected:
            continue
        raw_blocks = [{"block_number": str(block.number), "timecode": f"{block.start} --> {block.end}", "source_japanese": block.text} for block in selected]
        scenes.append(ReviewScene(row["scene_id"], timecode_to_seconds(row["start_time"]), timecode_to_seconds(row["end_time"]), raw_blocks, row.get("highest_band", "P4"), row.get("original_audio", ""), row.get("dialogue_audio", "")))
    return scenes
