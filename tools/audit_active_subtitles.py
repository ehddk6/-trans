from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from translation_forensics.srt import JAPANESE_RE, SRTError, parse_srt


BOILERPLATE = (
    "시청해 주셔서 감사합니다",
    "시청해주셔서 감사합니다",
    "봐 주셔서 감사합니다",
    "다음 영상에서 만나요",
)
REFUSAL_PLACEHOLDER = "[안전상 번역 불가]"
NON_SEMANTIC_RE = re.compile(
    r"^[\s.…·!！?？,，~〜ー\-]*(?:아|앗|하|핫|후|흐|응|음|으|어|오|와|야|에|헤|흠|읏|헉|하아|아아|으응|후훗|후후|하하)+[\s.…·!！?？,，~〜ー\-]*$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def semantic(text: str) -> bool:
    value = normalized(text)
    if not value or value in {"…", "...", "-"}:
        return False
    return not NON_SEMANTIC_RE.fullmatch(value)


def audit_file(
    path: Path,
    *,
    image_count: int,
    audio_count: int,
    project_audio_count: int = 0,
    project_asset_folder: Path | None = None,
) -> dict[str, Any]:
    total_audio_count = audio_count + project_audio_count
    try:
        blocks, encoding, newline = parse_srt(path)
    except SRTError as exc:
        return {
            "title": path.stem,
            "srt": str(path),
            "sha256": sha256(path),
            "parse_error": str(exc),
            "image_count": image_count,
            "audio_count": total_audio_count,
            "evidence_image_count": image_count,
            "evidence_audio_count": audio_count,
            "project_audio_count": project_audio_count,
            "project_asset_folder": str(project_asset_folder) if project_asset_folder else None,
            "risk_score": 10_000,
        }

    text = "\n".join(block.text for block in blocks)
    semantic_counts = Counter(normalized(block.text) for block in blocks if semantic(block.text))
    repeated = [
        {"text": value, "count": count}
        for value, count in semantic_counts.most_common()
        if count >= 4
    ][:20]
    boilerplate = [
        {"phrase": phrase, "count": text.count(phrase)}
        for phrase in BOILERPLATE
        if phrase in text
    ]
    invalid = [block.number for block in blocks if block.end_seconds <= block.start_seconds]
    micro = [
        block.number
        for block in blocks
        if block.duration < 0.25 and semantic(block.text)
    ]
    long_semantic = [
        block.number
        for block in blocks
        if block.duration > 30 and semantic(block.text)
    ]
    overlong_lines = [
        block.number
        for block in blocks
        if max(len(line) for line in block.lines) > 32
    ]
    too_many_lines = [block.number for block in blocks if len(block.lines) > 2]
    adjacent_duplicates = [
        right.number
        for left, right in zip(blocks, blocks[1:])
        if semantic(left.text) and normalized(left.text) == normalized(right.text)
    ]
    out_of_order = [
        right.number
        for left, right in zip(blocks, blocks[1:])
        if right.start_seconds < left.start_seconds
    ]
    overlaps = [
        right.number
        for left, right in zip(blocks, blocks[1:])
        if right.start_seconds < left.end_seconds
    ]
    japanese_count = len(JAPANESE_RE.findall(text))
    refusal_placeholder_count = text.count(REFUSAL_PLACEHOLDER)
    repeated_excess = sum(max(0, row["count"] - 3) for row in repeated)
    score = (
        japanese_count * 20
        + refusal_placeholder_count * 20
        + sum(row["count"] for row in boilerplate) * 12
        + len(invalid) * 5
        + len(out_of_order) * 8
        + len(overlaps) * 2
        + len(micro) * 3
        + len(adjacent_duplicates) * 3
        + len(long_semantic) * 2
        + repeated_excess
        + len(overlong_lines)
        + len(too_many_lines) * 2
    )
    return {
        "title": path.stem,
        "srt": str(path),
        "sha256": sha256(path),
        "encoding": encoding,
        "newline": newline,
        "block_count": len(blocks),
        "image_count": image_count,
        "audio_count": total_audio_count,
        "evidence_image_count": image_count,
        "evidence_audio_count": audio_count,
        "project_audio_count": project_audio_count,
        "project_asset_folder": str(project_asset_folder) if project_asset_folder else None,
        "japanese_character_count": japanese_count,
        "refusal_placeholder_count": refusal_placeholder_count,
        "boilerplate": boilerplate,
        "invalid_duration_count": len(invalid),
        "invalid_duration_blocks": invalid[:100],
        "semantic_micro_cue_count": len(micro),
        "semantic_micro_cue_blocks": micro[:100],
        "long_semantic_cue_count": len(long_semantic),
        "long_semantic_cue_blocks": long_semantic[:100],
        "overlong_line_count": len(overlong_lines),
        "overlong_line_blocks": overlong_lines[:100],
        "too_many_lines_count": len(too_many_lines),
        "too_many_lines_blocks": too_many_lines[:100],
        "adjacent_semantic_duplicate_count": len(adjacent_duplicates),
        "adjacent_semantic_duplicate_blocks": adjacent_duplicates[:100],
        "out_of_order_count": len(out_of_order),
        "out_of_order_blocks": out_of_order[:100],
        "overlap_count": len(overlaps),
        "overlap_blocks": overlaps[:100],
        "repeated_semantic_text": repeated,
        "risk_score": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit active SRT files that have matching work asset folders.")
    parser.add_argument("--subtitles-root", required=True, type=Path)
    parser.add_argument("--assets-root", required=True, type=Path)
    parser.add_argument(
        "--project-assets-root",
        type=Path,
        help="Optional original 작품 asset root. Exact title folders contribute source image/audio counts.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    subtitles_root = args.subtitles_root.expanduser().resolve()
    assets_root = args.assets_root.expanduser().resolve()
    project_assets_root = (
        args.project_assets_root.expanduser().resolve() if args.project_assets_root else None
    )
    srt_by_title = {path.stem.casefold(): path for path in subtitles_root.glob("*.srt")}
    project_assets: dict[str, tuple[Path, int]] = {}
    if project_assets_root:
        audio_suffixes = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
        rg_command = ["rg", "--files", str(project_assets_root)]
        for suffix in sorted(audio_suffixes):
            rg_command.extend(("-g", f"*{suffix}"))
        try:
            result = subprocess.run(
                rg_command,
                check=False,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
            )
            audio_paths = [Path(line) for line in result.stdout.splitlines()]
        except FileNotFoundError:
            audio_paths = [
                path
                for suffix in audio_suffixes
                for path in project_assets_root.rglob(f"*{suffix}")
            ]
        for audio_path in audio_paths:
            if not audio_path.is_file():
                continue
            title = audio_path.stem.casefold()
            folder = audio_path.parent
            if title not in srt_by_title:
                matched_title = None
                for ancestor in audio_path.parents:
                    if ancestor == project_assets_root:
                        break
                    candidate = ancestor.name.casefold()
                    if candidate in srt_by_title:
                        matched_title = candidate
                        folder = ancestor
                        break
                if matched_title is None:
                    continue
                title = matched_title
            previous = project_assets.get(title)
            project_assets[title] = (
                folder,
                (previous[1] if previous else 0) + 1,
            )
    rows: list[dict[str, Any]] = []
    for folder in sorted((path for path in assets_root.iterdir() if path.is_dir()), key=lambda path: path.name.casefold()):
        srt = srt_by_title.get(folder.name.casefold())
        if srt is None:
            continue
        files = [path for path in folder.rglob("*") if path.is_file()]
        image_count = sum(path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp"} for path in files)
        audio_count = sum(path.suffix.casefold() in {".mp3", ".wav", ".m4a", ".aac", ".flac"} for path in files)
        project_asset = project_assets.get(folder.name.casefold())
        rows.append(
            audit_file(
                srt,
                image_count=image_count,
                audio_count=audio_count,
                project_audio_count=project_asset[1] if project_asset else 0,
                project_asset_folder=project_asset[0] if project_asset else None,
            )
        )

    rows.sort(key=lambda row: (-int(row["risk_score"]), str(row["title"]).casefold()))
    report = {
        "schema_name": "translation-forensics/active-subtitle-risk-audit",
        "schema_version": "1",
        "subtitles_root": str(subtitles_root),
        "assets_root": str(assets_root),
        "project_assets_root": str(project_assets_root) if project_assets_root else None,
        "title_count": len(rows),
        "parse_error_count": sum("parse_error" in row for row in rows),
        "japanese_character_count": sum(int(row.get("japanese_character_count", 0)) for row in rows),
        "refusal_placeholder_count": sum(int(row.get("refusal_placeholder_count", 0)) for row in rows),
        "invalid_duration_count": sum(int(row.get("invalid_duration_count", 0)) for row in rows),
        "out_of_order_count": sum(int(row.get("out_of_order_count", 0)) for row in rows),
        "overlap_count": sum(int(row.get("overlap_count", 0)) for row in rows),
        "image_count": sum(int(row.get("image_count", 0)) for row in rows),
        "audio_count": sum(int(row.get("audio_count", 0)) for row in rows),
        "evidence_image_count": sum(int(row.get("evidence_image_count", 0)) for row in rows),
        "evidence_audio_count": sum(int(row.get("evidence_audio_count", 0)) for row in rows),
        "project_audio_count": sum(int(row.get("project_audio_count", 0)) for row in rows),
        "project_audio_title_count": sum(
            int(row.get("project_audio_count", 0)) > 0 for row in rows
        ),
        "missing_audio_titles": [row["title"] for row in rows if int(row["audio_count"]) == 0],
        "titles": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "title_count": report["title_count"],
                "parse_error_count": report["parse_error_count"],
                "japanese_character_count": report["japanese_character_count"],
                "refusal_placeholder_count": report["refusal_placeholder_count"],
                "invalid_duration_count": report["invalid_duration_count"],
                "out_of_order_count": report["out_of_order_count"],
                "overlap_count": report["overlap_count"],
                "image_count": report["image_count"],
                "audio_count": report["audio_count"],
                "project_audio_count": report["project_audio_count"],
                "project_audio_title_count": report["project_audio_title_count"],
                "missing_audio_titles": report["missing_audio_titles"],
                "top_risk": [
                    {"title": row["title"], "risk_score": row["risk_score"]}
                    for row in rows[:20]
                ],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
