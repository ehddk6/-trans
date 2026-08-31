from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
LOCAL_MEDIA_SUFFIXES = (".mp4", ".mkv", ".mov", ".m4v")


def evenly_spaced(values: list[int], limit: int) -> list[int]:
    if limit <= 0 or not values:
        return []
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    positions = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[index] for index in positions]


def selected_blocks(row: dict[str, Any], *, duplicate_sample: int) -> dict[int, list[str]]:
    reasons: dict[int, list[str]] = {}
    for block in row.get("long_semantic_cue_blocks", []):
        reasons.setdefault(int(block), []).append("long_semantic_cue")
    for block in row.get("semantic_micro_cue_blocks", []):
        reasons.setdefault(int(block), []).append("semantic_micro_cue")
    duplicates = [int(block) for block in row.get("adjacent_semantic_duplicate_blocks", [])]
    for block in evenly_spaced(duplicates, duplicate_sample):
        reasons.setdefault(block, []).append("sampled_adjacent_duplicate")
    return reasons


def direct_audio_files(folder: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.casefold() in AUDIO_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a hash-guarded, source-MP3 ASR queue from active subtitle risk findings."
    )
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--subtitles-root", required=True, type=Path)
    parser.add_argument("--titles", required=True, help="Comma-separated exact active title names.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--local-media-root",
        type=Path,
        help="Optional directory of local <title>.mp4 files used when a 작품 source MP3 is absent.",
    )
    parser.add_argument("--duplicate-sample", type=int, default=6)
    args = parser.parse_args()

    if args.duplicate_sample < 0:
        raise ValueError("--duplicate-sample must be non-negative")
    audit_path = args.audit.expanduser().resolve()
    subtitles_root = args.subtitles_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    ledger_path = args.ledger.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    local_media_root = (
        args.local_media_root.expanduser().resolve() if args.local_media_root else None
    )
    requested_titles = [title.strip() for title in args.titles.split(",") if title.strip()]
    if not requested_titles:
        raise ValueError("--titles must contain at least one title")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_rows = {str(row["title"]).casefold(): row for row in audit["titles"]}
    ledger_rows: list[dict[str, Any]] = []
    manifest_items: list[dict[str, str]] = []
    summary_rows: list[dict[str, Any]] = []
    for title in requested_titles:
        row = audit_rows.get(title.casefold())
        if row is None:
            raise ValueError(f"Title absent from audit: {title}")
        srt = subtitles_root / f"{row['title']}.srt"
        if not srt.is_file():
            raise FileNotFoundError(f"Active SRT not found: {srt}")
        project_folder_value = row.get("project_asset_folder")
        project_folder = Path(str(project_folder_value)) if project_folder_value else None
        audio_files = direct_audio_files(project_folder) if project_folder else []
        if len(audio_files) == 1:
            source_audio = audio_files[0]
            source_kind = "source_mp3"
        elif local_media_root:
            media_files = [
                local_media_root / f"{row['title']}{suffix}"
                for suffix in LOCAL_MEDIA_SUFFIXES
                if (local_media_root / f"{row['title']}{suffix}").is_file()
            ]
            if len(media_files) != 1:
                raise ValueError(
                    f"Expected exactly one local title media file for {row['title']}, found {len(media_files)}"
                )
            source_audio = media_files[0]
            source_kind = "local_video"
        elif project_folder is None:
            raise ValueError(f"No exact 작품 source folder for: {row['title']}")
        else:
            raise ValueError(
                f"Expected exactly one direct source audio file for {row['title']}, found {len(audio_files)}"
            )
        reasons = selected_blocks(row, duplicate_sample=args.duplicate_sample)
        if not reasons:
            continue
        file_name = srt.name
        for block, tags in sorted(reasons.items()):
            ledger_rows.append(
                {
                    "file": file_name,
                    "block": block,
                    "reason": ",".join(tags),
                    "source": f"active-subtitle-audit/{source_kind}",
                }
            )
        output_dir = output_root / str(row["title"]) / f"risk-asr-{source_kind}"
        manifest_items.append(
            {
                "title": str(row["title"]),
                "srt": str(srt),
                "audio": str(source_audio),
                "output_dir": str(output_dir),
                "file_name": file_name,
            }
        )
        summary_rows.append(
            {
                "title": row["title"],
                "risk_score": row["risk_score"],
                "selected_block_count": len(reasons),
                "source_audio": str(source_audio),
                "source_kind": source_kind,
                "reasons": {
                    tag: sum(tag in tags for tags in reasons.values())
                    for tag in ("long_semantic_cue", "semantic_micro_cue", "sampled_adjacent_duplicate")
                },
            }
        )

    if not manifest_items:
        raise ValueError("No reviewable blocks were selected")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ledger_rows),
        encoding="utf-8",
        newline="\n",
    )
    manifest_path.write_text(
        json.dumps({"items": manifest_items}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    summary_path = output_root / "queue-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "audit": str(audit_path),
                "title_count": len(manifest_items),
                "block_count": len(ledger_rows),
                "titles": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "title_count": len(manifest_items),
                "block_count": len(ledger_rows),
                "ledger": str(ledger_path),
                "manifest": str(manifest_path),
                "summary": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
