from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transcribe_risk_windows import audio_duration  # noqa: E402
from translation_forensics.srt import parse_srt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare each manifest source audio duration to its active SRT time axis."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", help="Restrict the report to one exact manifest title.")
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    items = manifest["items"]
    if args.title:
        items = [item for item in items if item["title"].casefold() == args.title.casefold()]
        if not items:
            raise ValueError(f"Title absent from manifest: {args.title}")
    for item in items:
        srt = Path(item["srt"]).expanduser().resolve()
        audio = Path(item["audio"]).expanduser().resolve()
        blocks, _, _ = parse_srt(srt)
        if not blocks:
            raise ValueError(f"No SRT blocks: {srt}")
        source_duration = audio_duration(audio, ffmpeg=ffmpeg)
        subtitle_end = blocks[-1].end_seconds
        rows.append(
            {
                "title": item["title"],
                "srt": str(srt),
                "audio": str(audio),
                "audio_duration_seconds": round(source_duration, 3),
                "srt_end_seconds": round(subtitle_end, 3),
                "duration_delta_seconds": round(source_duration - subtitle_end, 3),
                "within_30_seconds": abs(source_duration - subtitle_end) <= 30.0,
            }
        )
    report = {
        "manifest": str(manifest_path),
        "ffmpeg": str(ffmpeg),
        "title_count": len(rows),
        "within_30_seconds_count": sum(row["within_30_seconds"] for row in rows),
        "titles": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
