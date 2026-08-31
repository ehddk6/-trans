from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


_TIMECODE_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2}),(\d{3})")
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subtitle_duration_seconds(subtitle: Path) -> float:
    text = subtitle.read_text(encoding="utf-8-sig", errors="replace")
    values: list[float] = []
    for hours, minutes, seconds, milliseconds in _TIMECODE_RE.findall(text):
        if int(minutes) >= 60 or int(seconds) >= 60:
            continue
        values.append(
            int(hours) * 3600
            + int(minutes) * 60
            + int(seconds)
            + int(milliseconds) / 1000
        )
    if not values:
        raise ValueError(f"Subtitle has no usable timecodes: {subtitle}")
    return max(values)


def _run(command: list[str], output: Path) -> None:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(
            f"Evidence extraction failed for {output}: {completed.stderr[-500:]}"
        )


def _valid_frame(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size >= 1024
        and path.read_bytes()[:2] == b"\xff\xd8"
    )


def _valid_audio(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 44:
        return False
    header = path.read_bytes()[:12]
    return header[:4] == b"RIFF" and header[8:12] == b"WAVE"


def _source_candidates(active_root: Path, title: str, canonical_srt: Path) -> list[Path]:
    candidates: list[tuple[int, int, Path]] = []
    for path in active_root.glob("*.srt"):
        if path == canonical_srt or title.casefold() not in path.stem.casefold():
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        japanese = len(_JAPANESE_RE.findall(text))
        if japanese:
            candidates.append((japanese, path.stat().st_size, path))
    return [row[2] for row in sorted(candidates, reverse=True)]


def _create_title_assets(
    *,
    title: str,
    active_root: Path,
    assets_root: Path,
    ffmpeg: Path,
    force: bool,
    register_only: bool,
    register_reason: str,
) -> dict[str, Any]:
    media = active_root / f"{title}.mp4"
    subtitle = active_root / f"{title}.srt"
    if not media.is_file():
        raise FileNotFoundError(f"Missing media for {title}: {media}")
    if not subtitle.is_file():
        raise FileNotFoundError(f"Missing subtitle for {title}: {subtitle}")

    title_root = assets_root / title
    title_root.mkdir(parents=True, exist_ok=True)
    duration = _subtitle_duration_seconds(subtitle)
    if duration < 30:
        raise ValueError(f"Media is too short for representative evidence: {media}")
    source_candidates = _source_candidates(active_root, title, subtitle)
    copied_source: Path | None = None
    if source_candidates:
        copied_source = title_root / f"{title}.source_ja.srt"
        if force or not copied_source.exists():
            shutil.copy2(source_candidates[0], copied_source)

    if register_only:
        report = {
            "schema_name": "title-evidence-assets",
            "schema_version": 1,
            "status": "registered-without-extraction",
            "reason": register_reason,
            "title": title,
            "media": str(media),
            "subtitle": str(subtitle),
            "subtitle_timeline_seconds": round(duration, 3),
            "source_japanese": str(copied_source) if copied_source else None,
            "source_japanese_candidates": [str(path) for path in source_candidates],
            "assets": [],
        }
        manifest = title_root / "evidence-manifest.json"
        manifest.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        report["manifest"] = str(manifest)
        return report

    evidence_root = title_root / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    sample_points = [
        ("opening", max(5.0, duration * 0.10)),
        ("middle", duration * 0.50),
        ("closing", min(duration - 5.0, duration * 0.90)),
    ]

    assets: list[dict[str, Any]] = []
    for label, center in sample_points:
        frame = evidence_root / f"{title}.{label}.jpg"
        audio = evidence_root / f"{title}.{label}.wav"
        if force or not _valid_frame(frame):
            _run(
                [
                    str(ffmpeg),
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{center:.3f}",
                    "-i",
                    str(media),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale='min(1280,iw)':-2",
                    "-q:v",
                    "3",
                    "-y",
                    str(frame),
                ],
                frame,
            )
        audio_start = max(0.0, center - 7.5)
        if force or not _valid_audio(audio):
            _run(
                [
                    str(ffmpeg),
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{audio_start:.3f}",
                    "-i",
                    str(media),
                    "-t",
                    "15",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    "-y",
                    str(audio),
                ],
                audio,
            )
        assets.extend(
            [
                {
                    "kind": "frame",
                    "label": label,
                    "center_seconds": round(center, 3),
                    "path": str(frame),
                    "sha256": _sha256(frame),
                    "bytes": frame.stat().st_size,
                },
                {
                    "kind": "audio",
                    "label": label,
                    "start_seconds": round(audio_start, 3),
                    "duration_seconds": 15.0,
                    "path": str(audio),
                    "sha256": _sha256(audio),
                    "bytes": audio.stat().st_size,
                },
            ]
        )

    report = {
        "schema_name": "title-evidence-assets",
        "schema_version": 1,
        "status": "complete",
        "title": title,
        "media": str(media),
        "subtitle": str(subtitle),
        "subtitle_timeline_seconds": round(duration, 3),
        "source_japanese": str(copied_source) if copied_source else None,
        "source_japanese_candidates": [str(path) for path in source_candidates],
        "assets": assets,
    }
    manifest = title_root / "evidence-manifest.json"
    manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report["manifest"] = str(manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create representative local frame/audio evidence for subtitle review."
    )
    parser.add_argument("--active-root", required=True, type=Path)
    parser.add_argument("--assets-root", required=True, type=Path)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--titles", required=True, nargs="+")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument(
        "--register-reason",
        default="media extraction deferred",
        help="Reason recorded when --register-only is used.",
    )
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    if not args.ffmpeg.is_file():
        raise FileNotFoundError(f"ffmpeg was not found: {args.ffmpeg}")
    rows = [
        _create_title_assets(
            title=title.upper(),
            active_root=args.active_root,
            assets_root=args.assets_root,
            ffmpeg=args.ffmpeg,
            force=args.force,
            register_only=args.register_only,
            register_reason=args.register_reason,
        )
        for title in args.titles
    ]
    payload = {
        "schema_name": "title-evidence-assets-batch",
        "schema_version": 1,
        "title_count": len(rows),
        "titles": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
