from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import wave
from pathlib import Path
from typing import Any


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def representative_indices(count: int, limit: int = 3) -> list[int]:
    if count <= limit:
        return list(range(count))
    values = [0, count // 2, count - 1]
    return list(dict.fromkeys(values))[:limit]


def validate_wav(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        sample_width = handle.getsampwidth()
    duration = frames / sample_rate if sample_rate else 0.0
    if channels != 1 or sample_rate != 16_000 or sample_width != 2 or duration <= 0:
        raise ValueError(
            f"Unexpected WAV format for {path}: channels={channels}, "
            f"rate={sample_rate}, width={sample_width}, duration={duration}"
        )
    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "sample_width": sample_width,
        "duration_seconds": round(duration, 3),
    }


def create_clip(ffmpeg: Path, source: Path, output: Path) -> dict[str, Any]:
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-t",
        "15",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {source}: {completed.stderr.strip()}"
        )
    details = validate_wav(output)
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "output_bytes": output.stat().st_size,
        **details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create compact representative evidence audio from existing local workspace clips."
    )
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--workspaces-root", required=True, type=Path)
    parser.add_argument("--assets-root", required=True, type=Path)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for audit_row in sorted(audit["titles"], key=lambda row: row["title"].casefold()):
        title = audit_row["title"]
        title_workspace = args.workspaces_root / title
        title_assets = args.assets_root / title
        existing_audio = [
            path
            for path in title_assets.rglob("*")
            if path.is_file() and path.suffix.casefold() in AUDIO_EXTENSIONS
        ] if title_assets.exists() else []
        sources = [
            path
            for path in title_workspace.rglob("*")
            if path.is_file() and path.suffix.casefold() in AUDIO_EXTENSIONS
        ] if title_workspace.exists() else []
        sources.sort(key=lambda path: str(path).casefold())
        row: dict[str, Any] = {
            "title": title,
            "existing_audio_count": len(existing_audio),
            "workspace_audio_count": len(sources),
            "created": [],
            "status": "skipped",
        }
        if existing_audio:
            row["status"] = "existing evidence retained"
            rows.append(row)
            continue
        if not sources:
            row["status"] = "no local workspace audio"
            rows.append(row)
            continue
        evidence = title_assets / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        selected = [sources[index] for index in representative_indices(len(sources))]
        for index, source in enumerate(selected, 1):
            output = evidence / f"workspace-audio-{index:02d}.wav"
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite existing evidence: {output}")
            row["created"].append(create_clip(args.ffmpeg, source, output))
        row["status"] = "created from local workspace audio"
        manifest = evidence / "workspace-audio-manifest.json"
        if manifest.exists():
            raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest}")
        manifest.write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        rows.append(row)

    report = {
        "schema_name": "translation-forensics/workspace-audio-evidence",
        "schema_version": 1,
        "title_count": len(rows),
        "created_title_count": sum(row["status"].startswith("created") for row in rows),
        "created_audio_count": sum(len(row["created"]) for row in rows),
        "existing_evidence_title_count": sum(row["status"].startswith("existing") for row in rows),
        "no_local_audio_title_count": sum(row["status"].startswith("no local") for row in rows),
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
