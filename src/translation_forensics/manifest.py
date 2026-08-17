from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .srt import SRTError, parse_srt


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_text_encoding(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            raw.decode("utf-8-sig")
            return "utf-8-sig"
        except UnicodeDecodeError:
            return "unknown"
    for encoding in ("utf-8", "cp932", "cp949"):
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "unknown"


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def file_record(path: Path, *, role: str, relative_to: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    record: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "modified_at": _mtime(path),
    }
    if relative_to:
        try:
            record["relative_path"] = str(path.relative_to(relative_to.resolve()))
        except ValueError:
            record["relative_path"] = str(path)
    if path.suffix.lower() in {".srt", ".csv", ".txt", ".md", ".json"}:
        record["encoding"] = detect_text_encoding(path)
    if path.suffix.lower() == ".srt":
        try:
            blocks, encoding, newline = parse_srt(path)
            record.update({
                "encoding": encoding,
                "newline": newline,
                "srt_block_count": len(blocks),
                "first_timecode": blocks[0].start if blocks else None,
                "last_timecode": blocks[-1].end if blocks else None,
            })
        except SRTError as exc:
            record["srt_error"] = str(exc)
    return record


def tool_versions(project_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "translation_forensics": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ffmpeg": None,
    }
    if shutil.which("ffmpeg"):
        try:
            completed = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)
            result["ffmpeg"] = completed.stdout.splitlines()[0] if completed.stdout else "available"
        except OSError:
            result["ffmpeg"] = "available"
    return result


def build_project_manifest(title: str, project_root: Path, files_by_role: dict[str, Path | None], *, structure_diff: dict[str, Any] | None = None, validation_status: str = "미검증", unresolved_roles: list[str] | None = None, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    records = []
    for role, path in files_by_role.items():
        if path is not None and path.exists():
            records.append(file_record(path, role=role, relative_to=project_root))
    return {
        "schema_version": "translation-forensics/project-manifest/1",
        "title": title,
        "project_root": str(project_root.resolve()),
        "inputs": records,
        "structure_diff": structure_diff or {"status": "미검증"},
        "provided_roles": sorted({str(record["role"]) for record in records}),
        "unresolved_roles": unresolved_roles or [],
        "tool_versions": tool_versions(project_root),
        "validation_status": validation_status,
        "execution_history": history or [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def append_history(manifest_path: Path, event: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    data.setdefault("execution_history", []).append(event)
    write_json(manifest_path, data)
    return data
