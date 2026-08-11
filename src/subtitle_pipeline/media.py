from __future__ import annotations

import hashlib
import json
import math
import subprocess
import wave
from pathlib import Path
from typing import Any, Mapping


MEDIA_BINDING_SCHEMA = "subtitle-pipeline/source-media-binding"
MEDIA_BINDING_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_media_duration(path: Path) -> float:
    """Return a finite, positive media duration without decoding the full file."""

    path = Path(path).resolve()
    # Prefer the standard-library WAV reader by content.  Several workflows use
    # container-like filenames for normalized fixtures, and format sniffing also
    # avoids starting a subprocess for ordinary WAV input.
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            duration = handle.getnframes() / sample_rate if sample_rate else math.nan
    except (EOFError, OSError, wave.Error):
        try:
            completed = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=nw=1:nk=1",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            duration = float(completed.stdout.strip())
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            raise RuntimeError(f"Unable to probe media duration: {path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Media duration must be finite and positive: {path}")
    return duration


def build_media_binding(path: Path, *, known_sha256: str | None = None) -> dict[str, Any]:
    """Build the content identity used to bind a subtitle bundle to its media."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    digest = known_sha256 or sha256_file(path)
    if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
        raise ValueError("known_sha256 must be a 64-character hexadecimal digest")
    return {
        "schema_name": MEDIA_BINDING_SCHEMA,
        "schema_version": MEDIA_BINDING_VERSION,
        "path": str(path),
        "sha256": digest.lower(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "duration_seconds": round(probe_media_duration(path), 6),
    }


def validate_media_binding(value: Mapping[str, Any]) -> list[str]:
    """Return deterministic schema errors for a serialized media binding."""

    errors: list[str] = []
    if value.get("schema_name") != MEDIA_BINDING_SCHEMA:
        errors.append("invalid_source_media_schema")
    if value.get("schema_version") != MEDIA_BINDING_VERSION:
        errors.append("invalid_source_media_version")
    digest = value.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        errors.append("invalid_source_media_sha256")
    try:
        size = int(value.get("size_bytes"))
        duration = float(value.get("duration_seconds"))
    except (TypeError, ValueError):
        errors.append("invalid_source_media_metrics")
    else:
        if size < 0 or not math.isfinite(duration) or duration <= 0:
            errors.append("invalid_source_media_metrics")
    if not isinstance(value.get("path"), str) or not str(value.get("path")).strip():
        errors.append("invalid_source_media_path")
    return errors


def write_media_binding(path: Path, value: Mapping[str, Any]) -> None:
    """Write a binding only after its complete JSON representation is ready."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "MEDIA_BINDING_SCHEMA",
    "MEDIA_BINDING_VERSION",
    "build_media_binding",
    "probe_media_duration",
    "sha256_file",
    "validate_media_binding",
    "write_media_binding",
]
