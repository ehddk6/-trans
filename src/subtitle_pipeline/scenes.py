from __future__ import annotations

import re
import subprocess
from pathlib import Path


def detect_scene_changes(input_path: Path, max_duration: float | None = None, threshold: float = 0.40) -> list[float]:
    """Return FFmpeg scene-change candidates; callers must treat them only as soft evidence."""
    command = ["ffmpeg", "-hide_banner", "-v", "info", "-i", str(input_path)]
    if max_duration:
        command += ["-t", str(max_duration)]
    command += ["-vf", f"select='gt(scene,{threshold})',metadata=print", "-an", "-f", "null", "-"]
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return sorted({float(value) for value in re.findall(r"pts_time:([0-9.]+)", completed.stderr)})
