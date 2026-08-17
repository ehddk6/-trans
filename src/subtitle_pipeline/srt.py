from __future__ import annotations

import re
from pathlib import Path

from .models import Cue


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def parse_timestamp(value: str) -> float:
    # Accept the usual HH:MM:SS,mmm form plus common SRT variants that omit
    # a zero hour (MM:SS,mmm) or use a one-digit hour. The source subtitle
    # text and timestamps are preserved in the emitted faithful SRT; this
    # only makes reference parsing tolerant of real-world files.
    match = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{1,2}),(\d{3})", value)
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value}")
    hours_text, minutes_text, seconds_text, milliseconds_text = match.groups()
    hours = int(hours_text or 0)
    minutes = int(minutes_text)
    seconds = int(seconds_text)
    milliseconds = int(milliseconds_text)
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def write_srt(path: Path, cues: list[Cue]) -> None:
    blocks = []
    for number, cue in enumerate(cues, 1):
        blocks.append(f"{number}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{cue.text}")
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def read_srt_rows(path: Path) -> list[dict[str, object]]:
    """Read a user-supplied SRT for comparison without modifying its text."""
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        # ``utf-16`` consumes either BOM and selects the corresponding byte order.
        text = data.decode("utf-16")
    else:
        text = ""
        # This is a Japanese-only pipeline, so Japanese Windows/Shift-JIS
        # encodings take precedence over Korean legacy fallbacks when bytes are
        # ambiguous between the code pages.
        for encoding in ("utf-8-sig", "utf-8", "cp932", "shift_jis", "cp949", "euc-kr"):
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
    if not text:
        text = data.decode("utf-8", errors="replace")
    rows: list[dict[str, object]] = []
    for block in re.split(r"\r?\n\r?\n+", text.strip()):
        lines = block.splitlines()
        timing_index = next((index for index, line in enumerate(lines) if " --> " in line), None)
        if timing_index is None:
            continue
        start_text, end_text = lines[timing_index].split(" --> ", 1)
        rows.append({
            "id": lines[0].strip() if lines and lines[0].strip().isdigit() else len(rows) + 1,
            "start": parse_timestamp(start_text.strip().replace(".", ",")),
            "end": parse_timestamp(end_text.strip().replace(".", ",")),
            "text": "\n".join(lines[timing_index + 1:]).strip(),
        })
    return rows


def validate_cues(cues: list[Cue]) -> list[str]:
    errors: list[str] = []
    previous_end = -1.0
    for expected_id, cue in enumerate(cues, 1):
        if cue.id != expected_id:
            errors.append("non_contiguous_ids")
        if cue.end <= cue.start:
            errors.append(f"non_positive_duration:{cue.id}")
        if cue.start < previous_end - 0.001:
            errors.append(f"overlap:{cue.id}")
        previous_end = max(previous_end, cue.end)
        if not cue.text.strip():
            errors.append(f"empty_text:{cue.id}")
    return errors
