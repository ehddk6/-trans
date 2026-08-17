from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


TIME_RE = re.compile(r"^(\d{2,}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2,}:\d{2}:\d{2},\d{3})$")
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


class SRTError(ValueError):
    """SRT 구조가 규격에 맞지 않을 때 발생한다."""


@dataclass(frozen=True)
class SubtitleBlock:
    number: int
    start: str
    end: str
    text: str
    start_seconds: float
    end_seconds: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)

    @property
    def lines(self) -> list[str]:
        return self.text.split("\n") if self.text else [""]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def timecode_to_seconds(value: str) -> float:
    match = re.fullmatch(r"(\d{2,}):(\d{2}):(\d{2}),(\d{3})", value.strip())
    if not match:
        raise SRTError(f"잘못된 타임코드: {value!r}")
    hours, minutes, seconds, millis = map(int, match.groups())
    if minutes >= 60 or seconds >= 60:
        raise SRTError(f"범위를 벗어난 타임코드: {value!r}")
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def seconds_to_timecode(value: float) -> str:
    total = max(0, int(round(value * 1000)))
    hours, rem = divmod(total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _decode_utf8(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw.decode("utf-8-sig"), "utf-8-sig"
        except UnicodeDecodeError as exc:
            raise SRTError(f"UTF-8로 읽을 수 없습니다: {path}") from exc
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError as exc:
        raise SRTError(f"UTF-8로 읽을 수 없습니다: {path}") from exc


def parse_srt_text(text: str, *, source: str = "<text>") -> list[SubtitleBlock]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not normalized.strip():
        raise SRTError(f"빈 SRT: {source}")
    raw_blocks = re.split(r"\n\s*\n", normalized)
    blocks: list[SubtitleBlock] = []
    for index, raw in enumerate(raw_blocks, 1):
        lines = raw.split("\n")
        if len(lines) < 2:
            raise SRTError(f"블록 {index}의 헤더가 없습니다: {source}")
        try:
            number = int(lines[0].strip())
        except ValueError as exc:
            raise SRTError(f"블록 {index} 번호가 정수가 아닙니다: {source}") from exc
        match = TIME_RE.fullmatch(lines[1].strip())
        if not match:
            raise SRTError(f"블록 {number} 타임코드가 잘못되었습니다: {source}")
        start, end = match.groups()
        start_seconds = timecode_to_seconds(start)
        end_seconds = timecode_to_seconds(end)
        body = "\n".join(lines[2:]).strip()
        blocks.append(SubtitleBlock(number, start, end, body, start_seconds, end_seconds))
    return blocks


def parse_srt(path: Path) -> tuple[list[SubtitleBlock], str, str]:
    text, encoding = _decode_utf8(path)
    newline = "CRLF" if "\r\n" in text else "LF"
    return parse_srt_text(text, source=str(path)), encoding, newline


def render_srt(blocks: Iterable[SubtitleBlock]) -> str:
    parts = []
    for block in blocks:
        parts.append(f"{block.number}\n{block.start} --> {block.end}\n{block.text}")
    return "\n\n".join(parts) + "\n"


def write_srt(path: Path, blocks: Iterable[SubtitleBlock]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_srt(blocks), encoding="utf-8", newline="\n")


def structure_signature(blocks: Iterable[SubtitleBlock]) -> list[dict[str, object]]:
    return [{"number": b.number, "start": b.start, "end": b.end} for b in blocks]


def lock_structure(blocks: Iterable[SubtitleBlock]) -> tuple[dict[str, object], ...]:
    return tuple(structure_signature(blocks))


def compare_structure(reference: list[SubtitleBlock], candidate: list[SubtitleBlock]) -> dict[str, object]:
    issues: list[dict[str, object]] = []
    if len(reference) != len(candidate):
        issues.append({"code": "block_count", "reference": len(reference), "candidate": len(candidate)})
    for index, (left, right) in enumerate(zip(reference, candidate), 1):
        if left.number != right.number:
            issues.append({"code": "number_changed", "index": index, "reference": left.number, "candidate": right.number})
        if left.start != right.start or left.end != right.end:
            issues.append({"code": "timecode_changed", "index": index, "reference": f"{left.start} --> {left.end}", "candidate": f"{right.start} --> {right.end}"})
    if [b.number for b in candidate] != sorted(b.number for b in candidate):
        issues.append({"code": "order_changed", "detail": "번호가 오름차순이 아닙니다."})
    return {"pass": not issues, "issues": issues}


def has_japanese(text: str) -> bool:
    return bool(JAPANESE_RE.search(text))
