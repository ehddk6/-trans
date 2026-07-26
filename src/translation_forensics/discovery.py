from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".m2ts"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".m2ts"}
ROLE_SUFFIXES = {
    "structure": ".structure.srt",
    "ja": ".ja.srt",
    "previous_ko": ".previous-ko.srt",
    "photos": ".photos.zip",
    "review_queue": ".review-queue",
    "asr_candidates": ".asr-candidates.csv",
    "work_audio": ".work_audio.zip",
}


class DiscoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    role: str
    path: Path
    reason: str


def ignored_path(path: Path) -> bool:
    lowered = {part.lower() for part in path.parts}
    return bool(lowered & {"vendor", ".git", ".pytest_cache", ".source-inspection"})


def title_from_filename(name: str) -> str:
    stem = Path(name).stem
    patterns = [
        r"\.review-queue(?:\.[^.]+)?-v\d+$",
        r"\.asr-candidates$",
        r"\.work_audio$",
        r"\.photos$",
        r"\.previous-ko$",
        r"\.structure$",
        r"\.ja$",
    ]
    for pattern in patterns:
        stem = re.sub(pattern, "", stem, flags=re.IGNORECASE)
    return stem or "untitled"


def _all_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and not ignored_path(p)]


def candidates_for_role(root: Path, role: str, title: str | None = None) -> list[Candidate]:
    if role not in {"structure", "ja", "previous_ko", "audio", "video", "photos", "review_queue", "asr_candidates", "work_audio"}:
        raise ValueError(f"지원하지 않는 역할: {role}")
    result: list[Candidate] = []
    for path in _all_files(root):
        name = path.name.lower()
        if title and not (title.lower() in name or path.parent.name.lower() == title.lower()):
            continue
        match = False
        reason = ""
        if role == "structure" and path.suffix.lower() == ".srt" and ".structure" in name:
            match, reason = True, "표준 .structure.srt 이름"
        elif role == "ja" and path.suffix.lower() == ".srt" and ".ja" in name:
            match, reason = True, "표준 .ja.srt 이름"
        elif role == "previous_ko" and path.suffix.lower() == ".srt" and ".previous-ko" in name:
            match, reason = True, "표준 .previous-ko.srt 이름"
        elif role == "audio" and path.suffix.lower() in AUDIO_EXTENSIONS and "clips" not in {p.lower() for p in path.parts}:
            match, reason = True, "지원하는 음성·영상 확장자"
        elif role == "video" and path.suffix.lower() in VIDEO_EXTENSIONS and "clips" not in {p.lower() for p in path.parts}:
            match, reason = True, "지원하는 영상 확장자"
        elif role == "photos" and path.suffix.lower() == ".zip" and ".photos" in name:
            match, reason = True, "표준 .photos.zip 이름"
        elif role == "review_queue" and path.suffix.lower() == ".csv" and "review-queue" in name:
            match, reason = True, "review-queue 파일명"
        elif role == "asr_candidates" and path.suffix.lower() == ".csv" and ".asr-candidates" in name:
            match, reason = True, "표준 .asr-candidates.csv 이름"
        elif role == "work_audio" and path.suffix.lower() == ".zip" and ".work_audio" in name:
            match, reason = True, "표준 .work_audio.zip 이름"
        if match:
            result.append(Candidate(role, path.resolve(), reason))
    return sorted(result, key=lambda item: str(item.path).lower())


def resolve_role(root: Path, role: str, explicit: Path | None = None, title: str | None = None, *, required: bool = False) -> Path | None:
    if explicit:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise DiscoveryError(f"{role} 파일이 없습니다: {path}")
        return path
    candidates = candidates_for_role(root, role, title)
    if not candidates:
        if required:
            raise DiscoveryError(f"{role} 후보를 찾지 못했습니다. 명시적 경로를 지정하세요.")
        return None
    if len(candidates) > 1:
        listing = "\n".join(f"- {c.path} ({c.reason})" for c in candidates)
        raise DiscoveryError(f"{role} 후보가 여러 개라 자동 선택하지 않았습니다.\n{listing}")
    return candidates[0].path


def inspect_roles(root: Path, title: str | None = None) -> dict[str, object]:
    roles: dict[str, object] = {}
    for role in ("structure", "ja", "previous_ko", "audio", "video", "photos", "review_queue", "asr_candidates", "work_audio"):
        found = candidates_for_role(root, role, title)
        roles[role] = [{"path": str(c.path), "reason": c.reason} for c in found]
    return {"root": str(root.resolve()), "title": title, "roles": roles, "ambiguous_roles": [r for r, values in roles.items() if isinstance(values, list) and len(values) > 1]}
