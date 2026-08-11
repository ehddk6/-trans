from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    target_line_length: int = 18
    recommended_line_limit: int = 22
    absolute_line_limit: int = 26
    min_duration: float = 0.8
    target_min_duration: float = 1.2
    target_max_duration: float = 6.0
    max_duration: float = 7.0
    max_lines: int = 2
    preserve_disfluency: bool = False
    compactness: float = 1.0


BASE = Profile("viewer_ja")
PROFILES = {
    "viewer_ja": BASE,
    "cinema_ja": replace(BASE, name="cinema_ja", target_min_duration=1.4),
    "anime_ja": replace(BASE, name="anime_ja", target_line_length=16, target_min_duration=1.0, compactness=0.75),
    "lecture_ja": replace(BASE, name="lecture_ja", target_line_length=20, target_max_duration=6.5),
    "verbatim_ja": replace(BASE, name="verbatim_ja", preserve_disfluency=True, target_line_length=20),
    "compact_ja": replace(BASE, name="compact_ja", target_line_length=21, recommended_line_limit=24, compactness=1.35),
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown profile: {name}. Choices: {', '.join(PROFILES)}") from exc
