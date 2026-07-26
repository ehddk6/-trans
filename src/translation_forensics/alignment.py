from __future__ import annotations

from dataclasses import dataclass

from .srt import SubtitleBlock


@dataclass(frozen=True)
class Alignment:
    source_number: int
    candidate_number: int | None
    overlap_seconds: float
    candidate_start_distance: float | None
    confidence: str


def overlap_seconds(left: SubtitleBlock, right: SubtitleBlock) -> float:
    return max(0.0, min(left.end_seconds, right.end_seconds) - max(left.start_seconds, right.start_seconds))


def align_by_overlap(source: list[SubtitleBlock], candidate: list[SubtitleBlock], *, min_overlap: float = 0.0) -> list[Alignment]:
    result: list[Alignment] = []
    for block in source:
        scored = []
        for other in candidate:
            overlap = overlap_seconds(block, other)
            distance = abs(block.start_seconds - other.start_seconds)
            if overlap > min_overlap:
                scored.append((overlap, -distance, other, distance))
        if not scored:
            result.append(Alignment(block.number, None, 0.0, None, "unresolved"))
            continue
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_overlap, _, best, distance = scored[0]
        ratio = best_overlap / max(0.001, block.duration)
        confidence = "high" if ratio >= 0.8 else "medium" if ratio >= 0.35 else "low"
        result.append(Alignment(block.number, best.number, round(best_overlap, 3), round(distance, 3), confidence))
    return result


def align_to_dicts(source: list[SubtitleBlock], candidate: list[SubtitleBlock]) -> list[dict[str, object]]:
    return [alignment.__dict__ for alignment in align_by_overlap(source, candidate)]
