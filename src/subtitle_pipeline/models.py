from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Word:
    text: str
    start: float
    end: float
    probability: float | None = None
    backend: str | None = None
    source_chunk: int | None = None
    source_interval: int | None = None


@dataclass(slots=True)
class SourceSegment:
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    speaker: str = "speaker_unknown"
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Utterance:
    id: str
    source_segment_ids: list[int]
    start: float
    end: float
    text_raw: str
    text_normalized: str
    words: list[Word] = field(default_factory=list)
    speaker: str = "speaker_unknown"
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def json(self) -> dict[str, Any]:
        data = asdict(self)
        data["utterance_id"] = data.pop("id")
        data["asr_metrics"] = data.pop("metrics")
        return data


@dataclass(slots=True)
class Cue:
    id: int
    start: float
    end: float
    text: str
    normalized_text: str
    source_segment_ids: list[int]
    speaker: str = "speaker_unknown"
    split_reasons: list[str] = field(default_factory=list)
    merge_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    adjacent_silence: float | None = None
    scene_distance: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)
