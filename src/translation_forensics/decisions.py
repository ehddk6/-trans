from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VERIFICATION_STATES = ("structure-validated", "text-crosschecked", "audio-asr-crosschecked", "audio-human-verified", "evaluation-validated", "final")
SCENE_VERDICTS = ("stable", "normalized", "context-resolved", "nonverbal", "unresolved")


@dataclass
class MeaningFrame:
    block_number: int
    speaker: str = ""
    addressee: str = ""
    speech_act: str = ""
    polarity: str = ""
    question_or_statement: str = ""
    request_or_command: str = ""
    action: str = ""
    target: str = ""
    location: str = ""
    direction: str = ""
    tense: str = ""
    result: str = ""
    emotion: str = ""
    sexual_semantic_category: str = ""
    confirmed: list[str] = field(default_factory=list)
    uncertain: list[str] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    confidence: str = "low"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneVerdict:
    scene_id: str
    verdict: str
    confidence: str
    evidence_summary: str = ""
    unresolved_scope: str = ""
    direct_human_listening: bool = False

    def __post_init__(self) -> None:
        if self.verdict not in SCENE_VERDICTS:
            raise ValueError(f"알 수 없는 장면 판정: {self.verdict}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_state_transition(current: str, requested: str, *, all_blocks_reviewed: bool = False, direct_listening: bool = False, qa_passed: bool = False) -> tuple[bool, str]:
    if current not in VERIFICATION_STATES or requested not in VERIFICATION_STATES:
        return False, "알 수 없는 검증 상태입니다."
    order = {value: index for index, value in enumerate(VERIFICATION_STATES)}
    if order[requested] < order[current]:
        return False, "검증 상태를 뒤로 되돌리는 전이는 허용하지 않습니다."
    if requested == "audio-asr-crosschecked" and current == "text-crosschecked":
        return True, "ASR 교차검토 자료가 연결되었습니다."
    if requested == "audio-human-verified" and not direct_listening:
        return False, "직접 원음 청취가 확인되지 않아 audio-human-verified를 부여할 수 없습니다."
    if requested == "final" and (not all_blocks_reviewed or not qa_passed):
        return False, "전체 블록 검수와 QA 통과가 모두 필요합니다."
    return True, "전이 가능"
