from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any

_ALIGNMENT_PRIORITY = {
    "utterance-timestamp": 3,
    "boundary-expanded-window-rerun": 2,
    "overlapping-window": 1,
}


_SRT_TIMESTAMP = re.compile(
    r"^(?P<hours>\d{2,}):(?P<minutes>\d{2}):(?P<seconds>\d{2})"
    r"[,\.](?P<milliseconds>\d{3})$"
)


def _timestamp_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _SRT_TIMESTAMP.fullmatch(str(value).strip())
    if match is None:
        return None
    return (
        int(match.group("hours")) * 3600
        + int(match.group("minutes")) * 60
        + int(match.group("seconds"))
        + int(match.group("milliseconds")) / 1000.0
    )


def normalize_asr_text(value: Any) -> str:
    """Normalize surface text only for deterministic comparison."""
    text = str(value or "").casefold()
    return "".join(character for character in text if character.isalnum())


def _alignment_priority(row: Mapping[str, Any]) -> int:
    return _ALIGNMENT_PRIORITY.get(str(row.get("alignment_scope") or ""), 0)


def _evidence_ref(row: Mapping[str, Any]) -> str:
    family = str(row.get("source_family") or "unknown")
    utterance_id = str(row.get("utterance_id") or "")
    if utterance_id:
        return f"utterance:{utterance_id}:{family}"
    window_id = str(row.get("window_id") or "")
    return f"asr:{window_id}:{family}" if window_id else f"asr:{family}"


def _overlap_seconds(
    row: Mapping[str, Any],
    *,
    block_start: float | None,
    block_end: float | None,
) -> float:
    if block_start is None or block_end is None:
        return 0.0
    start = row.get("start_seconds", row.get("clip_start_seconds"))
    end = row.get("end_seconds", row.get("clip_end_seconds"))
    if start is None or end is None:
        return 0.0
    return max(0.0, min(float(end), block_end) - max(float(start), block_start))


def _deduplicated_text_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        normalized = normalize_asr_text(row.get("text"))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(row)
    return selected


def _family_hypothesis(
    family: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    block_start: float | None,
    block_end: float | None,
) -> dict[str, Any]:
    best_priority = max((_alignment_priority(row) for row in rows), default=0)
    candidates = [row for row in rows if _alignment_priority(row) == best_priority]

    if best_priority >= _ALIGNMENT_PRIORITY["utterance-timestamp"]:
        candidates.sort(
            key=lambda row: (
                float(row.get("start_seconds", 0.0) or 0.0),
                float(row.get("end_seconds", 0.0) or 0.0),
                str(row.get("utterance_id") or ""),
            )
        )
        selected = _deduplicated_text_rows(candidates)
    else:
        # Wide windows are alternate context views, not sequential utterances.
        # Select one stable best candidate instead of concatenating them.
        selected = []
        usable = _deduplicated_text_rows(candidates)
        if usable:
            best = max(
                usable,
                key=lambda row: (
                    _overlap_seconds(row, block_start=block_start, block_end=block_end),
                    float(row.get("alignment_confidence", 0.0) or 0.0),
                    -abs(
                        float(row.get("clip_end_seconds", 0.0) or 0.0)
                        - float(row.get("clip_start_seconds", 0.0) or 0.0)
                    ),
                    normalize_asr_text(row.get("text")),
                ),
            )
            selected = [best]

    texts = [
        re.sub(r"\s+", " ", str(row.get("text") or "").strip())
        for row in selected
        if str(row.get("text") or "").strip()
    ]
    joined = " ".join(texts).strip()
    return {
        "source_family": family,
        "text": joined,
        "normalized_text": normalize_asr_text(joined),
        "alignment_scope": (
            str(selected[0].get("alignment_scope") or "") if selected else ""
        ),
        "alignment_priority": best_priority,
        "block_aligned": best_priority >= _ALIGNMENT_PRIORITY["utterance-timestamp"],
        "evidence_refs": sorted({_evidence_ref(row) for row in selected}),
    }


def _surface_risk_flags(value: Any) -> dict[str, bool]:
    text = re.sub(r"\s+", "", str(value or ""))
    return {
        "negative": bool(
            re.search(
                r"(?:ない|なく|なかった|ません|じゃない|ではない|ぬ)"
                r"(?:よ|ね|ぞ|から|けど|って)?(?:[。！？!?…]*)$",
                text,
            )
        ),
        "question": bool(re.search(r"(?:[?？]|か[。！？!?…]*)$", text)),
        "refusal": any(token in text for token in ("いや", "だめ", "ダメ", "無理", "やめ")),
        "permission": any(token in text for token in ("どうぞ", "いいよ", "構わない", "かまわない")),
        "stop": any(token in text for token in ("やめ", "止め", "とめ")),
        "continue": any(token in text for token in ("続け", "つづけ", "そのまま")),
    }


def _meaning_flip_risks(left: Any, right: Any) -> list[str]:
    left_flags = _surface_risk_flags(left)
    right_flags = _surface_risk_flags(right)
    risks: list[str] = []
    if left_flags["negative"] != right_flags["negative"]:
        risks.append("polarity-marker-divergence")
    if left_flags["question"] != right_flags["question"]:
        risks.append("question-marker-divergence")
    if (
        left_flags["refusal"] and right_flags["permission"]
    ) or (
        right_flags["refusal"] and left_flags["permission"]
    ):
        risks.append("refusal-permission-marker-divergence")
    if (
        left_flags["stop"] and right_flags["continue"]
    ) or (
        right_flags["stop"] and left_flags["continue"]
    ):
        risks.append("stop-continue-marker-divergence")
    return risks


def _compatible(left: str, right: str, similarity: float) -> bool:
    if not left or not right:
        return False
    return (
        left == right
        or (
            min(len(left), len(right)) >= 2
            and (left in right or right in left or similarity >= 0.55)
        )
    )


def _shared_spans(left: str, right: str) -> list[str]:
    if not left or not right:
        return []
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    return [
        left[block.a : block.a + block.size]
        for block in matcher.get_matching_blocks()
        if block.size >= 2
    ]


def fuse_asr_transcripts(
    transcripts: Sequence[Mapping[str, Any]],
    *,
    block_start: float | None = None,
    block_end: float | None = None,
) -> dict[str, Any]:
    """Summarize independent ASR evidence without inventing a fused transcript.

    Rows from one source family collapse into one hypothesis and therefore count
    as one vote. Raw rows remain authoritative and are never overwritten.
    """

    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for row in transcripts:
        family = str(row.get("source_family") or "").strip()
        if family:
            by_family.setdefault(family, []).append(row)

    hypotheses = [
        _family_hypothesis(
            family,
            rows,
            block_start=block_start,
            block_end=block_end,
        )
        for family, rows in sorted(by_family.items())
    ]
    hypotheses = [row for row in hypotheses if row["normalized_text"]]
    if not hypotheses:
        return {
            "schema_name": "translation-forensics/asr-fusion",
            "schema_version": "1",
            "state": "empty",
            "family_hypotheses": [],
            "shared_spans": [],
            "pairwise_similarities": [],
            "risk_codes": [],
            "alignment_strength": "none",
        }

    priorities = [int(row["alignment_priority"]) for row in hypotheses]
    alignment_strength = (
        "block-aligned"
        if min(priorities) >= 3
        else "expanded"
        if min(priorities) >= 2
        else "context-only"
    )
    if len(hypotheses) == 1:
        return {
            "schema_name": "translation-forensics/asr-fusion",
            "schema_version": "1",
            "state": "single_family",
            "family_hypotheses": hypotheses,
            "shared_spans": [],
            "pairwise_similarities": [],
            "risk_codes": [],
            "alignment_strength": alignment_strength,
        }

    similarities: list[dict[str, Any]] = []
    risk_codes: set[str] = set()
    compatible = True
    exact = True
    shared: set[str] | None = None
    for left_index, left_row in enumerate(hypotheses):
        for right_row in hypotheses[left_index + 1 :]:
            left = str(left_row["normalized_text"])
            right = str(right_row["normalized_text"])
            similarity = SequenceMatcher(None, left, right, autojunk=False).ratio()
            pair_risks = _meaning_flip_risks(left_row["text"], right_row["text"])
            risk_codes.update(pair_risks)
            similarities.append(
                {
                    "left_family": left_row["source_family"],
                    "right_family": right_row["source_family"],
                    "similarity": round(similarity, 6),
                }
            )
            exact = exact and left == right
            compatible = compatible and not pair_risks and _compatible(left, right, similarity)
            pair_shared = set(_shared_spans(left, right))
            shared = pair_shared if shared is None else shared & pair_shared

    if risk_codes:
        state = "dual_conflict"
    elif exact:
        state = "dual_agreement"
    elif compatible:
        state = "dual_compatible"
    else:
        state = "dual_conflict"
    return {
        "schema_name": "translation-forensics/asr-fusion",
        "schema_version": "1",
        "state": state,
        "family_hypotheses": hypotheses,
        "shared_spans": sorted(shared or set()),
        "pairwise_similarities": similarities,
        "risk_codes": sorted(risk_codes),
        "alignment_strength": alignment_strength,
    }


def add_asr_fusion(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of one block evidence record with an additive fusion summary."""
    output = dict(record)
    output["schema_version"] = "2"
    output["asr_fusion"] = fuse_asr_transcripts(
        record.get("transcripts") or [],
        block_start=_timestamp_seconds(
            record.get("start_seconds", record.get("start"))
        ),
        block_end=_timestamp_seconds(
            record.get("end_seconds", record.get("end"))
        ),
    )
    return output
