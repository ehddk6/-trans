from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .checks import annotate_asr_warnings
from .models import SourceSegment
from .qwen import QwenRuntime, _media_duration, transcribe_qwen
from .srt import format_timestamp, read_srt_rows
from .text import content_signature, normalize_japanese


REVIEW_POLICY = {
    "policy_version": "qwen-anomaly-review-v1",
    "target_anomalies": ["possible_periodic_repetition"],
    "core_seconds": 4.0,
    "context_seconds": 8.0,
    "use_vad": False,
    "primary_source": "whisper",
}


QwenRunner = Callable[..., tuple[list[SourceSegment], float]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_transcript(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_cue_indexes(rows: list[dict[str, Any]], source_rows: list[dict[str, object]]) -> dict[int, int]:
    """Map source segment ids to source-SRT cue ordinals without trusting cue counts."""
    mapping: dict[int, int] = {}
    cue_index = 0
    for row in rows:
        text = str(row.get("text_normalized", row.get("text_raw", "")))
        if not text:
            continue
        if cue_index >= len(source_rows):
            break
        for source_id in row.get("source_segment_ids", []):
            try:
                mapping[int(source_id)] = cue_index
            except (TypeError, ValueError):
                continue
        cue_index += 1
    return mapping


def _segments_from_transcript(rows: list[dict[str, Any]]) -> list[SourceSegment]:
    segments: list[SourceSegment] = []
    for index, row in enumerate(rows):
        source_ids = row.get("source_segment_ids", [index])
        source_id = int(source_ids[0]) if source_ids else index
        segments.append(SourceSegment(
            source_id,
            float(row["start"]),
            float(row["end"]),
            str(row.get("text_raw", row.get("text_normalized", ""))),
            warnings=list(row.get("warnings", [])),
        ))
    annotate_asr_warnings(segments)
    return segments


def collect_source_review_targets(
    transcript_path: Path,
    source_rows: list[dict[str, object]],
    max_targets: int = 8,
) -> list[dict[str, Any]]:
    rows = _load_transcript(transcript_path)
    segments = _segments_from_transcript(rows)
    cue_indexes = _source_cue_indexes(rows, source_rows)
    candidates = []
    for segment in segments:
        if "possible_periodic_repetition" not in segment.warnings:
            continue
        cue_index = cue_indexes.get(segment.id)
        if cue_index is None:
            continue
        candidates.append({
            "source_segment_id": segment.id,
            "cue_index": cue_index,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "anomaly": "possible_periodic_repetition",
        })
    if max_targets <= 0 or len(candidates) <= max_targets:
        return candidates
    if max_targets == 1:
        return [candidates[len(candidates) // 2]]
    indexes = {
        round(index * (len(candidates) - 1) / (max_targets - 1))
        for index in range(max_targets)
    }
    return [candidate for index, candidate in enumerate(candidates) if index in indexes]


def _window(start: float, end: float, duration: float, width: float) -> dict[str, float]:
    midpoint = (start + end) / 2.0
    actual_width = min(width, duration)
    left = min(max(0.0, midpoint - actual_width / 2.0), max(0.0, duration - actual_width))
    return {"start": left, "end": left + actual_width}


def _qwen_text_for_target(
    segments: list[SourceSegment], start: float, end: float, fallback_interval: dict[str, float],
    source_interval: int | None = None,
) -> str:
    if source_interval is not None:
        segments = [
            segment for segment in segments
            if segment.words and segment.words[0].source_interval == source_interval
        ]
    overlapping = [
        segment for segment in segments
        if min(end, segment.end) - max(start, segment.start) >= 0.05
    ]
    if not overlapping:
        overlapping = [
            segment for segment in segments
            if segment.end > fallback_interval["start"] and segment.start < fallback_interval["end"]
        ]
    return normalize_japanese("".join(segment.text for segment in overlapping), "strict")


def _write_candidate(path: Path, rows: list[dict[str, object]]) -> None:
    blocks = []
    for index, row in enumerate(rows, 1):
        blocks.append(
            f"{index}\n{format_timestamp(float(row['start']))} --> "
            f"{format_timestamp(float(row['end']))}\n{row['text']}"
        )
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def build_source_review_candidate(
    transcript_path: Path,
    input_path: Path,
    output_dir: Path,
    runtime: QwenRuntime,
    *,
    max_targets: int = 8,
    duration: float | None = None,
    qwen_runner: QwenRunner = transcribe_qwen,
) -> dict[str, Any]:
    source_path = output_dir / "source_faithful_ja.srt"
    candidate_path = output_dir / "source_review_candidate_ja.srt"
    evidence_path = output_dir / "source_review_evidence.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source-faithful SRT: {source_path}")
    if source_path.resolve() == candidate_path.resolve():
        raise ValueError("review candidate must not overwrite source-faithful SRT")

    source_hash_before = _sha256(source_path)
    source_rows = read_srt_rows(source_path)
    targets = collect_source_review_targets(transcript_path, source_rows, max_targets)
    candidate_rows = [dict(row) for row in source_rows]
    evidence_targets: list[dict[str, Any]] = []
    target_windows: list[tuple[dict[str, Any], dict[str, float], dict[str, float]]] = []
    all_intervals: list[dict[str, object]] = []
    duration = _media_duration(input_path) if duration is None else duration
    for target in targets:
        start, end = float(target["start"]), float(target["end"])
        core = _window(start, end, duration, REVIEW_POLICY["core_seconds"])
        context = _window(start, end, duration, REVIEW_POLICY["context_seconds"])
        target_windows.append((target, core, context))
        all_intervals.extend([
            {**core, "allow_overlap": True, "reasons": [target["anomaly"], "core"]},
            {**context, "allow_overlap": True, "reasons": [target["anomaly"], "context"]},
        ])

    qwen_segments: list[SourceSegment] = []
    qwen_error: str | None = None
    if all_intervals:
        try:
            qwen_segments, _ = qwen_runner(
                input_path, runtime, language="ja", targeted_intervals=all_intervals,
                reuse_alignment_cache=False,
            )
        except Exception as exc:  # candidate generation must fail closed per target
            qwen_error = type(exc).__name__ + ": " + str(exc)

    for target_index, (target, core, context) in enumerate(target_windows):
        start, end = float(target["start"]), float(target["end"])
        source_cue = candidate_rows[int(target["cue_index"])]
        row_evidence = dict(target)
        row_evidence.update({
            "whisper_text": str(source_cue["text"]),
            "core_interval": core,
            "context_interval": context,
            "decision": "whisper_source_preserved",
            "agreement": False,
        })
        try:
            if qwen_error:
                raise RuntimeError(qwen_error)
            core_segments = [
                segment for segment in qwen_segments
                if segment.words and segment.words[0].source_interval == target_index * 2
            ]
            context_segments = [
                segment for segment in qwen_segments
                if segment.words and segment.words[0].source_interval == target_index * 2 + 1
            ]
            core_text = _qwen_text_for_target(core_segments, start, end, core, target_index * 2)
            context_text = _qwen_text_for_target(context_segments, start, end, context, target_index * 2 + 1)
            row_evidence.update({"qwen_core_text": core_text, "qwen_context_text": context_text})
            if core_text and content_signature(core_text) == content_signature(context_text):
                candidate_rows[int(target["cue_index"])] ["text"] = core_text
                row_evidence.update({"candidate_text": core_text, "decision": "qwen_proposal", "agreement": True})
            else:
                row_evidence.update({
                    "candidate_text": str(source_cue["text"]),
                    "decision": "disagreement_review_required",
                })
        except Exception as exc:  # candidate generation must fail closed per target
            row_evidence.update({
                "candidate_text": str(source_cue["text"]),
                "decision": "qwen_failed_source_preserved",
                "error": type(exc).__name__ + ": " + str(exc),
            })
        evidence_targets.append(row_evidence)

    _write_candidate(candidate_path, candidate_rows)
    source_hash_after = _sha256(source_path)
    if source_hash_before != source_hash_after:
        raise RuntimeError("source-faithful SRT changed while building review candidate")
    evidence = {
        "policy": REVIEW_POLICY,
        "source_srt": str(source_path),
        "source_srt_sha256": source_hash_after,
        "input_path": str(input_path),
        "review_required": True,
        "targets": evidence_targets,
        "candidate_srt": str(candidate_path),
    }
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "candidate_path": str(candidate_path),
        "evidence_path": str(evidence_path),
        "source_srt_sha256": source_hash_after,
        "target_count": len(targets),
        "proposal_count": sum(item.get("decision") == "qwen_proposal" for item in evidence_targets),
        "review_count": sum(item.get("decision") != "qwen_proposal" for item in evidence_targets),
    }
