from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from .manifest import sha256_file, write_json
from .srt import SubtitleBlock, compare_structure, parse_srt
from .timeline import media_duration_seconds, read_timeline_anchors


ALIGNMENT_SCHEMA_NAME = "translation-forensics/pilot-timeline-alignment"
AUDIO_REVIEW_SCHEMA_NAME = "translation-forensics/pilot-audio-review-manifest"
SCHEMA_VERSION = "1"


class PilotAlignmentError(ValueError):
    """Raised when a full-title listening packet cannot safely use the timeline."""


def _file_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _load_offset_map(path: Path | None, *, last_end_seconds: float) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    if path is None or not path.exists():
        return None, [], ["사람이 승인한 offset map이 없습니다."]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [], [f"offset map을 읽을 수 없습니다: {exc}"]
    if not isinstance(value, dict):
        return None, [], ["offset map 최상위 값은 객체여야 합니다."]

    errors: list[str] = []
    if value.get("approval_status") != "approved":
        errors.append("offset map approval_status는 approved여야 합니다.")
    if value.get("human_verified") is not True:
        errors.append("offset map human_verified는 true여야 합니다.")
    if not str(value.get("approved_by", "")).strip():
        errors.append("offset map approved_by가 비어 있습니다.")
    evidence_refs = value.get("evidence_refs")
    if not isinstance(evidence_refs, list) or not evidence_refs:
        errors.append("offset map에는 비어 있지 않은 evidence_refs가 필요합니다.")

    raw_segments = value.get("segments")
    if raw_segments is None and "offset_seconds" in value:
        raw_segments = [{
            "segment_id": "full-title",
            "srt_start_seconds": 0.0,
            "srt_end_seconds": last_end_seconds,
            "offset_seconds": value.get("offset_seconds"),
        }]
    if not isinstance(raw_segments, list) or not raw_segments:
        errors.append("offset map에는 하나 이상의 segments 또는 offset_seconds가 필요합니다.")
        return value, [], errors

    segments: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_segments, 1):
        if not isinstance(raw, dict):
            errors.append(f"offset segment {index}은 객체여야 합니다.")
            continue
        try:
            start = float(raw["srt_start_seconds"])
            end = float(raw["srt_end_seconds"])
            offset = float(raw["offset_seconds"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"offset segment {index}의 시간값이 잘못되었습니다.")
            continue
        if not all(math.isfinite(value) for value in (start, end, offset)):
            errors.append(f"offset segment {index}의 시간값은 유한한 숫자여야 합니다.")
            continue
        if start < 0 or end <= start:
            errors.append(f"offset segment {index}의 SRT 범위가 잘못되었습니다.")
            continue
        segments.append({
            "segment_id": str(raw.get("segment_id") or f"segment-{index}"),
            "srt_start_seconds": start,
            "srt_end_seconds": end,
            "offset_seconds": offset,
        })

    segments.sort(key=lambda item: item["srt_start_seconds"])
    if segments:
        if segments[0]["srt_start_seconds"] > 0:
            errors.append("offset map이 SRT 시작 시각 0초부터 적용되지 않습니다.")
        if segments[-1]["srt_end_seconds"] < last_end_seconds:
            errors.append("offset map이 마지막 자막 종료 시각까지 적용되지 않습니다.")
        for left, right in zip(segments, segments[1:]):
            if abs(left["srt_end_seconds"] - right["srt_start_seconds"]) > 1e-6:
                errors.append("offset map segment 사이에 공백 또는 겹침이 있습니다.")
    return value, segments, errors


def _segment_for(seconds: float, segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    for index, segment in enumerate(segments):
        start = float(segment["srt_start_seconds"])
        end = float(segment["srt_end_seconds"])
        if start <= seconds < end or (index == len(segments) - 1 and start <= seconds <= end):
            return segment
    return None


def _human_anchor_errors(anchors: list[dict[str, Any]], *, last_end_seconds: float) -> list[str]:
    errors: list[str] = []
    if len(anchors) < 3:
        errors.append("초·중·후반을 확인한 사람 청취 앵커가 최소 3개 필요합니다.")
        return errors
    if any("human" not in str(anchor.get("source", "")).lower() for anchor in anchors):
        errors.append("모든 앵커 source에는 human 직접 확인 계보가 명시돼야 합니다.")
    thirds = [False, False, False]
    for anchor in anchors:
        srt_time = float(anchor["srt_time_seconds"])
        media_time = float(anchor["media_time_seconds"])
        offset = float(anchor["offset_seconds"])
        if not all(math.isfinite(value) for value in (srt_time, media_time, offset)):
            errors.append(f"앵커 {anchor.get('anchor_id')}의 시간값은 유한한 숫자여야 합니다.")
            continue
        if srt_time < 0 or srt_time > last_end_seconds:
            errors.append(f"앵커 {anchor.get('anchor_id')}가 SRT 범위를 벗어납니다.")
            continue
        bucket = min(2, int((srt_time / max(last_end_seconds, 1e-9)) * 3))
        thirds[bucket] = True
    if not all(thirds):
        errors.append("사람 청취 앵커가 작품의 초·중·후반을 모두 포함하지 않습니다.")
    return errors


def evaluate_pilot_alignment(
    title_id: str,
    structure_path: Path,
    media_path: Path,
    anchors_path: Path | None,
    offset_map_path: Path | None,
    *,
    expected_blocks: int,
    tolerance_seconds: float = 0.25,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Evaluate the hard gate for a human-confirmed, full-title timeline.

    The result is safe to persist even when unresolved. Only ``status=resolved``
    permits downstream clip generation.
    """
    if expected_blocks < 1:
        raise ValueError("expected_blocks는 1 이상이어야 합니다.")
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds는 0 이상이어야 합니다.")
    blocks, _, _ = parse_srt(structure_path)
    last_end = max((block.end_seconds for block in blocks), default=0.0)
    errors: list[str] = []
    if len(blocks) != expected_blocks:
        errors.append(f"자막 블록 수가 {expected_blocks}개가 아닙니다: {len(blocks)}")
    if [block.number for block in blocks] != list(range(1, expected_blocks + 1)):
        errors.append(f"자막 번호가 1부터 {expected_blocks}까지 연속되지 않습니다.")
    if not media_path.exists():
        errors.append("원음 파일이 없습니다.")

    anchors: list[dict[str, Any]] = []
    if anchors_path is None or not anchors_path.exists():
        errors.append("사람 확인 timeline anchor 파일이 없습니다.")
    else:
        try:
            anchors = read_timeline_anchors(anchors_path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    errors.extend(_human_anchor_errors(anchors, last_end_seconds=last_end))

    offset_value, segments, offset_errors = _load_offset_map(offset_map_path, last_end_seconds=last_end)
    errors.extend(offset_errors)
    anchor_ids = {str(anchor.get("anchor_id", "")) for anchor in anchors}
    evidence_refs = set(offset_value.get("evidence_refs", [])) if isinstance(offset_value, dict) and isinstance(offset_value.get("evidence_refs"), list) else set()
    missing_anchor_refs = sorted(anchor_ids - evidence_refs)
    if missing_anchor_refs:
        errors.append(f"offset map evidence_refs에 앵커가 누락됐습니다: {', '.join(missing_anchor_refs)}")

    media_duration: float | None = None
    probe: dict[str, Any] | None = None
    if media_path.exists():
        try:
            if duration_seconds is None:
                media_duration, probe = media_duration_seconds(media_path)
            else:
                media_duration, probe = float(duration_seconds), {"source": "provided"}
            if not math.isfinite(media_duration) or media_duration <= 0:
                errors.append("원음 duration은 0보다 커야 합니다.")
        except RuntimeError as exc:
            errors.append(str(exc))

    for anchor in anchors:
        segment = _segment_for(float(anchor["srt_time_seconds"]), segments)
        if segment is None:
            errors.append(f"앵커 {anchor['anchor_id']}에 적용할 offset segment가 없습니다.")
            continue
        predicted = float(anchor["srt_time_seconds"]) + float(segment["offset_seconds"])
        if abs(predicted - float(anchor["media_time_seconds"])) > tolerance_seconds:
            errors.append(f"앵커 {anchor['anchor_id']}와 offset map의 차이가 허용 오차를 넘습니다.")

    mappings: list[dict[str, Any]] = []
    if segments:
        for block in blocks:
            start_segment = _segment_for(block.start_seconds, segments)
            end_segment = _segment_for(block.end_seconds, segments)
            if start_segment is None or end_segment is None:
                errors.append(f"블록 {block.number}에 적용할 offset segment가 없습니다.")
                continue
            media_start = block.start_seconds + float(start_segment["offset_seconds"])
            media_end = block.end_seconds + float(end_segment["offset_seconds"])
            if media_start < 0 or media_end <= media_start or (media_duration is not None and media_end > media_duration + tolerance_seconds):
                errors.append(f"블록 {block.number}의 매핑된 원음 범위가 유효하지 않습니다.")
                continue
            mappings.append({
                "block_number": block.number,
                "srt_start_seconds": block.start_seconds,
                "srt_end_seconds": block.end_seconds,
                "media_start_seconds": round(media_start, 6),
                "media_end_seconds": round(media_end, 6),
                "start_segment_id": start_segment["segment_id"],
                "end_segment_id": end_segment["segment_id"],
            })
    if len(mappings) != len(blocks):
        errors.append(f"전 블록 원음 매핑이 완결되지 않았습니다: {len(mappings)}/{len(blocks)}")

    resolved = not errors
    return {
        "schema_name": ALIGNMENT_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "title_id": title_id,
        "status": "resolved" if resolved else "unresolved",
        "clip_preparation_allowed": resolved,
        "expected_block_count": expected_blocks,
        "mapped_block_count": len(mappings),
        "tolerance_seconds": tolerance_seconds,
        "structure": {**_file_ref(structure_path), "block_count": len(blocks), "last_end_seconds": last_end},
        "media": {**_file_ref(media_path), "duration_seconds": media_duration, "probe": probe} if media_path.exists() else None,
        "human_anchors": anchors,
        "human_anchor_source": _file_ref(anchors_path) if anchors_path is not None and anchors_path.exists() else None,
        "offset_map": {
            "source": _file_ref(offset_map_path),
            "approval": offset_value,
            "segments": segments,
        } if offset_map_path is not None and offset_map_path.exists() else None,
        "block_mappings": mappings,
        "errors": errors,
    }


def write_pilot_alignment(output_path: Path, report: dict[str, Any]) -> None:
    if output_path.exists():
        raise FileExistsError(f"기존 pilot timeline alignment를 덮어쓰지 않습니다: {output_path}")
    write_json(output_path, report)


def write_blocked_audio_review_manifest(
    title_id: str,
    alignment_path: Path,
    output_dir: Path,
    *,
    expected_blocks: int,
    reason: str,
) -> dict[str, Any]:
    """Persist an auditable stop record without creating clips or review rows."""
    if output_dir.exists():
        raise FileExistsError(f"기존 audio review 디렉터리를 덮어쓰지 않습니다: {output_dir}")
    manifest = {
        "schema_name": AUDIO_REVIEW_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "title_id": title_id,
        "status": "blocked",
        "packet_scope": "full-title",
        "local_only": True,
        "external_delivery_performed": False,
        "clip_preparation_allowed": False,
        "expected_block_count": expected_blocks,
        "block_count": 0,
        "all_blocks_have_audio": False,
        "all_blocks_have_japanese": False,
        "all_blocks_have_scene_context": False,
        "inputs": {"alignment": _file_ref(alignment_path)} if alignment_path.exists() else {},
        "generation_parameters": None,
        "records": None,
        "clips": [],
        "errors": [reason],
        "resume_condition": "사람 확인 앵커와 승인 offset map으로 status=resolved인 새 alignment가 필요합니다.",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _validate_alignment_for_packet(alignment: dict[str, Any], structure_path: Path, media_path: Path, expected_blocks: int) -> None:
    errors: list[str] = []
    if alignment.get("schema_name") != ALIGNMENT_SCHEMA_NAME:
        errors.append("pilot timeline alignment schema_name이 잘못되었습니다.")
    if alignment.get("status") != "resolved" or alignment.get("clip_preparation_allowed") is not True:
        errors.append("시간축이 resolved 상태가 아니어서 청취 패킷 생성을 중단합니다.")
    if alignment.get("expected_block_count") != expected_blocks:
        errors.append("alignment의 expected_block_count가 요청과 다릅니다.")
    if alignment.get("mapped_block_count") != expected_blocks or len(alignment.get("block_mappings", [])) != expected_blocks:
        errors.append("alignment에 전 블록 매핑이 없습니다.")
    if alignment.get("structure", {}).get("sha256") != sha256_file(structure_path):
        errors.append("alignment 생성 이후 구조 SRT가 변경되었습니다.")
    if alignment.get("media", {}).get("sha256") != sha256_file(media_path):
        errors.append("alignment 생성 이후 원음이 변경되었습니다.")
    if errors:
        raise PilotAlignmentError(" ".join(errors))


def _ffmpeg_extractor(media_path: Path, destination: Path, start_seconds: float, duration_seconds: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg를 찾을 수 없어 원음 클립을 생성할 수 없습니다.")
    command = [
        ffmpeg,
        "-v", "error",
        "-nostdin",
        "-ss", f"{start_seconds:.6f}",
        "-t", f"{duration_seconds:.6f}",
        "-i", str(media_path),
        "-map_metadata", "-1",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"ffmpeg 클립 생성 실패({completed.returncode}): {completed.stderr.strip()[:400]}")


def _context(blocks: list[SubtitleBlock], index: int, radius: int) -> list[dict[str, Any]]:
    start = max(0, index - radius)
    end = min(len(blocks), index + radius + 1)
    return [{
        "block_number": block.number,
        "timecode": f"{block.start} --> {block.end}",
        "japanese": block.text,
        "is_target": position == index,
    } for position, block in enumerate(blocks[start:end], start)]


def build_pilot_audio_review_packet(
    title_id: str,
    structure_path: Path,
    japanese_path: Path,
    media_path: Path,
    alignment_path: Path,
    output_dir: Path,
    *,
    expected_blocks: int,
    context_before_seconds: float = 2.0,
    context_after_seconds: float = 2.0,
    context_radius: int = 1,
    clip_extractor: Callable[[Path, Path, float, float], None] | None = None,
) -> dict[str, Any]:
    """Create one deterministic original-audio review item for every SRT block."""
    if min(context_before_seconds, context_after_seconds) < 0 or context_radius < 0:
        raise ValueError("청취 문맥 범위는 0 이상이어야 합니다.")
    if output_dir.exists():
        raise FileExistsError(f"기존 audio review 디렉터리를 덮어쓰지 않습니다: {output_dir}")
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    _validate_alignment_for_packet(alignment, structure_path, media_path, expected_blocks)
    structure, _, _ = parse_srt(structure_path)
    japanese, _, _ = parse_srt(japanese_path)
    comparison = compare_structure(structure, japanese)
    if not comparison["pass"]:
        raise ValueError("일본어 SRT가 동결 구조와 일치하지 않습니다.")
    if len(structure) != expected_blocks:
        raise ValueError(f"전수 청취 패킷에는 {expected_blocks}개 블록이 필요합니다.")

    mappings = {int(item["block_number"]): item for item in alignment["block_mappings"]}
    media_duration = float(alignment["media"]["duration_seconds"])
    extractor = clip_extractor or _ffmpeg_extractor
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for index, block in enumerate(japanese):
            mapping = mappings.get(block.number)
            if mapping is None:
                raise PilotAlignmentError(f"블록 {block.number}의 원음 매핑이 없습니다.")
            clip_start = max(0.0, float(mapping["media_start_seconds"]) - context_before_seconds)
            clip_end = min(media_duration, float(mapping["media_end_seconds"]) + context_after_seconds)
            if clip_end <= clip_start:
                raise PilotAlignmentError(f"블록 {block.number}의 클립 범위가 유효하지 않습니다.")
            clip_rel = Path("clips") / f"block-{block.number:04d}.wav"
            clip_path = output_dir / clip_rel
            extractor(media_path, clip_path, clip_start, clip_end - clip_start)
            if not clip_path.is_file() or clip_path.stat().st_size == 0:
                raise RuntimeError(f"블록 {block.number} 원음 클립이 생성되지 않았습니다.")
            records.append({
                "block_number": block.number,
                "srt_timecode": f"{block.start} --> {block.end}",
                "media_start_seconds": round(clip_start, 6),
                "media_end_seconds": round(clip_end, 6),
                "japanese": block.text,
                "scene_context": _context(japanese, index, context_radius),
                "audio": {"path": clip_rel.as_posix(), "sha256": sha256_file(clip_path), "size_bytes": clip_path.stat().st_size},
            })

        records_path = output_dir / "blocks.jsonl"
        records_path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8", newline="\n")
        manifest = {
            "schema_name": AUDIO_REVIEW_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "title_id": title_id,
            "status": "ready",
            "packet_scope": "full-title",
            "local_only": True,
            "external_delivery_performed": False,
            "clip_preparation_allowed": True,
            "expected_block_count": expected_blocks,
            "block_count": len(records),
            "all_blocks_have_audio": len(records) == expected_blocks and all(record["audio"]["sha256"] for record in records),
            "all_blocks_have_japanese": len(records) == expected_blocks and all("japanese" in record for record in records),
            "all_blocks_have_scene_context": len(records) == expected_blocks and all(record["scene_context"] for record in records),
            "inputs": {
                "structure": _file_ref(structure_path),
                "japanese": _file_ref(japanese_path),
                "media": _file_ref(media_path),
                "alignment": _file_ref(alignment_path),
            },
            "generation_parameters": {
                "context_before_seconds": context_before_seconds,
                "context_after_seconds": context_after_seconds,
                "context_radius": context_radius,
                "audio_format": "wav/pcm_s16le/mono/16000Hz",
                "clip_naming": "clips/block-{block_number:04d}.wav",
            },
            "records": {"path": "blocks.jsonl", "sha256": sha256_file(records_path)},
            "clips": [record["audio"] for record in records],
        }
        write_json(output_dir / "manifest.json", manifest)
        return manifest
    except Exception:
        # Partial clips are never a review packet: without a manifest the gate stays closed.
        raise
