from __future__ import annotations

import json
import csv
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import sha256_file
from .srt import parse_srt


PASS_STATUSES = {"exact-compatible", "manual-override"}
BLOCKING_STATUSES = {"media-too-short", "missing-media", "probe-failed", "unresolved"}


def media_duration_seconds(path: Path) -> tuple[float, dict[str, Any]]:
    """Read one media duration with ffprobe; never infer it from a filename."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe를 찾을 수 없어 미디어 시간축을 검증할 수 없습니다.")
    completed = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"ffprobe 실패({completed.returncode}): {completed.stderr.strip()[:400]}")
    try:
        payload = json.loads(completed.stdout)
        duration = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("ffprobe 출력에 유효한 duration이 없습니다.") from exc
    if duration <= 0:
        raise RuntimeError("미디어 duration이 0 이하입니다.")
    return duration, {"ffprobe": ffprobe, "raw": payload}


ANCHOR_FIELDS = ("anchor_id", "srt_time_seconds", "media_time_seconds", "source")


def read_timeline_anchors(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(ANCHOR_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"timeline anchor 필수 열 누락: {', '.join(sorted(missing))}")
        anchors: list[dict[str, Any]] = []
        for index, row in enumerate(reader, 2):
            try:
                srt_time, media_time = float(row["srt_time_seconds"]), float(row["media_time_seconds"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"timeline anchor {index}행 시간값이 잘못되었습니다.") from exc
            if min(srt_time, media_time) < 0:
                raise ValueError(f"timeline anchor {index}행 시간값은 0 이상이어야 합니다.")
            anchors.append({"anchor_id": row["anchor_id"], "srt_time_seconds": srt_time, "media_time_seconds": media_time, "source": row["source"], "offset_seconds": round(media_time - srt_time, 6)})
    if len({anchor["anchor_id"] for anchor in anchors}) != len(anchors):
        raise ValueError("timeline anchor_id가 중복됩니다.")
    return anchors


def initialize_timeline_anchor_template(structure_path: Path, output_path: Path) -> dict[str, Any]:
    """Suggest early/middle/late SRT anchor points; media times remain human input."""
    if output_path.exists():
        raise FileExistsError(f"기존 timeline anchor template을 덮어쓰지 않습니다: {output_path}")
    blocks, _, _ = parse_srt(structure_path)
    if len(blocks) < 3:
        raise ValueError("timeline anchor template에는 최소 3개 SRT 블록이 필요합니다.")
    indices = sorted({round((len(blocks) - 1) * ratio) for ratio in (0.1, 0.5, 0.9)})
    while len(indices) < 3:
        indices.append(len(indices))
    rows = []
    for label, index in zip(("early", "middle", "late"), indices):
        block = blocks[index]
        rows.append({"anchor_id": f"A-{label}", "srt_time_seconds": round((block.start_seconds + block.end_seconds) / 2, 3), "media_time_seconds": "", "source": "human-direct-listening-required", "suggested_block_number": block.number, "srt_timecode": f"{block.start} --> {block.end}"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=["anchor_id", "srt_time_seconds", "media_time_seconds", "source", "suggested_block_number", "srt_timecode"])
        writer.writeheader(); writer.writerows(rows)
    return {"status": "timeline-anchor-template", "output": str(output_path), "anchors": len(rows), "note": "media_time_seconds는 사람 직접 청취·확인 전에는 비워 둡니다."}


def validate_offset_map(path: Path, anchors: list[dict[str, Any]], *, tolerance_seconds: float) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for field in ("approved_by", "evidence_refs", "scope", "offset_seconds"):
        if field not in value:
            errors.append(f"approved offset map에 {field}가 없습니다.")
    if not isinstance(value.get("evidence_refs"), list) or not value.get("evidence_refs"):
        errors.append("approved offset map에는 evidence_refs가 필요합니다.")
    try:
        offset = float(value.get("offset_seconds"))
    except (TypeError, ValueError):
        errors.append("approved offset map의 offset_seconds가 숫자가 아닙니다."); offset = 0.0
    if anchors and any(abs(anchor["offset_seconds"] - offset) > tolerance_seconds for anchor in anchors):
        errors.append("approved offset map이 앵커 오프셋과 일치하지 않습니다.")
    return {"status": "pass" if not errors else "fail", "value": value, "errors": errors}


def validate_timeline(
    structure_path: Path,
    media_path: Path | None,
    *,
    tolerance_seconds: float = 0.25,
    duration_seconds: float | None = None,
    anchors: list[dict[str, Any]] | None = None,
    approved_offset_map: Path | None = None,
) -> dict[str, Any]:
    """Check the safety property needed before clipping: every SRT end fits media.

    A media file that is longer than the last subtitle is *not* declared an exact
    edit match. It is only range-compatible, so offset/drift still needs evidence.
    """
    if tolerance_seconds < 0:
        raise ValueError("timeline tolerance은 0 이상이어야 합니다.")
    blocks, _, _ = parse_srt(structure_path)
    last_end = max((block.end_seconds for block in blocks), default=0.0)
    report: dict[str, Any] = {
        "schema_name": "translation-forensics/timeline-validation",
        "schema_version": "2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "structure": {"path": str(structure_path.resolve()), "sha256": sha256_file(structure_path), "last_end_seconds": last_end, "block_count": len(blocks)},
        "media": None,
        "tolerance_seconds": tolerance_seconds,
        "offset_map": None,
        "automatic_offset_applied": False,
        "anchors": anchors or [],
    }
    if media_path is None or not media_path.exists():
        report.update({"status": "missing-media", "clip_preparation_allowed": False, "reason": "검증할 오디오/비디오 입력이 없습니다."})
        return report
    media_path = media_path.resolve()
    report["media"] = {"path": str(media_path), "sha256": sha256_file(media_path)}
    try:
        media_duration, probe = (duration_seconds, {"source": "provided-for-test"}) if duration_seconds is not None else media_duration_seconds(media_path)
    except RuntimeError as exc:
        report.update({"status": "probe-failed", "clip_preparation_allowed": False, "reason": str(exc)})
        return report
    if media_duration is None or media_duration <= 0:
        report.update({"status": "probe-failed", "clip_preparation_allowed": False, "reason": "유효한 미디어 duration이 없습니다."})
        return report
    delta = round(float(media_duration) - last_end, 6)
    report["media"].update({"duration_seconds": float(media_duration), "probe": probe})
    report["duration_delta_seconds"] = delta
    if float(media_duration) + tolerance_seconds < last_end:
        report.update({
            "status": "media-too-short",
            "clip_preparation_allowed": False,
            "reason": "구조 SRT의 마지막 종료 시각이 미디어 범위를 벗어납니다.",
        })
    else:
        offsets = [anchor["offset_seconds"] for anchor in anchors or []]
        if len(offsets) < 3:
            report.update({"status": "unresolved", "clip_preparation_allowed": False, "reason": "미디어 범위는 충분하지만 초·중·후반 다중 앵커가 없어 오프셋·드리프트를 검증할 수 없습니다."})
        elif max(offsets) - min(offsets) <= tolerance_seconds and max(abs(offset) for offset in offsets) <= tolerance_seconds:
            report.update({"status": "exact-compatible", "clip_preparation_allowed": True, "reason": "초·중·후반 앵커가 허용 오차 안에서 같은 시간축을 지지합니다."})
        elif max(offsets) - min(offsets) <= tolerance_seconds:
            offset_report = validate_offset_map(approved_offset_map, anchors or [], tolerance_seconds=tolerance_seconds) if approved_offset_map else {"status": "missing", "errors": ["승인된 offset map이 없습니다."]}
            report["offset_map"] = offset_report
            if offset_report["status"] == "pass":
                report.update({"status": "manual-override", "clip_preparation_allowed": True, "reason": "일정 오프셋은 승인된 offset map 범위 안에서만 적용됩니다."})
            else:
                report.update({"status": "constant-offset", "clip_preparation_allowed": False, "reason": "일정 오프셋이 감지됐지만 승인된 offset map이 없어 자동 적용하지 않습니다."})
        else:
            report.update({"status": "drift-suspected", "clip_preparation_allowed": False, "reason": "앵커 오프셋이 구간별로 달라 드리프트 또는 다른 편집본이 의심됩니다."})
    return report


def timeline_is_usable(report: dict[str, Any]) -> bool:
    return report.get("status") in PASS_STATUSES and report.get("clip_preparation_allowed") is True
