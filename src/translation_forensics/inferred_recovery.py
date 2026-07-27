from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .srt import SubtitleBlock, compare_structure, has_japanese, parse_srt, write_srt


QUEUE_FIELDS = [
    "block_number",
    "timecode",
    "source_japanese",
    "provisional_korean",
    "confidence",
    "uncertain_scope",
    "required_evidence",
    "forensics_risk",
    "review_band",
    "priority_action",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {index}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path} line {index}: record must be an object")
        rows.append(row)
    return rows


def held_block_numbers(path: Path, title: str) -> list[int]:
    """Return only v3 blocks deliberately marked as unsupported.

    The selection is deliberately tied to the v3 ledger rather than scanning
    for an ellipsis in an SRT: an ellipsis can be a legitimate subtitle.
    """

    numbers: list[int] = []
    for row in _read_jsonl(path):
        if row.get("title_id") != title or row.get("applied_method") != "targeted-hold-marker":
            continue
        try:
            number = int(row["block_number"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid held block number") from exc
        numbers.append(number)
    if not numbers:
        raise ValueError(f"{title}: v3 hold ledger contains no targeted-hold-marker blocks")
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"{title}: v3 hold ledger contains duplicate block numbers")
    return sorted(numbers)


def build_inference_audio_queue(
    *,
    title: str,
    structure_path: Path,
    hold_ledger_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write a review queue that extracts one exact clip for every held block."""

    blocks, _, _ = parse_srt(structure_path)
    by_number = {block.number: block for block in blocks}
    numbers = held_block_numbers(hold_ledger_path, title)
    missing = [number for number in numbers if number not in by_number]
    if missing:
        raise ValueError(f"{title}: held blocks absent from structure: {missing[:10]}")
    if output_path.exists():
        raise FileExistsError(f"inference queue already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_FIELDS)
        writer.writeheader()
        for number in numbers:
            block = by_number[number]
            writer.writerow(
                {
                    "block_number": number,
                    "timecode": f"{block.start} --> {block.end}",
                    "source_japanese": block.text,
                    "provisional_korean": "",
                    "confidence": "low",
                    "uncertain_scope": "japanese-reference-corruption",
                    "required_evidence": "exact-block-local-asr",
                    "forensics_risk": "inference-only",
                    "review_band": "P1",
                    "priority_action": "local-asr-inference",
                }
            )
    return {
        "status": "inference-audio-queue-built",
        "title_id": title,
        "held_blocks": len(numbers),
        "queue": str(output_path),
        "selection": "targeted-hold-marker from v3 ledger",
        "human_reviewed": False,
        "final_promotion_allowed": False,
    }


def _read_scenes(path: Path) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for raw in (row.get("block_numbers") or "").split(","):
                if not raw.strip():
                    continue
                number = int(raw)
                if number in result:
                    raise ValueError(f"{path}: block {number} is assigned to multiple ASR scenes")
                result[number] = dict(row)
    return result


def _read_asr_by_scene(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[str(row.get("scene_id") or "")].append(dict(row))
    return {scene: sorted(rows, key=lambda row: str(row.get("profile") or "")) for scene, rows in grouped.items()}


def build_inference_context(
    *,
    title: str,
    structure_path: Path,
    source_path: Path,
    viewer_path: Path,
    hold_ledger_path: Path,
    scenes_path: Path,
    asr_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Make a compact, reproducible model input from exact clips and local context."""

    structure, _, _ = parse_srt(structure_path)
    source, _, _ = parse_srt(source_path)
    viewer, _, _ = parse_srt(viewer_path)
    if not compare_structure(structure, source)["pass"] or not compare_structure(structure, viewer)["pass"]:
        raise ValueError("inference input SRTs do not match the Japanese structure")
    numbers = held_block_numbers(hold_ledger_path, title)
    scenes = _read_scenes(scenes_path)
    asr = _read_asr_by_scene(asr_path)
    by_number = {block.number: block for block in structure}
    source_by_number = {block.number: block.text for block in source}
    viewer_by_number = {block.number: block.text for block in viewer}
    missing_scenes = [number for number in numbers if number not in scenes]
    if missing_scenes:
        raise ValueError(f"{title}: ASR scene missing held blocks: {missing_scenes[:10]}")
    targets: list[dict[str, Any]] = []
    ordered = [block.number for block in structure]
    positions = {number: index for index, number in enumerate(ordered)}
    for number in numbers:
        block = by_number[number]
        scene = scenes[number]
        context: list[dict[str, Any]] = []
        start = max(0, positions[number] - 2)
        end = min(len(ordered), positions[number] + 3)
        for neighbor_number in ordered[start:end]:
            neighbor = by_number[neighbor_number]
            context.append(
                {
                    "block_number": neighbor_number,
                    "timecode": f"{neighbor.start} --> {neighbor.end}",
                    "japanese_srt": neighbor.text,
                    "current_source_korean": source_by_number[neighbor_number],
                    "current_viewer_korean": viewer_by_number[neighbor_number],
                    "is_target": neighbor_number == number,
                }
            )
        targets.append(
            {
                "title": title,
                "block_number": number,
                "timecode": f"{block.start} --> {block.end}",
                "japanese_srt": block.text,
                "asr_scene": {
                    "scene_id": scene["scene_id"],
                    "scene_timecode": f"{scene['start_time']} --> {scene['end_time']}",
                    "profiles": [
                        {
                            "profile": row.get("profile", ""),
                            "text": row.get("text", ""),
                            "avg_logprob": row.get("avg_logprob", ""),
                            "no_speech_prob": row.get("no_speech_prob", ""),
                            "error": row.get("error", ""),
                            "source_family": "whisper-family",
                        }
                        for row in asr.get(scene["scene_id"], [])
                    ],
                },
                "local_context": context,
            }
        )
    if output_path.exists():
        raise FileExistsError(f"inference context already exists: {output_path}")
    payload = {
        "schema_name": "translation-forensics/inferred-recovery-input",
        "schema_version": "1",
        "title_id": title,
        "target_count": len(targets),
        "evidence_limits": {
            "human_reviewed": False,
            "asr_is_single_family": True,
            "audio_is_local": True,
            "final_promotion_allowed": False,
            "purpose": "authorized speculative recovery of v3 hold markers",
        },
        "targets": targets,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {
        "status": "inference-context-built",
        "title_id": title,
        "targets": len(targets),
        "output": str(output_path),
        "human_reviewed": False,
        "final_promotion_allowed": False,
    }


def _read_inference_response(path: Path, title: str) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = raw.get("results") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path}: results must be an array")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or row.get("title") != title:
            continue
        try:
            number = int(row["block_number"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path} results[{index}]: invalid block_number") from exc
        decision = row.get("decision")
        source = str(row.get("source_faithful_korean", "")).strip()
        viewer = str(row.get("viewer_natural_korean", "")).strip()
        confidence = row.get("confidence")
        basis = row.get("basis")
        if not isinstance(basis, list) or not all(isinstance(item, str) for item in basis):
            raise ValueError(f"{path} block {number}: basis must be a string array")
        if decision == "infer-replace":
            if not source or not viewer or has_japanese(source) or has_japanese(viewer):
                raise ValueError(f"{path} block {number}: inferred replacement must be non-empty Korean text")
            if confidence not in {"medium", "low"}:
                raise ValueError(f"{path} block {number}: inferred replacement confidence must be medium or low")
            if "asr_same_family" not in basis:
                raise ValueError(f"{path} block {number}: inferred replacement must disclose ASR-family basis")
        elif decision == "infer-hold":
            if source or viewer or confidence != "low":
                raise ValueError(f"{path} block {number}: infer-hold contract violated")
        else:
            raise ValueError(f"{path} block {number}: decision must be infer-replace or infer-hold")
        if not str(row.get("reason") or "").strip():
            raise ValueError(f"{path} block {number}: reason is required")
        result.append({**row, "block_number": number})
    return result


def apply_inferred_recovery(
    *,
    title: str,
    structure_path: Path,
    source_path: Path,
    viewer_path: Path,
    hold_ledger_path: Path,
    response_paths: list[Path],
    output_dir: Path,
    hold_marker: str = "…",
    version: str = "v4",
) -> dict[str, Any]:
    """Apply user-authorized *inferred* replacements without relabeling them final."""

    if not hold_marker.strip():
        raise ValueError("hold marker must not be empty")
    structure, _, _ = parse_srt(structure_path)
    source, _, _ = parse_srt(source_path)
    viewer, _, _ = parse_srt(viewer_path)
    if not compare_structure(structure, source)["pass"] or not compare_structure(structure, viewer)["pass"]:
        raise ValueError("inference input SRTs do not match the Japanese structure")
    targets = set(held_block_numbers(hold_ledger_path, title))
    decisions: dict[int, dict[str, Any]] = {}
    for response_path in response_paths:
        for row in _read_inference_response(response_path, title):
            number = row["block_number"]
            if number not in targets:
                raise ValueError(f"{title}: response contains non-held block {number}")
            if number in decisions:
                raise ValueError(f"{title}: duplicate inferred response for block {number}")
            decisions[number] = row
    missing = sorted(targets - set(decisions))
    if missing:
        raise ValueError(f"{title}: inferred response omits held blocks: {missing[:10]}")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"inferred recovery output already exists: {output_dir}")
    source_by_number = {block.number: block for block in source}
    viewer_by_number = {block.number: block for block in viewer}
    new_source: list[SubtitleBlock] = []
    new_viewer: list[SubtitleBlock] = []
    ledger: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for block in structure:
        decision = decisions.get(block.number)
        if decision is None:
            source_text = source_by_number[block.number].text
            viewer_text = viewer_by_number[block.number].text
            method = "unchanged-from-v3"
        elif decision["decision"] == "infer-replace":
            source_text = str(decision["source_faithful_korean"]).strip()
            viewer_text = str(decision["viewer_natural_korean"]).strip()
            method = "inferred-audio-recovery"
        else:
            source_text = hold_marker
            viewer_text = hold_marker
            method = "inferred-recovery-hold-marker"
        counts[method] += 1
        new_source.append(SubtitleBlock(block.number, block.start, block.end, source_text, block.start_seconds, block.end_seconds))
        new_viewer.append(SubtitleBlock(block.number, block.start, block.end, viewer_text, block.start_seconds, block.end_seconds))
        if decision is not None:
            ledger.append(
                {
                    "title_id": title,
                    "block_number": block.number,
                    "timecode": f"{block.start} --> {block.end}",
                    "decision": decision["decision"],
                    "applied_method": method,
                    "confidence": decision["confidence"],
                    "basis": decision["basis"],
                    "reason": decision["reason"],
                    "inference": True,
                    "human_reviewed": False,
                    "machine_final_allowed": False,
                    "human_final_allowed": False,
                    "final_promotion_allowed": False,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=False)
    source_output = output_dir / f"{title}.source-faithful-ko.inferred-recovery-{version}.srt"
    viewer_output = output_dir / f"{title}.viewer-natural-ko.inferred-recovery-{version}.srt"
    ledger_path = output_dir / "inferred-recovery-decisions.jsonl"
    report_path = output_dir / "inferred-recovery-report.json"
    write_srt(source_output, new_source)
    write_srt(viewer_output, new_viewer)
    ledger_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ledger), encoding="utf-8", newline="\n")
    report = {
        "schema_name": "translation-forensics/inferred-recovery-report",
        "schema_version": "1",
        "title_id": title,
        "version": version,
        "status": "inferred-recovery-applied",
        "blocks": len(structure),
        "targeted_blocks": len(targets),
        "inferred_replacement_blocks": counts["inferred-audio-recovery"],
        "remaining_hold_marker_blocks": counts["inferred-recovery-hold-marker"],
        "unchanged_blocks": counts["unchanged-from-v3"],
        "hold_marker": hold_marker,
        "inference": True,
        "human_reviewed": False,
        "machine_final_allowed": False,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
        "note": "User-authorized local-audio inference. It is not human verification or a final release.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {
        "status": report["status"],
        "source_output": str(source_output),
        "viewer_output": str(viewer_output),
        "ledger": str(ledger_path),
        "report": str(report_path),
        "inferred_replacement_blocks": report["inferred_replacement_blocks"],
        "remaining_hold_marker_blocks": report["remaining_hold_marker_blocks"],
        "final_promotion_allowed": False,
    }
