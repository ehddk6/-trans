from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .alignment import align_by_overlap
from .asr_evidence import classify_scene
from .scenes import scenes_from_csv
from .srt import SubtitleBlock, parse_srt


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _by_number(blocks: list[SubtitleBlock]) -> dict[int, SubtitleBlock]:
    return {block.number: block for block in blocks}


def _aligned_texts(source: list[SubtitleBlock], candidate: list[SubtitleBlock]) -> dict[int, str]:
    by_number = _by_number(candidate)
    return {alignment.source_number: by_number[alignment.candidate_number].text for alignment in align_by_overlap(source, candidate) if alignment.candidate_number in by_number}


def build_change_log(reference: list[SubtitleBlock], previous: list[SubtitleBlock] | None, source: list[SubtitleBlock], viewer: list[SubtitleBlock], *, status: str, japanese: list[SubtitleBlock] | None = None) -> list[dict[str, object]]:
    previous_map = _by_number(previous or [])
    source_map = _by_number(source)
    viewer_map = _by_number(viewer)
    japanese_map = _by_number(japanese or [])
    previous_aligned = _aligned_texts(reference, previous or [])
    rows = []
    for block in reference:
        source_text = source_map.get(block.number, block).text
        viewer_text = viewer_map.get(block.number, block).text
        previous_text = previous_aligned.get(block.number, "")
        change_type = "unchanged" if source_text == previous_text and viewer_text == source_text else "reconstructed_or_naturalized"
        rows.append({
            "block_number": block.number, "start_time": block.start, "end_time": block.end,
            "source_japanese": japanese_map.get(block.number).text if block.number in japanese_map else "", "previous_korean": previous_text,
            "source_faithful_korean": source_text, "viewer_natural_korean": viewer_text,
            "change_type": change_type, "main_issue": "", "evidence_summary": "",
            "confidence": "low", "review_note": f"verification_status={status}",
        })
    return rows


def build_evidence_ledger(reference: list[SubtitleBlock], *, status: str, asr_rows: list[dict[str, str]] | None = None, japanese_available: bool = False, previous_available: bool = False, photos_available: bool = False, screen_available: bool = False) -> list[dict[str, object]]:
    asr_by_scene: dict[str, list[dict[str, str]]] = {}
    for row in asr_rows or []:
        asr_by_scene.setdefault(row.get("scene_id", ""), []).append(row)
    return [{
        "block_number": block.number,
        "timecode": f"{block.start} --> {block.end}",
        "japanese_reference": "available" if japanese_available else "not_provided",
        "alternative_japanese_asr": "available" if asr_rows else "not_provided",
        "original_unbiased_asr": "not_mapped",
        "dialogue_unbiased_asr": "not_mapped",
        "original_no_vad_asr": "not_mapped",
        "original_prompted_asr": "not_mapped",
        "photos": "available_as_path_only" if photos_available else "not_provided",
        "screen": "available_as_path_only" if screen_available else "not_provided",
        "neighboring_context": "review_required",
        "previous_korean": "available" if previous_available else "not_provided",
        "error_memory": "not_auto_applied",
        "verification_status": status,
        "evidence_independence_note": "동일 Whisper 모델 패스는 독립 증거로 세지 않음",
    } for block in reference]


def build_uncertainty_map(reference: list[SubtitleBlock], *, status: str) -> list[dict[str, object]]:
    return [{
        "block_number": block.number,
        "uncertain_block": "yes",
        "uncertain_slots": "meaning decision not supplied by code",
        "possible_meaning_range": "not automatically inferred",
        "adopted_broad_expression": "",
        "additional_evidence_needed": "원음·문맥·일본어 검토",
        "current_verification_status": status,
        "impact": "미확정 세부사항을 창작으로 채우지 않음",
    } for block in reference]


def build_asr_verdicts(scenes_path: Path, asr_path: Path | None) -> list[dict[str, object]]:
    scenes = scenes_from_csv(scenes_path)
    asr_rows = read_csv(asr_path) if asr_path and asr_path.exists() else []
    by_scene: dict[str, list[dict[str, str]]] = {}
    for row in asr_rows:
        by_scene.setdefault(row.get("scene_id", ""), []).append(row)
    return [{
        "scene_id": scene.get("scene_id", ""),
        "verdict": classify_scene(by_scene.get(scene.get("scene_id", ""), [])) if asr_rows else "unresolved",
        "confidence": "low" if not asr_rows else "medium",
        "evidence_summary": "ASR 결과 미제공" if not asr_rows else "동일 모델 다중 패스 후보",
        "unresolved_scope": "직접 청취·의미 판정 미수행" if not asr_rows else "자동 후보만 존재",
        "direct_human_listening": False,
    } for scene in scenes]


def build_scene_map(scenes_path: Path | None, asr_path: Path | None = None) -> list[dict[str, object]]:
    if not scenes_path or not scenes_path.exists():
        return []
    verdicts = {row["scene_id"]: row for row in build_asr_verdicts(scenes_path, asr_path)}
    result = []
    for scene in scenes_from_csv(scenes_path):
        verdict = verdicts.get(scene.get("scene_id", ""), {})
        result.append({
            "scene_id": scene.get("scene_id", ""),
            "start_time": scene.get("start_time", ""),
            "end_time": scene.get("end_time", ""),
            "duration_sec": scene.get("duration_sec", ""),
            "block_numbers": scene.get("block_numbers", ""),
            "highest_band": scene.get("highest_band", ""),
            "original_audio": scene.get("original_audio", ""),
            "dialogue_audio": scene.get("dialogue_audio", ""),
            "asr_verdict": verdict.get("verdict", "unresolved"),
            "asr_confidence": verdict.get("confidence", "low"),
        })
    return result


def build_review_context(ja_path: Path, previous_path: Path | None, scenes_path: Path, asr_path: Path | None, output_path: Path) -> dict[str, object]:
    ja_blocks, _, _ = parse_srt(ja_path)
    previous_blocks = parse_srt(previous_path)[0] if previous_path and previous_path.exists() else []
    previous_map = _aligned_texts(ja_blocks, previous_blocks)
    scenes = scenes_from_csv(scenes_path)
    asr_rows = read_csv(asr_path) if asr_path and asr_path.exists() else []
    asr_by_scene: dict[str, list[dict[str, str]]] = {}
    for row in asr_rows:
        asr_by_scene.setdefault(row.get("scene_id", ""), []).append(row)
    block_map = _by_number(ja_blocks)
    records = []
    for scene in scenes:
        numbers = [int(x) for x in scene.get("block_numbers", "").split(",") if x.strip().isdigit()]
        blocks = []
        for number in numbers:
            block = block_map.get(number)
            if not block:
                continue
            blocks.append({
                "block_number": number,
                "timecode": f"{block.start} --> {block.end}",
                "source_japanese": block.text,
                "previous_korean": previous_map.get(number, ""),
            })
        records.append({**scene, "blocks": blocks, "asr_candidates": asr_by_scene.get(scene.get("scene_id", ""), [])})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "created", "scenes": len(records), "asr_rows": len(asr_rows), "output": str(output_path)}


def write_qa_markdown(path: Path, validation: dict[str, object], *, title: str, stage: str, status: str, notes: list[str]) -> None:
    lines = [f"# {title} QA 보고서", "", f"- 단계: `{stage}`", f"- 검증 상태: `{status}`", f"- 자동 QA 상태: `{validation.get('status', '미검증')}`", "", "## 자동 검사", "", "```json", json.dumps(validation, ensure_ascii=False, indent=2), "```", "", "## 주의 및 미검증", ""]
    lines.extend(f"- {note}" for note in notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
