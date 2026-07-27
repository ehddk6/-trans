from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

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


def _decision_map(decisions: list[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for decision in decisions or []:
        try:
            result[int(decision.get("block_number"))] = decision
        except (TypeError, ValueError):
            continue
    return result


def _aligned_texts(source: list[SubtitleBlock], candidate: list[SubtitleBlock]) -> dict[int, str]:
    by_number = _by_number(candidate)
    return {
        alignment.source_number: by_number[alignment.candidate_number].text
        for alignment in align_by_overlap(source, candidate)
        if alignment.candidate_number in by_number
    }


def _list_text(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if str(item).strip())
    return str(value or "").strip()


def build_change_log(
    reference: list[SubtitleBlock],
    previous: list[SubtitleBlock] | None,
    source: list[SubtitleBlock],
    viewer: list[SubtitleBlock],
    *,
    status: str,
    japanese: list[SubtitleBlock] | None = None,
    decisions: list[dict[str, Any]] | None = None,
) -> list[dict[str, object]]:
    source_map = _by_number(source)
    viewer_map = _by_number(viewer)
    japanese_map = _by_number(japanese or [])
    previous_aligned = _aligned_texts(reference, previous or [])
    decisions_by_number = _decision_map(decisions)
    rows = []
    for block in reference:
        source_text = source_map.get(block.number, block).text
        viewer_text = viewer_map.get(block.number, block).text
        previous_text = previous_aligned.get(block.number, "")
        decision = decisions_by_number.get(block.number, {})
        change_type = "unchanged" if source_text == previous_text and viewer_text == source_text else "reconstructed_or_naturalized"
        issue_parts = [
            _list_text(decision.get("risk_codes")),
            _list_text(decision.get("uncertain_slots")),
            _list_text(decision.get("abstention_reasons")),
        ]
        main_issue = "; ".join(part for part in issue_parts if part)
        evidence_summary = _list_text(decision.get("evidence_refs"))
        confidence = str(decision.get("confidence") or "unverified")
        review_note = str(decision.get("review_note") or decision.get("reason") or f"verification_status={status}")
        rows.append({
            "block_number": block.number,
            "start_time": block.start,
            "end_time": block.end,
            "source_japanese": japanese_map.get(block.number).text if block.number in japanese_map else str(decision.get("source_japanese") or ""),
            "previous_korean": previous_text,
            "source_faithful_korean": source_text,
            "viewer_natural_korean": viewer_text,
            "change_type": change_type,
            "main_issue": main_issue,
            "evidence_summary": evidence_summary,
            "confidence": confidence,
            "review_note": review_note,
        })
    return rows


def _scene_blocks(scenes_path: Path | None) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    if not scenes_path or not scenes_path.exists():
        return result
    for scene in scenes_from_csv(scenes_path):
        scene_id = str(scene.get("scene_id", ""))
        for raw in str(scene.get("block_numbers", "")).split(","):
            if raw.strip().isdigit():
                result.setdefault(int(raw), []).append(scene_id)
    return result


def _asr_by_scene(asr_rows: list[dict[str, str]] | None) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for row in asr_rows or []:
        result.setdefault(str(row.get("scene_id", "")), []).append(row)
    return result


def _profile_text(rows: list[dict[str, str]], profile: str) -> str:
    values = [str(row.get("text") or "").strip() for row in rows if str(row.get("profile") or "") == profile and str(row.get("text") or "").strip()]
    return " | ".join(dict.fromkeys(values)) if values else "not_mapped"


def build_evidence_ledger(
    reference: list[SubtitleBlock],
    *,
    status: str,
    asr_rows: list[dict[str, str]] | None = None,
    scenes_path: Path | None = None,
    decisions: list[dict[str, Any]] | None = None,
    japanese: list[SubtitleBlock] | None = None,
    previous: list[SubtitleBlock] | None = None,
    japanese_available: bool = False,
    previous_available: bool = False,
    photos_available: bool = False,
    screen_available: bool = False,
) -> list[dict[str, object]]:
    scene_ids_by_block = _scene_blocks(scenes_path)
    asr_by_scene = _asr_by_scene(asr_rows)
    decisions_by_number = _decision_map(decisions)
    japanese_map = _by_number(japanese or [])
    previous_map = _aligned_texts(reference, previous or [])
    result: list[dict[str, object]] = []
    for index, block in enumerate(reference):
        scene_ids = scene_ids_by_block.get(block.number, [])
        block_asr = [row for scene_id in scene_ids for row in asr_by_scene.get(scene_id, [])]
        decision = decisions_by_number.get(block.number, {})
        neighbors = [
            japanese_map[reference[neighbor].number].text
            for neighbor in (index - 1, index + 1)
            if 0 <= neighbor < len(reference) and reference[neighbor].number in japanese_map
        ]
        result.append({
            "block_number": block.number,
            "timecode": f"{block.start} --> {block.end}",
            "japanese_reference": japanese_map.get(block.number).text if block.number in japanese_map else ("available" if japanese_available else "not_provided"),
            "alternative_japanese_asr": " | ".join(dict.fromkeys(str(row.get("text") or "").strip() for row in block_asr if str(row.get("text") or "").strip())) or "not_mapped",
            "original_unbiased_asr": _profile_text(block_asr, "original_unbiased"),
            "dialogue_unbiased_asr": _profile_text(block_asr, "dialogue_unbiased"),
            "original_no_vad_asr": _profile_text(block_asr, "original_no_vad"),
            "original_prompted_asr": _profile_text(block_asr, "original_prompted"),
            "photos": "available_as_path_only" if photos_available else "not_provided",
            "screen": "available_as_path_only" if screen_available else "not_provided",
            "neighboring_context": " | ".join(neighbors) or "not_mapped",
            "previous_korean": previous_map.get(block.number, "available" if previous_available else "not_provided"),
            "decision_status": str(decision.get("status") or decision.get("source_status") or "not_provided"),
            "decision_evidence_refs": _list_text(decision.get("evidence_refs")),
            "error_memory": "not_auto_applied",
            "verification_status": status,
            "evidence_independence_note": "동일 Whisper 모델 패스는 독립 증거로 세지 않음",
        })
    return result


def build_uncertainty_map(
    reference: list[SubtitleBlock],
    *,
    status: str,
    decisions: list[dict[str, Any]] | None = None,
) -> list[dict[str, object]]:
    decisions_by_number = _decision_map(decisions)
    rows: list[dict[str, object]] = []
    for block in reference:
        decision = decisions_by_number.get(block.number)
        if decision is None:
            rows.append({
                "block_number": block.number,
                "uncertain_block": "yes",
                "uncertain_slots": "meaning decision not supplied by code",
                "possible_meaning_range": "not automatically inferred",
                "adopted_broad_expression": "",
                "additional_evidence_needed": "원음·문맥·일본어 검토",
                "current_verification_status": status,
                "impact": "미확정 세부사항을 창작으로 채우지 않음",
            })
            continue
        decision_status = str(decision.get("status") or decision.get("source_status") or "")
        confidence = str(decision.get("confidence") or "")
        uncertain_slots = _list_text(decision.get("uncertain_slots") or decision.get("inferred_slots"))
        competing = _list_text(decision.get("competing_interpretations"))
        risks = _list_text(decision.get("risk_codes") or decision.get("abstention_reasons"))
        uncertain = decision_status in {"abstained", "untranslated", "hold"} or confidence == "low" or bool(uncertain_slots or competing or risks)
        rows.append({
            "block_number": block.number,
            "uncertain_block": "yes" if uncertain else "no",
            "uncertain_slots": uncertain_slots,
            "possible_meaning_range": competing,
            "adopted_broad_expression": str(decision.get("viewer_natural_korean") or decision.get("viewer_text") or ""),
            "additional_evidence_needed": risks if uncertain else "",
            "current_verification_status": decision_status or status,
            "impact": str(decision.get("review_note") or decision.get("reason") or ("미확정 세부사항을 창작으로 채우지 않음" if uncertain else "")),
        })
    return rows


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
