from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .alignment import align_by_overlap
from .srt import JAPANESE_RE, SubtitleBlock, has_japanese, parse_srt, write_srt
from .translation_model import DEFAULT_TRANSLATION_MODEL, resolve_translation_model


DECISION_FIELDS = [
    "block_number",
    "source_japanese",
    "previous_korean",
    "source_faithful_korean",
    "viewer_natural_korean",
    "translation_method",
    "translation_model",
    "status",
    "confidence",
    "uncertain_slots",
    "evidence_refs",
    "review_note",
]

_PLACEHOLDER_RE = re.compile(r"\[(?:번역 필요|불명|미확정|검토 필요)\]|번역 불가|알아들을 수 없")
_HANGUL_RE = re.compile(r"[가-힣]")
_QUESTION_RE = re.compile(r"[?？]|(?:か|かな|の)\s*$")
_NEGATION_RE = re.compile(r"(?:ない|ぬ|ん|ません|ず|だめ|ダメ|嫌|いや|やめ)")
_REQUEST_RE = re.compile(r"(?:ください|下さい|くれ|てくれ|なさい|ろ|て|で)\s*$")


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL {line_number}번 레코드는 객체여야 합니다: {path}")
        records.append(value)
    return records


def _by_number(blocks: Iterable[SubtitleBlock]) -> dict[int, SubtitleBlock]:
    return {block.number: block for block in blocks}


def _aligned_texts(reference: list[SubtitleBlock], candidate: list[SubtitleBlock]) -> dict[int, str]:
    by_number = _by_number(candidate)
    return {
        alignment.source_number: by_number[alignment.candidate_number].text
        for alignment in align_by_overlap(reference, candidate)
        if alignment.candidate_number in by_number
    }


def _aligned_candidates(reference: list[SubtitleBlock], candidate: list[SubtitleBlock]) -> dict[int, dict[str, Any]]:
    by_number = _by_number(candidate)
    return {
        alignment.source_number: {
            "candidate_number": alignment.candidate_number,
            "text": by_number[alignment.candidate_number].text if alignment.candidate_number in by_number else "",
            "confidence": alignment.confidence,
            "overlap_seconds": alignment.overlap_seconds,
            "candidate_start_distance": alignment.candidate_start_distance,
        }
        for alignment in align_by_overlap(reference, candidate)
    }


def _required_slots(japanese: str) -> list[str]:
    slots: list[str] = []
    if _QUESTION_RE.search(japanese.strip()):
        slots.append("question_or_statement")
    if _NEGATION_RE.search(japanese):
        slots.append("negation_refusal_or_prohibition")
    if _REQUEST_RE.search(japanese.strip()):
        slots.append("request_or_command")
    if not slots:
        slots.append("speaker_action_target")
    return slots


def _neighbor_rows(blocks: list[SubtitleBlock], index: int, radius: int = 2) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for neighbor_index in range(max(0, index - radius), min(len(blocks), index + radius + 1)):
        if neighbor_index == index:
            continue
        block = blocks[neighbor_index]
        result.append({"block_number": block.number, "timecode": f"{block.start} --> {block.end}", "text": block.text})
    return result


def _context_maps(review_context_path: Path | None) -> tuple[dict[int, list[str]], dict[int, list[dict[str, str]]]]:
    scene_ids: dict[int, list[str]] = {}
    asr_by_block: dict[int, list[dict[str, str]]] = {}
    for record in _read_jsonl(review_context_path):
        scene_id = str(record.get("scene_id", ""))
        candidates = [row for row in record.get("asr_candidates", []) if isinstance(row, dict)]
        for block in record.get("blocks", []):
            if not isinstance(block, dict):
                continue
            try:
                number = int(block["block_number"])
            except (KeyError, TypeError, ValueError):
                continue
            scene_ids.setdefault(number, []).append(scene_id)
            asr_by_block.setdefault(number, []).extend(candidates)
    for number, rows in asr_by_block.items():
        seen: set[tuple[str, str, str]] = set()
        unique: list[dict[str, str]] = []
        for row in rows:
            key = (row.get("profile", ""), row.get("text", ""), row.get("scene_id", ""))
            if key not in seen:
                seen.add(key)
                unique.append(row)
        asr_by_block[number] = unique
    return scene_ids, asr_by_block


def _band_map(review_queue_path: Path | None) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in _read_csv(review_queue_path):
        try:
            result[int(row.get("block_number", ""))] = row.get("review_band", "P3") or "P3"
        except ValueError:
            continue
    return result


def build_translation_queue(
    structure_path: Path,
    ja_path: Path,
    previous_ko_path: Path | None,
    output_path: Path,
    report_path: Path,
    *,
    review_context_path: Path | None = None,
    review_queue_path: Path | None = None,
    capture_index_path: Path | None = None,
    translation_model: str = DEFAULT_TRANSLATION_MODEL,
) -> dict[str, Any]:
    translation_model = resolve_translation_model(translation_model)
    structure, _, _ = parse_srt(structure_path)
    japanese, _, _ = parse_srt(ja_path)
    previous = parse_srt(previous_ko_path)[0] if previous_ko_path and previous_ko_path.exists() else []
    previous_map = _aligned_candidates(structure, previous)
    japanese_map = _aligned_candidates(structure, japanese)
    scene_ids, asr_by_block = _context_maps(review_context_path)
    bands = _band_map(review_queue_path)
    capture_available = bool(capture_index_path and capture_index_path.exists())
    structure_equals_ja = structure_path.resolve() == ja_path.resolve()
    records: list[dict[str, Any]] = []
    unresolved_source = 0
    for index, reference in enumerate(structure):
        ja_match = japanese_map.get(reference.number, {})
        japanese_text = str(ja_match.get("text", ""))
        if not japanese_text:
            unresolved_source += 1
        previous_match = previous_map.get(reference.number, {})
        band = bands.get(reference.number, "P3")
        asr_rows = asr_by_block.get(reference.number, [])
        evidence_refs = ["japanese_srt", "neighboring_context"]
        if previous_match.get("text"):
            evidence_refs.append("previous_korean_candidate")
        if asr_rows:
            evidence_refs.append("local_asr_candidates")
        if capture_available:
            evidence_refs.append("capture_index_only")
        if band in {"P1", "P2"}:
            evidence_refs.append("priority_scene_review")
        records.append({
            "block_number": reference.number,
            "timecode": f"{reference.start} --> {reference.end}",
            "review_band": band,
            "source_japanese": japanese_text,
            "source_japanese_alignment": ja_match,
            "previous_korean": previous_match.get("text", ""),
            "previous_korean_alignment": previous_match,
            "neighboring_source_japanese": _neighbor_rows(japanese, min(index, len(japanese) - 1)) if japanese else [],
            "asr_candidates": asr_rows,
            "scene_ids": scene_ids.get(reference.number, []),
            "required_semantic_slots": _required_slots(japanese_text),
            "evidence_refs": evidence_refs,
            "structure_locked_to": str(structure_path),
            "structure_equals_japanese_source": structure_equals_ja,
            "capture_semantic_interpretation": False if capture_available else None,
            "decision_status": "untranslated",
            "translation_method_required": "semantic_review_from_japanese",
            "translation_model_required": translation_model,
            "translation_constraints": [
                "preserve_question_negation_request_refusal_speaker_action_target_location_tense_intensity",
                "do_not_add_screen_only_actions_or_body_parts",
                "do_not_treat_previous_korean_as_truth",
                "write_source_faithful_before_viewer_natural",
            ],
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "status": "translation-queue-ready",
        "structure": str(structure_path),
        "japanese": str(ja_path),
        "previous_korean": str(previous_ko_path) if previous_ko_path else None,
        "output": str(output_path),
        "blocks": len(records),
        "unresolved_japanese_alignment": unresolved_source,
        "priority_blocks": sum(record["review_band"] in {"P1", "P2"} for record in records),
        "structure_equals_japanese_source": structure_equals_ja,
        "capture_semantic_interpretation": False if capture_available else None,
        "translation_model": translation_model,
        "next_required_action": "supply one semantic translation decision per block before SRT promotion",
        "final_promotion_allowed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def initialize_translation_decisions(queue_path: Path, output_path: Path, *, translation_model: str = DEFAULT_TRANSLATION_MODEL) -> dict[str, Any]:
    translation_model = resolve_translation_model(translation_model)
    queue = _read_jsonl(queue_path)
    records: list[dict[str, Any]] = []
    for item in queue:
        required_model = item.get("translation_model_required", translation_model)
        records.append({
            "block_number": item.get("block_number"),
            "source_japanese": item.get("source_japanese", ""),
            "previous_korean": item.get("previous_korean", ""),
            "source_faithful_korean": "",
            "viewer_natural_korean": "",
            "translation_method": "",
            "translation_model": resolve_translation_model(required_model),
            "status": "untranslated",
            "confidence": "",
            "uncertain_slots": item.get("required_semantic_slots", []),
            "evidence_refs": item.get("evidence_refs", []),
            "review_note": "",
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "status": "decision-template",
        "queue": str(queue_path),
        "output": str(output_path),
        "blocks": len(records),
        "translation_model": translation_model,
        "final_promotion_allowed": False,
    }


def merge_translation_decisions(base_path: Path, reviewed_path: Path, output_path: Path) -> dict[str, Any]:
    base = _read_jsonl(base_path)
    reviewed = _read_jsonl(reviewed_path)
    by_number: dict[int, dict[str, Any]] = {}
    for record in base:
        by_number[_decision_key(record.get("block_number"))] = record
    errors: list[str] = []
    reviewed_numbers: set[int] = set()
    for record in reviewed:
        number = _decision_key(record.get("block_number"))
        if number not in by_number:
            errors.append(f"구조 큐에 없는 block {number}")
            continue
        if number in reviewed_numbers:
            errors.append(f"검수 결정이 중복됨: block {number}")
            continue
        reviewed_numbers.add(number)
        merged = dict(by_number[number])
        merged.update(record)
        by_number[number] = merged
    if errors:
        return {"status": "fail", "errors": errors, "final_promotion_allowed": False}
    ordered = [by_number[number] for number in sorted(by_number)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    unresolved = [number for number, record in by_number.items() if str(record.get("status", "")) == "untranslated" or not _decision_text(record, "source_faithful_korean") or not _decision_text(record, "viewer_natural_korean")]
    return {
        "status": "merged",
        "base": str(base_path),
        "reviewed": str(reviewed_path),
        "output": str(output_path),
        "blocks": len(ordered),
        "reviewed_blocks": len(reviewed_numbers),
        "unresolved_blocks": len(unresolved),
        "final_promotion_allowed": not unresolved,
    }


def _decision_key(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"번역 결정의 block_number가 정수가 아닙니다: {value!r}") from exc


def _decision_text(record: dict[str, Any], field: str) -> str:
    value = record.get(field, "")
    return value.strip() if isinstance(value, str) else ""


def _decision_errors(record: dict[str, Any], *, strict: bool) -> list[str]:
    errors: list[str] = []
    block = record.get("block_number", "?")
    source = _decision_text(record, "source_faithful_korean")
    viewer = _decision_text(record, "viewer_natural_korean")
    status = str(record.get("status", "")).strip()
    method = str(record.get("translation_method", "")).strip()
    model = record.get("translation_model")
    confidence = str(record.get("confidence", "")).strip().lower()
    if not source or not viewer:
        errors.append(f"{block}: source_faithful_korean/viewer_natural_korean 누락")
    if strict and status not in {"translated", "reviewed", "approved"}:
        errors.append(f"{block}: 승인되지 않은 상태 {status!r}")
    if strict and method in {"", "carryover", "previous_korean_copy", "aligned_previous"}:
        errors.append(f"{block}: 의미 번역 방법이 아니거나 누락됨")
    if model is None or not str(model).strip():
        errors.append(f"{block}: translation_model 누락")
    else:
        try:
            resolve_translation_model(model)
        except ValueError as exc:
            errors.append(f"{block}: {exc}")
    if strict and confidence not in {"high", "medium"}:
        errors.append(f"{block}: 확신도는 high 또는 medium이어야 함")
    for field, text in (("source_faithful_korean", source), ("viewer_natural_korean", viewer)):
        if has_japanese(text):
            errors.append(f"{block}: {field}에 일본어 문자가 남음")
        if _PLACEHOLDER_RE.search(text):
            errors.append(f"{block}: {field}에 미완료 표식이 남음")
        if strict and not _HANGUL_RE.search(text) and text not in {"…", "...", "♪", "♪♪"}:
            errors.append(f"{block}: {field}에 한국어 문장이 없음")
    return errors


def validate_translation_decisions(structure_path: Path, decisions_path: Path, *, strict: bool = True) -> dict[str, Any]:
    structure, _, _ = parse_srt(structure_path)
    records = _read_jsonl(decisions_path)
    by_number: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    for record in records:
        try:
            number = _decision_key(record.get("block_number"))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if number in by_number:
            errors.append(f"{number}: 번역 결정이 중복됨")
        by_number[number] = record
        errors.extend(_decision_errors(record, strict=strict))
    expected = {block.number for block in structure}
    actual = set(by_number)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    errors.extend(f"누락된 번역 결정 block {number}" for number in missing)
    errors.extend(f"구조에 없는 번역 결정 block {number}" for number in extra)
    return {
        "status": "pass" if not errors else "fail",
        "strict": strict,
        "structure": str(structure_path),
        "decisions": str(decisions_path),
        "blocks": len(structure),
        "decision_records": len(records),
        "errors": errors,
        "missing_blocks": missing,
        "extra_blocks": extra,
        "translation_model": DEFAULT_TRANSLATION_MODEL,
        "final_promotion_allowed": not errors and strict,
    }


def apply_translation_decisions(
    structure_path: Path,
    decisions_path: Path,
    source_output_path: Path,
    viewer_output_path: Path,
    report_path: Path,
    *,
    strict: bool = True,
    translation_model: str = DEFAULT_TRANSLATION_MODEL,
) -> dict[str, Any]:
    translation_model = resolve_translation_model(translation_model)
    structure, _, _ = parse_srt(structure_path)
    records = _read_jsonl(decisions_path)
    by_number: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    for record in records:
        try:
            number = _decision_key(record.get("block_number"))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if number in by_number:
            errors.append(f"{number}: 번역 결정이 중복됨")
        by_number[number] = record
        errors.extend(_decision_errors(record, strict=strict))
    expected = {block.number for block in structure}
    actual = set(by_number)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    errors.extend(f"누락된 번역 결정 block {number}" for number in missing)
    errors.extend(f"구조에 없는 번역 결정 block {number}" for number in extra)
    if errors:
        report = {
            "status": "fail",
            "strict": strict,
            "structure": str(structure_path),
            "decisions": str(decisions_path),
            "errors": errors,
            "missing_blocks": missing,
            "extra_blocks": extra,
            "translation_model": translation_model,
            "final_promotion_allowed": False,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        return report
    source_blocks: list[SubtitleBlock] = []
    viewer_blocks: list[SubtitleBlock] = []
    for block in structure:
        record = by_number[block.number]
        source_blocks.append(SubtitleBlock(block.number, block.start, block.end, _decision_text(record, "source_faithful_korean"), block.start_seconds, block.end_seconds))
        viewer_blocks.append(SubtitleBlock(block.number, block.start, block.end, _decision_text(record, "viewer_natural_korean"), block.start_seconds, block.end_seconds))
    write_srt(source_output_path, source_blocks)
    write_srt(viewer_output_path, viewer_blocks)
    report = {
        "status": "text-crosschecked" if strict else "translation-draft",
        "strict": strict,
        "structure": str(structure_path),
        "decisions": str(decisions_path),
        "source_output": str(source_output_path),
        "viewer_output": str(viewer_output_path),
        "blocks": len(structure),
        "low_confidence_blocks": [block.number for block in structure if str(by_number[block.number].get("confidence", "")).lower() == "low"],
        "translation_model": translation_model,
        # SRT application preserves structure but cannot establish human audio
        # review, evidence completeness, or evaluation readiness on its own.
        "final_promotion_allowed": False,
        "next_required_action": "complete evidence, audio, and evaluation gates before final packaging",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report
