from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .srt import SubtitleBlock, compare_structure, has_japanese, parse_srt, write_srt


def _read_response(path: Path, title: str) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = raw.get("results") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"재번역 응답에 results 배열이 없습니다: {path}")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{path} results[{index}]가 객체가 아닙니다.")
        if row.get("title") != title:
            continue
        try:
            number = int(row["block_number"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path} results[{index}]의 block_number가 잘못되었습니다.") from exc
        decision = row.get("decision")
        source = str(row.get("source_faithful_korean", "")).strip()
        viewer = str(row.get("viewer_natural_korean", "")).strip()
        confidence = row.get("confidence")
        if decision == "replace":
            if not source or not viewer or has_japanese(source) or has_japanese(viewer):
                raise ValueError(f"{path} block {number}: replace 결과가 유효한 한국어 자막이 아닙니다.")
            if confidence not in {"high", "medium"}:
                raise ValueError(f"{path} block {number}: replace confidence가 high/medium이 아닙니다.")
        elif decision == "hold":
            if source or viewer or confidence != "low":
                raise ValueError(f"{path} block {number}: hold 출력 계약을 위반했습니다.")
        else:
            raise ValueError(f"{path} block {number}: decision이 replace/hold가 아닙니다.")
        result.append({**row, "block_number": number})
    return result


def _read_reviews(path: Path, title: str) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = raw.get("reviews") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"재번역 검토 응답에 reviews 배열이 없습니다: {path}")
    result: dict[int, str] = {}
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or row.get("title") != title:
            continue
        try:
            number = int(row["block_number"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path} reviews[{index}]의 block_number가 잘못되었습니다.") from exc
        verdict = row.get("verdict")
        if verdict not in {"keep", "revert"}:
            raise ValueError(f"{path} block {number}: review verdict가 keep/revert가 아닙니다.")
        if number in result:
            raise ValueError(f"{path} block {number}: review verdict가 중복되었습니다.")
        result[number] = verdict
    return result


def apply_targeted_retranslations(
    *,
    title: str,
    structure_path: Path,
    source_path: Path,
    viewer_path: Path,
    response_paths: list[Path],
    output_dir: Path,
    hold_marker: str = "…",
    review_paths: list[Path] | None = None,
    version: str = "v2",
) -> dict[str, Any]:
    """Apply supported machine corrections and replace unsupported text with an honest marker."""
    if not hold_marker.strip():
        raise ValueError("hold_marker는 비어 있을 수 없습니다.")
    if not version or not version.replace("-", "").isalnum():
        raise ValueError("재번역 version 형식이 잘못되었습니다.")
    structure, _, _ = parse_srt(structure_path)
    source, _, _ = parse_srt(source_path)
    viewer, _, _ = parse_srt(viewer_path)
    if not compare_structure(structure, source)["pass"] or not compare_structure(structure, viewer)["pass"]:
        raise ValueError("재번역 입력 SRT가 기준 구조와 일치하지 않습니다.")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"기존 재번역 출력 디렉터리를 덮어쓰지 않습니다: {output_dir}")
    decisions: dict[int, dict[str, Any]] = {}
    for response_path in response_paths:
        for row in _read_response(response_path.expanduser().resolve(), title):
            number = row["block_number"]
            if number in decisions:
                raise ValueError(f"block {number}에 대한 재번역 결과가 중복되었습니다.")
            decisions[number] = row
    reviews: dict[int, str] = {}
    for review_path in review_paths or []:
        for number, verdict in _read_reviews(review_path.expanduser().resolve(), title).items():
            if number in reviews:
                raise ValueError(f"block {number}에 대한 재번역 검토가 중복되었습니다.")
            reviews[number] = verdict
    expected = {block.number for block in structure}
    extra = sorted(set(decisions) - expected)
    if extra:
        raise ValueError(f"기준 SRT에 없는 재번역 block이 있습니다: {extra[:10]}")
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
            method = "unchanged"
        elif decision["decision"] == "replace" and reviews.get(block.number, "keep") == "keep":
            source_text = str(decision["source_faithful_korean"]).strip()
            viewer_text = str(decision["viewer_natural_korean"]).strip()
            method = "targeted-retranslation"
        else:
            source_text = hold_marker
            viewer_text = hold_marker
            method = "targeted-hold-marker"
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
                    "basis": decision.get("basis", []),
                    "reason": decision.get("reason", ""),
                    "review_verdict": reviews.get(block.number),
                    "human_reviewed": False,
                    "final_promotion_allowed": False,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=False)
    source_output = output_dir / f"{title}.source-faithful-ko.targeted-retranslation-{version}.srt"
    viewer_output = output_dir / f"{title}.viewer-natural-ko.targeted-retranslation-{version}.srt"
    ledger_path = output_dir / "targeted-retranslation-decisions.jsonl"
    report_path = output_dir / "targeted-retranslation-report.json"
    write_srt(source_output, new_source)
    write_srt(viewer_output, new_viewer)
    ledger_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ledger), encoding="utf-8", newline="\n")
    report = {
        "schema_name": "translation-forensics/targeted-retranslation-report",
        "schema_version": "1",
        "title_id": title,
        "version": version,
        "status": "targeted-retranslation-applied",
        "blocks": len(structure),
        "targeted_blocks": len(decisions),
        "replacement_blocks": counts["targeted-retranslation"],
        "hold_marker_blocks": counts["targeted-hold-marker"],
        "review_reverted_blocks": sum(1 for number, decision in decisions.items() if decision["decision"] == "replace" and reviews.get(number) == "revert"),
        "unchanged_blocks": counts["unchanged"],
        "hold_marker": hold_marker,
        "human_reviewed": False,
        "final_promotion_allowed": False,
        "note": "hold marker는 잘못된 기존 번역을 유지하지 않기 위한 표시이며 번역 완료를 뜻하지 않습니다.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {
        "status": report["status"],
        "source_output": str(source_output),
        "viewer_output": str(viewer_output),
        "ledger": str(ledger_path),
        "report": str(report_path),
        "replacement_blocks": report["replacement_blocks"],
        "hold_marker_blocks": report["hold_marker_blocks"],
        "review_reverted_blocks": report["review_reverted_blocks"],
        "final_promotion_allowed": False,
    }
