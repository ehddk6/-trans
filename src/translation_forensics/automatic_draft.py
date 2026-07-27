from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .srt import SubtitleBlock, compare_structure, parse_srt, write_srt


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_candidate(path: Path | None, reference: list[SubtitleBlock], label: str) -> dict[int, str]:
    if path is None:
        return {}
    blocks, _, _ = parse_srt(path)
    diff = compare_structure(reference, blocks)
    if not diff["pass"]:
        raise ValueError(f"{label}의 구조가 기준 SRT와 일치하지 않습니다.")
    return {block.number: block.text.strip() for block in blocks}


def _read_decisions(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None:
        return {}
    result: dict[int, dict[str, str]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
            number = int(record["block_number"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"결정 후보 {path}의 {line_number}행이 올바른 JSONL이 아닙니다.") from exc
        result[number] = {
            "source_faithful_korean": str(record.get("source_faithful_korean", "")).strip(),
            "viewer_natural_korean": str(record.get("viewer_natural_korean", "")).strip(),
        }
    return result


def _pick_text(
    number: int,
    field: str,
    preferred: dict[int, str],
    preferred_label: str,
    decisions: dict[int, dict[str, str]],
    fallback: dict[int, str],
) -> tuple[str, str]:
    if preferred.get(number):
        return preferred[number], preferred_label
    if decisions.get(number, {}).get(field):
        return decisions[number][field], "decision-candidate"
    if fallback.get(number):
        return fallback[number], "aligned-legacy-fallback"
    return "", "unresolved-empty"


def build_automatic_draft(
    *,
    title: str,
    structure_path: Path,
    output_dir: Path,
    source_candidate_path: Path | None = None,
    viewer_candidate_path: Path | None = None,
    single_candidate_path: Path | None = None,
    fallback_path: Path | None = None,
    decision_candidate_path: Path | None = None,
) -> dict[str, Any]:
    """Build a complete playback draft without claiming human validation.

    Candidate SRTs must preserve the source structure.  A partial decision JSONL
    can fill individual blocks, while a structure-aligned legacy SRT is used only
    as a final non-empty fallback.  The resulting ledger records that provenance
    per field so the draft cannot be mistaken for an approved translation.
    """
    structure_path = structure_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    candidates = [
        source_candidate_path,
        viewer_candidate_path,
        single_candidate_path,
        fallback_path,
        decision_candidate_path,
    ]
    candidates = [path.expanduser().resolve() for path in candidates if path is not None]
    if output_dir.exists():
        raise FileExistsError(f"기존 자동 초안 디렉터리를 덮어쓰지 않습니다: {output_dir}")
    if not candidates:
        raise ValueError("자동 초안을 만들 번역 후보 또는 fallback이 필요합니다.")

    reference, _, _ = parse_srt(structure_path)
    source_candidate = _read_candidate(source_candidate_path, reference, "source candidate")
    viewer_candidate = _read_candidate(viewer_candidate_path, reference, "viewer candidate")
    single_candidate = _read_candidate(single_candidate_path, reference, "single candidate")
    fallback = _read_candidate(fallback_path, reference, "fallback")
    decisions = _read_decisions(decision_candidate_path)
    if not (source_candidate or viewer_candidate or single_candidate or decisions or fallback):
        raise ValueError("비어 있지 않은 자동 초안 후보가 없습니다.")

    source_blocks: list[SubtitleBlock] = []
    viewer_blocks: list[SubtitleBlock] = []
    ledger: list[dict[str, Any]] = []
    source_methods: Counter[str] = Counter()
    viewer_methods: Counter[str] = Counter()
    unresolved: list[int] = []
    for block in reference:
        source_preferred = source_candidate or single_candidate
        viewer_preferred = viewer_candidate or single_candidate
        source_label = "source-candidate" if source_candidate else "single-candidate"
        viewer_label = "viewer-candidate" if viewer_candidate else "single-candidate"
        source_text, source_method = _pick_text(
            block.number,
            "source_faithful_korean",
            source_preferred,
            source_label,
            decisions,
            fallback,
        )
        viewer_text, viewer_method = _pick_text(
            block.number,
            "viewer_natural_korean",
            viewer_preferred,
            viewer_label,
            decisions,
            fallback,
        )
        if not source_text or not viewer_text:
            unresolved.append(block.number)
        source_methods[source_method] += 1
        viewer_methods[viewer_method] += 1
        source_blocks.append(SubtitleBlock(block.number, block.start, block.end, source_text, block.start_seconds, block.end_seconds))
        viewer_blocks.append(SubtitleBlock(block.number, block.start, block.end, viewer_text, block.start_seconds, block.end_seconds))
        ledger.append(
            {
                "schema_name": "translation-forensics/automatic-draft-decision",
                "schema_version": "1",
                "title_id": title,
                "block_number": block.number,
                "timecode": f"{block.start} --> {block.end}",
                "source_faithful_method": source_method,
                "viewer_natural_method": viewer_method,
                "source_faithful_korean": source_text,
                "viewer_natural_korean": viewer_text,
                "status": "automatic-draft",
                "human_reviewed": False,
                "final_promotion_allowed": False,
                "review_required": True,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    source_output = output_dir / f"{title}.source-faithful-ko.automatic-draft-v1.srt"
    viewer_output = output_dir / f"{title}.viewer-natural-ko.automatic-draft-v1.srt"
    ledger_path = output_dir / "automatic-draft-decisions.jsonl"
    report_path = output_dir / "automatic-draft-report.json"
    manifest_path = output_dir / "automatic-draft-manifest.json"
    write_srt(source_output, source_blocks)
    write_srt(viewer_output, viewer_blocks)
    ledger_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ledger), encoding="utf-8", newline="\n")
    report = {
        "schema_name": "translation-forensics/automatic-draft-report",
        "schema_version": "1",
        "title_id": title,
        "status": "automatic-draft-complete" if not unresolved else "automatic-draft-incomplete",
        "blocks": len(reference),
        "unresolved_blocks": unresolved,
        "source_method_counts": dict(sorted(source_methods.items())),
        "viewer_method_counts": dict(sorted(viewer_methods.items())),
        "human_reviewed": False,
        "final_promotion_allowed": False,
        "note": "자동 초안은 재생용 후보이며 사람 검수 또는 사람 정답과의 동일성을 주장하지 않습니다.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "schema_name": "translation-forensics/automatic-draft-manifest",
        "schema_version": "1",
        "title_id": title,
        "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in [structure_path, *candidates]],
        "outputs": [
            {"path": str(source_output), "sha256": _sha256(source_output)},
            {"path": str(viewer_output), "sha256": _sha256(viewer_output)},
            {"path": str(ledger_path), "sha256": _sha256(ledger_path)},
            {"path": str(report_path), "sha256": _sha256(report_path)},
        ],
        "human_reviewed": False,
        "final_promotion_allowed": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {
        "status": report["status"],
        "title": title,
        "blocks": len(reference),
        "unresolved_blocks": len(unresolved),
        "source_output": str(source_output),
        "viewer_output": str(viewer_output),
        "ledger": str(ledger_path),
        "report": str(report_path),
        "manifest": str(manifest_path),
        "final_promotion_allowed": False,
    }
