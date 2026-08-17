from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from .srt import SubtitleBlock, compare_structure, parse_srt, write_srt
from .validation import validate_pair, write_validation_report


ABSTAIN_MARKERS = {"[미확정]", "…", "..."}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8", newline="\n")


def _version_key(path: Path) -> tuple[int, str]:
    match = re.search(r"-v(\d+(?:\.\d+)?)", path.name)
    if not match:
        return (0, str(path))
    try:
        return (int(float(match.group(1)) * 1000), str(path))
    except ValueError:
        return (0, str(path))


def _latest_directory(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.is_dir()]
    return max(existing, key=_version_key) if existing else None


def _manifest_role_path(title_dir: Path, role: str) -> Path | None:
    manifest_path = title_dir / "metadata" / "project-manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    for record in manifest.get("inputs", []):
        if not isinstance(record, dict) or record.get("role") != role:
            continue
        raw = str(record.get("path") or "").strip()
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = title_dir / candidate
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_structure(title: str, title_dir: Path, machine_dir: Path | None) -> Path:
    candidates: list[Path] = []
    manifest_structure = _manifest_role_path(title_dir, "structure")
    if manifest_structure:
        candidates.append(manifest_structure)
    candidates.append(title_dir / "inputs" / f"{title}.structure.srt")
    candidates.extend(sorted((title_dir / "inputs").glob("*.structure.srt")) if (title_dir / "inputs").exists() else [])
    if machine_dir:
        candidates.append(machine_dir / "structure.srt")
    existing = [path.resolve() for path in candidates if path.exists()]
    unique = list(dict.fromkeys(existing))
    if not unique:
        raise FileNotFoundError(f"잠긴 structure SRT를 찾을 수 없습니다: {title_dir}")
    if len(unique) > 1:
        first = unique[0].read_bytes()
        if any(path.read_bytes() != first for path in unique[1:]):
            raise ValueError(f"서로 다른 structure SRT 후보가 있습니다: {[str(path) for path in unique]}")
    return unique[0]


def _candidate_map(path: Path | None, reference: list[SubtitleBlock], *, label: str) -> tuple[dict[int, SubtitleBlock], dict[str, Any]]:
    if path is None or not path.exists():
        return {}, {"label": label, "status": "missing", "path": str(path) if path else None}
    blocks, encoding, newline = parse_srt(path)
    structure = compare_structure(reference, blocks)
    report = {
        "label": label,
        "status": "usable" if structure["pass"] else "rejected-structure-mismatch",
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "encoding": encoding,
        "newline": newline,
        "structure": structure,
    }
    if not structure["pass"]:
        return {}, report
    return {block.number: block for block in blocks}, report


def _clean_text(block: SubtitleBlock | None, *, reject_unconfirmed: bool = False) -> str:
    if block is None:
        return ""
    text = block.text.strip()
    if not text or text in ABSTAIN_MARKERS:
        return ""
    if reject_unconfirmed and "[미확정]" in text:
        return ""
    return text


def _first_text(candidates: list[tuple[str, SubtitleBlock | None]], *, reject_unconfirmed: bool = False) -> tuple[str, str]:
    for origin, block in candidates:
        text = _clean_text(block, reject_unconfirmed=reject_unconfirmed)
        if text:
            return text, origin
    return "…", "abstained"


def _latest_file(directory: Path | None, pattern: str) -> Path | None:
    if directory is None:
        return None
    candidates = [path for path in directory.glob(pattern) if path.is_file()]
    return max(candidates, key=_version_key) if candidates else None


def _discover_paths(title: str, title_dir: Path) -> dict[str, Path | None]:
    machine_dirs = list(title_dir.glob("runs/*/machine-final-v*")) + list(title_dir.glob("machine-final-v*"))
    machine_dir = _latest_directory(machine_dirs)
    inferred_dirs = list(title_dir.glob("inferred-recovery-v*"))
    inferred_dir = _latest_directory(inferred_dirs)
    closed_dir = title_dir / "closed-world" / "closed-world-validated-v1"
    return {
        "machine_dir": machine_dir,
        "closed_source": closed_dir / "source-faithful.preview.srt",
        "closed_viewer": closed_dir / "viewer-natural.preview.srt",
        "inferred_source": _latest_file(inferred_dir, f"{title}.source-faithful-ko.inferred-recovery-v*.srt"),
        "inferred_viewer": _latest_file(inferred_dir, f"{title}.viewer-natural-ko.inferred-recovery-v*.srt"),
        "machine_source": _latest_file(machine_dir, f"{title}.source-faithful-ko.machine-final-v*.srt"),
        "machine_viewer": _latest_file(machine_dir, f"{title}.viewer-natural-ko.machine-final-v*.srt"),
    }


def build_offline_hybrid(titles_file: Path, workspace_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    if not titles_file.exists():
        raise FileNotFoundError(f"Titles file not found: {titles_file}")
    titles = [title.strip() for title in titles_file.read_text("utf-8-sig").splitlines() if title.strip()]
    results: list[dict[str, Any]] = []

    for title in titles:
        logging.info("Processing offline hybrid preview for %s", title)
        title_dir = workspace_root / title
        if not title_dir.exists():
            results.append({"title_id": title, "status": "skipped-missing-workspace", "workspace": str(title_dir)})
            continue

        paths = _discover_paths(title, title_dir)
        structure_path = _resolve_structure(title, title_dir, paths["machine_dir"])
        reference, _, _ = parse_srt(structure_path)

        closed_source, closed_source_report = _candidate_map(paths["closed_source"], reference, label="closed-world-source")
        closed_viewer, closed_viewer_report = _candidate_map(paths["closed_viewer"], reference, label="closed-world-viewer")
        inferred_source, inferred_source_report = _candidate_map(paths["inferred_source"], reference, label="inferred-source")
        inferred_viewer, inferred_viewer_report = _candidate_map(paths["inferred_viewer"], reference, label="inferred-viewer")
        machine_source, machine_source_report = _candidate_map(paths["machine_source"], reference, label="machine-source")
        machine_viewer, machine_viewer_report = _candidate_map(paths["machine_viewer"], reference, label="machine-viewer")

        package_dir = output_dir / title / "offline-hybrid-preview-v1"
        if package_dir.exists():
            raise FileExistsError(f"기존 오프라인 하이브리드 미리보기를 덮어쓰지 않습니다: {package_dir}")
        package_dir.mkdir(parents=True)

        source_blocks: list[SubtitleBlock] = []
        viewer_blocks: list[SubtitleBlock] = []
        decisions: list[dict[str, Any]] = []
        for block in reference:
            source_text, source_origin = _first_text([
                ("closed-world", closed_source.get(block.number)),
                ("inferred-recovery", inferred_source.get(block.number)),
                ("machine-final", machine_source.get(block.number)),
            ], reject_unconfirmed=True)
            viewer_text, viewer_origin = _first_text([
                ("closed-world", closed_viewer.get(block.number)),
                ("inferred-recovery", inferred_viewer.get(block.number)),
                ("machine-final", machine_viewer.get(block.number)),
                (f"source:{source_origin}", SubtitleBlock(block.number, block.start, block.end, source_text, block.start_seconds, block.end_seconds)),
            ], reject_unconfirmed=True)
            source_blocks.append(SubtitleBlock(block.number, block.start, block.end, source_text, block.start_seconds, block.end_seconds))
            viewer_blocks.append(SubtitleBlock(block.number, block.start, block.end, viewer_text, block.start_seconds, block.end_seconds))
            decisions.append({
                "schema_name": "translation-forensics/offline-hybrid-decision",
                "schema_version": "1",
                "title_id": title,
                "block_number": block.number,
                "source_origin": source_origin,
                "viewer_origin": viewer_origin,
                "source_text": source_text,
                "viewer_text": viewer_text,
                "human_reviewed": False,
                "human_final_allowed": False,
                "final_promotion_allowed": False,
            })

        source_output = package_dir / f"{title}.source-faithful-ko.offline-hybrid-preview-v1.srt"
        viewer_output = package_dir / f"{title}.viewer-complete-ko.offline-hybrid-preview-v1.srt"
        write_srt(source_output, source_blocks)
        write_srt(viewer_output, viewer_blocks)
        decisions_path = package_dir / "offline-hybrid-decisions.jsonl"
        _write_jsonl(decisions_path, decisions)

        validation = validate_pair(structure_path, source_output, viewer_output)
        validation_path = package_dir / "validation.json"
        write_validation_report(validation_path, validation)
        candidate_reports = [
            closed_source_report, closed_viewer_report, inferred_source_report,
            inferred_viewer_report, machine_source_report, machine_viewer_report,
        ]
        report = {
            "schema_name": "translation-forensics/offline-hybrid-preview-report",
            "schema_version": "1",
            "title_id": title,
            "status": "offline-hybrid-preview-packaged" if validation["status"] != "fail" else "offline-hybrid-preview-invalid",
            "blocks": len(reference),
            "source_origin_counts": {origin: sum(row["source_origin"] == origin for row in decisions) for origin in sorted({row["source_origin"] for row in decisions})},
            "viewer_origin_counts": {origin: sum(row["viewer_origin"] == origin for row in decisions) for origin in sorted({row["viewer_origin"] for row in decisions})},
            "candidate_reports": candidate_reports,
            "human_reviewed": False,
            "human_final_allowed": False,
            "final_promotion_allowed": False,
        }
        report_path = package_dir / "offline-hybrid-report.json"
        _write_json(report_path, report)
        artifacts = [source_output, viewer_output, decisions_path, validation_path, report_path]
        manifest = {
            "schema_name": "translation-forensics/offline-hybrid-preview-manifest",
            "schema_version": "1",
            "title_id": title,
            "structure": {"path": str(structure_path.resolve()), "sha256": _sha256(structure_path)},
            "inputs": [
                {"label": item["label"], "path": item.get("path"), "sha256": item.get("sha256"), "status": item["status"]}
                for item in candidate_reports
            ],
            "outputs": [{"path": path.relative_to(package_dir).as_posix(), "sha256": _sha256(path)} for path in artifacts],
            "human_reviewed": False,
            "human_final_allowed": False,
            "final_promotion_allowed": False,
        }
        manifest_path = package_dir / "offline-hybrid-manifest.json"
        _write_json(manifest_path, manifest)
        if validation["status"] == "fail":
            raise RuntimeError(f"오프라인 하이브리드 미리보기 검증 실패: {package_dir}")
        results.append({"title_id": title, "status": report["status"], "output": str(package_dir), "blocks": len(reference)})
    return results
