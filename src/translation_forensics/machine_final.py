from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .asr_evidence import read_asr_candidates
from .srt import compare_structure, parse_srt


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 객체가 아닙니다: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}의 {line_number}행 JSONL이 올바르지 않습니다.") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}의 {line_number}행이 객체가 아닙니다.")
        records.append(record)
    return records


def _utf8_bom_viewing_bytes(path: Path) -> bytes:
    """Return an Excel-friendly UTF-8 rendering without changing the canonical CSV."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"ASR CSV가 유효한 UTF-8이 아닙니다: {path}") from exc
    return b"\xef\xbb\xbf" + text.encode("utf-8")


def _write_viewing_copy(source_asr_path: Path, destination: Path) -> None:
    payload = _utf8_bom_viewing_bytes(source_asr_path)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise FileExistsError(f"기존 Excel 보기용 ASR 파일을 덮어쓰지 않습니다: {destination}")
        return
    destination.write_bytes(payload)


def _validate_machine_inputs(
    structure_path: Path,
    source_path: Path,
    viewer_path: Path,
    automatic_report_path: Path,
    automatic_ledger_path: Path,
    asr_path: Path,
    timeline_path: Path,
    proof_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    reference, _, _ = parse_srt(structure_path)
    source, _, _ = parse_srt(source_path)
    viewer, _, _ = parse_srt(viewer_path)
    source_diff = compare_structure(reference, source)
    viewer_diff = compare_structure(reference, viewer)
    if not source_diff["pass"] or not viewer_diff["pass"]:
        raise ValueError("machine-final 입력 SRT의 구조/번호/타임코드가 기준과 다릅니다.")
    empty = [
        block.number
        for block in [*source, *viewer]
        if not block.text.strip()
    ]
    if empty:
        raise ValueError(f"machine-final 입력에 빈 자막 블록이 있습니다: {sorted(set(empty))[:10]}")
    report = _read_json(automatic_report_path)
    if report.get("status") != "automatic-draft-complete" or report.get("unresolved_blocks"):
        raise ValueError("미해결 블록이 남은 자동 초안은 machine-final로 패키징할 수 없습니다.")
    ledger = _read_jsonl(automatic_ledger_path)
    by_number = {int(record.get("block_number")): record for record in ledger if str(record.get("block_number", "")).isdigit()}
    expected = {block.number for block in reference}
    if set(by_number) != expected:
        raise ValueError("automatic draft 원장의 블록 커버리지가 기준 SRT와 일치하지 않습니다.")
    for number, record in by_number.items():
        if record.get("status") != "automatic-draft":
            raise ValueError(f"{number}: automatic-draft 상태가 아닙니다.")
        if not str(record.get("source_faithful_korean", "")).strip() or not str(record.get("viewer_natural_korean", "")).strip():
            raise ValueError(f"{number}: 자동 초안 원장에 빈 한국어 텍스트가 있습니다.")
    asr_rows = read_asr_candidates(asr_path)
    if not asr_rows:
        raise ValueError("machine-final에는 ASR 후보가 필요합니다.")
    timeline = _read_json(timeline_path)
    proof = _read_json(proof_path)
    if proof.get("human_reference_equality") != "unidentifiable" or proof.get("100_percent_equal") is True:
        raise ValueError("사람 정답 동일성에 대한 잘못된 품질 주장이 감지되었습니다.")
    return timeline, ledger, len(asr_rows)


def package_machine_final(
    *,
    title: str,
    structure_path: Path,
    source_path: Path,
    viewer_path: Path,
    automatic_report_path: Path,
    automatic_ledger_path: Path,
    asr_path: Path,
    timeline_path: Path,
    proof_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Package a complete machine-only release without impersonating human final review."""
    paths = [
        structure_path,
        source_path,
        viewer_path,
        automatic_report_path,
        automatic_ledger_path,
        asr_path,
        timeline_path,
        proof_path,
    ]
    paths = [path.expanduser().resolve() for path in paths]
    structure_path, source_path, viewer_path, automatic_report_path, automatic_ledger_path, asr_path, timeline_path, proof_path = paths
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"기존 machine-final 패키지를 덮어쓰지 않습니다: {output_dir}")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"machine-final 입력이 없습니다: {path}")
    timeline, ledger, asr_rows = _validate_machine_inputs(
        structure_path,
        source_path,
        viewer_path,
        automatic_report_path,
        automatic_ledger_path,
        asr_path,
        timeline_path,
        proof_path,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    copies = {
        "source": output_dir / f"{title}.source-faithful-ko.machine-final-v1.srt",
        "viewer": output_dir / f"{title}.viewer-natural-ko.machine-final-v1.srt",
        "automatic_ledger": output_dir / "automatic-draft-decisions.jsonl",
        "automatic_report": output_dir / "automatic-draft-report.json",
        "asr": output_dir / "asr-candidates.csv",
        "timeline": output_dir / "machine-timeline.json",
        "closed_world_proof": output_dir / "closed-world-proof.json",
    }
    source_map = {
        "source": source_path,
        "viewer": viewer_path,
        "automatic_ledger": automatic_ledger_path,
        "automatic_report": automatic_report_path,
        "asr": asr_path,
        "timeline": timeline_path,
        "closed_world_proof": proof_path,
    }
    for key, destination in copies.items():
        shutil.copy2(source_map[key], destination)
    excel_asr = output_dir / "asr-candidates.for-excel.utf8bom.csv"
    _write_viewing_copy(copies["asr"], excel_asr)
    report = {
        "schema_name": "translation-forensics/machine-final-report",
        "schema_version": "1",
        "title_id": title,
        "release_kind": "machine-final",
        "status": "machine-final-packaged",
        "blocks": len(ledger),
        "asr_candidate_rows": asr_rows,
        "timeline_status": timeline.get("status"),
        "machine_final_allowed": True,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
        "human_reference_equality": "unidentifiable",
        "100_percent_equal": False,
        "limitations": [
            "사람 직접 청취 또는 사람 검수 기록이 없습니다.",
            "관측되지 않은 사람 정답과의 의미 또는 문자열 동일성은 판별할 수 없습니다.",
            "machine-final은 프로젝트의 human final 단계와 별도입니다.",
        ],
    }
    report_path = output_dir / "machine-final-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "schema_name": "translation-forensics/machine-final-manifest",
        "schema_version": "1",
        "title_id": title,
        "release_kind": "machine-final",
        "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in paths],
        "outputs": [{"path": str(path), "sha256": _sha256(path)} for path in [*copies.values(), excel_asr, report_path]],
        "machine_final_allowed": True,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
    }
    manifest_path = output_dir / "machine-final-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {
        "status": "machine-final-packaged",
        "title": title,
        "output": str(output_dir),
        "blocks": len(ledger),
        "machine_final_allowed": True,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
        "report": str(report_path),
        "manifest": str(manifest_path),
    }


def repair_machine_final_asr(*, source_asr_path: Path, package_dir: Path) -> dict[str, Any]:
    """Restore a canonical ASR copy from its manifest-locked source and add an Excel view."""
    source_asr_path = source_asr_path.expanduser().resolve()
    package_dir = package_dir.expanduser().resolve()
    manifest_path = package_dir / "machine-final-manifest.json"
    canonical = package_dir / "asr-candidates.csv"
    viewing = package_dir / "asr-candidates.for-excel.utf8bom.csv"
    if not source_asr_path.exists() or not manifest_path.exists() or not canonical.exists():
        raise FileNotFoundError("ASR 원본, machine-final manifest 또는 패키지 ASR 파일이 없습니다.")
    manifest = _read_json(manifest_path)
    expected = next(
        (str(item.get("sha256", "")) for item in manifest.get("outputs", []) if Path(str(item.get("path", ""))).name == canonical.name),
        "",
    )
    if not expected:
        raise ValueError("machine-final manifest에 canonical ASR 해시가 없습니다.")
    source_hash = _sha256(source_asr_path)
    if source_hash != expected:
        raise ValueError("지정한 ASR 원본이 manifest의 canonical 해시와 일치하지 않습니다.")
    before_hash = _sha256(canonical)
    repaired = before_hash != expected
    if repaired:
        shutil.copy2(source_asr_path, canonical)
    if _sha256(canonical) != expected:
        raise ValueError("ASR canonical 복구 후 해시가 manifest와 일치하지 않습니다.")
    _write_viewing_copy(canonical, viewing)
    output_entries = [item for item in manifest.get("outputs", []) if Path(str(item.get("path", ""))).name != viewing.name]
    output_entries.append({"path": str(viewing), "sha256": _sha256(viewing)})
    manifest["outputs"] = output_entries
    manifest["excel_viewing_asr"] = str(viewing)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {
        "status": "repaired" if repaired else "verified",
        "package": str(package_dir),
        "canonical_asr": str(canonical),
        "excel_viewing_asr": str(viewing),
        "canonical_sha256": expected,
    }
