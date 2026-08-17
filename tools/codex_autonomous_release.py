from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

try:
    from translation_forensics.autonomous_release import (
        _allowed_evidence_refs,
        _build_evidence_graph,
        _title_memory,
        _validate_decisions,
        build_autonomous_evidence,
        validate_autonomous_regressions,
    )
    from translation_forensics.srt import SubtitleBlock, compare_structure, parse_srt, write_srt
    from translation_forensics.validation import validate_pair
except ImportError as exc:  # pragma: no cover - CLI diagnostics
    print(f"Import error: {exc}", file=sys.stderr)
    print(
        "Run from the translation-forensics project root: "
        "python tools/codex_autonomous_release.py ...",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


ROOT = Path(__file__).resolve().parents[1]
WORKSPACES = ROOT / "workspaces"
DRAFT_KIND = "codex-assisted-draft"
DRAFT_STATUS = "codex-assisted-draft-packaged"
INVALID_STATUS = "codex-assisted-draft-invalid"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n" for value in values),
        encoding="utf-8",
        newline="\n",
    )


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL object required: {path}:{line_number}")
        rows.append(value)
    return rows


def _resolve_manifest_path(raw: str, *, workspace: Path, manifest_path: Path) -> Path:
    value = Path(raw).expanduser()
    if value.is_absolute():
        return value.resolve()
    candidates = [ROOT / value, workspace / value, manifest_path.parent / value]
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())


def _role_paths(workspace: Path) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    manifest_path = workspace / "metadata" / "project-manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"매니페스트 없음: {manifest_path}")
    manifest = _json(manifest_path)
    roles: dict[str, Path] = {}
    for item in manifest.get("inputs", []):
        if not isinstance(item, dict) or not str(item.get("role") or "").strip():
            continue
        path = _resolve_manifest_path(str(item.get("path") or ""), workspace=workspace, manifest_path=manifest_path)
        if path.exists():
            roles[str(item["role"])] = path
    return manifest_path.resolve(), roles, manifest


def _optional_workspace_file(workspace: Path, title_id: str, name: str) -> Path | None:
    candidates = [
        workspace / "inferred-recovery-audio-v4" / name,
        workspace / "intermediate" / f"{title_id}.work_audio" / name,
        workspace / name,
    ]
    return next((path.resolve() for path in candidates if path.exists()), None)


def _input_record(role: str, path: Path) -> dict[str, str]:
    return {"role": role, "path": str(path.resolve()), "sha256": _sha256(path)}


def _portable_input_record(record: dict[str, str], workspace: Path) -> dict[str, str]:
    path = Path(record["path"]).resolve()
    try:
        portable = path.relative_to(workspace.resolve()).as_posix()
        base = "workspace"
    except ValueError:
        portable = path.name
        base = "basename-only"
    return {
        "role": record["role"],
        "path": portable,
        "path_base": base,
        "sha256": record["sha256"],
    }


def _verify_prepared_inputs(info: dict[str, Any]) -> dict[str, Path]:
    rows = info.get("prepared_inputs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("prepare 입력 해시 기록이 없습니다. prepare를 다시 실행해야 합니다.")
    result: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("prepare 입력 기록 형식이 잘못되었습니다.")
        role = str(row.get("role") or "")
        path = Path(str(row.get("path") or "")).expanduser().resolve()
        expected = str(row.get("sha256") or "")
        if not role or not path.exists():
            raise FileNotFoundError(f"prepare 이후 입력이 사라졌습니다: role={role} path={path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"prepare 이후 입력이 변경되었습니다: role={role} path={path} "
                f"expected={expected} actual={actual}"
            )
        result[role] = path
    return result


def _validate_decision_schema(rows: list[dict[str, Any]]) -> None:
    schema_path = ROOT / "schemas" / "autonomous-decision.schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"결정 스키마 없음: {schema_path}")
    validator = Draft202012Validator(_json(schema_path))
    errors: list[str] = []
    for index, row in enumerate(rows, 1):
        first_error = next(iter(validator.iter_errors(row)), None)
        if first_error is not None:
            errors.append(f"decision {index}: {first_error.message}")
    if errors:
        raise ValueError("결정 JSON Schema 검증 실패: " + "; ".join(errors[:5]))


def phase_prepare(title_id: str, workspace: Path) -> dict[str, Any]:
    workspace = workspace.expanduser().resolve()
    bridge_dir = workspace / "codex-bridge"
    protected = [
        bridge_dir / "title-info.json",
        bridge_dir / "evidence.jsonl",
        bridge_dir / "machine-alignment.json",
        bridge_dir / "evidence-summary.json",
    ]
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            "기존 Codex bridge 산출물을 덮어쓰지 않습니다: " + ", ".join(str(path) for path in existing)
        )
    bridge_dir.mkdir(parents=True, exist_ok=True)

    manifest_path, roles, _ = _role_paths(workspace)
    structure_path = roles.get("structure")
    japanese_path = roles.get("ja")
    if structure_path is None or japanese_path is None:
        raise FileNotFoundError("structure/ja SRT를 찾을 수 없습니다.")
    previous_path = roles.get("previous_ko")
    scenes_path = _optional_workspace_file(workspace, title_id, "review-scenes.csv")
    local_asr_path = _optional_workspace_file(workspace, title_id, "asr-candidates.csv")

    evidence, alignment, _, _ = build_autonomous_evidence(
        title_id=title_id,
        structure_path=structure_path,
        japanese_path=japanese_path,
        previous_path=previous_path,
        scenes_path=scenes_path,
        local_asr_path=local_asr_path,
    )
    reference, _, _ = parse_srt(structure_path)

    evidence_path = bridge_dir / "evidence.jsonl"
    alignment_path = bridge_dir / "machine-alignment.json"
    _write_jsonl(evidence_path, evidence)
    _write_json(alignment_path, alignment)

    input_paths: list[tuple[str, Path]] = [
        ("project-manifest", manifest_path),
        ("structure", structure_path),
        ("ja", japanese_path),
    ]
    for role, path in (
        ("previous_ko", previous_path),
        ("review-scenes", scenes_path),
        ("asr-candidates", local_asr_path),
    ):
        if path is not None:
            input_paths.append((role, path))

    info = {
        "schema_name": "translation-forensics/codex-bridge-preparation",
        "schema_version": "2",
        "title_id": title_id,
        "total_blocks": len(reference),
        "prepared_inputs": [_input_record(role, path) for role, path in input_paths],
        "evidence_path": str(evidence_path.resolve()),
        "evidence_sha256": _sha256(evidence_path),
        "alignment_path": str(alignment_path.resolve()),
        "alignment_sha256": _sha256(alignment_path),
        "decision_schema_path": str((ROOT / "schemas" / "autonomous-decision.schema.json").resolve()),
        "decision_schema_sha256": _sha256(ROOT / "schemas" / "autonomous-decision.schema.json"),
        "alignment_status": alignment.get("status"),
        "alignment_coverage": alignment.get("coverage", 0),
        "human_reviewed": False,
        "final_promotion_allowed": False,
    }
    _write_json(bridge_dir / "title-info.json", info)

    risk_counts = Counter(
        code for item in evidence for code in item.get("risk_codes", [])
    )
    _write_json(
        bridge_dir / "evidence-summary.json",
        {
            "title_id": title_id,
            "blocks": len(reference),
            "risk_distribution": dict(risk_counts.most_common()),
            "instructions": [
                "evidence.jsonl만 근거로 decisions.jsonl을 작성합니다.",
                "기존 한국어 후보와 표면형 규칙을 정답으로 취급하지 않습니다.",
                "prepare 이후 입력 또는 evidence 해시가 달라지면 finalize는 중단됩니다.",
                "이 경로는 모델 호출 provenance와 독립 비평을 기록하지 않으므로 draft 등급입니다.",
            ],
        },
    )
    return {
        "status": "prepared",
        "title_id": title_id,
        "blocks": len(reference),
        "bridge_dir": str(bridge_dir),
        "evidence_path": str(evidence_path),
        "decisions_path": str(bridge_dir / "decisions.jsonl"),
        "alignment": alignment.get("status"),
        "final_promotion_allowed": False,
    }


def phase_finalize(title_id: str, workspace: Path, output_dir: Path | None = None) -> dict[str, Any]:
    workspace = workspace.expanduser().resolve()
    bridge_dir = workspace / "codex-bridge"
    info_path = bridge_dir / "title-info.json"
    evidence_path = bridge_dir / "evidence.jsonl"
    decisions_path = bridge_dir / "decisions.jsonl"
    alignment_path = bridge_dir / "machine-alignment.json"
    for path in (info_path, evidence_path, decisions_path, alignment_path):
        if not path.exists():
            raise FileNotFoundError(f"필수 bridge 파일 없음: {path}")

    info = _json(info_path)
    if info.get("title_id") != title_id:
        raise ValueError("prepare title_id와 finalize title_id가 다릅니다.")
    if _sha256(evidence_path) != info.get("evidence_sha256"):
        raise ValueError("prepare 이후 evidence.jsonl이 변경되었습니다. prepare를 다시 실행하십시오.")
    if _sha256(alignment_path) != info.get("alignment_sha256"):
        raise ValueError("prepare 이후 machine-alignment.json이 변경되었습니다.")
    schema_path = Path(str(info.get("decision_schema_path") or "")).resolve()
    if not schema_path.exists() or _sha256(schema_path) != info.get("decision_schema_sha256"):
        raise ValueError("prepare 이후 autonomous decision schema가 변경되었습니다.")

    prepared = _verify_prepared_inputs(info)
    structure_path = prepared.get("structure")
    japanese_path = prepared.get("ja")
    if structure_path is None or japanese_path is None:
        raise ValueError("prepare 입력 기록에 structure/ja가 없습니다.")

    reference, _, _ = parse_srt(structure_path)
    japanese, _, _ = parse_srt(japanese_path)
    if not compare_structure(reference, japanese)["pass"]:
        raise ValueError("prepare 입력의 structure와 Japanese SRT 구조가 다릅니다.")

    evidence = _jsonl(evidence_path)
    decisions = _jsonl(decisions_path)
    validated = _validate_decisions(
        title_id,
        [block.number for block in reference],
        decisions,
        allowed_evidence_refs=_allowed_evidence_refs(evidence),
    )
    alignment = _json(alignment_path)
    japanese_by_number = {block.number: block.text for block in japanese}

    ledger: list[dict[str, Any]] = []
    source_blocks: list[SubtitleBlock] = []
    viewer_blocks: list[SubtitleBlock] = []
    for block, decision in zip(reference, validated):
        source_text = str(decision.get("source_faithful_korean") or "").strip() or "…"
        viewer_text = str(decision.get("viewer_natural_korean") or "").strip() or "…"
        source_blocks.append(
            SubtitleBlock(block.number, block.start, block.end, source_text, block.start_seconds, block.end_seconds)
        )
        viewer_blocks.append(
            SubtitleBlock(block.number, block.start, block.end, viewer_text, block.start_seconds, block.end_seconds)
        )
        ledger.append(
            {
                **decision,
                "schema_name": "translation-forensics/autonomous-decision",
                "schema_version": "1",
                "source_srt_text": source_text,
                "human_reviewed": False,
                "human_final_allowed": False,
                "final_promotion_allowed": False,
            }
        )

    _validate_decision_schema(ledger)

    output_dir = (output_dir or workspace / "codex-assisted-draft-v1").expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"기존 Codex draft 패키지를 덮어쓰지 않습니다: {output_dir}")
    output_dir.mkdir(parents=True)

    structure_copy = output_dir / "structure.srt"
    source_output = output_dir / f"{title_id}.source-faithful-ko.codex-assisted-v1.srt"
    viewer_output = output_dir / f"{title_id}.viewer-complete-ko.codex-assisted-v1.srt"
    shutil.copy2(structure_path, structure_copy)
    write_srt(source_output, source_blocks)
    write_srt(viewer_output, viewer_blocks)

    decisions_out = output_dir / "codex-decisions.jsonl"
    evidence_out = output_dir / "codex-evidence.jsonl"
    uncertainty_path = output_dir / "uncertainty-map.jsonl"
    memory_path = output_dir / "title-memory.json"
    graph_path = output_dir / "evidence-graph.json"
    qa_path = output_dir / "qa-report.json"
    report_path = output_dir / "codex-release-report.json"
    proof_path = output_dir / "codex-proof.json"
    manifest_path = output_dir / "codex-release-manifest.json"

    _write_jsonl(decisions_out, ledger)
    _write_jsonl(evidence_out, evidence)
    _write_jsonl(uncertainty_path, [row for row in ledger if row.get("source_status") == "abstained"])
    _write_json(
        memory_path,
        {
            "schema_name": "translation-forensics/title-memory",
            "schema_version": "1",
            "title_id": title_id,
            "records": _title_memory(validated, japanese_by_number),
        },
    )
    shutil.copy2(alignment_path, output_dir / "machine-alignment.json")
    _write_json(graph_path, _build_evidence_graph(evidence, ledger, []))

    srt_qa = validate_pair(structure_copy, source_output, viewer_output, project_root=ROOT)
    regression = validate_autonomous_regressions(
        title_id=title_id,
        decisions=validated,
        critiques=[],
        suite_path=ROOT / "regressions" / "autonomous-release-v1.json",
    )
    critical_conflicts = sum(
        1
        for row in ledger
        if row.get("source_status") == "accepted"
        and any(str(code).startswith("critical:") for code in row.get("risk_codes", []))
    )
    qa_status = "fail" if (
        srt_qa.get("status") == "fail"
        or regression.get("status") == "fail"
        or critical_conflicts
    ) else srt_qa.get("status")
    qa = {
        "schema_name": "translation-forensics/autonomous-qa-report",
        "schema_version": "1",
        "status": qa_status,
        "structure_same": bool(srt_qa.get("structure_same")),
        "srt_validation": srt_qa,
        "semantic_slot_coverage": sum(bool(row.get("semantic_slots")) for row in ledger) / max(1, len(ledger)),
        "accepted_critical_conflicts": critical_conflicts,
        "regression": regression,
        "independent_critic_review": "not-performed",
        "model_call_provenance": "not-recorded",
    }
    _write_json(qa_path, qa)

    counts = Counter(row.get("source_status") for row in ledger)
    viewer_counts = Counter(row.get("viewer_status") for row in ledger)
    valid = qa_status != "fail" and critical_conflicts == 0
    report = {
        "schema_name": "translation-forensics/autonomous-release-report",
        "schema_version": "1",
        "title_id": title_id,
        "release_kind": DRAFT_KIND,
        "status": DRAFT_STATUS if valid else INVALID_STATUS,
        "blocks": len(reference),
        "source_accepted": counts["accepted"],
        "source_abstained": counts["abstained"],
        "viewer_status_counts": dict(sorted(viewer_counts.items())),
        "critical_conflicts_in_accepted": critical_conflicts,
        "model_calls": 0,
        "estimated_cost_usd": 0.0,
        "external_transfer_calls": None,
        "all_model_calls_traceable": False,
        "codex_session_provenance_recorded": False,
        "independent_critic_review": False,
        "alignment_status": alignment.get("status", "unknown"),
        "human_reviewed": False,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
    }
    _write_json(report_path, report)
    proof = {
        "schema_name": "translation-forensics/autonomous-proof",
        "schema_version": "1",
        "title_id": title_id,
        "release_kind": DRAFT_KIND,
        "structure_preserved": bool(srt_qa.get("structure_same")),
        "viewer_block_coverage": 1.0,
        "source_decision_coverage": 1.0,
        "source_acceptance_rate": round(counts["accepted"] / max(1, len(reference)), 6),
        "source_abstention_rate": round(counts["abstained"] / max(1, len(reference)), 6),
        "critical_conflicts_in_accepted": critical_conflicts,
        "all_model_calls_traceable": False,
        "human_reference_equality": "unidentifiable",
        "100_percent_equal": False,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
    }
    _write_json(proof_path, proof)

    artifact_paths = [
        structure_copy,
        source_output,
        viewer_output,
        decisions_out,
        evidence_out,
        uncertainty_path,
        memory_path,
        output_dir / "machine-alignment.json",
        graph_path,
        qa_path,
        report_path,
        proof_path,
    ]
    prepared_rows = [row for row in info.get("prepared_inputs", []) if isinstance(row, dict)]
    manifest = {
        "schema_name": "translation-forensics/codex-release-manifest",
        "schema_version": "2",
        "title_id": title_id,
        "release_kind": DRAFT_KIND,
        "inputs": [_portable_input_record(row, workspace) for row in prepared_rows],
        "bridge_inputs": [
            {"path": "codex-bridge/evidence.jsonl", "sha256": _sha256(evidence_path)},
            {"path": "codex-bridge/decisions.jsonl", "sha256": _sha256(decisions_path)},
        ],
        "outputs": [
            {"path": path.relative_to(output_dir).as_posix(), "sha256": _sha256(path)}
            for path in artifact_paths
        ],
        "network_activity_tracked": False,
        "codex_session_provenance_recorded": False,
        "human_reviewed": False,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
    }
    _write_json(manifest_path, manifest)
    return {
        "status": report["status"],
        "title_id": title_id,
        "output": str(output_dir),
        "blocks": len(reference),
        "source_accepted": counts["accepted"],
        "source_abstained": counts["abstained"],
        "qa_status": qa_status,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Codex 결정 파일 검증·draft 패키징 브리지")
    parser.add_argument("phase", choices=["prepare", "finalize"])
    parser.add_argument("--title", required=True)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve() if args.workspace else (WORKSPACES / args.title)
    result = (
        phase_prepare(args.title, workspace)
        if args.phase == "prepare"
        else phase_finalize(args.title, workspace, args.output)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") not in {INVALID_STATUS, "validation-failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
