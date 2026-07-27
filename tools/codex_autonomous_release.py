import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# --- 기존 모듈 임포트 ---
try:
    from translation_forensics.autonomous_release import (
        build_autonomous_evidence,
        _validate_decisions,
        _allowed_evidence_refs,
        _title_memory,
        _build_evidence_graph,
        estimate_machine_alignment,
        AUTONOMOUS_RELEASE_KIND,
        SOURCE_STATUSES,
        VIEWER_STATUSES,
        CONFIDENCES,
        SEMANTIC_SLOT_KEYS,
    )
    from translation_forensics.srt import parse_srt, write_srt, SubtitleBlock, compare_structure
    from translation_forensics.validation import validate_pair
    from translation_forensics.cli import _workspace, _project_root
except ImportError as exc:
    print(f"Import error: {exc}", file=sys.stderr)
    print("Run from the translation-forensics project root with: python -m tools.codex_autonomous_release ...", file=sys.stderr)
    sys.exit(1)


ROOT = Path(__file__).resolve().parents[1]  # translation-forensics/
WORKSPACES = ROOT / "workspaces"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(v, ensure_ascii=False, sort_keys=True, default=str) + "\n" for v in values),
        encoding="utf-8",
        newline="\n",
    )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            result.append(json.loads(line))
    return result


def _semantic_slots_empty() -> dict[str, Any]:
    keys = [
        "question", "polarity", "refusal_permission", "command_strength",
        "speaker", "actor", "action", "target", "location", "tense_aspect",
        "direction", "intensity",
    ]
    return {k: None for k in keys}


# ---------- 1단계: 증거 준비 ----------

def phase_prepare(title_id: str, workspace: Path) -> dict[str, Any]:
    """증거를 준비하고 브릿지 디렉터리에 저장합니다."""
    from translation_forensics.discovery import resolve_role

    bridge_dir = workspace / "codex-bridge"
    bridge_dir.mkdir(parents=True, exist_ok=True)

    # 입력 파일 찾기
    manifest_path = workspace / "metadata" / "project-manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"매니페스트 없음: {manifest_path}")
    manifest = _json(manifest_path)

    inputs = manifest.get("inputs", [])
    def path_for_role(role: str) -> Path | None:
        for inp in inputs:
            if isinstance(inp, dict) and inp.get("role") == role:
                p = Path(str(inp["path"]))
                return p if p.exists() else None
        return None

    structure_path = path_for_role("structure")
    japanese_path = path_for_role("ja")
    previous_path = path_for_role("previous_ko")
    audio_path = path_for_role("audio")

    if structure_path is None or japanese_path is None:
        raise FileNotFoundError(f"structure/ja SRT를 찾을 수 없습니다.")

    # 장면 CSV 및 로컬 ASR 경로
    scenes_path = None
    for candidate in [
        workspace / "inferred-recovery-audio-v4" / "review-scenes.csv",
        workspace / "intermediate" / f"{title_id}.work_audio" / "review-scenes.csv",
        workspace / "review-scenes.csv",
    ]:
        if candidate.exists():
            scenes_path = candidate
            break

    local_asr_path = None
    for candidate in [
        workspace / "inferred-recovery-audio-v4" / "asr-candidates.csv",
        workspace / "intermediate" / f"{title_id}.work_audio" / "asr-candidates.csv",
        workspace / "asr-candidates.csv",
    ]:
        if candidate.exists():
            local_asr_path = candidate
            break

    # 증거 수집 (로컬 전용)
    evidence, alignment, scenes_by_id, asr_by_scene = build_autonomous_evidence(
        title_id=title_id,
        structure_path=structure_path,
        japanese_path=japanese_path,
        previous_path=previous_path,
        scenes_path=scenes_path,
        local_asr_path=local_asr_path,
    )

    reference, _, _ = parse_srt(structure_path)
    japanese, _, _ = parse_srt(japanese_path)
    japanese_by_number = {b.number: b.text for b in japanese}

    # 브릿지 데이터 저장
    info = {
        "title_id": title_id,
        "total_blocks": len(reference),
        "structure_path": str(structure_path),
        "japanese_path": str(japanese_path),
        "scenes_path": str(scenes_path),
        "local_asr_path": str(local_asr_path),
        "alignment_status": alignment["status"],
        "alignment_coverage": alignment.get("coverage", 0),
    }
    _write_json(bridge_dir / "title-info.json", info)
    _write_jsonl(bridge_dir / "evidence.jsonl", evidence)
    _write_json(bridge_dir / "machine-alignment.json", alignment)
    _write_json(bridge_dir / "title-info.json", info)

    # 블록별 요약 (내가 읽기 쉽게)
    summary_lines = []
    summary_lines.append(f"=== {title_id} === {len(reference)} 블록")
    summary_lines.append(f"정렬 상태: {alignment['status']} (커버리지: {alignment.get('coverage', 0):.1%})")
    summary_lines.append(f"")

    risk_counts = Counter()
    for item in evidence:
        for code in item.get("risk_codes", []):
            risk_counts[code] += 1
    if risk_counts:
        summary_lines.append("리스크 분포:")
        for code, count in risk_counts.most_common():
            summary_lines.append(f"  {code}: {count}")
        summary_lines.append("")

    summary_lines.append(f"증거 파일: {bridge_dir / 'evidence.jsonl'}")
    summary_lines.append(f"블록당 증거 구조:")
    summary_lines.append(f"  - block_number: 블록 번호")
    summary_lines.append(f"  - japanese.text: 일본어 원문")
    summary_lines.append(f"  - local_asr[]: 로컬 ASR 후보 (timeline_compatible인 경우)")
    summary_lines.append(f"  - local_context[]: 앞뒤 2블록 컨텍스트")
    summary_lines.append(f"  - previous_korean_candidate: 이전 한국어 초안 (있는 경우)")
    summary_lines.append(f"  - risk_codes: 위험 코드 목록")
    summary_lines.append(f"  - timeline_compatible: 타임라인 정렬 여부")
    summary_lines.append("")

    _write_json(bridge_dir / "evidence-summary.json", {
        "summary": "\n".join(summary_lines),
        "info": info,
        "risk_distribution": dict(risk_counts.most_common()),
    })

    return {
        "status": "prepared",
        "title_id": title_id,
        "blocks": len(reference),
        "evidence_path": str(bridge_dir / "evidence.jsonl"),
        "decisions_path": str(bridge_dir / "decisions.jsonl"),
        "bridge_dir": str(bridge_dir),
        "alignment": alignment["status"],
    }


# ---------- 2단계: 결정 파일 읽기 및 검증 ----------

def phase_finalize(title_id: str, workspace: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """내가 작성한 결정을 읽고 검증·패키징합니다."""
    bridge_dir = workspace / "codex-bridge"
    evidence_path = bridge_dir / "evidence.jsonl"
    decisions_path = bridge_dir / "decisions.jsonl"

    if not evidence_path.exists():
        raise FileNotFoundError(f"증거 파일 없음: {evidence_path}")
    if not decisions_path.exists():
        raise FileNotFoundError(f"결정 파일 없음: {decisions_path}")

    # 결정 읽기
    decisions = _jsonl(decisions_path)
    evidence = _jsonl(evidence_path)
    info = _json(bridge_dir / "title-info.json") if (bridge_dir / "title-info.json").exists() else {}
    alignment = _json(bridge_dir / "machine-alignment.json") if (bridge_dir / "machine-alignment.json").exists() else {}

    # 구조/일본어 SRT 로드
    manifest_path = workspace / "metadata" / "project-manifest.json"
    manifest = _json(manifest_path)
    inputs = manifest.get("inputs", [])
    def path_for_role(role: str) -> Path | None:
        for inp in inputs:
            if isinstance(inp, dict) and inp.get("role") == role:
                p = Path(str(inp["path"]))
                return p if p.exists() else None
        return None

    structure_path = path_for_role("structure")
    japanese_path = path_for_role("ja")

    if structure_path is None or japanese_path is None:
        # manifest에서 못 찾으면 직접 경로 시도
        for inp in manifest.get("inputs", []):
            if isinstance(inp, dict):
                p = Path(str(inp["path"]))
                if p.exists():
                    if inp.get("role") == "structure":
                        structure_path = p
                    elif inp.get("role") == "ja":
                        japanese_path = p

    reference, _, _ = parse_srt(structure_path)
    japanese, _, _ = parse_srt(japanese_path)
    japanese_by_number = {b.number: b.text for b in japanese}
    allowed_refs = _allowed_evidence_refs(evidence)

    # 결정 검증
    try:
        validated = _validate_decisions(
            title_id,
            [b.number for b in reference],
            decisions,
            allowed_evidence_refs=allowed_refs,
        )
    except ValueError as exc:
        return {"status": "validation-failed", "error": str(exc)}

    # 각 결정에 ledger 필드 추가
    ledger = []
    for decision in validated:
        source_text = str(decision.get("source_faithful_korean") or "").strip() or "…"
        viewer_text = str(decision["viewer_natural_korean"]).strip()
        ledger.append({
            "schema_name": "translation-forensics/autonomous-decision",
            "schema_version": "1",
            **decision,
            "source_srt_text": source_text,
            "human_reviewed": False,
            "human_final_allowed": False,
            "final_promotion_allowed": False,
        })

    # ---------- 출력 패키징 ----------
    if output_dir is None:
        output_dir = workspace / "autonomous-release-codex-v1"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # SRT 생성
    source_blocks: list[SubtitleBlock] = []
    viewer_blocks: list[SubtitleBlock] = []
    for block, decision in zip(reference, validated):
        source_text = str(decision.get("source_faithful_korean") or "").strip() or "…"
        viewer_text = str(decision["viewer_natural_korean"]).strip()
        source_blocks.append(SubtitleBlock(
            block.number, block.start, block.end, source_text,
            block.start_seconds, block.end_seconds,
        ))
        viewer_blocks.append(SubtitleBlock(
            block.number, block.start, block.end, viewer_text,
            block.start_seconds, block.end_seconds,
        ))

    source_output = output_dir / f"{title_id}.source-faithful-ko.codex-release-v1.srt"
    viewer_output = output_dir / f"{title_id}.viewer-complete-ko.codex-release-v1.srt"
    structure_copy = output_dir / "structure.srt"

    write_srt(source_output, source_blocks)
    write_srt(viewer_output, viewer_blocks)
    shutil.copy2(structure_path, structure_copy)

    # JSONL 출력
    decisions_path_out = output_dir / "codex-decisions.jsonl"
    evidence_path_out = output_dir / "codex-evidence.jsonl"
    uncertainty_path = output_dir / "uncertainty-map.jsonl"

    _write_jsonl(decisions_path_out, ledger)
    _write_jsonl(evidence_path_out, evidence)
    _write_jsonl(uncertainty_path, [r for r in ledger if r["source_status"] == "abstained"])

    # title-memory
    memory = _title_memory(validated, {n: japanese_by_number.get(n, "") for n in japanese_by_number})
    memory_path = output_dir / "title-memory.json"
    _write_json(memory_path, {
        "schema_name": "translation-forensics/title-memory",
        "schema_version": "1",
        "title_id": title_id,
        "records": memory,
    })

    # alignment
    alignment_path = output_dir / "machine-alignment.json"
    _write_json(alignment_path, alignment)

    # evidence graph (빈 critiques 목록 사용)
    graph = _build_evidence_graph(evidence, validated, [])
    _write_json(output_dir / "evidence-graph.json", graph)

    # QA
    srt_qa = validate_pair(structure_copy, source_output, viewer_output, project_root=ROOT)
    counts = Counter(r["source_status"] for r in ledger)
    viewer_counts = Counter(r["viewer_status"] for r in ledger)
    critical_conflicts = sum(
        1 for r in ledger
        if r["source_status"] == "accepted"
        and any(str(c).startswith("critical:") for c in r.get("risk_codes", []))
    )

    qa = {
        "schema_name": "translation-forensics/autonomous-qa-report",
        "schema_version": "1",
        "status": srt_qa.get("status"),
        "structure_same": bool(srt_qa.get("structure_same")),
        "srt_validation": srt_qa,
        "semantic_slot_coverage": sum(1 for r in ledger if r.get("semantic_slots")) / max(1, len(ledger)),
        "accepted_critical_conflicts": critical_conflicts,
        "regression": {"status": "skipped", "reason": "codex-powered, no regression suite"},
        "readability": {
            "source_status": srt_qa.get("source_faithful", {}).get("status") if isinstance(srt_qa.get("source_faithful"), dict) else None,
            "viewer_status": srt_qa.get("viewer_natural", {}).get("status") if isinstance(srt_qa.get("viewer_natural"), dict) else None,
        },
    }
    _write_json(output_dir / "qa-report.json", qa)

    # Report
    report = {
        "schema_name": "translation-forensics/autonomous-release-report",
        "schema_version": "1",
        "title_id": title_id,
        "release_kind": "codex-autonomous-release",
        "status": "codex-release-packaged" if qa.get("status") != "fail" and critical_conflicts == 0 else "codex-release-invalid",
        "blocks": len(reference),
        "source_accepted": counts["accepted"],
        "source_abstained": counts["abstained"],
        "viewer_status_counts": dict(sorted(viewer_counts.items())),
        "critical_conflicts_in_accepted": critical_conflicts,
        "model_calls": 0,
        "estimated_cost_usd": 0.0,
        "external_transfer_calls": 0,
        "cache_replay_calls": 0,
        "all_model_calls_traceable": False,
        "alignment_status": alignment.get("status", "unknown"),
        "human_reviewed": False,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
        "powered_by": "codex-gpt-5.6",
        "codex_prompt": "translation-forensics autonomous-release (Codex-powered)",
    }
    _write_json(output_dir / "codex-release-report.json", report)

    # Proof
    proof = {
        "schema_name": "translation-forensics/autonomous-proof",
        "schema_version": "1",
        "title_id": title_id,
        "release_kind": "codex-autonomous-release",
        "structure_preserved": bool(srt_qa.get("structure_same")),
        "viewer_block_coverage": 1.0,
        "source_decision_coverage": 1.0,
        "source_acceptance_rate": round(counts["accepted"] / max(1, len(reference)), 6),
        "source_abstention_rate": round(counts["abstained"] / max(1, len(reference)), 6),
        "critical_conflicts_in_accepted": critical_conflicts,
        "all_model_calls_traceable": False,
        "external_transfer_calls": 0,
        "human_reference_equality": "unidentifiable",
        "100_percent_equal": False,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
        "powered_by": "codex-gpt-5.6",
    }
    _write_json(output_dir / "codex-proof.json", proof)

    # Manifest
    artifact_paths = [
        structure_copy, source_output, viewer_output,
        decisions_path_out, evidence_path_out, uncertainty_path,
        memory_path, alignment_path,
        output_dir / "evidence-graph.json",
        output_dir / "qa-report.json",
        output_dir / "codex-release-report.json",
        output_dir / "codex-proof.json",
    ]
    manifest = {
        "schema_name": "translation-forensics/codex-release-manifest",
        "schema_version": "1",
        "title_id": title_id,
        "release_kind": "codex-autonomous-release",
        "inputs": [
            {"path": str(p), "sha256": _sha256(p)}
            for p in [structure_path, japanese_path]
            if p and p.exists()
        ],
        "outputs": [
            {"path": p.relative_to(output_dir).as_posix(), "sha256": _sha256(p)}
            for p in artifact_paths if p.exists()
        ],
        "network_enabled": False,
        "powered_by": "codex-gpt-5.6",
        "human_reviewed": False,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
    }
    _write_json(output_dir / "codex-release-manifest.json", manifest)

    return {
        "status": report["status"],
        "title_id": title_id,
        "output": str(output_dir),
        "blocks": len(reference),
        "source_accepted": counts["accepted"],
        "source_abstained": counts["abstained"],
        "viewer_complete": True,
        "critical_conflicts": critical_conflicts,
        "qa_status": qa.get("status"),
    }


# ---------- CLI ----------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Codex 기반 autonomous-release 브릿지")
    parser.add_argument("phase", choices=["prepare", "finalize"])
    parser.add_argument("--title", required=True)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = ROOT
    workspace = args.workspace.resolve() if args.workspace else (WORKSPACES / args.title)

    if args.phase == "prepare":
        result = phase_prepare(args.title, workspace)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()
        print("=" * 60)
        print(f"증거 준비 완료: {result['bridge_dir']}")
        print("evidence.jsonl을 읽고 블록별 결정을 decisions.jsonl로 작성해주세요.")
        print("결정 형식은 evidence-summary.json을 참고하세요.")
        print(f"결정 작성 후: python tools/codex_autonomous_release.py finalize --title {args.title}")
        print("=" * 60)
    elif args.phase == "finalize":
        result = phase_finalize(args.title, workspace, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("status") == "codex-release-packaged":
            print(f"\n릴리스 패키지 생성 완료: {result['output']}")
        elif result.get("status") == "validation-failed":
            print(f"\n검증 실패: {result.get('error')}")
        else:
            print(f"\n상태: {result.get('status')}")


if __name__ == "__main__":
    main()
