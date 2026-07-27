from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .asr_evidence import read_asr_candidates
from .reporting import build_asr_verdicts, build_change_log, build_evidence_ledger, build_scene_map, build_uncertainty_map, write_csv, write_qa_markdown
from .semantic_translation import validate_translation_decisions
from .forensic_model import validate_forensic_records
from .mqm import validate_mqm_csv
from .evidence_artifacts import validate_alignment_evidence, validate_backtranslation_check, validate_speaker_state
from .evaluation import validate_evaluation_summary
from .run_manifest import create_run_manifest
from .timeline import timeline_is_usable
from .srt import parse_srt
from .validation import validate_pair, write_validation_report


STAGES = ("structure-validated", "text-crosschecked", "audio-asr-crosschecked", "closed-world-validated", "audio-human-verified", "evaluation-validated", "final")
ARTIFACT_KINDS = ("source-faithful-ko", "viewer-natural-ko", "change-log", "evidence-ledger", "uncertainty-map", "asr-scene-verdicts", "scene-map", "regression-check", "qa-report", "semantic-frames", "hypothesis-ledger", "speaker-state", "alignment-evidence", "mqm-errors", "backtranslation-check", "evaluation-summary", "blind-review-pack", "release-gate", "run-manifest")


def next_version(output_dir: Path, title: str, stage: str) -> int:
    pattern = re.compile(rf"^{re.escape(title)}\.(?:{'|'.join(map(re.escape, ARTIFACT_KINDS))})\.{re.escape(stage)}-v(\d+)\.")
    versions = []
    if output_dir.exists():
        for path in output_dir.iterdir():
            match = pattern.match(path.name)
            if match:
                versions.append(int(match.group(1)))
    return max(versions, default=0) + 1


def _copy_new(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"기존 산출물을 덮어쓰지 않습니다: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL object required: {path}:{line_number}")
        rows.append(value)
    return rows


def package_title_outputs(title: str, reference_path: Path, source_path: Path, viewer_path: Path, output_dir: Path, *, stage: str = "text-crosschecked", version: int | None = None, japanese_path: Path | None = None, previous_path: Path | None = None, photos_path: Path | None = None, scenes_path: Path | None = None, asr_path: Path | None = None, translation_decisions_path: Path | None = None, semantic_frames_path: Path | None = None, hypothesis_ledger_path: Path | None = None, speaker_state_path: Path | None = None, alignment_evidence_path: Path | None = None, mqm_errors_path: Path | None = None, backtranslation_check_path: Path | None = None, evaluation_summary_path: Path | None = None, blind_review_pack_path: Path | None = None, release_gate_path: Path | None = None, timeline_validation_path: Path | None = None, project_root: Path | None = None, notes: list[str] | None = None, all_blocks_reviewed: bool = False, direct_human_listening: bool = False, evidence_complete: bool = False) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"검증 단계가 잘못되었습니다: {stage}")
    if stage == "closed-world-validated":
        raise RuntimeError("closed-world-validated는 사람 검증 패키지와 분리됩니다. run-closed-world 명령을 사용하세요.")
    reference_blocks, _, _ = parse_srt(reference_path)
    validation = validate_pair(reference_path, source_path, viewer_path, project_root=project_root)
    if validation.get("status") == "fail":
        raise RuntimeError("최종 패키징 전 자동 QA가 실패했습니다.")
    if stage == "text-crosschecked" and (not japanese_path or not japanese_path.exists()):
        raise RuntimeError("text-crosschecked에는 일본어 기준본 경로가 필요합니다.")
    if stage in {"audio-asr-crosschecked", "audio-human-verified", "evaluation-validated", "final"} and (not asr_path or not asr_path.exists()):
        raise RuntimeError(f"{stage}에는 ASR 결과 경로가 필요합니다.")
    if stage in {"audio-asr-crosschecked", "audio-human-verified", "evaluation-validated", "final"}:
        if not timeline_validation_path or not timeline_validation_path.exists():
            raise RuntimeError(f"{stage}에는 통과한 timeline-validation 보고서가 필요합니다.")
        timeline = json.loads(timeline_validation_path.read_text(encoding="utf-8"))
        if not timeline_is_usable(timeline):
            raise RuntimeError("timeline-validation이 클립·승격을 허용하지 않습니다.")
    asr_rows = read_asr_candidates(asr_path) if asr_path and asr_path.exists() else []
    if stage == "audio-human-verified" and not direct_human_listening:
        raise RuntimeError("audio-human-verified에는 직접 원음 청취 확인이 필요합니다.")
    if stage in {"evaluation-validated", "final"}:
        if not (evaluation_summary_path and evaluation_summary_path.exists() and blind_review_pack_path and blind_review_pack_path.exists()):
            raise RuntimeError(f"{stage}에는 완료된 blind-review-pack과 evaluation-summary가 필요합니다.")
        evaluation_report = validate_evaluation_summary(evaluation_summary_path)
        if evaluation_report["status"] != "pass":
            raise RuntimeError("사람 검수가 완료되지 않은 evaluation-summary로는 평가 검증 상태에 올릴 수 없습니다.")
    if stage == "final" and not (all_blocks_reviewed and direct_human_listening and evidence_complete):
        raise RuntimeError("final에는 전체 블록 검수, 직접 청취 확인, 증거·QA 완료 확인이 모두 필요합니다.")
    if stage == "final":
        if not release_gate_path or not release_gate_path.exists():
            raise RuntimeError("final에는 통과한 release-gate 보고서가 필요합니다.")
        release_gate = json.loads(release_gate_path.read_text(encoding="utf-8"))
        if release_gate.get("status") != "pass" or release_gate.get("release_allowed") is not True:
            raise RuntimeError("release-gate가 통과하지 않아 final 패키지를 만들 수 없습니다.")
    if stage == "final" and not translation_decisions_path:
        raise RuntimeError("final에는 의미 번역 결정 JSONL이 필요합니다.")
    if translation_decisions_path:
        decision_report = validate_translation_decisions(reference_path, translation_decisions_path, strict=True)
        if decision_report.get("status") != "pass":
            raise RuntimeError("의미 번역 결정 검증이 실패했습니다. apply-translations --strict 결과를 먼저 확인하세요.")
    forensic_paths = (semantic_frames_path, hypothesis_ledger_path)
    if stage in {"evaluation-validated", "final"} and not all(path and path.exists() for path in forensic_paths):
        raise RuntimeError(f"{stage}에는 semantic frame과 hypothesis ledger가 필요합니다.")
    if semantic_frames_path and hypothesis_ledger_path:
        forensic_report = validate_forensic_records(semantic_frames_path, hypothesis_ledger_path)
        if forensic_report["status"] != "pass":
            raise RuntimeError("의미 프레임 또는 가설 원장 검증이 실패했습니다.")
        if stage in {"evaluation-validated", "final"} and (forensic_report["unresolved_frames"] or forensic_report["critical_conflict_escalations"]):
            raise RuntimeError("미해결 semantic frame 또는 critical slot 충돌이 남아 있어 평가 검증 상태로 패키징할 수 없습니다.")
        if stage == "final" and forensic_report["reviewed_frames"] != len(reference_blocks):
            raise RuntimeError("final에는 모든 구조 블록의 reviewed semantic frame이 필요합니다.")
    if mqm_errors_path and mqm_errors_path.exists():
        mqm_report = validate_mqm_csv(mqm_errors_path)
        if mqm_report["status"] != "pass":
            raise RuntimeError("MQM 오류 원장 형식이 잘못되었습니다.")
        if stage in {"evaluation-validated", "final"} and mqm_report["critical_errors"]:
            raise RuntimeError("critical MQM 오류가 남아 있어 평가 검증 상태로 패키징할 수 없습니다.")
    artifact_validators = {
        "speaker-state": (speaker_state_path, validate_speaker_state),
        "alignment-evidence": (alignment_evidence_path, validate_alignment_evidence),
        "backtranslation-check": (backtranslation_check_path, validate_backtranslation_check),
    }
    for label, (artifact_path, validator) in artifact_validators.items():
        if artifact_path and artifact_path.exists():
            artifact_report = validator(artifact_path)
            if artifact_report["status"] != "pass":
                raise RuntimeError(f"{label} artifact validation failed: {artifact_report['errors'][:3]}")
    version = version or next_version(output_dir, title, stage)
    prefix = f"{title}."
    paths = {
        "source": output_dir / f"{prefix}source-faithful-ko.{stage}-v{version}.srt",
        "viewer": output_dir / f"{prefix}viewer-natural-ko.{stage}-v{version}.srt",
        "change": output_dir / f"{prefix}change-log.{stage}-v{version}.csv",
        "evidence": output_dir / f"{prefix}evidence-ledger.{stage}-v{version}.csv",
        "uncertainty": output_dir / f"{prefix}uncertainty-map.{stage}-v{version}.csv",
        "asr_verdicts": output_dir / f"{prefix}asr-scene-verdicts.{stage}-v{version}.csv",
        "scene_map": output_dir / f"{prefix}scene-map.{stage}-v{version}.json",
        "regression": output_dir / f"{prefix}regression-check.{stage}-v{version}.json",
        "qa": output_dir / f"{prefix}qa-report.{stage}-v{version}.md",
        "run_manifest": output_dir / f"{prefix}run-manifest.{stage}-v{version}.json",
    }
    optional_artifacts = {
        "semantic_frames": semantic_frames_path,
        "hypothesis_ledger": hypothesis_ledger_path,
        "speaker_state": speaker_state_path,
        "alignment_evidence": alignment_evidence_path,
        "mqm_errors": mqm_errors_path,
        "backtranslation_check": backtranslation_check_path,
        "evaluation_summary": evaluation_summary_path,
        "blind_review_pack": blind_review_pack_path,
        "release_gate": release_gate_path,
        "timeline_validation": timeline_validation_path,
    }
    suffixes = {"semantic_frames": ".jsonl", "hypothesis_ledger": ".jsonl", "speaker_state": ".json", "alignment_evidence": ".csv", "mqm_errors": ".csv", "backtranslation_check": ".csv", "evaluation_summary": ".json", "blind_review_pack": ".zip", "release_gate": ".json", "timeline_validation": ".json"}
    labels = {"semantic_frames": "semantic-frames", "hypothesis_ledger": "hypothesis-ledger", "speaker_state": "speaker-state", "alignment_evidence": "alignment-evidence", "mqm_errors": "mqm-errors", "backtranslation_check": "backtranslation-check", "evaluation_summary": "evaluation-summary", "blind_review_pack": "blind-review-pack", "release_gate": "release-gate", "timeline_validation": "timeline-validation"}
    for key, input_path in optional_artifacts.items():
        if input_path and input_path.exists():
            paths[key] = output_dir / f"{prefix}{labels[key]}.{stage}-v{version}{suffixes[key]}"
    _copy_new(source_path, paths["source"])
    _copy_new(viewer_path, paths["viewer"])
    for key, input_path in optional_artifacts.items():
        if key in paths and input_path:
            _copy_new(input_path, paths[key])
    reference = reference_blocks
    japanese = parse_srt(japanese_path)[0] if japanese_path and japanese_path.exists() else None
    source, _, _ = parse_srt(source_path)
    viewer, _, _ = parse_srt(viewer_path)
    previous = parse_srt(previous_path)[0] if previous_path and previous_path.exists() else None
    decisions = _read_jsonl(translation_decisions_path)
    write_csv(
        paths["change"],
        build_change_log(reference, previous, source, viewer, status=stage, japanese=japanese, decisions=decisions),
        ["block_number", "start_time", "end_time", "source_japanese", "previous_korean", "source_faithful_korean", "viewer_natural_korean", "change_type", "main_issue", "evidence_summary", "confidence", "review_note"],
    )
    write_csv(
        paths["evidence"],
        build_evidence_ledger(
            reference,
            status=stage,
            asr_rows=asr_rows,
            scenes_path=scenes_path,
            decisions=decisions,
            japanese=japanese,
            previous=previous,
            japanese_available=japanese is not None,
            previous_available=previous is not None,
            photos_available=photos_path is not None,
            screen_available=False,
        ),
        ["block_number", "timecode", "japanese_reference", "alternative_japanese_asr", "original_unbiased_asr", "dialogue_unbiased_asr", "original_no_vad_asr", "original_prompted_asr", "photos", "screen", "neighboring_context", "previous_korean", "decision_status", "decision_evidence_refs", "error_memory", "verification_status", "evidence_independence_note"],
    )
    write_csv(
        paths["uncertainty"],
        build_uncertainty_map(reference, status=stage, decisions=decisions),
        ["block_number", "uncertain_block", "uncertain_slots", "possible_meaning_range", "adopted_broad_expression", "additional_evidence_needed", "current_verification_status", "impact"],
    )
    if scenes_path and scenes_path.exists():
        verdicts = build_asr_verdicts(scenes_path, asr_path)
    else:
        verdicts = []
    write_csv(paths["asr_verdicts"], verdicts, ["scene_id", "verdict", "confidence", "evidence_summary", "unresolved_scope", "direct_human_listening"])
    paths["scene_map"].write_text(json.dumps(build_scene_map(scenes_path, asr_path), ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    validation_path = paths["regression"]
    write_validation_report(validation_path, validation)
    write_qa_markdown(paths["qa"], validation, title=title, stage=stage, status=stage, notes=notes or ["의미 판정·직접 청취·전체 블록 검수 여부는 코드가 자동 확정하지 않습니다.", "필요한 증거가 없으면 final 상태로 올리지 않습니다."])
    manifest_inputs = [reference_path, source_path, viewer_path]
    if translation_decisions_path and translation_decisions_path.exists():
        manifest_inputs.append(translation_decisions_path)
    if asr_path and asr_path.exists():
        manifest_inputs.append(asr_path)
    if scenes_path and scenes_path.exists():
        manifest_inputs.append(scenes_path)
    manifest_artifacts = [path for key, path in paths.items() if key != "run_manifest" and path.exists()]
    create_run_manifest(project_root or reference_path.parent, paths["run_manifest"], title=title, stage=stage, inputs=manifest_inputs, artifacts=manifest_artifacts)
    return {"status": "packaged", "title": title, "stage": stage, "version": version, "output_dir": str(output_dir), "files": {key: str(value) for key, value in paths.items()}, "validation": validation}
