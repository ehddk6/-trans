from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .audio_adapter import detect_device, ingest_asr, prepare_audio, run_asr, write_run_summary, zip_work_audio
from .audit_sampling import sample_audit, summarize_audit
from .alignment import align_to_dicts
from .automatic_draft import build_automatic_draft
from .autonomous_release import prove_autonomous_claim, run_autonomous_release, validate_autonomous_release
from .asr_evidence import read_asr_candidates
from .closed_world import prove_quality_claim, run_closed_world, validate_closed_world_package
from .codex_exec_provider import CodexExecError, CodexExecProvider, CodexUsageLimitError
from .codex_quality import CodexQualityError, evaluate_codex_quality, evaluate_evidence_ceiling, run_codex_quality_title, validate_codex_quality
from .consistency import initialize_consistency_ledger, validate_consistency_ledger
from .discovery import DiscoveryError, inspect_roles, resolve_role
from .drafts import build_korean_aligned_draft
from .forensics_adapter import analyze_title
from .forensic_artifacts import build_slot_conflicts, initialize_phonetic_candidates, initialize_speaker_state
from .forensic_model import initialize_forensic_records, validate_forensic_records
from .identity import migrate_jsonl_identity, validate_identity_records
from .inferred_recovery import apply_inferred_recovery, build_inference_audio_queue, build_inference_context
from .evaluation import build_blind_review_pack, initialize_gold_layout, initialize_gold_record, migrate_gold_layout, summarize_blind_review, validate_evaluation_summary, validate_gold_record, validate_gold_suite
from .evidence_artifacts import validate_alignment_evidence, validate_backtranslation_check, validate_speaker_state
from .evidence_graph import build_evidence_graph
from .mqm import validate_mqm_csv
from .manifest import append_history, build_project_manifest, write_json
from .memory_ledger import initialize_memory_ledger, validate_memory_ledger
from .machine_final import package_machine_final, repair_machine_final_asr
from .local_asr import (
    FOCUSED_REPAIR_POLICY,
    LocalASRError,
    build_timestamped_utterance_evidence,
    merge_repaired_acoustic_evidence,
    run_full_local_asr,
    run_pre_ceiling_evidence_repair,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
from .openai_provider import BudgetTracker, OpenAIProvider, ProviderUnavailableError, BudgetExceededError
from .outputs import STAGES, package_title_outputs
from .pilot_audio_review import PilotAlignmentError, build_pilot_audio_review_packet, evaluate_pilot_alignment, write_blocked_audio_review_manifest, write_pilot_alignment
from .pilot_blind_review import (
    PilotBlindReviewBlocked,
    PilotBlindReviewError,
    adjudicate_pilot_reviews,
    build_pilot_blind_review_packets,
    initialize_pilot_blind_internal_key,
    validate_pilot_reviewer_submission,
    write_blocked_pilot_review_artifacts,
)
from .pilot_candidate import PilotCandidateError, build_pilot_candidate, validate_pilot_candidate, write_pilot_sol_reviews
from .pilot_automated_validation import (
    PilotAutomatedValidationError,
    validate_pilot_automated,
    write_pilot_automated_validation,
)
from .pilot_metrics import (
    MAXIMUM_NATURALNESS_LOSS_PERCENT,
    MINIMUM_ERROR_BLOCK_REDUCTION_PERCENT,
    MINIMUM_NATURALNESS_WIN_PERCENT,
    PilotMetricsError,
    evaluate_critical_major_error_block_reduction_file,
    evaluate_naturalness_comparison_file,
    evaluate_new_critical_semantic_errors_file,
    evaluate_pilot_balance_contract_file,
    write_error_block_reduction_result,
    write_naturalness_comparison_result,
    write_new_critical_semantic_error_result,
    write_pilot_balance_contract_results,
)
from .pilot_report import (
    BASELINE_SHA256 as PILOT_BASELINE_SHA256,
    PilotReportError,
    build_pilot_evaluation_report,
    write_pilot_evaluation_report,
)
from .process_title import ProcessTitleConfig, process_title
from .prompt_contract import validate_prompt_contract
from .review_pack import build_review_pack, validate_review_decisions
from .reverse_check import initialize_reverse_check, validate_reverse_check
from .release_metrics import calculate_review_budget_metrics, validate_release_gate
from .run_manifest import create_run_manifest
from .reporting import build_asr_verdicts, build_review_context, build_scene_map, write_csv
from .review_prioritization import build_uncertainty_review_queue
from .semantic_translation import apply_translation_decisions, build_translation_queue, initialize_translation_decisions, merge_translation_decisions
from .offline_hybrid import build_offline_hybrid
from .translation_model import ALLOWED_TRANSLATION_MODELS, DEFAULT_TRANSLATION_MODEL
from .translation_quality import validate_quality_regression_suite
from .scenes import build_review_scenes, read_review_queue, write_review_scenes
from .srt import compare_structure, parse_srt
from .timeline import initialize_timeline_anchor_template, read_timeline_anchors, timeline_is_usable, validate_timeline
from .sol_review import initialize_sol_review_records, validate_sol_review_records
from .source_quality import write_source_quality_audit
from .terminology import initialize_terminology, terminology_conflicts, validate_terminology
from .targeted_retranslation import apply_targeted_retranslations
from .validation import validate_pair, write_validation_report
from .workspace_audit import audit_workspace


LOG = logging.getLogger("translation_forensics")


def _project_root(args: argparse.Namespace) -> Path:
    return Path(args.project_root).expanduser().resolve() if args.project_root else Path(__file__).resolve().parents[2]


def _workspace(root: Path, title: str, explicit: str | None = None) -> Path:
    return Path(explicit).expanduser().resolve() if explicit else root / "workspaces" / title


def _emit(value: Any, args: argparse.Namespace) -> None:
    text: str
    if args.json:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    try:
        print(text)
    except UnicodeEncodeError:
        # Windows legacy terminals can still select cp949 even though the
        # artifacts are UTF-8.  A reporting failure must not turn a completed
        # ASR run into a failed process or corrupt the source CSV.
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is None:
            raise
        buffer.write((text + "\n").encode("utf-8"))
        buffer.flush()


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, help="translation-forensics 프로젝트 루트")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="JSON으로 결과 출력")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))


def _add_title(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument("--title", required=required)
    parser.add_argument("--workspace", type=Path)


def _input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--structure", type=Path)
    parser.add_argument("--ja", type=Path)
    parser.add_argument("--previous-ko", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--photos", type=Path)
    parser.add_argument("--review-queue", type=Path)
    parser.add_argument("--asr-candidates", type=Path)
    parser.add_argument("--work-audio", type=Path)


def _resolve_inputs(root: Path, title: str, args: argparse.Namespace) -> dict[str, Path | None]:
    explicit = {
        "structure": args.structure,
        "ja": args.ja,
        "previous_ko": args.previous_ko,
        "audio": args.audio,
        "video": getattr(args, "video", None),
        "photos": args.photos,
        "review_queue": args.review_queue,
        "asr_candidates": args.asr_candidates,
        "work_audio": args.work_audio,
    }
    return {role: resolve_role(root, role, path, title, required=role in {"structure", "ja"}) for role, path in explicit.items()}


def _manifest_path(workspace: Path) -> Path:
    return workspace / "metadata" / "project-manifest.json"


def _timeline_report_path(workspace: Path, title: str) -> Path:
    return workspace / "metadata" / f"{title}.timeline-validation-v1.json"


def _manifest_role_path(workspace: Path, role: str) -> Path | None:
    manifest_path = _manifest_path(workspace)
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for record in manifest.get("inputs", []):
        if isinstance(record, dict) and record.get("role") == role:
            path = Path(str(record.get("path", ""))).expanduser()
            if path.exists():
                return path.resolve()
    return None


def _closed_world_structure_from_package(package_dir: Path) -> Path:
    manifest_path = package_dir / "run-manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"폐쇄형 run manifest가 없습니다: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest.get("inputs", []):
        if isinstance(record, dict) and record.get("role") == "structure":
            path = Path(str(record.get("path", "")))
            if path.exists():
                return path.resolve()
    raise ValueError("폐쇄형 run manifest에서 사용 가능한 structure 입력을 찾지 못했습니다.")


def _discover_closed_world_candidates(workspace: Path) -> list[Path]:
    intermediate = workspace / "intermediate"
    patterns = (
        "*.gpt56-direct*.source-faithful.srt",
        "*.gpt56-direct*.viewer-natural.srt",
        "*.machine-assisted*.srt",
        "*.ko-aligned-draft*.srt",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path.resolve() for path in intermediate.glob(pattern) if path.is_file())
    return sorted(paths, key=lambda path: str(path).lower())


def cmd_doctor(args: argparse.Namespace) -> int:
    root = _project_root(args)
    checks: dict[str, dict[str, Any]] = {}
    checks["python"] = {"required": True, "ok": sys.version_info >= (3, 10), "value": sys.version.split()[0]}
    for name in ("numpy", "pandas", "scipy", "sklearn", "joblib", "faster_whisper", "pytest"):
        checks[f"package:{name}"] = {"required": name in {"pytest"}, "ok": importlib.util.find_spec(name) is not None}
    checks["ffmpeg"] = {"required": False, "ok": shutil.which("ffmpeg") is not None, "value": shutil.which("ffmpeg") or "missing"}
    checks["cuda-tool"] = {"required": False, "ok": shutil.which("nvidia-smi") is not None, "value": shutil.which("nvidia-smi") or "missing"}
    references = [root / "references" / name for name in ("translation-prompt-v6.txt", "subtitle-audio-crosscheck-method-v1.md", "subtitle-forensics-io-schema-v1.md")]
    checks["reference-documents"] = {"required": True, "ok": all(path.exists() for path in references), "files": [str(path) for path in references]}
    vendor = root / "vendor"
    checks["vendor-forensics"] = {"required": True, "ok": (vendor / "subtitle_forensics_v1" / "subtitle_forensics.py").exists()}
    checks["vendor-audio-runner"] = {"required": False, "ok": (vendor / "subtitle_audio_forensics_runner_v1" / "run_asr_adaptive.py").exists()}
    checks["model-file"] = {"required": False, "ok": (vendor / "subtitle_forensics_v1" / "model" / "risk_model.joblib").exists()}
    checks["write-permission"] = {"required": True, "ok": False}
    if not args.dry_run:
        try:
            root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=".doctor-", dir=root, delete=True):
                pass
            checks["write-permission"]["ok"] = True
        except OSError as exc:
            checks["write-permission"]["error"] = str(exc)
    else:
        checks["write-permission"]["ok"] = True
    checks["vendor-self-test"] = {"required": False, "ok": False, "status": "not-run"}
    if not args.dry_run:
        test_script = vendor / "subtitle_forensics_v1" / "self_test.py"
        if test_script.exists():
            completed = subprocess.run([sys.executable, str(test_script)], cwd=str(test_script.parent), capture_output=True, text=True, timeout=180, check=False)
            checks["vendor-self-test"] = {"required": False, "ok": completed.returncode == 0, "returncode": completed.returncode, "output_tail": completed.stdout[-1000:]}
    required_failures = [name for name, value in checks.items() if value.get("required") and not value.get("ok")]
    result = {"project_root": str(root), "version": __version__, "checks": checks, "status": "pass" if not required_failures else "fail", "required_failures": required_failures}
    _emit(result, args)
    return 0 if not required_failures else 1


def cmd_init_title(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    planned = {"workspace": str(workspace), "directories": ["inputs", "intermediate", "audio", "final", "metadata", "logs"]}
    if args.dry_run:
        _emit({"status": "dry-run", **planned}, args); return 0
    if _manifest_path(workspace).exists():
        _emit({"status": "exists", "manifest": str(_manifest_path(workspace)), "note": "기존 매니페스트를 덮어쓰지 않았습니다."}, args); return 2
    for directory in planned["directories"]:
        (workspace / directory).mkdir(parents=True, exist_ok=True)
    manifest = build_project_manifest(args.title, workspace, {}, unresolved_roles=["structure", "ja", "previous_ko", "audio", "photos", "review_queue", "asr_candidates"], validation_status="미검증")
    write_json(_manifest_path(workspace), manifest)
    _emit({"status": "created", **planned, "manifest": str(_manifest_path(workspace))}, args)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        inputs = _resolve_inputs(input_root, args.title, args)
    except DiscoveryError as exc:
        result = {"status": "fail", "error": str(exc), "inspection": inspect_roles(input_root, args.title)}
        _emit(result, args); return 2
    if args.dry_run:
        _emit({"status": "dry-run", "inputs": {key: str(value) if value else None for key, value in inputs.items()}}, args); return 0
    structure_diff: dict[str, Any] = {"status": "미검증"}
    if inputs["structure"] and inputs["previous_ko"]:
        try:
            structure_blocks, _, _ = parse_srt(inputs["structure"])
            previous_blocks, _, _ = parse_srt(inputs["previous_ko"])
            structure_diff = {
                "status": "checked",
                "reference": str(inputs["structure"]),
                "candidate": str(inputs["previous_ko"]),
                "diff": compare_structure(structure_blocks, previous_blocks),
                "alignment_by_time_overlap": align_to_dicts(structure_blocks, previous_blocks),
            }
            write_json(workspace / "metadata" / "alignment-report.json", structure_diff)
        except (OSError, ValueError) as exc:
            structure_diff = {"status": "failed", "error": str(exc)}
    unresolved = [role for role, value in inputs.items() if value is None]
    manifest = build_project_manifest(args.title, workspace, inputs, structure_diff=structure_diff, unresolved_roles=unresolved)
    write_json(_manifest_path(workspace), manifest)
    result = {"status": "inspected", "workspace": str(workspace), "inputs": {key: str(value) if value else None for key, value in inputs.items()}, "unresolved_roles": unresolved, "manifest": str(_manifest_path(workspace))}
    _emit(result, args); return 0


def cmd_validate_timeline(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        structure = resolve_role(input_root, "structure", args.structure, args.title, required=True)
        media = resolve_role(input_root, "audio", args.audio, args.title, required=False)
        if media is None:
            media = resolve_role(input_root, "video", args.video, args.title, required=False)
    except DiscoveryError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    try:
        anchors = read_timeline_anchors(args.anchors.expanduser().resolve()) if args.anchors else []
        report = validate_timeline(structure, media, tolerance_seconds=args.tolerance, anchors=anchors, approved_offset_map=args.approved_offset_map.expanduser().resolve() if args.approved_offset_map else None)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    if args.dry_run:
        _emit({"status": "dry-run", "report": report}, args); return 0
    output = args.output or _timeline_report_path(workspace, args.title)
    try:
        if output.exists():
            raise FileExistsError(f"기존 시간축 보고서를 덮어쓰지 않습니다: {output}")
        write_json(output, report)
        anchor_output = None
        if anchors:
            anchor_output = workspace / "metadata" / f"{args.title}.timeline-anchors-v1.csv"
            if anchor_output.exists():
                raise FileExistsError(f"기존 timeline anchors를 덮어쓰지 않습니다: {anchor_output}")
            write_csv(anchor_output, anchors, ["anchor_id", "srt_time_seconds", "media_time_seconds", "source", "offset_seconds"])
        offset_output = None
        if args.approved_offset_map:
            offset_output = workspace / "metadata" / f"{args.title}.approved-offset-map-v1.json"
            if offset_output.exists():
                raise FileExistsError(f"기존 approved offset map을 덮어쓰지 않습니다: {offset_output}")
            write_json(offset_output, report.get("offset_map"))
        manifest_path = _manifest_path(workspace)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["timeline_validation"] = {
                "status": report["status"],
                "clip_preparation_allowed": report["clip_preparation_allowed"],
                "report": str(output),
                "reason": report["reason"],
            }
            write_json(manifest_path, manifest)
    except (OSError, FileExistsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit({**report, "output": str(output), "anchors_output": str(anchor_output) if anchor_output else None, "approved_offset_map_output": str(offset_output) if offset_output else None}, args)
    return 0 if timeline_is_usable(report) else 1


def cmd_init_timeline_anchors(args: argparse.Namespace) -> int:
    root = _project_root(args); workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        structure = resolve_role(input_root, "structure", args.structure, args.title, required=True)
        output = args.output or workspace / "metadata" / f"{args.title}.timeline-anchors.template-v1.csv"
        result = initialize_timeline_anchor_template(structure, output)
    except (OSError, ValueError, FileExistsError, DiscoveryError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_build_pilot_alignment(args: argparse.Namespace) -> int:
    try:
        report = evaluate_pilot_alignment(
            args.title,
            args.structure.expanduser().resolve(),
            args.audio.expanduser().resolve(),
            args.anchors.expanduser().resolve(),
            args.offset_map.expanduser().resolve(),
            expected_blocks=args.expected_blocks,
            tolerance_seconds=args.tolerance,
        )
        if not args.dry_run:
            write_pilot_alignment(args.output.expanduser().resolve(), report)
    except (OSError, ValueError, json.JSONDecodeError, FileExistsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(report, args)
    return 0 if report["status"] == "resolved" else 1


def cmd_build_pilot_audio_review(args: argparse.Namespace) -> int:
    try:
        if args.dry_run:
            alignment = json.loads(args.alignment.expanduser().resolve().read_text(encoding="utf-8"))
            result = {
                "status": "dry-run" if alignment.get("status") == "resolved" and alignment.get("clip_preparation_allowed") is True else "blocked",
                "alignment_status": alignment.get("status"),
                "expected_block_count": args.expected_blocks,
                "output": str(args.output.expanduser().resolve()),
            }
        else:
            result = build_pilot_audio_review_packet(
                args.title,
                args.structure.expanduser().resolve(),
                args.ja.expanduser().resolve(),
                args.audio.expanduser().resolve(),
                args.alignment.expanduser().resolve(),
                args.output.expanduser().resolve(),
                expected_blocks=args.expected_blocks,
                context_before_seconds=args.context_before,
                context_after_seconds=args.context_after,
                context_radius=args.context_radius,
            )
    except PilotAlignmentError as exc:
        try:
            blocked = write_blocked_audio_review_manifest(
                args.title,
                args.alignment.expanduser().resolve(),
                args.output.expanduser().resolve(),
                expected_blocks=args.expected_blocks,
                reason=str(exc),
            )
        except (OSError, FileExistsError) as write_exc:
            _emit({"status": "blocked", "error": str(exc), "manifest_error": str(write_exc)}, args); return 2
        _emit(blocked, args); return 2
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, FileExistsError) as exc:
        _emit({"status": "blocked", "error": str(exc)}, args); return 2
    _emit(result, args)
    return 0 if result.get("status") in {"ready", "dry-run"} else 2


def cmd_build_pilot_candidate(args: argparse.Namespace) -> int:
    try:
        if args.dry_run:
            result = {
                "status": "dry-run",
                "title_id": args.title,
                "expected_block_count": args.expected_blocks,
                "pipeline_order": [
                    "terra-translation-decision",
                    "sol-independent-critique-repair",
                    "semantic-recheck",
                ],
                "output": str(args.output.expanduser().resolve()),
            }
        else:
            result = build_pilot_candidate(
                title_id=args.title,
                structure_path=args.structure.expanduser().resolve(),
                baseline_path=args.baseline.expanduser().resolve(),
                terra_decisions_path=args.terra_decisions.expanduser().resolve(),
                sol_reviews_path=args.sol_reviews.expanduser().resolve(),
                terra_prompt_path=args.terra_prompt.expanduser().resolve(),
                sol_prompt_path=args.sol_prompt.expanduser().resolve(),
                provenance_schema_path=args.provenance_schema.expanduser().resolve(),
                output_dir=args.output.expanduser().resolve(),
                project_root=_project_root(args),
                expected_blocks=args.expected_blocks,
                expected_baseline_sha256=args.expected_baseline_sha256,
            )
    except (OSError, ValueError, json.JSONDecodeError, FileExistsError, PilotCandidateError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args)
    return 0


def cmd_record_pilot_sol_reviews(args: argparse.Namespace) -> int:
    try:
        if args.dry_run:
            result = {
                "status": "dry-run",
                "title_id": args.title,
                "reviewer_model": "gpt-5.6-sol",
                "expected_block_count": args.expected_blocks,
                "output": str(args.output.expanduser().resolve()),
            }
        else:
            result = write_pilot_sol_reviews(
                terra_decisions_path=args.terra_decisions.expanduser().resolve(),
                review_plan_path=args.review_plan.expanduser().resolve(),
                output_path=args.output.expanduser().resolve(),
                expected_blocks=args.expected_blocks,
            )
    except (OSError, ValueError, json.JSONDecodeError, FileExistsError, PilotCandidateError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args)
    return 0


def cmd_validate_pilot_candidate(args: argparse.Namespace) -> int:
    try:
        result = validate_pilot_candidate(
            output_dir=args.candidate.expanduser().resolve(),
            structure_path=args.structure.expanduser().resolve(),
            baseline_path=args.baseline.expanduser().resolve(),
            provenance_schema_path=args.provenance_schema.expanduser().resolve(),
            project_root=_project_root(args),
            expected_blocks=args.expected_blocks,
            expected_baseline_sha256=args.expected_baseline_sha256,
        )
    except (OSError, ValueError, json.JSONDecodeError, PilotCandidateError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args)
    return 0 if result.get("status") == "pass" else 2


def cmd_validate_pilot_automated(args: argparse.Namespace) -> int:
    try:
        result = validate_pilot_automated(
            title_id=args.title,
            structure_path=args.structure,
            baseline_path=args.baseline,
            candidate_dir=args.candidate,
            provenance_schema_path=args.provenance_schema,
            terra_manifest_path=args.terra_manifest,
            autonomous_manifest_path=args.autonomous_manifest,
            regression_suite_path=args.regressions,
            report_schema_path=args.report_schema,
            project_root=_project_root(args),
            expected_blocks=args.expected_blocks,
            expected_baseline_sha256=args.expected_baseline_sha256,
        )
        if not args.dry_run:
            write_pilot_automated_validation(args.output.expanduser().resolve(), result)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        FileExistsError,
        PilotAutomatedValidationError,
    ) as exc:
        _emit({"status": "fail", "error": str(exc)}, args)
        return 2
    _emit(result, args)
    return 0 if result.get("status") == "pass" else 2


def cmd_init_pilot_blind_key(args: argparse.Namespace) -> int:
    try:
        if args.dry_run:
            result = {
                "status": "dry-run",
                "title_id": args.title,
                "expected_block_count": args.expected_blocks,
                "output": str(args.output.expanduser().resolve()),
                "candidate_must_not_exist": str(args.candidate_expected.expanduser().resolve()),
            }
        else:
            result = initialize_pilot_blind_internal_key(
                title_id=args.title,
                baseline_path=args.baseline.expanduser().resolve(),
                candidate_expected_path=args.candidate_expected.expanduser().resolve(),
                evaluation_contract_path=args.evaluation_contract.expanduser().resolve(),
                mqm_schema_path=args.mqm_schema.expanduser().resolve(),
                output_path=args.output.expanduser().resolve(),
                project_root=_project_root(args),
                random_seed=args.random_seed,
                expected_blocks=args.expected_blocks,
                expected_baseline_sha256=args.expected_baseline_sha256,
            )
    except (OSError, ValueError, json.JSONDecodeError, FileExistsError, PilotBlindReviewError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args)
    return 0


def cmd_build_pilot_blind_review(args: argparse.Namespace) -> int:
    try:
        if args.dry_run:
            audio_manifest = json.loads(args.audio_review_manifest.expanduser().resolve().read_text(encoding="utf-8"))
            ready = audio_manifest.get("status") == "ready" and audio_manifest.get("clip_preparation_allowed") is True
            result = {
                "status": "dry-run" if ready else "blocked",
                "title_id": args.title,
                "expected_block_count": args.expected_blocks,
                "output": str(args.output.expanduser().resolve()),
                "audio_review_status": audio_manifest.get("status"),
            }
        else:
            audio_manifest = json.loads(args.audio_review_manifest.expanduser().resolve().read_text(encoding="utf-8"))
            ready = (
                audio_manifest.get("status") == "ready"
                and audio_manifest.get("clip_preparation_allowed") is True
                and audio_manifest.get("block_count") == args.expected_blocks
            )
            if ready:
                result = build_pilot_blind_review_packets(
                    title_id=args.title,
                    structure_path=args.structure.expanduser().resolve(),
                    baseline_path=args.baseline.expanduser().resolve(),
                    candidate_path=args.candidate.expanduser().resolve(),
                    candidate_provenance_path=args.candidate_provenance.expanduser().resolve(),
                    audio_manifest_path=args.audio_review_manifest.expanduser().resolve(),
                    internal_key_path=args.internal_key.expanduser().resolve(),
                    output_dir=args.output.expanduser().resolve(),
                    expected_blocks=args.expected_blocks,
                    expected_baseline_sha256=args.expected_baseline_sha256,
                )
            else:
                reason = (
                    "timeline/audio review gate is not ready: "
                    f"status={audio_manifest.get('status')}, "
                    f"clip_preparation_allowed={audio_manifest.get('clip_preparation_allowed')}, "
                    f"block_count={audio_manifest.get('block_count')}/{args.expected_blocks}"
                )
                result = write_blocked_pilot_review_artifacts(
                    title_id=args.title,
                    structure_path=args.structure.expanduser().resolve(),
                    baseline_path=args.baseline.expanduser().resolve(),
                    candidate_path=args.candidate.expanduser().resolve(),
                    candidate_provenance_path=args.candidate_provenance.expanduser().resolve(),
                    audio_manifest_path=args.audio_review_manifest.expanduser().resolve(),
                    evaluation_contract_path=args.evaluation_contract.expanduser().resolve(),
                    mqm_schema_path=args.mqm_schema.expanduser().resolve(),
                    blind_review_dir=args.output.expanduser().resolve(),
                    internal_key_path=args.internal_key.expanduser().resolve(),
                    adjudication_output_path=args.adjudication_output.expanduser().resolve(),
                    project_root=_project_root(args),
                    random_seed=args.random_seed,
                    reason=reason,
                    expected_blocks=args.expected_blocks,
                    expected_baseline_sha256=args.expected_baseline_sha256,
                )
    except (OSError, ValueError, json.JSONDecodeError, FileExistsError, PilotBlindReviewError, PilotBlindReviewBlocked) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args)
    return 0


def cmd_validate_pilot_reviewer_submission(args: argparse.Namespace) -> int:
    try:
        result = validate_pilot_reviewer_submission(
            pack_path=args.pack.expanduser().resolve(),
            attestation_path=args.attestation.expanduser().resolve(),
            decisions_path=args.decisions.expanduser().resolve(),
            expected_blocks=args.expected_blocks,
        )
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile, PilotBlindReviewError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    public = {key: value for key, value in result.items() if key not in {"attestation", "decisions"}}
    _emit(public, args)
    return 0


def cmd_adjudicate_pilot_reviews(args: argparse.Namespace) -> int:
    try:
        result = adjudicate_pilot_reviews(
            reviewer_1_pack_path=args.reviewer_1_pack.expanduser().resolve(),
            reviewer_1_attestation_path=args.reviewer_1_attestation.expanduser().resolve(),
            reviewer_1_decisions_path=args.reviewer_1_decisions.expanduser().resolve(),
            reviewer_2_pack_path=args.reviewer_2_pack.expanduser().resolve(),
            reviewer_2_attestation_path=args.reviewer_2_attestation.expanduser().resolve(),
            reviewer_2_decisions_path=args.reviewer_2_decisions.expanduser().resolve(),
            internal_key_path=args.internal_key.expanduser().resolve(),
            consensus_path=args.consensus.expanduser().resolve(),
            output_path=args.output.expanduser().resolve(),
            expected_blocks=args.expected_blocks,
            expected_baseline_sha256=args.expected_baseline_sha256,
        )
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile, FileExistsError, PilotBlindReviewError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args)
    return 0 if result.get("adjudication_complete") is True else 1


def cmd_evaluate_pilot_new_critical(args: argparse.Namespace) -> int:
    try:
        if args.dry_run:
            result = {
                "status": "dry-run",
                "adjudication": str(args.adjudication.expanduser().resolve()),
                "expected_block_count": args.expected_blocks,
                "output": str(args.output.expanduser().resolve()),
            }
        else:
            result = evaluate_new_critical_semantic_errors_file(
                args.adjudication.expanduser().resolve(),
                expected_blocks=args.expected_blocks,
            )
            write_new_critical_semantic_error_result(args.output.expanduser().resolve(), result)
    except (OSError, ValueError, json.JSONDecodeError, FileExistsError, PilotMetricsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args)
        return 2
    _emit(result, args)
    return 0 if result.get("gate_passed") is True or result.get("status") == "dry-run" else 1


def cmd_evaluate_pilot_error_reduction(args: argparse.Namespace) -> int:
    try:
        if args.dry_run:
            result = {
                "status": "dry-run",
                "adjudication": str(args.adjudication.expanduser().resolve()),
                "expected_block_count": args.expected_blocks,
                "minimum_reduction_percent": MINIMUM_ERROR_BLOCK_REDUCTION_PERCENT,
                "output": str(args.output.expanduser().resolve()),
            }
        else:
            result = evaluate_critical_major_error_block_reduction_file(
                args.adjudication.expanduser().resolve(),
                expected_blocks=args.expected_blocks,
            )
            write_error_block_reduction_result(args.output.expanduser().resolve(), result)
    except (OSError, ValueError, json.JSONDecodeError, FileExistsError, PilotMetricsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args)
        return 2
    _emit(result, args)
    return 0 if result.get("gate_passed") is True or result.get("status") == "dry-run" else 1


def cmd_evaluate_pilot_naturalness(args: argparse.Namespace) -> int:
    try:
        if args.dry_run:
            result = {
                "status": "dry-run",
                "adjudication": str(args.adjudication.expanduser().resolve()),
                "expected_block_count": args.expected_blocks,
                "minimum_improvement_win_percent": MINIMUM_NATURALNESS_WIN_PERCENT,
                "maximum_improvement_loss_percent": MAXIMUM_NATURALNESS_LOSS_PERCENT,
                "output": str(args.output.expanduser().resolve()),
            }
        else:
            result = evaluate_naturalness_comparison_file(
                args.adjudication.expanduser().resolve(),
                expected_blocks=args.expected_blocks,
            )
            write_naturalness_comparison_result(args.output.expanduser().resolve(), result)
    except (OSError, ValueError, json.JSONDecodeError, FileExistsError, PilotMetricsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args)
        return 2
    _emit(result, args)
    return 0 if result.get("gate_passed") is True or result.get("status") == "dry-run" else 1


def cmd_evaluate_pilot_balance_contract(args: argparse.Namespace) -> int:
    try:
        if args.dry_run:
            result = {
                "status": "dry-run",
                "adjudication": str(args.adjudication.expanduser().resolve()),
                "expected_block_count": args.expected_blocks,
                "metrics_output": str(args.metrics_output.expanduser().resolve()),
                "error_ledger_output": str(args.error_ledger_output.expanduser().resolve()),
            }
        else:
            result, error_ledger = evaluate_pilot_balance_contract_file(
                args.adjudication.expanduser().resolve(),
                expected_blocks=args.expected_blocks,
            )
            write_pilot_balance_contract_results(
                args.metrics_output.expanduser().resolve(),
                args.error_ledger_output.expanduser().resolve(),
                result,
                error_ledger,
            )
    except (OSError, ValueError, json.JSONDecodeError, FileExistsError, PilotMetricsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args)
        return 2
    _emit(result, args)
    return 0 if result.get("all_required_gates_passed") is True or result.get("status") == "dry-run" else 1


def cmd_build_pilot_evaluation_report(args: argparse.Namespace) -> int:
    root = _project_root(args)
    try:
        manifest, report = build_pilot_evaluation_report(
            project_root=root,
            baseline_path=args.baseline.expanduser().resolve(),
            candidate_source_path=args.candidate_source.expanduser().resolve(),
            candidate_viewer_path=args.candidate_viewer.expanduser().resolve(),
            candidate_provenance_path=args.candidate_provenance.expanduser().resolve(),
            timeline_alignment_path=args.timeline_alignment.expanduser().resolve(),
            audio_review_manifest_path=args.audio_review_manifest.expanduser().resolve(),
            blind_review_packet_paths=[path.expanduser().resolve() for path in args.blind_review_packet],
            internal_key_path=args.internal_key.expanduser().resolve(),
            reviewer_attestation_paths=[path.expanduser().resolve() for path in args.reviewer_attestation],
            reviewer_decision_paths=[path.expanduser().resolve() for path in args.reviewer_decisions],
            adjudication_path=args.adjudication.expanduser().resolve(),
            metrics_path=args.metrics.expanduser().resolve(),
            error_ledger_path=args.error_ledger.expanduser().resolve(),
            automated_validation_path=args.automated_validation.expanduser().resolve(),
            report_output_path=args.report_output.expanduser().resolve(),
            expected_blocks=args.expected_blocks,
            expected_baseline_sha256=args.expected_baseline_sha256,
        )
        if not args.dry_run:
            write_pilot_evaluation_report(
                args.manifest_output.expanduser().resolve(),
                args.report_output.expanduser().resolve(),
                manifest,
                report,
                schema_path=Path(__file__).resolve().parents[2]
                / "schemas"
                / "pilot-evaluation-manifest.schema.json",
            )
    except (OSError, ValueError, json.JSONDecodeError, FileExistsError, PilotReportError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args)
        return 2
    result = {
        "status": "dry-run" if args.dry_run else manifest["status"],
        "result": manifest["result"],
        "lifecycle_status": manifest["lifecycle_status"],
        "manifest": str(args.manifest_output.expanduser().resolve()),
        "report": str(args.report_output.expanduser().resolve()),
    }
    _emit(result, args)
    return 0


def _default_queue(workspace: Path, title: str) -> Path:
    base = workspace / "intermediate" / f"{title}.review-queue.structure-fallback-v1.csv"
    if not base.exists():
        return base
    index = 2
    while True:
        candidate = workspace / "intermediate" / f"{title}.review-queue.structure-fallback-v{index}.csv"
        if not candidate.exists():
            return candidate
        index += 1


def cmd_analyze(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        ja = resolve_role(input_root, "ja", args.ja, args.title, required=True)
        previous = resolve_role(input_root, "previous_ko", args.previous_ko, args.title)
    except DiscoveryError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    queue = args.output or _default_queue(workspace, args.title)
    if queue.exists() and not args.dry_run:
        _emit({"status": "fail", "error": f"기존 review queue를 덮어쓰지 않습니다: {queue}", "suggested_output": str(_default_queue(workspace, args.title))}, args); return 2
    vendor_root = args.vendor_root.expanduser().resolve() if args.vendor_root else None
    result = analyze_title(ja, previous, queue, vendor_root=vendor_root, vendor_output=workspace / "intermediate" / "vendor-forensics", dry_run=args.dry_run)
    _emit({"title": args.title, "queue": str(queue), **result}, args); return 0 if result.get("status") != "failed" else 1


def cmd_build_korean_draft(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        ja = resolve_role(input_root, "ja", args.ja, args.title, required=True)
        previous = resolve_role(input_root, "previous_ko", args.previous_ko, args.title, required=True)
    except DiscoveryError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    output = args.output or workspace / "intermediate" / f"{args.title}.ko-aligned-draft-v1.srt"
    report = args.report or workspace / "intermediate" / f"{args.title}.ko-aligned-draft-v1.report.json"
    if args.dry_run:
        _emit({"status": "dry-run", "ja": str(ja), "previous_ko": str(previous), "output": str(output), "report": str(report)}, args); return 0
    if output.exists() or report.exists():
        _emit({"status": "fail", "error": "기존 번역 초안을 덮어쓰지 않습니다.", "output": str(output), "report": str(report)}, args); return 2
    try:
        result = build_korean_aligned_draft(ja, previous, output, report)
    except (OSError, ValueError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_build_automatic_draft(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        structure = resolve_role(input_root, "structure", args.structure, args.title, required=True)
    except DiscoveryError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    output = args.output or workspace / "automatic-draft-v1"
    paths = {
        "source_candidate": args.source_candidate,
        "viewer_candidate": args.viewer_candidate,
        "single_candidate": args.single_candidate,
        "fallback": args.fallback,
        "decision_candidate": args.decision_candidate,
    }
    resolved = {name: path.expanduser().resolve() for name, path in paths.items() if path is not None}
    missing = [name for name, path in resolved.items() if not path.exists()]
    if missing:
        _emit({"status": "fail", "error": f"자동 초안 후보 파일이 없습니다: {', '.join(missing)}"}, args); return 2
    if args.dry_run:
        _emit({"status": "dry-run", "structure": str(structure), "output": str(output), "candidates": {name: str(path) for name, path in resolved.items()}}, args); return 0
    try:
        result = build_automatic_draft(
            title=args.title,
            structure_path=structure,
            output_dir=output,
            source_candidate_path=resolved.get("source_candidate"),
            viewer_candidate_path=resolved.get("viewer_candidate"),
            single_candidate_path=resolved.get("single_candidate"),
            fallback_path=resolved.get("fallback"),
            decision_candidate_path=resolved.get("decision_candidate"),
        )
    except (FileExistsError, OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["status"] == "automatic-draft-complete" else 1


def cmd_package_machine_final(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        structure = resolve_role(input_root, "structure", args.structure, args.title, required=True)
    except DiscoveryError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    output = args.output or workspace / "machine-final-v1"
    required = {
        "source_faithful": args.source_faithful,
        "viewer_natural": args.viewer_natural,
        "automatic_report": args.automatic_report,
        "automatic_ledger": args.automatic_ledger,
        "asr": args.asr,
        "timeline_validation": args.timeline_validation,
        "closed_world_proof": args.closed_world_proof,
    }
    missing = [name for name, path in required.items() if path is None]
    if missing:
        _emit({"status": "fail", "error": f"machine-final 필수 입력이 없습니다: {', '.join(missing)}"}, args); return 2
    resolved = {name: path.expanduser().resolve() for name, path in required.items()}
    absent = [name for name, path in resolved.items() if not path.exists()]
    if absent:
        _emit({"status": "fail", "error": f"machine-final 입력 파일이 없습니다: {', '.join(absent)}"}, args); return 2
    if args.dry_run:
        _emit({"status": "dry-run", "output": str(output), "inputs": {name: str(path) for name, path in resolved.items()}}, args); return 0
    try:
        result = package_machine_final(
            title=args.title,
            structure_path=structure,
            source_path=resolved["source_faithful"],
            viewer_path=resolved["viewer_natural"],
            automatic_report_path=resolved["automatic_report"],
            automatic_ledger_path=resolved["automatic_ledger"],
            asr_path=resolved["asr"],
            timeline_path=resolved["timeline_validation"],
            proof_path=resolved["closed_world_proof"],
            output_dir=output,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_repair_machine_final_asr(args: argparse.Namespace) -> int:
    if args.dry_run:
        _emit({"status": "dry-run", "source_asr": str(args.source_asr), "package": str(args.package)}, args); return 0
    try:
        result = repair_machine_final_asr(source_asr_path=args.source_asr, package_dir=args.package)
    except (FileExistsError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_apply_targeted_retranslations(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        structure = resolve_role(input_root, "structure", args.structure, args.title, required=True)
    except DiscoveryError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    source = args.source_faithful.expanduser().resolve()
    viewer = args.viewer_natural.expanduser().resolve()
    responses = [path.expanduser().resolve() for path in args.response]
    reviews = [path.expanduser().resolve() for path in (args.review or [])]
    output = args.output or workspace / "targeted-retranslation-v2"
    if args.dry_run:
        _emit({"status": "dry-run", "structure": str(structure), "source": str(source), "viewer": str(viewer), "responses": [str(path) for path in responses], "reviews": [str(path) for path in reviews], "version": args.version, "output": str(output)}, args); return 0
    try:
        result = apply_targeted_retranslations(title=args.title, structure_path=structure, source_path=source, viewer_path=viewer, response_paths=responses, output_dir=output, hold_marker=args.hold_marker, review_paths=reviews, version=args.version)
    except (FileExistsError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_prepare_inference_audio(args: argparse.Namespace) -> int:
    """Cut one local ASR clip per v3 hold marker, without changing subtitles."""
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    try:
        structure = args.structure.expanduser().resolve() if args.structure else _manifest_role_path(workspace, "structure")
        audio = args.audio.expanduser().resolve() if args.audio else _manifest_role_path(workspace, "audio")
        if structure is None or audio is None:
            raise DiscoveryError("프로젝트 매니페스트에서 structure 또는 audio 입력을 찾지 못했습니다.")
    except DiscoveryError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    hold_ledger = args.hold_ledger.expanduser().resolve()
    output = (args.output or workspace / "inferred-recovery-audio-v4").expanduser().resolve()
    queue = output.parent / f"{args.title}.inferred-recovery-v4.queue.csv"
    if args.dry_run:
        _emit({"status": "dry-run", "structure": str(structure), "audio": str(audio), "hold_ledger": str(hold_ledger), "queue": str(queue), "output": str(output), "padding": args.padding, "merge_gap": args.merge_gap}, args); return 0
    if output.exists() or queue.exists():
        _emit({"status": "fail", "error": "기존 추론 복구용 음성 작업물을 덮어쓰지 않습니다.", "queue": str(queue), "output": str(output)}, args); return 2
    try:
        queue_result = build_inference_audio_queue(title=args.title, structure_path=structure, hold_ledger_path=hold_ledger, output_path=queue)
        audio_result = prepare_audio(root, queue, audio, output, bands="P1", padding=args.padding, merge_gap=args.merge_gap, max_scene=args.max_scene, include_audio=True, force=False)
        if audio_result.get("status") == "failed":
            _emit({"status": "fail", "queue": queue_result, "audio": audio_result}, args); return 1
        write_json(output / "inference-audio-preparation.json", {
            "schema_name": "translation-forensics/inferred-recovery-audio-preparation",
            "schema_version": "1",
            "title_id": args.title,
            "queue": str(queue),
            "hold_ledger": str(hold_ledger),
            "padding_seconds": args.padding,
            "merge_gap_seconds": args.merge_gap,
            "max_scene_seconds": args.max_scene,
            "audio_result": audio_result,
            "human_reviewed": False,
            "final_promotion_allowed": False,
        })
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit({"status": "inference-audio-prepared", "queue": queue_result, "audio": audio_result, "output": str(output), "human_reviewed": False, "final_promotion_allowed": False}, args); return 0


def cmd_build_inference_context(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    try:
        structure = args.structure.expanduser().resolve() if args.structure else _manifest_role_path(workspace, "structure")
        if structure is None:
            raise DiscoveryError("프로젝트 매니페스트에서 structure 입력을 찾지 못했습니다.")
        result = build_inference_context(
            title=args.title,
            structure_path=structure,
            source_path=args.source_faithful.expanduser().resolve(),
            viewer_path=args.viewer_natural.expanduser().resolve(),
            hold_ledger_path=args.hold_ledger.expanduser().resolve(),
            scenes_path=args.scenes.expanduser().resolve(),
            asr_path=args.asr.expanduser().resolve(),
            output_path=args.output.expanduser().resolve(),
        )
    except (DiscoveryError, FileExistsError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_apply_inferred_recovery(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    try:
        structure = args.structure.expanduser().resolve() if args.structure else _manifest_role_path(workspace, "structure")
        if structure is None:
            raise DiscoveryError("프로젝트 매니페스트에서 structure 입력을 찾지 못했습니다.")
        result = apply_inferred_recovery(
            title=args.title,
            structure_path=structure,
            source_path=args.source_faithful.expanduser().resolve(),
            viewer_path=args.viewer_natural.expanduser().resolve(),
            hold_ledger_path=args.hold_ledger.expanduser().resolve(),
            response_paths=[path.expanduser().resolve() for path in args.response],
            output_dir=(args.output or workspace / "inferred-recovery-v4").expanduser().resolve(),
            hold_marker=args.hold_marker,
            version=args.version,
        )
    except (DiscoveryError, FileExistsError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def _autonomous_titles(args: argparse.Namespace) -> list[str]:
    values = [str(value).strip() for value in (args.title or []) if str(value).strip()]
    if args.titles_file:
        for line in args.titles_file.expanduser().resolve().read_text(encoding="utf-8-sig").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                values.append(value)
    titles = list(dict.fromkeys(values))
    if not titles:
        raise ValueError("--title 또는 --titles-file로 작품을 하나 이상 지정해야 합니다.")
    return titles


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path.resolve() for path in paths if path.exists()), None)


def cmd_audit_source_quality(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, None)
    japanese = args.ja.expanduser().resolve() if args.ja else _manifest_role_path(workspace, "ja")
    if japanese is None:
        _emit({"status": "fail", "error": f"{args.title}: Japanese SRT를 찾을 수 없습니다."}, args)
        return 2
    output = (args.output or workspace / "autonomous-quality-v1" / "source-quality").expanduser().resolve()
    if args.dry_run:
        _emit(
            {
                "status": "dry-run",
                "title_id": args.title,
                "japanese": str(japanese),
                "asr_evidence": str(args.asr_evidence.expanduser().resolve()) if args.asr_evidence else None,
                "output": str(output),
            },
            args,
        )
        return 0
    try:
        result = write_source_quality_audit(
            title_id=args.title,
            japanese_path=japanese,
            asr_evidence_path=args.asr_evidence.expanduser().resolve() if args.asr_evidence else None,
            output_dir=output,
            resume=args.resume,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args)
        return 2
    _emit(result, args)
    return 0


def _quality_titles(raw: str) -> list[str]:
    titles = list(dict.fromkeys(value.strip() for value in raw.split(",") if value.strip()))
    if not titles:
        raise ValueError("--titles must contain at least one title")
    return titles


def _write_codex_quota_checkpoint(
    *,
    title_id: str,
    package_dir: Path,
    cache_dir: Path,
    error: CodexUsageLimitError,
) -> dict[str, Any]:
    package_receipts = list(package_dir.glob("scenes/**/*.receipt.json")) if package_dir.is_dir() else []
    cached_receipts: list[dict[str, Any]] = []
    if cache_dir.is_dir():
        for receipt_path in cache_dir.glob("*/receipt.json"):
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if receipt.get("title_id") == title_id and receipt.get("status") == "succeeded":
                cached_receipts.append(receipt)
    role_counts: dict[str, int] = {}
    for receipt in cached_receipts:
        role = str(receipt.get("role") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
    checkpoint = {
        "schema_name": "translation-forensics/codex-quality-execution-checkpoint",
        "schema_version": "1",
        "status": "awaiting-codex-quota",
        "title_id": title_id,
        "failed_call": {"call_id": error.call_id, "role": error.role},
        "retry_after_reported_by_service": error.retry_after,
        "completed_model_calls_in_cache": len(cached_receipts),
        "completed_model_calls_materialized_in_package": len(package_receipts),
        "completed_model_calls_by_role": dict(sorted(role_counts.items())),
        "resume_supported": True,
        "resume_command": f"translation-forensics run-codex-quality --titles {title_id} --resume",
        "api_key_used": False,
        "model_fallback_used": False,
        "release_kind": "autonomous-quality-candidate",
        "human_equal": False,
        "human_final": False,
        "final_promotion_allowed": False,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(package_dir / "execution-checkpoint.json", checkpoint)
    return checkpoint


def cmd_run_codex_quality(args: argparse.Namespace) -> int:
    root = _project_root(args)
    try:
        titles = _quality_titles(args.titles)
    except ValueError as exc:
        _emit({"status": "fail", "error": str(exc)}, args)
        return 2
    plans: list[dict[str, Any]] = []
    for title in titles:
        workspace = _workspace(root, title, None)
        structure = _manifest_role_path(workspace, "structure")
        japanese = _manifest_role_path(workspace, "ja")
        audio = _manifest_role_path(workspace, "audio")
        package = (
            args.output.expanduser().resolve() / title
            if args.output
            else workspace / "autonomous-quality-v1" / "package-v3"
        )
        plans.append(
            {
                "title_id": title,
                "workspace": workspace,
                "structure": structure,
                "japanese": japanese,
                "audio": audio,
                "package": package,
            }
        )
    missing = [
        f"{plan['title_id']}:{field}"
        for plan in plans
        for field in ("structure", "japanese", "audio")
        if plan[field] is None
    ]
    if missing:
        _emit({"status": "blocked", "error": "Missing required inputs", "missing": missing}, args)
        return 2
    if args.dry_run:
        _emit(
            {
                "status": "dry-run",
                "titles": titles,
                "full_local_asr": args.full_local_asr,
                "repair_evidence": args.repair_evidence,
                "resume": args.resume,
                "api_key_required": False,
                "external_codex_transfer": True,
                "runs": [
                    {key: str(value) if isinstance(value, Path) else value for key, value in plan.items()}
                    for plan in plans
                ],
            },
            args,
        )
        return 0

    cache_dir = (args.cache_dir or root / ".cache" / "codex-quality").expanduser().resolve()
    provider = CodexExecProvider(
        cache_dir=cache_dir,
        timeout_seconds=args.codex_timeout,
    )
    preflight = provider.preflight()
    if preflight["status"] != "pass":
        _emit({"status": "blocked", "phase": "codex-preflight", "preflight": preflight}, args)
        return 2

    results: list[dict[str, Any]] = []
    for plan in plans:
        title = str(plan["title_id"])
        workspace = plan["workspace"]
        quality_root = workspace / "autonomous-quality-v1"
        local_asr_dir = quality_root / "local-asr-v2"
        utterance_dir = quality_root / "utterance-evidence-v3"
        acoustic_path = utterance_dir / "block-acoustic-evidence.jsonl"
        try:
            if args.full_local_asr:
                asr_result = run_full_local_asr(
                    title_id=title,
                    audio_path=plan["audio"],
                    structure_path=plan["structure"],
                    output_dir=local_asr_dir,
                    force_cpu=args.cpu,
                    allow_model_download=not args.offline,
                    resume=args.resume,
                    max_windows=args.max_windows,
                )
            elif acoustic_path.is_file():
                asr_result = {"status": "reused", "output": str(local_asr_dir)}
            else:
                local_manifest = local_asr_dir / "local-asr-manifest.json"
                if not local_manifest.is_file():
                    raise FileNotFoundError(
                        f"{title}: local ASR evidence is missing; rerun with --full-local-asr"
                    )
                asr_result = {"status": "reused", "output": str(local_asr_dir)}
            utterance_result = build_timestamped_utterance_evidence(
                title_id=title,
                structure_path=plan["structure"],
                local_asr_dir=local_asr_dir,
                output_dir=utterance_dir,
                force_cpu=args.cpu,
                allow_model_download=not args.offline,
                resume=args.resume,
            )
            source_quality_dir = quality_root / "source-quality-v3"
            source_map_path = source_quality_dir / "source-quality-map.jsonl"
            source_result = write_source_quality_audit(
                title_id=title,
                japanese_path=plan["japanese"],
                asr_evidence_path=acoustic_path,
                output_dir=source_quality_dir,
                resume=args.resume,
            )
            effective_acoustic_path = acoustic_path
            effective_source_map_path = source_map_path
            repair_result: dict[str, Any] | None = None
            effective_repair_report_path: Path | None = None
            if args.repair_evidence:
                base_acoustic = {
                    int(row["block_number"]): row
                    for row in (
                        json.loads(line)
                        for line in acoustic_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                }
                structure_blocks, _, _ = parse_srt(plan["structure"])
                source_quality = {
                    int(row["block_number"]): row
                    for row in (
                        json.loads(line)
                        for line in source_map_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                }
                initial_ceiling = evaluate_evidence_ceiling(
                    title_id=title,
                    expected_blocks=[block.number for block in structure_blocks],
                    source_quality=source_quality,
                    acoustic=base_acoustic,
                )
                if initial_ceiling["status"] != "pass":
                    ineligible = [
                        block
                        for block in structure_blocks
                        if block.number not in set(initial_ceiling["eligible_block_numbers"])
                    ]
                    focused_repair = bool(getattr(args, "focused_repair", False))
                    repair_kind = "focused" if focused_repair else "pre-ceiling"
                    repair_dir = quality_root / ("focused-repair-v1" if focused_repair else "pre-ceiling-repair-v4")
                    repair_policy = FOCUSED_REPAIR_POLICY if focused_repair else None
                    repair_lineage = {
                        "audio_sha256": _sha256_file(plan["audio"]),
                        "structure_sha256": _sha256_file(plan["structure"]),
                        "base_acoustic_sha256": _sha256_file(acoustic_path),
                        "base_source_quality_sha256": _sha256_file(source_map_path),
                        "japanese_sha256": _sha256_file(plan["japanese"]),
                        "title_id": title,
                    }
                    repaired_rows = run_pre_ceiling_evidence_repair(
                        title_id=title,
                        audio_path=plan["audio"],
                        blocks=ineligible,
                        base_acoustic=base_acoustic,
                        output_dir=repair_dir,
                        force_cpu=args.cpu,
                        allow_model_download=not args.offline,
                        resume=args.resume,
                        lineage=repair_lineage,
                        attribution_blocks=structure_blocks,
                        policy=repair_policy,
                        repair_kind=repair_kind,
                    )
                    merged_acoustic_path = repair_dir / "block-acoustic-evidence-merged.jsonl"
                    repair_report_path = repair_dir / "repair-report.json"
                    repair_report_for_merge = json.loads(repair_report_path.read_text(encoding="utf-8"))
                    merge_identity = {
                        "cache_identity": repair_report_for_merge.get("cache_identity"),
                        "base_acoustic_sha256": _sha256_file(acoustic_path),
                        "repaired_evidence_sha256": repair_report_for_merge.get("evidence_sha256"),
                    }
                    merge_meta_path = merged_acoustic_path.with_suffix(".meta.json")
                    reusable_merge = False
                    if args.resume and merged_acoustic_path.is_file() and merge_meta_path.is_file():
                        try:
                            existing_merge_meta = json.loads(merge_meta_path.read_text(encoding="utf-8"))
                            reusable_merge = (
                                all(existing_merge_meta.get(key) == value for key, value in merge_identity.items())
                                and existing_merge_meta.get("merged_acoustic_sha256") == _sha256_file(merged_acoustic_path)
                            )
                        except (OSError, TypeError, ValueError, json.JSONDecodeError):
                            reusable_merge = False
                    if reusable_merge:
                        effective_acoustic_path = merged_acoustic_path
                    else:
                        if merged_acoustic_path.exists():
                            merged_acoustic_path = repair_dir / (
                                f"block-acoustic-evidence-merged-{str(merge_identity['cache_identity'])[:12]}.jsonl"
                            )
                            merge_meta_path = merged_acoustic_path.with_suffix(".meta.json")
                        effective_acoustic_path = merge_repaired_acoustic_evidence(
                            base_acoustic_path=acoustic_path,
                            repaired_rows=repaired_rows,
                            output_path=merged_acoustic_path,
                        )
                        merge_identity["merged_acoustic_sha256"] = _sha256_file(effective_acoustic_path)
                        merge_meta_path.write_text(
                            json.dumps(merge_identity, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                            encoding="utf-8",
                            newline="\n",
                        )
                    source_quality_dir = quality_root / "source-quality-v4-repaired"
                    effective_source_map_path = source_quality_dir / "source-quality-map.jsonl"
                    source_result = write_source_quality_audit(
                        title_id=title,
                        japanese_path=plan["japanese"],
                        asr_evidence_path=effective_acoustic_path,
                        output_dir=source_quality_dir,
                        resume=args.resume,
                    )
                    repair_report = repair_dir / "repair-report.json"
                    effective_repair_report_path = repair_report
                    repair_result = json.loads(repair_report.read_text(encoding="utf-8"))
                    final_acoustic = {
                        int(row["block_number"]): row
                        for row in (
                            json.loads(line)
                            for line in effective_acoustic_path.read_text(encoding="utf-8").splitlines()
                            if line.strip()
                        )
                    }
                    final_source_quality = {
                        int(row["block_number"]): row
                        for row in (
                            json.loads(line)
                            for line in effective_source_map_path.read_text(encoding="utf-8").splitlines()
                            if line.strip()
                        )
                    }
                    final_ceiling = evaluate_evidence_ceiling(
                        title_id=title,
                        expected_blocks=[block.number for block in structure_blocks],
                        source_quality=final_source_quality,
                        acoustic=final_acoustic,
                    )
                    repair_result.update(
                        {
                            "initial_ceiling": {
                                "eligible_block_count": initial_ceiling["eligible_block_count"],
                                "eligible_block_numbers": initial_ceiling["eligible_block_numbers"],
                                "maximum_possible_accepted_rate": initial_ceiling["maximum_possible_accepted_rate"],
                            },
                            "final_ceiling": {
                                "eligible_block_count": final_ceiling["eligible_block_count"],
                                "eligible_block_numbers": final_ceiling["eligible_block_numbers"],
                                "maximum_possible_accepted_rate": final_ceiling["maximum_possible_accepted_rate"],
                                "status": final_ceiling["status"],
                            },
                            "net_new_eligible_block_numbers": sorted(
                                set(final_ceiling["eligible_block_numbers"])
                                - set(initial_ceiling["eligible_block_numbers"])
                            ),
                            "merged_acoustic_path": effective_acoustic_path.name,
                            "merged_acoustic_sha256": _sha256_file(effective_acoustic_path),
                            "merged_acoustic_meta_path": merge_meta_path.name,
                            "merged_acoustic_meta_sha256": _sha256_file(merge_meta_path),
                            "regenerated_source_quality_path": effective_source_map_path.name,
                            "regenerated_source_quality_sha256": _sha256_file(effective_source_map_path),
                            "source_quality_audit_input": {
                                "japanese_sha256": _sha256_file(plan["japanese"]),
                                "acoustic_sha256": _sha256_file(effective_acoustic_path),
                                "audit_policy_version": "source-quality-v4-repaired",
                            },
                        }
                    )
                    repair_report.write_text(
                        json.dumps(repair_result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                        newline="\n",
                    )
            package_result = run_codex_quality_title(
                title_id=title,
                structure_path=plan["structure"],
                source_quality_map_path=effective_source_map_path,
                acoustic_evidence_path=effective_acoustic_path,
                audio_path=plan["audio"],
                output_dir=plan["package"],
                provider=provider,
                prompt_dir=root / "prompts",
                schema_dir=root / "schemas",
                evidence_repair_path=effective_repair_report_path,
                max_scene_blocks=args.max_scene_blocks,
                maximum_scene_gap_seconds=args.max_scene_gap,
                max_repairs=args.max_repairs,
                force_cpu=args.cpu,
                allow_model_download=not args.offline,
                resume=args.resume,
                local_asr_call_count=int((repair_result or {}).get("local_asr_call_count", 0)),
                partial_evidence_evaluation=bool(getattr(args, "partial_evidence_evaluation", False)),
            )
            results.append(
                {
                    "title_id": title,
                    "status": package_result.get("status"),
                    "source_quality": source_result.get("status_counts"),
                    "local_asr_status": asr_result.get("status"),
                    "utterance_evidence_status": utterance_result.get("status"),
                    "evidence_repair": repair_result,
                    "package": package_result.get("output"),
                }
            )
        except CodexUsageLimitError as exc:
            checkpoint = _write_codex_quota_checkpoint(
                title_id=title,
                package_dir=plan["package"],
                cache_dir=cache_dir,
                error=exc,
            )
            _emit(
                {
                    "status": "awaiting-codex-quota",
                    "title_id": title,
                    "error": str(exc),
                    "checkpoint": str(plan["package"] / "execution-checkpoint.json"),
                    "retry_after_reported_by_service": exc.retry_after,
                    "completed": results,
                },
                args,
            )
            return 2
        except (
            CodexExecError,
            CodexQualityError,
            LocalASRError,
            FileExistsError,
            FileNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            _emit({"status": "fail", "title_id": title, "error": str(exc), "completed": results}, args)
            return 2
    _emit({"status": "completed", "results": results}, args)
    return 0


def cmd_validate_codex_quality(args: argparse.Namespace) -> int:
    try:
        result = validate_codex_quality(args.package.expanduser().resolve())
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args)
        return 2
    _emit(result, args)
    return 0 if result["status"] == "pass" else 1


def cmd_evaluate_codex_quality(args: argparse.Namespace) -> int:
    root = _project_root(args)
    provider = CodexExecProvider(
        cache_dir=(args.cache_dir or root / ".cache" / "codex-quality").expanduser().resolve(),
        timeout_seconds=args.codex_timeout,
    )
    if args.dry_run:
        _emit(
            {
                "status": "dry-run",
                "package": str(args.package.expanduser().resolve()),
                "baseline": str(args.baseline.expanduser().resolve()),
                "sample_size": args.sample_size,
                "evaluators": ["gpt-5.6-terra", "gpt-5.6-sol"],
                "engineering_proxy_only": True,
            },
            args,
        )
        return 0
    preflight = provider.preflight()
    if preflight["status"] != "pass":
        _emit({"status": "blocked", "phase": "codex-preflight", "preflight": preflight}, args)
        return 2
    try:
        result = evaluate_codex_quality(
            package_dir=args.package.expanduser().resolve(),
            baseline=args.baseline.expanduser().resolve(),
            provider=provider,
            prompt_dir=root / "prompts",
            schema_dir=root / "schemas",
            sample_size=args.sample_size,
            batch_size=args.batch_size,
            resume=args.resume,
        )
    except (
        CodexExecError,
        CodexQualityError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        _emit({"status": "fail", "error": str(exc)}, args)
        return 2
    _emit(result, args)
    return 0 if result["status"] == "pass" else 1


def cmd_run_autonomous_release(args: argparse.Namespace) -> int:
    root = _project_root(args)
    try:
        titles = _autonomous_titles(args)
    except (OSError, ValueError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    explicit_inputs = [args.structure, args.ja, args.previous_ko, args.scenes, args.local_asr]
    if len(titles) > 1 and any(explicit_inputs):
        _emit({"status": "fail", "error": "여러 작품 실행에서는 작품별 매니페스트/기본 경로를 사용해야 하므로 단일 입력 경로 옵션을 함께 쓸 수 없습니다."}, args); return 2
    prompt_root = root / "prompts"
    schema_root = root / "schemas"
    dry_plan: list[dict[str, Any]] = []

    def resolve_title(title: str) -> dict[str, Path | None]:
        workspace = _workspace(root, title, None)
        structure = args.structure.expanduser().resolve() if args.structure else _manifest_role_path(workspace, "structure")
        japanese = args.ja.expanduser().resolve() if args.ja else _manifest_role_path(workspace, "ja")
        previous = args.previous_ko.expanduser().resolve() if args.previous_ko else _manifest_role_path(workspace, "previous_ko")
        scenes = args.scenes.expanduser().resolve() if args.scenes else _first_existing([
            workspace / "inferred-recovery-audio-v4" / "review-scenes.csv",
            workspace / "intermediate" / f"{title}.work_audio" / "review-scenes.csv",
        ])
        local_asr = args.local_asr.expanduser().resolve() if args.local_asr else _first_existing([
            workspace / "inferred-recovery-audio-v4" / "asr-candidates.csv",
            workspace / "intermediate" / f"{title}.work_audio" / "asr-candidates.csv",
        ])
        if structure is None or japanese is None:
            raise FileNotFoundError(f"{title}: structure 또는 Japanese SRT를 프로젝트 매니페스트에서 찾지 못했습니다.")
        if len(titles) == 1 and args.output:
            output = args.output.expanduser().resolve()
        elif args.output:
            output = args.output.expanduser().resolve() / title / "autonomous-release-v1"
        else:
            output = workspace / "autonomous-release-v1"
        return {"workspace": workspace, "structure": structure, "japanese": japanese, "previous": previous, "scenes": scenes, "local_asr": local_asr, "output": output}

    try:
        for title in titles:
            paths = resolve_title(title)
            dry_plan.append({"title_id": title, **{key: str(value) if value else None for key, value in paths.items()}})
    except (OSError, ValueError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    if args.dry_run:
        _emit({
            "status": "dry-run",
            "titles": titles,
            "runs": dry_plan,
            "allow_network": args.allow_network,
            "max_cost_usd": args.max_cost_usd,
            "max_workers": args.max_workers,
            "network_preflight_required": True,
        }, args)
        return 0
    if not args.allow_network or args.max_cost_usd is None or args.max_cost_usd <= 0:
        _emit({"status": "blocked", "error": "실제 autonomous-release에는 --allow-network와 양수 --max-cost-usd가 모두 필요합니다."}, args); return 2
    budget = BudgetTracker(float(args.max_cost_usd))
    provider = OpenAIProvider(
        cache_dir=args.cache_dir or root / ".cache" / "autonomous-release",
        budget=budget,
        allow_network=True,
        max_retries=2,
    )
    preflight = provider.preflight()
    if preflight["status"] != "pass":
        _emit({"status": "blocked", "preflight": preflight}, args); return 2

    prepared_paths: dict[str, dict[str, Path | None]] = {}
    try:
        for title in titles:
            paths = resolve_title(title)
            if paths["local_asr"] is None and paths["scenes"] is not None:
                local_asr = paths["workspace"] / "intermediate" / f"{title}.autonomous-local-asr.csv"  # type: ignore[operator]
                if not local_asr.exists():
                    local_asr.parent.mkdir(parents=True, exist_ok=True)
                    execution = run_asr(
                        root,
                        paths["scenes"],  # type: ignore[arg-type]
                        local_asr,
                        model=args.local_asr_model,
                        force_cpu=args.cpu,
                        offline=not args.allow_local_model_download,
                    )
                    if execution.get("status") != "completed" or not local_asr.exists():
                        raise RuntimeError(f"{title}: local Whisper execution failed: {execution.get('stderr') or execution}")
                paths["local_asr"] = local_asr
            prepared_paths[title] = paths
    except (OSError, RuntimeError, ValueError) as exc:
        _emit({"status": "blocked", "error": str(exc), "phase": "local-asr"}, args); return 2

    def execute(title: str) -> dict[str, Any]:
        paths = prepared_paths[title]
        return run_autonomous_release(
            title_id=title,
            structure_path=paths["structure"],  # type: ignore[arg-type]
            japanese_path=paths["japanese"],  # type: ignore[arg-type]
            previous_path=paths["previous"],
            scenes_path=paths["scenes"],
            local_asr_path=paths["local_asr"],
            output_dir=paths["output"],  # type: ignore[arg-type]
            provider=provider,
            decision_prompt_path=prompt_root / "autonomous-subtitle-decision-v1.md",
            critic_prompt_path=prompt_root / "autonomous-subtitle-critic-v1.md",
            decision_schema_path=schema_root / "autonomous-decision-response.schema.json",
            critique_schema_path=schema_root / "autonomous-critique-response.schema.json",
            batch_size=args.batch_size,
            max_repairs=args.max_repairs,
            resume=args.resume,
        )

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
            future_map = {pool.submit(execute, title): title for title in titles}
            for future in as_completed(future_map):
                title = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    failures.append({"title_id": title, "error": str(exc)})
    except (OSError, ValueError, ProviderUnavailableError, BudgetExceededError) as exc:
        _emit({"status": "fail", "error": str(exc), "results": results}, args); return 2
    results.sort(key=lambda row: str(row.get("title_id")))
    failures.sort(key=lambda row: row["title_id"])
    status = "completed" if not failures else "partial-failure"
    _emit({"status": status, "results": results, "failures": failures, "spent_usd": round(budget.spent_usd, 6), "max_cost_usd": budget.maximum_usd}, args)
    return 0 if not failures else 1


def cmd_validate_autonomous_release(args: argparse.Namespace) -> int:
    result = validate_autonomous_release(args.package.expanduser().resolve())
    if args.output and not args.dry_run:
        output = args.output.expanduser().resolve()
        if output.exists():
            _emit({"status": "fail", "error": f"기존 검증 보고서를 덮어쓰지 않습니다: {output}"}, args); return 2
        write_json(output, result)
    _emit(result, args)
    return 0 if result["status"] == "pass" else 1


def cmd_prove_autonomous_claim(args: argparse.Namespace) -> int:
    try:
        result = prove_autonomous_claim(args.package, None if args.dry_run else args.output)
    except (FileExistsError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args)
    return 0 if result["status"] == "pass" else 1


def cmd_build_translation_queue(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        structure = resolve_role(input_root, "structure", args.structure, args.title, required=True)
        ja = resolve_role(input_root, "ja", args.ja, args.title, required=True)
        previous = resolve_role(input_root, "previous_ko", args.previous_ko, args.title)
        photos = resolve_role(input_root, "photos", args.photos, args.title)
    except DiscoveryError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    review_context = args.review_context or workspace / "intermediate" / "review-context.jsonl"
    review_queue = args.review_queue or workspace / "intermediate" / f"{args.title}.review-queue.structure-fallback-v4.csv"
    capture_index = args.capture_index
    consistency_ledger = args.consistency_ledger or workspace / "intermediate" / f"{args.title}.translation-consistency-v1.jsonl"
    terminology = args.terminology
    speaker_state = args.speaker_state or workspace / "intermediate" / f"{args.title}.speaker-state-v1.json"
    for label, explicit in (("consistency_ledger", args.consistency_ledger), ("terminology", terminology), ("speaker_state", args.speaker_state)):
        if explicit and not explicit.expanduser().exists():
            _emit({"status": "fail", "error": f"{label} 파일이 없습니다: {explicit}"}, args); return 2
    output = args.output or workspace / "intermediate" / f"{args.title}.translation-queue-v1.jsonl"
    report = args.report or workspace / "intermediate" / f"{args.title}.translation-queue-v1.report.json"
    if args.dry_run:
        _emit({"status": "dry-run", "structure": str(structure), "ja": str(ja), "previous_ko": str(previous) if previous else None, "photos": str(photos) if photos else None, "consistency_ledger": str(consistency_ledger), "terminology": str(terminology) if terminology else None, "speaker_state": str(speaker_state), "output": str(output), "report": str(report), "translation_model": args.translation_model}, args); return 0
    if output.exists() or report.exists():
        _emit({"status": "fail", "error": "기존 translation queue를 덮어쓰지 않습니다.", "output": str(output), "report": str(report)}, args); return 2
    try:
        result = build_translation_queue(structure, ja, previous, output, report, review_context_path=review_context if review_context.exists() else None, review_queue_path=review_queue if review_queue.exists() else None, capture_index_path=capture_index if capture_index and capture_index.exists() else None, consistency_ledger_path=consistency_ledger if consistency_ledger.exists() else None, terminology_path=terminology.expanduser().resolve() if terminology and terminology.exists() else None, speaker_state_path=speaker_state if speaker_state.exists() else None, translation_model=args.translation_model, title_id=args.title)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_apply_translations(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        structure = resolve_role(input_root, "structure", args.structure, args.title, required=True)
    except DiscoveryError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    decisions = args.decisions.expanduser().resolve()
    translation_queue = args.translation_queue or workspace / "intermediate" / f"{args.title}.translation-queue-v1.jsonl"
    source_output = args.source_output or workspace / "intermediate" / f"{args.title}.source-faithful-ko.text-crosschecked-v1.srt"
    viewer_output = args.viewer_output or workspace / "intermediate" / f"{args.title}.viewer-natural-ko.text-crosschecked-v1.srt"
    report = args.report or workspace / "intermediate" / f"{args.title}.translation-application-v1.report.json"
    if args.dry_run:
        _emit({"status": "dry-run", "structure": str(structure), "decisions": str(decisions), "translation_queue": str(translation_queue), "source_output": str(source_output), "viewer_output": str(viewer_output), "strict": args.strict, "translation_model": args.translation_model}, args); return 0
    if not decisions.exists():
        _emit({"status": "fail", "error": f"번역 결정 파일이 없습니다: {decisions}"}, args); return 2
    if args.strict and not translation_queue.exists():
        _emit({"status": "fail", "error": "--strict 적용에는 결정의 evidence_refs를 검증할 translation queue가 필요합니다.", "translation_queue": str(translation_queue)}, args); return 2
    if source_output.exists() or viewer_output.exists() or report.exists():
        _emit({"status": "fail", "error": "기존 번역 적용 결과를 덮어쓰지 않습니다.", "source_output": str(source_output), "viewer_output": str(viewer_output), "report": str(report)}, args); return 2
    try:
        result = apply_translation_decisions(structure, decisions, source_output, viewer_output, report, strict=args.strict, translation_model=args.translation_model, translation_queue_path=translation_queue if args.strict else None)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result.get("status") != "fail" else 1


def cmd_init_translation_decisions(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    queue = args.queue or workspace / "intermediate" / f"{args.title}.translation-queue-v1.jsonl"
    output = args.output or workspace / "intermediate" / f"{args.title}.translation-decisions-template-v1.jsonl"
    if args.dry_run:
        _emit({"status": "dry-run", "queue": str(queue), "output": str(output), "translation_model": args.translation_model}, args); return 0
    if not queue.exists():
        _emit({"status": "fail", "error": f"translation queue가 없습니다: {queue}"}, args); return 2
    if output.exists():
        _emit({"status": "fail", "error": "기존 번역 결정 템플릿을 덮어쓰지 않습니다.", "output": str(output)}, args); return 2
    try:
        result = initialize_translation_decisions(queue, output, translation_model=args.translation_model)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_init_consistency_ledger(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    output = args.output or workspace / "intermediate" / f"{args.title}.translation-consistency-v1.jsonl"
    if args.dry_run:
        _emit({"status": "dry-run", "output": str(output)}, args); return 0
    try:
        result = initialize_consistency_ledger(output)
    except (OSError, ValueError, FileExistsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_validate_consistency_ledger(args: argparse.Namespace) -> int:
    try:
        result = validate_consistency_ledger(args.input.expanduser().resolve())
        if args.output and not args.dry_run:
            output = args.output.expanduser().resolve()
            if output.exists():
                raise FileExistsError(f"기존 consistency ledger 보고서를 덮어쓰지 않습니다: {output}")
            write_json(output, result)
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["status"] == "pass" else 1


def cmd_validate_quality_regressions(args: argparse.Namespace) -> int:
    try:
        result = validate_quality_regression_suite(args.input.expanduser().resolve())
        if args.output and not args.dry_run:
            output = args.output.expanduser().resolve()
            if output.exists():
                raise FileExistsError(f"기존 quality regression 보고서를 덮어쓰지 않습니다: {output}")
            write_json(output, result)
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["status"] == "pass" else 1


def cmd_build_uncertainty_review_queue(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        structure = resolve_role(input_root, "structure", args.structure, args.title, required=True)
    except DiscoveryError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    decisions = args.decisions or workspace / "intermediate" / f"{args.title}.translation-decisions-template-v1.jsonl"
    translation_queue = args.translation_queue or workspace / "intermediate" / f"{args.title}.translation-queue-v1.jsonl"
    forensics_queue = args.forensics_queue or workspace / "intermediate" / f"{args.title}.review-queue.structure-fallback-v4.csv"
    output = args.output or workspace / "intermediate" / f"{args.title}.review-queue.uncertainty-v1.csv"
    source = args.source_faithful.expanduser().resolve() if args.source_faithful else None
    viewer = args.viewer_natural.expanduser().resolve() if args.viewer_natural else None
    if args.dry_run:
        _emit({"status": "dry-run", "structure": str(structure), "decisions": str(decisions), "translation_queue": str(translation_queue), "forensics_queue": str(forensics_queue), "source_faithful": str(source) if source else None, "viewer_natural": str(viewer) if viewer else None, "output": str(output)}, args); return 0
    try:
        result = build_uncertainty_review_queue(
            structure,
            output,
            decisions_path=decisions if decisions.exists() else None,
            translation_queue_path=translation_queue if translation_queue.exists() else None,
            forensics_queue_path=forensics_queue if forensics_queue.exists() else None,
            source_path=source,
            viewer_path=viewer,
            project_root=root,
        )
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_merge_translation_decisions(args: argparse.Namespace) -> int:
    base = args.base.expanduser().resolve()
    reviewed = args.reviewed.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.dry_run:
        _emit({"status": "dry-run", "base": str(base), "reviewed": str(reviewed), "output": str(output)}, args); return 0
    if not base.exists() or not reviewed.exists():
        _emit({"status": "fail", "error": "base 또는 reviewed 번역 결정 파일이 없습니다."}, args); return 2
    if output.exists():
        _emit({"status": "fail", "error": "기존 병합 결과를 덮어쓰지 않습니다.", "output": str(output)}, args); return 2
    try:
        result = merge_translation_decisions(base, reviewed, output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result.get("status") != "fail" else 1


def cmd_init_forensic_records(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    queue = args.queue or workspace / "intermediate" / f"{args.title}.translation-queue-v1.jsonl"
    frames = args.frames or workspace / "intermediate" / f"{args.title}.semantic-frames.unreviewed-v1.jsonl"
    hypotheses = args.hypotheses or workspace / "intermediate" / f"{args.title}.hypothesis-ledger.unreviewed-v1.jsonl"
    if args.dry_run:
        _emit({"status": "dry-run", "queue": str(queue), "frames": str(frames), "hypotheses": str(hypotheses)}, args); return 0
    if not queue.exists():
        _emit({"status": "fail", "error": f"translation queue가 없습니다: {queue}"}, args); return 2
    if frames.exists() or hypotheses.exists():
        _emit({"status": "fail", "error": "기존 forensics record를 덮어쓰지 않습니다."}, args); return 2
    try:
        result = initialize_forensic_records(queue, frames, hypotheses)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_validate_forensic_records(args: argparse.Namespace) -> int:
    report = args.output.expanduser().resolve() if args.output else None
    try:
        result = validate_forensic_records(args.frames.expanduser().resolve(), args.hypotheses.expanduser().resolve())
        if report and not args.dry_run:
            if report.exists():
                raise FileExistsError(f"기존 보고서를 덮어쓰지 않습니다: {report}")
            write_json(report, result)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["status"] == "pass" else 1


def cmd_init_gold(args: argparse.Namespace) -> int:
    root = _project_root(args)
    if args.dry_run:
        _emit({"status": "dry-run", "gold_root": str(root / "evaluation" / "gold")}, args); return 0
    try:
        result = initialize_gold_layout(root)
    except (OSError, FileExistsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_init_gold_record(args: argparse.Namespace) -> int:
    root = _project_root(args)
    block_ids = [int(value) for value in args.block_ids.split(",") if value.strip()]
    if args.dry_run:
        _emit({"status": "dry-run", "gold_id": args.gold_id, "title_id": args.title_id, "scene_id": args.scene_id, "block_ids": block_ids, "split": args.split}, args); return 0
    try:
        result = initialize_gold_record(root, gold_id=args.gold_id, title_id=args.title_id, scene_id=args.scene_id, block_ids=block_ids, split=args.split)
    except (OSError, ValueError, FileExistsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_validate_gold_record(args: argparse.Namespace) -> int:
    try:
        result = validate_gold_record(args.input.expanduser().resolve())
        if args.output and not args.dry_run:
            output = args.output.expanduser().resolve()
            if output.exists():
                raise FileExistsError(f"기존 보고서를 덮어쓰지 않습니다: {output}")
            write_json(output, result)
    except (OSError, ValueError, json.JSONDecodeError, FileExistsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["status"] == "pass" else 1


def cmd_validate_gold_suite(args: argparse.Namespace) -> int:
    try:
        result = validate_gold_suite(_project_root(args))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["status"] == "pass" else 1


def cmd_migrate_gold_layout(args: argparse.Namespace) -> int:
    try:
        result = migrate_gold_layout(_project_root(args))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_build_blind_review_pack(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        structure = resolve_role(input_root, "structure", args.structure, args.title, required=True)
        source = args.source_faithful.expanduser().resolve()
        viewer = args.viewer_natural.expanduser().resolve()
        if not source.exists() or not viewer.exists():
            raise ValueError("--source-faithful와 --viewer-natural 파일이 모두 필요합니다.")
        output = args.output or workspace / "intermediate" / f"{args.title}.blind-review-pack-v1.zip"
        if args.dry_run:
            _emit({"status": "dry-run", "structure": str(structure), "source": str(source), "viewer": str(viewer), "output": str(output), "random_seed": args.random_seed}, args); return 0
        result = build_blind_review_pack(args.title, structure, source, viewer, output, random_seed=args.random_seed)
    except (OSError, ValueError, FileExistsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_build_review_pack(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    context = args.context or workspace / "intermediate" / "review-context.jsonl"
    output = args.output or workspace / "intermediate" / f"{args.title}.review-pack-v1"
    audio_root = args.audio_root or workspace / "intermediate" / f"{args.title}.work_audio"
    if args.dry_run:
        _emit({"status": "dry-run", "context": str(context), "output": str(output), "audio_root": str(audio_root)}, args); return 0
    try:
        result = build_review_pack(args.title, context, output, audio_root=audio_root)
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_validate_review_decisions(args: argparse.Namespace) -> int:
    try:
        result = validate_review_decisions(args.context.expanduser().resolve(), args.input.expanduser().resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["status"] == "pass" else 1


def cmd_sample_audit(args: argparse.Namespace) -> int:
    root = _project_root(args); workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    queue = args.queue or workspace / "intermediate" / f"{args.title}.translation-queue-v1.jsonl"
    output = args.output or workspace / "intermediate" / f"{args.title}.audit-sample-v1.csv"
    bands = {value.strip() for value in args.bands.split(",") if value.strip()}
    if args.dry_run:
        _emit({"status": "dry-run", "queue": str(queue), "output": str(output), "rate": args.rate, "bands": sorted(bands), "seed": args.seed}, args); return 0
    try:
        result = sample_audit(queue, output, title=args.title, seed=args.seed, rate=args.rate, bands=bands)
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_summarize_audit(args: argparse.Namespace) -> int:
    try:
        result = summarize_audit(args.input.expanduser().resolve(), args.output.expanduser().resolve())
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_init_terminology(args: argparse.Namespace) -> int:
    try:
        result = initialize_terminology(args.output.expanduser().resolve()) if not args.dry_run else {"status": "dry-run", "output": str(args.output)}
    except (OSError, FileExistsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_validate_terminology(args: argparse.Namespace) -> int:
    try:
        result = validate_terminology(args.input.expanduser().resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["status"] == "pass" else 1


def cmd_terminology_conflicts(args: argparse.Namespace) -> int:
    try:
        result = terminology_conflicts(args.input.expanduser().resolve(), args.output.expanduser().resolve())
    except (OSError, ValueError, json.JSONDecodeError, FileExistsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_build_evidence_graph(args: argparse.Namespace) -> int:
    root = _project_root(args); workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    queue = args.queue or workspace / "intermediate" / f"{args.title}.translation-queue-v1.jsonl"
    output = args.output or workspace / "intermediate" / f"{args.title}.evidence-graph-v1.jsonl"
    if args.dry_run:
        _emit({"status": "dry-run", "queue": str(queue), "output": str(output)}, args); return 0
    try:
        result = build_evidence_graph(args.title, queue, output, decisions_path=args.decisions, frames_path=args.frames, hypotheses_path=args.hypotheses, evaluation_path=args.evaluation)
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_summarize_blind_review(args: argparse.Namespace) -> int:
    try:
        result = summarize_blind_review(args.pack.expanduser().resolve(), args.reviewed.expanduser().resolve(), args.key.expanduser().resolve(), args.output.expanduser().resolve())
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["evaluation_status"] == "human-reviewed" else 1


def cmd_validate_evaluation_summary(args: argparse.Namespace) -> int:
    try:
        result = validate_evaluation_summary(args.input.expanduser().resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["status"] == "pass" else 1


def cmd_create_run_manifest(args: argparse.Namespace) -> int:
    root = _project_root(args)
    try:
        result = create_run_manifest(root, args.output.expanduser().resolve(), title=args.title, stage=args.stage, inputs=[path.expanduser().resolve() for path in args.inputs], artifacts=[path.expanduser().resolve() for path in args.artifacts], prompt_manifest=args.prompt_manifest.expanduser().resolve() if args.prompt_manifest else None, parent_run_id=args.parent_run_id, run_id=args.run_id)
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_init_reverse_check(args: argparse.Namespace) -> int:
    try:
        result = initialize_reverse_check(args.decisions.expanduser().resolve(), args.output.expanduser().resolve())
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_validate_reverse_check(args: argparse.Namespace) -> int:
    try:
        result = validate_reverse_check(args.decisions.expanduser().resolve(), args.input.expanduser().resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["status"] == "pass" else 1


def cmd_migrate_identity(args: argparse.Namespace) -> int:
    try:
        result = migrate_jsonl_identity(args.input.expanduser().resolve(), args.output.expanduser().resolve(), title_id=args.title, kind=args.kind)
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_validate_identity(args: argparse.Namespace) -> int:
    try:
        from .forensic_model import read_jsonl
        result = validate_identity_records(read_jsonl(args.input.expanduser().resolve()), title_id=args.title)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["status"] == "pass" else 1


def cmd_init_memory_ledger(args: argparse.Namespace) -> int:
    try:
        result = initialize_memory_ledger(args.output.expanduser().resolve(), kind=args.kind)
    except (OSError, ValueError, FileExistsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_validate_memory_ledger(args: argparse.Namespace) -> int:
    try:
        result = validate_memory_ledger(args.input.expanduser().resolve(), kind=args.kind)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["status"] == "pass" else 1


def cmd_init_speaker_state(args: argparse.Namespace) -> int:
    try:
        result = initialize_speaker_state(args.scenes.expanduser().resolve(), args.output.expanduser().resolve())
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_init_phonetic_candidates(args: argparse.Namespace) -> int:
    try:
        result = initialize_phonetic_candidates(args.queue.expanduser().resolve(), args.output.expanduser().resolve())
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_analyze_slot_conflicts(args: argparse.Namespace) -> int:
    try:
        result = build_slot_conflicts(args.hypotheses.expanduser().resolve(), args.output.expanduser().resolve())
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_evaluate_review_budget(args: argparse.Namespace) -> int:
    try:
        from .forensic_model import read_jsonl
        result = calculate_review_budget_metrics(read_jsonl(args.input.expanduser().resolve()), gold_positive_count=args.gold_positive_count, budgets=tuple(args.budgets))
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"기존 review budget metrics를 덮어쓰지 않습니다: {output}")
        write_json(output, result)
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["metrics_status"] == "demonstrated" else 1


def _load_json_evidence(path: Path | None) -> dict[str, Any] | None:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8")) if path else None


def cmd_validate_release_gate(args: argparse.Namespace) -> int:
    try:
        evidence = {"gold_suite": _load_json_evidence(args.gold_suite), "blind_review": _load_json_evidence(args.blind_review), "audit_summary": _load_json_evidence(args.audit_summary), "review_metrics": _load_json_evidence(args.review_metrics), "release_approval": _load_json_evidence(args.release_approval), "sol_experiment": _load_json_evidence(args.sol_experiment)}
        result = validate_release_gate(evidence, require_human_approval=not args.no_human_approval_required, require_sol_experiment=args.require_sol_experiment)
        if args.output:
            output = args.output.expanduser().resolve()
            if output.exists():
                raise FileExistsError(f"기존 release gate 보고서를 덮어쓰지 않습니다: {output}")
            write_json(output, result)
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["status"] == "pass" else 1


def cmd_init_sol_review(args: argparse.Namespace) -> int:
    try:
        result = initialize_sol_review_records(args.queue.expanduser().resolve(), args.output.expanduser().resolve(), title_id=args.title)
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_validate_sol_review(args: argparse.Namespace) -> int:
    result = validate_sol_review_records(args.input.expanduser().resolve())
    _emit(result, args); return 0 if result["status"] == "pass" else 1


def cmd_audit_current(args: argparse.Namespace) -> int:
    root = _project_root(args); workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    result = audit_workspace(root, args.title, workspace)
    if args.output and not args.dry_run:
        output = args.output.expanduser().resolve()
        if output.exists():
            _emit({"status": "fail", "error": f"기존 audit 보고서를 덮어쓰지 않습니다: {output}"}, args); return 2
        write_json(output, result)
    _emit(result, args); return 0 if result["status"] == "ready-for-final-audit" else 1


def cmd_validate_mqm(args: argparse.Namespace) -> int:
    try:
        result = validate_mqm_csv(args.input.expanduser().resolve())
        if args.output and not args.dry_run:
            output = args.output.expanduser().resolve()
            if output.exists():
                raise FileExistsError(f"기존 보고서를 덮어쓰지 않습니다: {output}")
            write_json(output, result)
    except (OSError, ValueError, FileExistsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["status"] == "pass" else 1


def cmd_validate_evidence_artifact(args: argparse.Namespace) -> int:
    path = args.input.expanduser().resolve()
    validators = {
        "speaker-state": validate_speaker_state,
        "alignment-evidence": validate_alignment_evidence,
        "backtranslation-check": validate_backtranslation_check,
    }
    try:
        result = validators[args.kind](path)
        if args.output and not args.dry_run:
            output = args.output.expanduser().resolve()
            if output.exists():
                raise FileExistsError(f"existing report will not be overwritten: {output}")
            write_json(output, result)
    except (OSError, ValueError, json.JSONDecodeError, FileExistsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["status"] == "pass" else 1


def cmd_validate_prompt_contract(args: argparse.Namespace) -> int:
    try:
        result = validate_prompt_contract(args.manifest.expanduser().resolve())
        if args.output and not args.dry_run:
            output = args.output.expanduser().resolve()
            if output.exists():
                raise FileExistsError(f"기존 보고서를 덮어쓰지 않습니다: {output}")
            write_json(output, result)
    except (OSError, ValueError, json.JSONDecodeError, FileExistsError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result["status"] == "pass" else 1


def cmd_prepare_audio(args: argparse.Namespace) -> int:
    root = _project_root(args); workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None); input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        queue = resolve_role(input_root, "review_queue", args.review_queue, args.title, required=True)
        audio = resolve_role(input_root, "audio", args.audio, args.title, required=True)
    except DiscoveryError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    timeline_path = args.timeline_report or _timeline_report_path(workspace, args.title)
    if not args.allow_unvalidated_timeline:
        if not timeline_path.exists():
            _emit({"status": "blocked", "error": "prepare-audio 전에 validate-timeline을 실행해야 합니다.", "timeline_report": str(timeline_path)}, args); return 2
        try:
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _emit({"status": "blocked", "error": f"시간축 보고서를 읽을 수 없습니다: {exc}", "timeline_report": str(timeline_path)}, args); return 2
        if not timeline_is_usable(timeline):
            _emit({"status": "blocked", "error": "시간축 검증이 클립 생성을 허용하지 않습니다.", "timeline_status": timeline.get("status"), "timeline_report": str(timeline_path)}, args); return 2
    out_dir = args.output or workspace / "intermediate" / f"{args.title}.work_audio"
    try:
        result = prepare_audio(root, queue, audio, out_dir, bands=args.bands, padding=args.padding, merge_gap=args.merge_gap, max_scene=args.max_scene, include_audio=not args.no_audio, dry_run=args.dry_run, force=args.force)
    except (RuntimeError, ValueError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result.get("status") not in {"failed"} else 1


def cmd_run_asr(args: argparse.Namespace) -> int:
    root = _project_root(args); workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None); work = workspace / "intermediate" / f"{args.title}.work_audio"
    scenes = args.scenes or work / "review-scenes.csv"; out = args.output or work / "asr-candidates.csv"
    prompt = args.prompt or root / "vendor" / "subtitle_audio_forensics_runner_v1" / "asr_prompt_ja.txt"
    try:
        result = run_asr(root, scenes, out, model=args.model, force_cpu=args.cpu, prompt=prompt if prompt.exists() else None, threshold=args.threshold, max_scenes=args.max_scenes, dry_run=args.dry_run, offline=args.offline)
    except RuntimeError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result.get("status") != "failed" else 1


def cmd_ingest_asr(args: argparse.Namespace) -> int:
    root = _project_root(args); workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None); source = args.source.expanduser().resolve(); dest = args.output or workspace / "intermediate" / f"{args.title}.work_audio"
    if args.dry_run:
        _emit({"status": "dry-run", "source": str(source), "destination": str(dest)}, args); return 0
    try:
        result = ingest_asr(source, dest, force=args.force)
    except (ValueError, RuntimeError, OSError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_package_audio(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    work = args.work_dir or workspace / "intermediate" / f"{args.title}.work_audio"
    asr = work / "asr-candidates.csv"
    scenes = work / "review-scenes.csv"
    output = args.output or workspace / f"{args.title}.work_audio.zip"
    if args.dry_run:
        _emit({"status": "dry-run", "work_dir": str(work), "output": str(output)}, args); return 0
    if not scenes.exists() or not asr.exists():
        _emit({"status": "fail", "error": "review-scenes.csv와 asr-candidates.csv가 모두 필요합니다."}, args); return 2
    try:
        rows = read_asr_candidates(asr)
        device, compute_type = detect_device(False)
        write_run_summary(work, title=args.title, model=args.model, device=device, compute_type=compute_type, asr_path=asr, status="audio-asr-crosschecked-input", returncode=0)
        zip_work_audio(work, output, include_clips=not args.no_clips)
    except (ValueError, FileExistsError, OSError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit({"status": "packaged", "rows": len(rows), "work_dir": str(work), "output": str(output), "direct_human_listening": False}, args); return 0


def cmd_build_review_context(args: argparse.Namespace) -> int:
    root = _project_root(args); workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None); input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        ja = resolve_role(input_root, "ja", args.ja, args.title, required=True)
        previous = resolve_role(input_root, "previous_ko", args.previous_ko, args.title)
    except DiscoveryError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    scenes = args.scenes or workspace / "intermediate" / f"{args.title}.work_audio" / "review-scenes.csv"
    asr = args.asr or workspace / "intermediate" / f"{args.title}.work_audio" / "asr-candidates.csv"
    output = args.output or workspace / "intermediate" / "review-context.jsonl"
    if args.dry_run:
        _emit({"status": "dry-run", "ja": str(ja), "scenes": str(scenes), "asr": str(asr), "output": str(output)}, args); return 0
    try:
        result = build_review_context(ja, previous, scenes, asr if asr.exists() else None, output)
        verdicts_path = output.parent / "asr-scene-verdicts.csv"
        scene_map_path = output.parent / "scene-map.json"
        verdicts = build_asr_verdicts(scenes, asr if asr.exists() else None)
        write_csv(verdicts_path, verdicts, ["scene_id", "verdict", "confidence", "evidence_summary", "unresolved_scope", "direct_human_listening"])
        write_json(scene_map_path, build_scene_map(scenes, asr if asr.exists() else None))
        result.update({"asr_verdicts": str(verdicts_path), "scene_map": str(scene_map_path)})
    except (OSError, ValueError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_validate(args: argparse.Namespace) -> int:
    root = _project_root(args); workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None); input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        reference = resolve_role(input_root, "structure", args.structure, args.title, required=True)
    except DiscoveryError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    if not args.source_faithful or not args.viewer_natural:
        _emit({"status": "fail", "error": "--source-faithful와 --viewer-natural이 모두 필요합니다."}, args); return 2
    report = validate_pair(reference, args.source_faithful.expanduser().resolve(), args.viewer_natural.expanduser().resolve(), project_root=root)
    if args.output and not args.dry_run:
        write_validation_report(args.output, report)
    _emit(report, args); return 0 if report.get("status") != "fail" else 1


def cmd_run_closed_world(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    structure = args.structure.expanduser().resolve() if args.structure else _manifest_role_path(workspace, "structure")
    japanese = args.ja.expanduser().resolve() if args.ja else _manifest_role_path(workspace, "ja")
    previous = args.previous_ko.expanduser().resolve() if args.previous_ko else _manifest_role_path(workspace, "previous_ko")
    if not structure or not japanese:
        _emit({"status": "fail", "error": "structure와 ja 입력을 명시하거나 project-manifest에 기록해야 합니다."}, args)
        return 2
    candidates = [path.expanduser().resolve() for path in (args.candidate_srt or [])]
    if not candidates:
        candidates = _discover_closed_world_candidates(workspace)
    review_context = args.review_context or workspace / "intermediate" / "review-context.jsonl"
    asr = args.asr or workspace / "intermediate" / f"{args.title}.work_audio" / "asr-candidates.csv"
    scenes = workspace / "intermediate" / f"{args.title}.work_audio" / "review-scenes.csv"
    timeline = args.timeline_validation or _timeline_report_path(workspace, args.title)
    default_output = (
        workspace / "network-enabled" / "closed-world-validated-v1"
        if args.allow_model_download
        else workspace / "closed-world" / "closed-world-validated-v1"
    )
    output = args.output or default_output
    if args.dry_run:
        _emit({
            "status": "dry-run",
            "title": args.title,
            "structure": str(structure),
            "japanese": str(japanese),
            "previous_korean_candidate": str(previous) if previous else None,
            "candidate_srts": [str(path) for path in candidates],
            "review_context": str(review_context),
            "asr": str(asr),
            "will_run_local_asr": not asr.exists() and scenes.exists(),
            "model_download_permission": args.allow_model_download,
            "timeline_validation": str(timeline),
            "output": str(output),
            "network_model_download_permitted": args.allow_model_download,
            "human_labels_used": False,
        }, args)
        return 0
    asr_execution: dict[str, Any] = {
        "status": "reused" if asr.exists() else "not-available",
        "offline": not args.allow_model_download,
        "model_download_permitted": args.allow_model_download,
    }
    if not asr.exists() and scenes.exists():
        prompt = root / "vendor" / "subtitle_audio_forensics_runner_v1" / "asr_prompt_ja.txt"
        try:
            asr_execution = run_asr(
                root,
                scenes,
                asr,
                model=args.asr_model,
                force_cpu=args.cpu,
                prompt=prompt if prompt.exists() else None,
                max_scenes=args.max_scenes,
                offline=not args.allow_model_download,
            )
        except RuntimeError as exc:
            asr_execution = {
                "status": "failed",
                "offline": not args.allow_model_download,
                "model_download_permitted": args.allow_model_download,
                "error": str(exc),
            }
    asr_usable = False
    if asr.exists():
        try:
            read_asr_candidates(asr)
            asr_usable = True
        except (OSError, ValueError):
            asr_execution = {**asr_execution, "status": "failed", "error": "생성된 ASR CSV가 규격 검증을 통과하지 못했습니다."}
    try:
        result = run_closed_world(
            title=args.title,
            project_root=root,
            structure_path=structure,
            japanese_path=japanese,
            output_dir=output,
            candidate_paths=candidates,
            previous_path=previous,
            review_context_path=review_context if review_context.exists() else None,
            asr_path=asr if asr_usable else None,
            timeline_validation_path=timeline if timeline.exists() else None,
            network_model_download_permitted=args.allow_model_download,
        )
    except (OSError, ValueError, RuntimeError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args)
        return 2
    result["asr_execution"] = asr_execution
    _emit(result, args)
    return 0


def cmd_validate_closed_world(args: argparse.Namespace) -> int:
    package_dir = args.package.expanduser().resolve()
    try:
        structure = args.structure.expanduser().resolve() if args.structure else _closed_world_structure_from_package(package_dir)
        result = validate_closed_world_package(structure, package_dir)
        if args.output and not args.dry_run:
            output = args.output.expanduser().resolve()
            if output.exists():
                raise FileExistsError(f"기존 검증 보고서를 덮어쓰지 않습니다: {output}")
            write_json(output, result)
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args)
        return 2
    _emit(result, args)
    return 0 if result["status"] == "pass" else 1


def cmd_prove_quality_claim(args: argparse.Namespace) -> int:
    package_dir = args.package.expanduser().resolve()
    try:
        structure = args.structure.expanduser().resolve() if args.structure else _closed_world_structure_from_package(package_dir)
        human_reference = args.human_reference.expanduser().resolve() if args.human_reference else None
        result = prove_quality_claim(structure, package_dir, human_reference_path=human_reference)
        if args.output and not args.dry_run:
            output = args.output.expanduser().resolve()
            if output.exists():
                raise FileExistsError(f"기존 품질 증명 보고서를 덮어쓰지 않습니다: {output}")
            write_json(output, result)
    except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args)
        return 2
    _emit(result, args)
    return 0 if result["status"] == "pass" else 1


def cmd_package(args: argparse.Namespace) -> int:
    root = _project_root(args); workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None); input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        reference = resolve_role(input_root, "structure", args.structure, args.title, required=True)
    except DiscoveryError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    output = args.output or workspace / "final"
    translation_queue = args.translation_queue or (workspace / "intermediate" / f"{args.title}.translation-queue-v1.jsonl" if args.decisions else None)
    if args.dry_run:
        _emit({"status": "dry-run", "output": str(output), "stage": args.stage, "translation_queue": str(translation_queue) if translation_queue else None}, args); return 0
    try:
        result = package_title_outputs(args.title, reference, args.source_faithful.expanduser().resolve(), args.viewer_natural.expanduser().resolve(), output, stage=args.stage, version=args.version, japanese_path=args.ja, previous_path=args.previous_ko, photos_path=args.photos, scenes_path=args.scenes, asr_path=args.asr, translation_decisions_path=args.decisions, translation_queue_path=translation_queue, semantic_frames_path=args.semantic_frames, hypothesis_ledger_path=args.hypothesis_ledger, speaker_state_path=args.speaker_state, alignment_evidence_path=args.alignment_evidence, mqm_errors_path=args.mqm_errors, backtranslation_check_path=args.backtranslation_check, evaluation_summary_path=args.evaluation_summary, blind_review_pack_path=args.blind_review_pack, release_gate_path=args.release_gate, timeline_validation_path=args.timeline_validation, project_root=root, all_blocks_reviewed=args.all_blocks_reviewed, direct_human_listening=args.direct_human_listening, evidence_complete=args.evidence_complete)
    except (RuntimeError, ValueError, FileExistsError, OSError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = _project_root(args); workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    if args.dry_run:
        _emit({"status": "dry-run", "steps": ["inspect", "analyze", "prepare-audio (원음 제공 시)", "run-asr (모델 제공 시)", "build-review-context", "validate/package는 의미 판정 후"]}, args); return 0
    inspect_args = argparse.Namespace(**vars(args)); inspect_args.structure = args.structure; inspect_args.ja = args.ja; inspect_args.previous_ko = args.previous_ko; inspect_args.audio = args.audio; inspect_args.video = args.video; inspect_args.photos = None; inspect_args.review_queue = None; inspect_args.asr_candidates = None; inspect_args.work_audio = None; inspect_args.json = True
    inspect_status = cmd_inspect(inspect_args)
    if inspect_status not in (0, 2):
        return inspect_status
    if not args.ja:
        _emit({"status": "blocked", "reason": "일본어 기준본이 필요합니다."}, args); return 2
    analyze_args = argparse.Namespace(**vars(args)); analyze_args.output = None; analyze_args.vendor_root = None; analyze_args.json = True
    analyze_status = cmd_analyze(analyze_args)
    _emit({"status": "ready_for_review", "workspace": str(workspace), "inspect_exit": inspect_status, "analyze_exit": analyze_status, "note": "의미 판정·번역·최종 패키징은 자동으로 만들지 않았습니다."}, args)
    return analyze_status


def cmd_process_title(args: argparse.Namespace) -> int:
    root = _project_root(args)
    config = ProcessTitleConfig(
        project_root=root,
        title_id=args.title,
        media=args.media,
        reference_ja=args.reference_ja,
        reference_ja_approved=args.reference_ja_approved,
        japanese_bundle=args.japanese_bundle,
        legacy_captures=args.legacy_captures,
        translation_policy=args.translation_policy,
        visual_policy=args.visual_policy,
        max_visual_units=args.max_visual_units,
        max_frames_per_unit=args.max_frames_per_unit,
        auto_capture_frames=args.auto_capture_frames,
        quality_policy=args.quality_policy,
        translation_batch_size=args.translation_batch_size,
        model_batch_workers=args.model_batch_workers,
        qwen_root=args.qwen_root,
        review_decisions=args.review_decisions,
        resume=args.resume,
        codex_timeout_seconds=args.codex_timeout,
        audit_attempt=args.audit_attempt,
        output_root=args.output_root,
    )
    if args.dry_run:
        try:
            config.validate()
        except ValueError as exc:
            _emit({"status": "fail", "error": str(exc)}, args)
            return 2
        _emit(
            {
                "status": "dry-run",
                "title_id": args.title,
                "media": str(args.media.expanduser().resolve()),
                "japanese_source": (
                    "existing-bundle" if args.japanese_bundle else
                    "approved-reference" if args.reference_ja else "ensemble"
                ),
                "translation_policy": args.translation_policy,
                "visual_policy": args.visual_policy,
                "max_visual_units": args.max_visual_units,
                "max_frames_per_unit": args.max_frames_per_unit,
                "external_image_transfer_authorized": args.visual_policy == "targeted",
                "final_promotion_allowed": False,
            },
            args,
        )
        return 0
    try:
        result = process_title(config)
    except CodexUsageLimitError as exc:
        _emit(
            {
                "status": "blocked",
                "reason": "codex-usage-limit",
                "role": exc.role,
                "call_id": exc.call_id,
                "retry_after": exc.retry_after,
                "error": str(exc),
            },
            args,
        )
        return 2
    except (CodexExecError, RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args)
        return 2
    _emit(result, args)
    return 0


def cmd_build_offline_hybrid(args: argparse.Namespace) -> int:
    build_offline_hybrid(args.titles_file, args.workspace_root, args.output)
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="translation-forensics", description="일본어 자막 복원·번역·검증 통합 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="공용 자료와 실행 환경 검사"); _add_common(p); p.set_defaults(func=cmd_doctor)
    p = sub.add_parser("init-title", help="작품별 작업 디렉터리 생성"); _add_common(p); _add_title(p); p.set_defaults(func=cmd_init_title)
    p = sub.add_parser("inspect", help="입력 역할과 SRT 구조 검사"); _add_common(p); _add_title(p); _input_args(p); p.set_defaults(func=cmd_inspect)
    p = sub.add_parser("validate-timeline", help="구조 SRT와 미디어의 초·중·후반 시간축 앵커 검증"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--audio", type=Path); p.add_argument("--video", type=Path); p.add_argument("--anchors", type=Path, help="anchor_id,srt_time_seconds,media_time_seconds,source CSV"); p.add_argument("--approved-offset-map", type=Path); p.add_argument("--tolerance", type=float, default=0.25); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_validate_timeline)
    p = sub.add_parser("init-timeline-anchors", help="사람 확인용 초·중·후반 시간축 앵커 템플릿 생성"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_init_timeline_anchors)
    p = sub.add_parser("build-pilot-alignment", help="사람 청취 앵커와 승인 offset map으로 파일럿 시간축 하드 게이트 생성"); _add_common(p); _add_title(p); p.add_argument("--structure", required=True, type=Path); p.add_argument("--audio", required=True, type=Path); p.add_argument("--anchors", required=True, type=Path); p.add_argument("--offset-map", required=True, type=Path); p.add_argument("--expected-blocks", type=int, default=298); p.add_argument("--tolerance", type=float, default=0.25); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_build_pilot_alignment)
    p = sub.add_parser("build-pilot-audio-review", help="resolved 시간축에서 전 블록 원음·일본어·장면 문맥 청취 패킷 생성"); _add_common(p); _add_title(p); p.add_argument("--structure", required=True, type=Path); p.add_argument("--ja", required=True, type=Path); p.add_argument("--audio", required=True, type=Path); p.add_argument("--alignment", required=True, type=Path); p.add_argument("--expected-blocks", type=int, default=298); p.add_argument("--context-before", type=float, default=2.0); p.add_argument("--context-after", type=float, default=2.0); p.add_argument("--context-radius", type=int, default=1); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_build_pilot_audio_review)
    p = sub.add_parser("build-pilot-candidate", help="Terra 결정, Sol 독립 비평·수리, 의미 재검사를 거친 SSIS-908 후보 생성"); _add_common(p); _add_title(p); p.add_argument("--structure", required=True, type=Path); p.add_argument("--baseline", required=True, type=Path); p.add_argument("--terra-decisions", required=True, type=Path); p.add_argument("--sol-reviews", required=True, type=Path); p.add_argument("--terra-prompt", required=True, type=Path); p.add_argument("--sol-prompt", required=True, type=Path); p.add_argument("--provenance-schema", required=True, type=Path); p.add_argument("--expected-blocks", type=int, default=298); p.add_argument("--expected-baseline-sha256", default="8653a42dc952152994c75e9d43265c49eddc1b7d581cf05d8012fd87e3c3e25b"); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_build_pilot_candidate)
    p = sub.add_parser("record-pilot-sol-reviews", help="완료된 독립 Sol 전 블록 비평·수리 계획을 검증 가능한 레코드로 기록"); _add_common(p); _add_title(p); p.add_argument("--terra-decisions", required=True, type=Path); p.add_argument("--review-plan", required=True, type=Path); p.add_argument("--expected-blocks", type=int, default=298); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_record_pilot_sol_reviews)
    p = sub.add_parser("validate-pilot-candidate", help="SSIS-908 후보 구조·해시·Terra/Sol/의미 재검사 계보 검증"); _add_common(p); _add_title(p); p.add_argument("--candidate", required=True, type=Path); p.add_argument("--structure", required=True, type=Path); p.add_argument("--baseline", required=True, type=Path); p.add_argument("--provenance-schema", required=True, type=Path); p.add_argument("--expected-blocks", type=int, default=298); p.add_argument("--expected-baseline-sha256", default="8653a42dc952152994c75e9d43265c49eddc1b7d581cf05d8012fd87e3c3e25b"); p.set_defaults(func=cmd_validate_pilot_candidate)
    p = sub.add_parser("validate-pilot-automated", help="SSIS-908 구조·문자·가독성·계보·근거·프롬프트·합성 회귀 통합 검증"); _add_common(p); _add_title(p); p.add_argument("--structure", required=True, type=Path); p.add_argument("--baseline", required=True, type=Path); p.add_argument("--candidate", required=True, type=Path); p.add_argument("--provenance-schema", required=True, type=Path); p.add_argument("--terra-manifest", required=True, type=Path); p.add_argument("--autonomous-manifest", required=True, type=Path); p.add_argument("--regressions", required=True, type=Path); p.add_argument("--report-schema", required=True, type=Path); p.add_argument("--expected-blocks", type=int, default=298); p.add_argument("--expected-baseline-sha256", default="8653a42dc952152994c75e9d43265c49eddc1b7d581cf05d8012fd87e3c3e25b"); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_validate_pilot_automated)
    p = sub.add_parser("init-pilot-blind-key", help="후보 생성 전에 평가 계약·MQM·무작위 배치 내부 키 봉인"); _add_common(p); _add_title(p); p.add_argument("--baseline", required=True, type=Path); p.add_argument("--candidate-expected", required=True, type=Path); p.add_argument("--evaluation-contract", required=True, type=Path); p.add_argument("--mqm-schema", required=True, type=Path); p.add_argument("--random-seed", required=True); p.add_argument("--expected-blocks", type=int, default=298); p.add_argument("--expected-baseline-sha256", default="8653a42dc952152994c75e9d43265c49eddc1b7d581cf05d8012fd87e3c3e25b"); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_init_pilot_blind_key)
    p = sub.add_parser("build-pilot-blind-review", help="resolved 원음 패킷에서 두 독립 검수자용 동일 배치 A/B 패킷 생성"); _add_common(p); _add_title(p); p.add_argument("--structure", required=True, type=Path); p.add_argument("--baseline", required=True, type=Path); p.add_argument("--candidate", required=True, type=Path); p.add_argument("--candidate-provenance", required=True, type=Path); p.add_argument("--audio-review-manifest", required=True, type=Path); p.add_argument("--evaluation-contract", required=True, type=Path); p.add_argument("--mqm-schema", required=True, type=Path); p.add_argument("--internal-key", required=True, type=Path); p.add_argument("--random-seed", required=True); p.add_argument("--expected-blocks", type=int, default=298); p.add_argument("--expected-baseline-sha256", default="8653a42dc952152994c75e9d43265c49eddc1b7d581cf05d8012fd87e3c3e25b"); p.add_argument("--adjudication-output", required=True, type=Path); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_build_pilot_blind_review)
    p = sub.add_parser("validate-pilot-reviewer-submission", help="검수자 전 블록 직접 청취·독립성·MQM 제출 완전성 검사"); _add_common(p); p.add_argument("--pack", required=True, type=Path); p.add_argument("--attestation", required=True, type=Path); p.add_argument("--decisions", required=True, type=Path); p.add_argument("--expected-blocks", type=int, default=298); p.set_defaults(func=cmd_validate_pilot_reviewer_submission)
    p = sub.add_parser("adjudicate-pilot-reviews", help="두 독립 제출의 불일치 합의를 검증하고 봉인 키로 최종 결정 기록"); _add_common(p); p.add_argument("--reviewer-1-pack", required=True, type=Path); p.add_argument("--reviewer-1-attestation", required=True, type=Path); p.add_argument("--reviewer-1-decisions", required=True, type=Path); p.add_argument("--reviewer-2-pack", required=True, type=Path); p.add_argument("--reviewer-2-attestation", required=True, type=Path); p.add_argument("--reviewer-2-decisions", required=True, type=Path); p.add_argument("--internal-key", required=True, type=Path); p.add_argument("--consensus", required=True, type=Path); p.add_argument("--expected-blocks", type=int, default=298); p.add_argument("--expected-baseline-sha256", default="8653a42dc952152994c75e9d43265c49eddc1b7d581cf05d8012fd87e3c3e25b"); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_adjudicate_pilot_reviews)
    p = sub.add_parser("evaluate-pilot-new-critical", help="최종 조정 MQM에서 개선본에 새로 생긴 critical 의미·화행 오류 0건 게이트 판정"); _add_common(p); p.add_argument("--adjudication", required=True, type=Path); p.add_argument("--expected-blocks", type=int, default=298); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_evaluate_pilot_new_critical)
    p = sub.add_parser("evaluate-pilot-error-reduction", help="기준본 대비 critical/major 의미·맥락 오류 고유 블록 50%% 감소 게이트 판정"); _add_common(p); p.add_argument("--adjudication", required=True, type=Path); p.add_argument("--expected-blocks", type=int, default=298); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_evaluate_pilot_error_reduction)
    p = sub.add_parser("evaluate-pilot-naturalness", help="동률 포함 전체 조정 블록 기준 개선본 자연스러움 승률 65%%·패배율 15%% 게이트 판정"); _add_common(p); p.add_argument("--adjudication", required=True, type=Path); p.add_argument("--expected-blocks", type=int, default=298); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_evaluate_pilot_naturalness)
    p = sub.add_parser("analyze", help="Subtitle Forensics 실행 또는 구조 기반 검토 큐 생성"); _add_common(p); _add_title(p); p.add_argument("--ja", type=Path); p.add_argument("--previous-ko", type=Path); p.add_argument("--output", type=Path); p.add_argument("--vendor-root", type=Path); p.set_defaults(func=cmd_analyze)
    p = sub.add_parser("build-korean-draft", help="일본어 구조에 맞춘 한국어 번역 초안 생성"); _add_common(p); _add_title(p); p.add_argument("--ja", type=Path); p.add_argument("--previous-ko", type=Path); p.add_argument("--output", type=Path); p.add_argument("--report", type=Path); p.set_defaults(func=cmd_build_korean_draft)
    p = sub.add_parser("build-automatic-draft", help="기존 자동 후보와 정렬 fallback으로 전 블록 재생용 초안 생성"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--source-candidate", type=Path); p.add_argument("--viewer-candidate", type=Path); p.add_argument("--single-candidate", type=Path); p.add_argument("--fallback", type=Path); p.add_argument("--decision-candidate", type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_build_automatic_draft)
    p = sub.add_parser("build-translation-queue", help="블록별 의미 번역 결정 큐 생성"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--ja", type=Path); p.add_argument("--previous-ko", type=Path); p.add_argument("--photos", type=Path); p.add_argument("--review-context", type=Path); p.add_argument("--review-queue", type=Path); p.add_argument("--capture-index", type=Path); p.add_argument("--consistency-ledger", type=Path, help="confirmed 장편 말투·호칭·이전 선택 문맥 원장"); p.add_argument("--terminology", type=Path, help="기존 승인 용어집; 파일 복사 없이 queue 문맥으로 어댑터 연결"); p.add_argument("--speaker-state", type=Path, help="scene별 검수된 화자·상대 문맥"); p.add_argument("--output", type=Path); p.add_argument("--report", type=Path); p.add_argument("--translation-model", default=DEFAULT_TRANSLATION_MODEL, choices=ALLOWED_TRANSLATION_MODELS, help="한국어 의미 번역 모델(고정값)"); p.set_defaults(func=cmd_build_translation_queue)
    p = sub.add_parser("apply-translations", help="번역 결정 JSONL을 구조 고정 SRT 두 종으로 적용"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--decisions", required=True, type=Path); p.add_argument("--translation-queue", type=Path, help="--strict에서 decision evidence_refs를 대조할 원본 번역 큐"); p.add_argument("--source-output", type=Path); p.add_argument("--viewer-output", type=Path); p.add_argument("--report", type=Path); p.add_argument("--strict", action="store_true", default=False); p.add_argument("--translation-model", default=DEFAULT_TRANSLATION_MODEL, choices=ALLOWED_TRANSLATION_MODELS, help="한국어 의미 번역 모델(고정값)"); p.set_defaults(func=cmd_apply_translations)
    p = sub.add_parser("init-translation-decisions", help="번역 결정 템플릿 생성"); _add_common(p); _add_title(p); p.add_argument("--queue", type=Path); p.add_argument("--output", type=Path); p.add_argument("--translation-model", default=DEFAULT_TRANSLATION_MODEL, choices=ALLOWED_TRANSLATION_MODELS, help="한국어 의미 번역 모델(고정값)"); p.set_defaults(func=cmd_init_translation_decisions)
    p = sub.add_parser("init-consistency-ledger", help="사람 검수형 장편 말투·호칭·용어 원장 생성"); _add_common(p); _add_title(p); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_init_consistency_ledger)
    p = sub.add_parser("validate-consistency-ledger", help="장편 일관성 원장의 형식·충돌 보고 검증"); _add_common(p); p.add_argument("--input", required=True, type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_validate_consistency_ledger)
    p = sub.add_parser("validate-quality-regressions", help="합성 일본어→한국어 품질 회귀 제약 묶음 검증"); _add_common(p); p.add_argument("--input", required=True, type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_validate_quality_regressions)
    p = sub.add_parser("build-uncertainty-review-queue", help="결정·포렌식·자동 QA 이유를 묶은 투명한 검수 우선순위 큐 생성"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--decisions", type=Path); p.add_argument("--translation-queue", type=Path); p.add_argument("--forensics-queue", type=Path); p.add_argument("--source-faithful", type=Path); p.add_argument("--viewer-natural", type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_build_uncertainty_review_queue)
    p = sub.add_parser("merge-translation-decisions", help="검수 완료 블록을 결정 템플릿에 병합"); _add_common(p); p.add_argument("--base", required=True, type=Path); p.add_argument("--reviewed", required=True, type=Path); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_merge_translation_decisions)
    p = sub.add_parser("init-forensic-records", help="의미 프레임·가설 원장을 명시적 미검수 상태로 초기화"); _add_common(p); _add_title(p); p.add_argument("--queue", type=Path); p.add_argument("--frames", type=Path); p.add_argument("--hypotheses", type=Path); p.set_defaults(func=cmd_init_forensic_records)
    p = sub.add_parser("validate-forensic-records", help="의미 프레임·가설 원장과 critical conflict escalation 검사"); _add_common(p); p.add_argument("--frames", required=True, type=Path); p.add_argument("--hypotheses", required=True, type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_validate_forensic_records)
    p = sub.add_parser("init-gold", help="정답을 만들지 않는 gold evaluation 디렉터리 골격 생성"); _add_common(p); p.set_defaults(func=cmd_init_gold)
    p = sub.add_parser("init-gold-record", help="사람 청취용 미완성 gold 레코드 템플릿 생성"); _add_common(p); p.add_argument("--gold-id", required=True); p.add_argument("--title-id", required=True); p.add_argument("--scene-id", required=True); p.add_argument("--block-ids", required=True, help="쉼표로 구분한 구조 블록 번호"); p.add_argument("--split", required=True, choices=("train", "development", "locked-test")); p.set_defaults(func=cmd_init_gold_record)
    p = sub.add_parser("validate-gold-record", help="사람 작성 gold 레코드의 직접 청취·분할 계약 검증"); _add_common(p); p.add_argument("--input", required=True, type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_validate_gold_record)
    p = sub.add_parser("validate-gold-suite", help="작품 단위 gold split 누출과 골드 레코드 계약 검증"); _add_common(p); p.set_defaults(func=cmd_validate_gold_suite)
    p = sub.add_parser("migrate-gold-layout", help="기존 gold 골격을 release/sealed split 구조로 비파괴 마이그레이션"); _add_common(p); p.set_defaults(func=cmd_migrate_gold_layout)
    p = sub.add_parser("build-blind-review-pack", help="출처를 숨긴 로컬 A/B 검수 패킷 생성"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--source-faithful", required=True, type=Path); p.add_argument("--viewer-natural", required=True, type=Path); p.add_argument("--output", type=Path); p.add_argument("--random-seed", type=int, required=True); p.set_defaults(func=cmd_build_blind_review_pack)
    p = sub.add_parser("summarize-blind-review", help="완료된 사람 A/B 판정을 내부 키로 해제·요약"); _add_common(p); p.add_argument("--pack", required=True, type=Path); p.add_argument("--reviewed", required=True, type=Path); p.add_argument("--key", required=True, type=Path); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_summarize_blind_review)
    p = sub.add_parser("validate-evaluation-summary", help="사람 검수 완료 평가 요약의 승격 계약 검증"); _add_common(p); p.add_argument("--input", required=True, type=Path); p.set_defaults(func=cmd_validate_evaluation_summary)
    p = sub.add_parser("create-run-manifest", help="입력·프롬프트·산출물 해시를 묶은 재현 실행 기록 생성"); _add_common(p); p.add_argument("--title", required=True); p.add_argument("--stage", required=True); p.add_argument("--inputs", required=True, nargs="+", type=Path); p.add_argument("--artifacts", nargs="*", default=[], type=Path); p.add_argument("--prompt-manifest", type=Path); p.add_argument("--parent-run-id"); p.add_argument("--run-id"); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_create_run_manifest)
    p = sub.add_parser("init-reverse-check", help="두 한국어안의 사람 작성 역의미 검증 템플릿 생성"); _add_common(p); p.add_argument("--decisions", required=True, type=Path); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_init_reverse_check)
    p = sub.add_parser("validate-reverse-check", help="역의미 검증의 누락·의미 반전·추가를 차단"); _add_common(p); p.add_argument("--decisions", required=True, type=Path); p.add_argument("--input", required=True, type=Path); p.set_defaults(func=cmd_validate_reverse_check)
    p = sub.add_parser("migrate-identity", help="기존 JSONL을 안정 ID·스키마 v2 산출물로 비파괴 마이그레이션"); _add_common(p); p.add_argument("--title", required=True); p.add_argument("--kind", required=True, choices=("queue", "decision", "semantic-frame", "hypothesis")); p.add_argument("--input", required=True, type=Path); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_migrate_identity)
    p = sub.add_parser("validate-identity", help="JSONL의 title/block 안정 ID와 참조 일관성 검사"); _add_common(p); p.add_argument("--title"); p.add_argument("--input", required=True, type=Path); p.set_defaults(func=cmd_validate_identity)
    p = sub.add_parser("init-memory-ledger", help="자동 적용을 하지 않는 장면·번역·오류·표현 정책 메모리 생성"); _add_common(p); p.add_argument("--kind", required=True, choices=("scene", "translation", "error", "style-policy")); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_init_memory_ledger)
    p = sub.add_parser("validate-memory-ledger", help="메모리 범위·승인·근거 및 자동 적용 금지 검증"); _add_common(p); p.add_argument("--kind", required=True, choices=("scene", "translation", "error", "style-policy")); p.add_argument("--input", required=True, type=Path); p.set_defaults(func=cmd_validate_memory_ledger)
    p = sub.add_parser("init-speaker-state", help="검수자 작성 화자·행동 상태 템플릿 생성"); _add_common(p); p.add_argument("--scenes", required=True, type=Path); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_init_speaker_state)
    p = sub.add_parser("init-phonetic-candidates", help="근거 없는 음가 추정을 만들지 않는 후보 원장 템플릿 생성"); _add_common(p); p.add_argument("--queue", required=True, type=Path); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_init_phonetic_candidates)
    p = sub.add_parser("analyze-slot-conflicts", help="경쟁 가설의 명시적 의미 슬롯 충돌 보고서 생성"); _add_common(p); p.add_argument("--hypotheses", required=True, type=Path); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_analyze_slot_conflicts)
    p = sub.add_parser("evaluate-review-budget", help="완전한 사람 라벨·골드 분모가 있을 때만 검수 예산별 recall/precision 계산"); _add_common(p); p.add_argument("--input", required=True, type=Path, help="record_id/rank/reviewer_label JSONL"); p.add_argument("--gold-positive-count", required=True, type=int); p.add_argument("--budgets", nargs="+", type=int, default=[5, 10, 20]); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_evaluate_review_budget)
    p = sub.add_parser("validate-release-gate", help="골드·블라인드 검수·감사·지표·책임자 기반 릴리스 게이트 검증"); _add_common(p); p.add_argument("--gold-suite", type=Path); p.add_argument("--blind-review", type=Path); p.add_argument("--audit-summary", type=Path); p.add_argument("--review-metrics", type=Path); p.add_argument("--release-approval", type=Path); p.add_argument("--sol-experiment", type=Path); p.add_argument("--require-sol-experiment", action="store_true"); p.add_argument("--no-human-approval-required", action="store_true"); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_validate_release_gate)
    p = sub.add_parser("init-sol-review", help="외부 호출 없이 선택적 Sol 반증 검토 원장 템플릿 생성"); _add_common(p); p.add_argument("--title", required=True); p.add_argument("--queue", required=True, type=Path); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_init_sol_review)
    p = sub.add_parser("validate-sol-review", help="Sol 반증 기록의 승인·비용·사람 판정·자동 최종판정 금지 검증"); _add_common(p); p.add_argument("--input", required=True, type=Path); p.set_defaults(func=cmd_validate_sol_review)
    p = sub.add_parser("audit-current", help="작품별 입력·시간축·결정·골드·최종 산출물 완료 요건 감사"); _add_common(p); _add_title(p); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_audit_current)
    p = sub.add_parser("validate-mqm", help="subtitle MQM 오류 원장 형식과 critical 잔존 여부 검사"); _add_common(p); p.add_argument("--input", required=True, type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_validate_mqm)
    p = sub.add_parser("validate-evidence-artifact", help="reviewer-authored evidence artifact schema validation"); _add_common(p); p.add_argument("--kind", required=True, choices=("speaker-state", "alignment-evidence", "backtranslation-check")); p.add_argument("--input", required=True, type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_validate_evidence_artifact)
    p = sub.add_parser("validate-prompt-contract", help="Terra 번역 프롬프트 템플릿 계약·시험 사례 검증"); _add_common(p); p.add_argument("--manifest", required=True, type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_validate_prompt_contract)
    p = sub.add_parser("prepare-audio", help="P1/P2 장면과 음성 클립 준비"); _add_common(p); _add_title(p); p.add_argument("--review-queue", type=Path); p.add_argument("--audio", type=Path); p.add_argument("--output", type=Path); p.add_argument("--bands", default="P1,P2"); p.add_argument("--padding", type=float, default=2.5); p.add_argument("--merge-gap", type=float, default=1.5); p.add_argument("--max-scene", type=float, default=45.0); p.add_argument("--no-audio", action="store_true"); p.add_argument("--timeline-report", type=Path); p.add_argument("--allow-unvalidated-timeline", action="store_true", help="예외적인 기존 작업의 명시적 우회; 결과를 시간축 검증됨으로 표기하지 않음"); p.add_argument("--force", action="store_true"); p.set_defaults(func=cmd_prepare_audio)
    p = sub.add_parser("run-asr", help="적응형 다중 ASR 실행"); _add_common(p); _add_title(p); p.add_argument("--scenes", type=Path); p.add_argument("--output", type=Path); p.add_argument("--prompt", type=Path); p.add_argument("--model", default="large-v3"); p.add_argument("--cpu", action="store_true"); p.add_argument("--offline", action="store_true", help="캐시된 ASR 모델만 사용하고 네트워크 다운로드를 차단"); p.add_argument("--threshold", type=float, default=0.82); p.add_argument("--max-scenes", type=int, default=0); p.set_defaults(func=cmd_run_asr)
    p = sub.add_parser("ingest-asr", help="ASR CSV/ZIP 검사 및 연결"); _add_common(p); _add_title(p); p.add_argument("--source", required=True, type=Path); p.add_argument("--output", type=Path); p.add_argument("--force", action="store_true"); p.set_defaults(func=cmd_ingest_asr)
    p = sub.add_parser("package-audio", help="완료된 음성 검토 작업을 work_audio ZIP으로 패키징"); _add_common(p); _add_title(p); p.add_argument("--work-dir", type=Path); p.add_argument("--output", type=Path); p.add_argument("--model", default="large-v3"); p.add_argument("--no-clips", action="store_true"); p.set_defaults(func=cmd_package_audio)
    p = sub.add_parser("build-review-context", help="장면 검토 패킷 생성"); _add_common(p); _add_title(p); p.add_argument("--ja", type=Path); p.add_argument("--previous-ko", type=Path); p.add_argument("--scenes", type=Path); p.add_argument("--asr", type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_build_review_context)
    p = sub.add_parser("build-review-pack", help="한 화면의 로컬 검수 HTML·결정 템플릿 생성"); _add_common(p); _add_title(p); p.add_argument("--context", type=Path); p.add_argument("--audio-root", type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_build_review_pack)
    p = sub.add_parser("validate-review-decisions", help="검수 결정의 블록 커버리지·근거 연결·상태 검증"); _add_common(p); p.add_argument("--context", required=True, type=Path); p.add_argument("--input", required=True, type=Path); p.set_defaults(func=cmd_validate_review_decisions)
    p = sub.add_parser("sample-audit", help="P3/P4 등의 시드 고정 무작위 감사 표본 생성"); _add_common(p); _add_title(p); p.add_argument("--queue", type=Path); p.add_argument("--output", type=Path); p.add_argument("--rate", required=True, type=float); p.add_argument("--bands", default="P3,P4"); p.add_argument("--seed", required=True, type=int); p.set_defaults(func=cmd_sample_audit)
    p = sub.add_parser("summarize-audit", help="사람이 채운 무작위 감사 표본의 기술 요약 생성"); _add_common(p); p.add_argument("--input", required=True, type=Path); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_summarize_audit)
    p = sub.add_parser("init-terminology", help="자동 치환을 하지 않는 승인형 용어집 생성"); _add_common(p); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_init_terminology)
    p = sub.add_parser("validate-terminology", help="승인형 용어집의 범위·근거 계약 검증"); _add_common(p); p.add_argument("--input", required=True, type=Path); p.set_defaults(func=cmd_validate_terminology)
    p = sub.add_parser("terminology-conflicts", help="같은 범위의 승인 용어 충돌 보고서 생성"); _add_common(p); p.add_argument("--input", required=True, type=Path); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_terminology_conflicts)
    p = sub.add_parser("build-evidence-graph", help="블록·근거·결정·가설·평가의 역추적 그래프 생성"); _add_common(p); _add_title(p); p.add_argument("--queue", type=Path); p.add_argument("--decisions", type=Path); p.add_argument("--frames", type=Path); p.add_argument("--hypotheses", type=Path); p.add_argument("--evaluation", type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_build_evidence_graph)
    p = sub.add_parser("validate", help="최종 SRT 구조·문자·가독성·회귀 검사"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--source-faithful", type=Path); p.add_argument("--viewer-natural", type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_validate)
    p = sub.add_parser("run-closed-world", help="모든 블록을 accepted/abstained로 판정하고 별도 패키징"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--ja", type=Path); p.add_argument("--previous-ko", type=Path); p.add_argument("--candidate-srt", action="append", type=Path, help="검증할 로컬 한국어 후보 SRT. 여러 번 지정할 수 있습니다."); p.add_argument("--review-context", type=Path); p.add_argument("--asr", type=Path); p.add_argument("--asr-model", default="large-v3", help="faster-whisper 모델 이름"); p.add_argument("--allow-model-download", action="store_true", help="ASR CSV가 없을 때 모델 다운로드를 명시적으로 허용하고 network-enabled 패키지에 기록"); p.add_argument("--cpu", action="store_true"); p.add_argument("--max-scenes", type=int, default=0); p.add_argument("--timeline-validation", type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_run_closed_world)
    p = sub.add_parser("validate-closed-world", help="폐쇄형 패키지의 결정 커버리지·구조·해시·승격 차단 검증"); _add_common(p); p.add_argument("--package", required=True, type=Path); p.add_argument("--structure", type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_validate_closed_world)
    p = sub.add_parser("prove-quality-claim", help="폐쇄형 자동 검증의 보장 범위와 사람 정답 동일성 비식별 상태 보고"); _add_common(p); p.add_argument("--package", required=True, type=Path); p.add_argument("--structure", type=Path); p.add_argument("--human-reference", type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_prove_quality_claim)
    p = sub.add_parser("package-machine-final", help="사람 final과 분리된 machine-final 패키지 생성"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--source-faithful", type=Path); p.add_argument("--viewer-natural", type=Path); p.add_argument("--automatic-report", type=Path); p.add_argument("--automatic-ledger", type=Path); p.add_argument("--asr", type=Path); p.add_argument("--timeline-validation", type=Path); p.add_argument("--closed-world-proof", type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_package_machine_final)
    p = sub.add_parser("repair-machine-final-asr", help="손상된 canonical ASR 복구 및 Excel UTF-8 BOM 보기용 CSV 생성"); _add_common(p); p.add_argument("--source-asr", required=True, type=Path); p.add_argument("--package", required=True, type=Path); p.set_defaults(func=cmd_repair_machine_final_asr)
    p = sub.add_parser("apply-targeted-retranslations", help="구조화된 자동 재번역 후보를 v2 자막에 적용"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--source-faithful", required=True, type=Path); p.add_argument("--viewer-natural", required=True, type=Path); p.add_argument("--response", required=True, type=Path, action="append"); p.add_argument("--review", type=Path, action="append"); p.add_argument("--output", type=Path); p.add_argument("--version", default="v2"); p.add_argument("--hold-marker", default="…"); p.set_defaults(func=cmd_apply_targeted_retranslations)
    p = sub.add_parser("prepare-inference-audio", help="v3 보류 블록을 블록별 로컬 ASR 음성으로 추출"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--audio", type=Path); p.add_argument("--hold-ledger", required=True, type=Path); p.add_argument("--output", type=Path); p.add_argument("--padding", type=float, default=0.25); p.add_argument("--merge-gap", type=float, default=-10.0); p.add_argument("--max-scene", type=float, default=60.0); p.set_defaults(func=cmd_prepare_inference_audio)
    p = sub.add_parser("build-inference-context", help="블록별 ASR·문맥을 추론 복구 모델 입력으로 고정"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--source-faithful", required=True, type=Path); p.add_argument("--viewer-natural", required=True, type=Path); p.add_argument("--hold-ledger", required=True, type=Path); p.add_argument("--scenes", required=True, type=Path); p.add_argument("--asr", required=True, type=Path); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_build_inference_context)
    p = sub.add_parser("apply-inferred-recovery", help="사용자 승인 음성 추론 복구본을 v4 SRT에 적용"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--source-faithful", required=True, type=Path); p.add_argument("--viewer-natural", required=True, type=Path); p.add_argument("--hold-ledger", required=True, type=Path); p.add_argument("--response", required=True, type=Path, action="append"); p.add_argument("--output", type=Path); p.add_argument("--version", default="v4"); p.add_argument("--hold-marker", default="…"); p.set_defaults(func=cmd_apply_inferred_recovery)
    p = sub.add_parser("run-autonomous-release", help="사람 final과 분리된 하이브리드 무인 번역·반증·패키징 실행"); _add_common(p); p.add_argument("--title", action="append"); p.add_argument("--titles-file", type=Path); p.add_argument("--structure", type=Path); p.add_argument("--ja", type=Path); p.add_argument("--previous-ko", type=Path); p.add_argument("--scenes", type=Path); p.add_argument("--local-asr", type=Path); p.add_argument("--local-asr-model", default="large-v3"); p.add_argument("--cpu", action="store_true"); p.add_argument("--allow-local-model-download", action="store_true"); p.add_argument("--output", type=Path); p.add_argument("--allow-network", action="store_true"); p.add_argument("--max-cost-usd", type=float); p.add_argument("--cache-dir", type=Path); p.add_argument("--resume", action="store_true"); p.add_argument("--max-workers", type=int, default=2); p.add_argument("--batch-size", type=int, default=20); p.add_argument("--max-repairs", type=int, default=2); p.set_defaults(func=cmd_run_autonomous_release)
    p = sub.add_parser("audit-source-quality", help="일본어 SRT의 구조와 텍스트 신뢰도를 분리해 trusted/suspect/unusable 지도를 생성"); _add_common(p); _add_title(p); p.add_argument("--ja", type=Path); p.add_argument("--asr-evidence", type=Path); p.add_argument("--output", type=Path); p.add_argument("--resume", action="store_true"); p.set_defaults(func=cmd_audit_source_quality)
    p = sub.add_parser("run-codex-quality", help="Terra 생성·Sol 독립 반증 기반 autonomous-quality-candidate 실행"); _add_common(p); p.add_argument("--titles", required=True, help="쉼표로 구분한 작품 ID"); p.add_argument("--full-local-asr", action="store_true"); p.add_argument("--repair-evidence", action="store_true", help="초기 evidence ceiling 탈락 블록에 한해 native timestamp ASR 복구 후 ceiling 재평가"); p.add_argument("--focused-repair", action="store_true", help="--repair-evidence와 함께 4~8초 블록 중심 native ASR 복구를 사용"); p.add_argument("--partial-evidence-evaluation", action="store_true", help="title gate가 실패해도 증거 통과 블록만 진단 실행; promotion은 항상 차단"); p.add_argument("--resume", action="store_true"); p.add_argument("--offline", action="store_true", help="로컬 ASR 모델 다운로드 금지"); p.add_argument("--cpu", action="store_true"); p.add_argument("--max-windows", type=int, default=0, help="개발용 ASR 창 제한; 0은 전체"); p.add_argument("--max-scene-blocks", type=int, default=60); p.add_argument("--max-scene-gap", type=float, default=60.0); p.add_argument("--max-repairs", type=int, default=2); p.add_argument("--codex-timeout", type=int, default=600); p.add_argument("--cache-dir", type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_run_codex_quality)
    p = sub.add_parser("validate-codex-quality", help="autonomous-quality-candidate의 구조·근거·모델 분리·충돌·해시 게이트 검증"); _add_common(p); p.add_argument("--package", required=True, type=Path); p.set_defaults(func=cmd_validate_codex_quality)
    p = sub.add_parser("evaluate-codex-quality", help="고정 시드 120블록을 Terra/Sol 익명 A/B 대리평가"); _add_common(p); p.add_argument("--package", required=True, type=Path); p.add_argument("--baseline", required=True, type=Path); p.add_argument("--sample-size", type=int, default=120); p.add_argument("--batch-size", type=int, default=20); p.add_argument("--resume", action="store_true"); p.add_argument("--codex-timeout", type=int, default=600); p.add_argument("--cache-dir", type=Path); p.set_defaults(func=cmd_evaluate_codex_quality)
    p = sub.add_parser("validate-autonomous-release", help="autonomous-release 구조·커버리지·해시·사람 final 경계 검증"); _add_common(p); p.add_argument("--package", required=True, type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_validate_autonomous_release)
    p = sub.add_parser("prove-autonomous-claim", help="autonomous-release가 보장하는 속성과 식별 불가능한 사람 정답 주장을 분리"); _add_common(p); p.add_argument("--package", required=True, type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_prove_autonomous_claim)
    p = sub.add_parser("package", help="새 버전으로 최종 산출물 패키징"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--ja", type=Path, required=True); p.add_argument("--photos", type=Path); p.add_argument("--source-faithful", required=True, type=Path); p.add_argument("--viewer-natural", required=True, type=Path); p.add_argument("--previous-ko", type=Path); p.add_argument("--scenes", type=Path); p.add_argument("--asr", type=Path); p.add_argument("--decisions", type=Path); p.add_argument("--translation-queue", type=Path, help="결정의 evidence_refs를 대조할 원본 번역 큐"); p.add_argument("--semantic-frames", type=Path); p.add_argument("--hypothesis-ledger", type=Path); p.add_argument("--speaker-state", type=Path); p.add_argument("--alignment-evidence", type=Path); p.add_argument("--mqm-errors", type=Path); p.add_argument("--backtranslation-check", type=Path); p.add_argument("--evaluation-summary", type=Path); p.add_argument("--blind-review-pack", type=Path); p.add_argument("--release-gate", type=Path); p.add_argument("--timeline-validation", type=Path); p.add_argument("--output", type=Path); p.add_argument("--stage", choices=STAGES, default="text-crosschecked"); p.add_argument("--version", type=int); p.add_argument("--all-blocks-reviewed", action="store_true"); p.add_argument("--direct-human-listening", action="store_true"); p.add_argument("--evidence-complete", action="store_true"); p.set_defaults(func=cmd_package)
    p = sub.add_parser("run", help="결정적 단계만 수행하고 의미 판정 전 중단"); _add_common(p); _add_title(p); _input_args(p); p.set_defaults(func=cmd_run)
    p = sub.add_parser("process-title", help="영상에서 일본어 원문을 복원하고 Terra/Sol 한국어 이중 산출을 패키징")
    _add_common(p)
    _add_title(p)
    p.add_argument("--media", required=True, type=Path, help="작품 코드가 포함된 원본 영상")
    p.add_argument("--reference-ja", type=Path, help="승인할 일본어 참조 SRT")
    p.add_argument("--reference-ja-approved", action="store_true", help="참조 SRT를 일본어 원문 근거로 명시 승인")
    p.add_argument("--japanese-bundle", type=Path, help="검증된 기존 일본어 자막 번들 디렉터리")
    p.add_argument("--legacy-captures", type=Path, help="timestamp가 파일명에 포함된 기존 프레임 디렉터리")
    p.add_argument("--translation-policy", choices=("dual",), default="dual")
    p.add_argument("--visual-policy", choices=("off", "metadata", "targeted"), default="targeted")
    p.add_argument("--max-visual-units", type=int, default=20)
    p.add_argument("--max-frames-per-unit", type=int, default=3)
    p.add_argument(
        "--auto-capture-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="사진이 없으면 모호한 번역 단위의 대표 프레임을 영상에서 자동 추출 (기본값: 활성)",
    )
    p.add_argument(
        "--quality-policy",
        choices=("automated", "legacy"),
        default="automated",
        help="사람 승인 없이 자동 품질 게이트를 사용 (기본값: automated)",
    )
    p.add_argument("--translation-batch-size", type=int, default=40)
    p.add_argument("--model-batch-workers", type=int, default=1, help="동시에 실행할 독립 모델 배치 수 (기본값: 1)")
    p.add_argument("--qwen-root", type=Path, help="Qwen3ASR 런타임 루트; 환경변수도 지원")
    p.add_argument("--review-decisions", type=Path, help="사람 검수 결정 JSONL")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--codex-timeout", type=int, default=600)
    p.add_argument(
        "--audit-attempt",
        type=int,
        default=0,
        help="사용량 제한·timeout 뒤 Sol 감사를 새 immutable run으로 재시도할 번호",
    )
    p.add_argument("--output-root", type=Path, help="기본값: workspaces/<TITLE>/integrated")
    p.set_defaults(func=cmd_process_title)
    p = sub.add_parser("build-offline-hybrid", help="API 호출 없이 로컬 결과물을 병합한 재생용 미리보기 생성 (human final 아님)")
    _add_common(p)
    p.add_argument("--titles-file", type=Path, required=True, help="작품 목록 텍스트 파일 경로")
    p.add_argument("--workspace-root", type=Path, default=Path("workspaces"), help="워크스페이스 루트 경로")
    p.add_argument("--output", type=Path, required=True, help="출력 디렉터리 경로")
    p.set_defaults(func=cmd_build_offline_hybrid)

    p = sub.add_parser(
        "evaluate-pilot-balance-contract",
        help="세 의미·자연스러움 하드 게이트를 결합해 SSIS-908 파일럿 판정",
    )
    _add_common(p)
    p.add_argument("--adjudication", required=True, type=Path)
    p.add_argument("--expected-blocks", type=int, default=298)
    p.add_argument("--metrics-output", required=True, type=Path)
    p.add_argument("--error-ledger-output", required=True, type=Path)
    p.set_defaults(func=cmd_evaluate_pilot_balance_contract)

    p = sub.add_parser(
        "build-pilot-evaluation-report",
        help="SSIS-908 파일럿 증거 해시·판정·주장 경계를 담은 보고서와 매니페스트 생성",
    )
    _add_common(p)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--candidate-source", required=True, type=Path)
    p.add_argument("--candidate-viewer", required=True, type=Path)
    p.add_argument("--candidate-provenance", required=True, type=Path)
    p.add_argument("--timeline-alignment", required=True, type=Path)
    p.add_argument("--audio-review-manifest", required=True, type=Path)
    p.add_argument("--blind-review-packet", required=True, action="append", type=Path)
    p.add_argument("--internal-key", required=True, type=Path)
    p.add_argument("--reviewer-attestation", action="append", default=[], type=Path)
    p.add_argument("--reviewer-decisions", action="append", default=[], type=Path)
    p.add_argument("--adjudication", required=True, type=Path)
    p.add_argument("--metrics", required=True, type=Path)
    p.add_argument("--error-ledger", required=True, type=Path)
    p.add_argument("--automated-validation", required=True, type=Path)
    p.add_argument("--expected-blocks", type=int, default=298)
    p.add_argument("--expected-baseline-sha256", default=PILOT_BASELINE_SHA256)
    p.add_argument("--manifest-output", required=True, type=Path)
    p.add_argument("--report-output", required=True, type=Path)
    p.set_defaults(func=cmd_build_pilot_evaluation_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    try:
        return int(args.func(args))
    except (DiscoveryError, ValueError, OSError) as exc:
        if getattr(args, "json", False):
            print(json.dumps({"status": "fail", "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"오류: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
