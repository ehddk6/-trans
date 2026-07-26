from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .audio_adapter import detect_device, ingest_asr, prepare_audio, run_asr, write_run_summary, zip_work_audio
from .alignment import align_to_dicts
from .asr_evidence import read_asr_candidates
from .discovery import DiscoveryError, inspect_roles, resolve_role
from .drafts import build_korean_aligned_draft
from .forensics_adapter import analyze_title
from .forensic_model import initialize_forensic_records, validate_forensic_records
from .evaluation import initialize_gold_layout
from .evidence_artifacts import validate_alignment_evidence, validate_backtranslation_check, validate_speaker_state
from .mqm import validate_mqm_csv
from .manifest import append_history, build_project_manifest, write_json
from .outputs import STAGES, package_title_outputs
from .reporting import build_asr_verdicts, build_review_context, build_scene_map, write_csv
from .semantic_translation import apply_translation_decisions, build_translation_queue, initialize_translation_decisions, merge_translation_decisions
from .scenes import build_review_scenes, read_review_queue, write_review_scenes
from .srt import compare_structure, parse_srt
from .validation import validate_pair, write_validation_report


LOG = logging.getLogger("translation_forensics")


def _project_root(args: argparse.Namespace) -> Path:
    return Path(args.project_root).expanduser().resolve() if args.project_root else Path(__file__).resolve().parents[2]


def _workspace(root: Path, title: str, explicit: str | None = None) -> Path:
    return Path(explicit).expanduser().resolve() if explicit else root / "workspaces" / title


def _emit(value: Any, args: argparse.Namespace) -> None:
    if args.json:
        print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    elif isinstance(value, str):
        print(value)
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


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
    output = args.output or workspace / "intermediate" / f"{args.title}.translation-queue-v1.jsonl"
    report = args.report or workspace / "intermediate" / f"{args.title}.translation-queue-v1.report.json"
    if args.dry_run:
        _emit({"status": "dry-run", "structure": str(structure), "ja": str(ja), "previous_ko": str(previous) if previous else None, "photos": str(photos) if photos else None, "output": str(output), "report": str(report)}, args); return 0
    if output.exists() or report.exists():
        _emit({"status": "fail", "error": "기존 translation queue를 덮어쓰지 않습니다.", "output": str(output), "report": str(report)}, args); return 2
    try:
        result = build_translation_queue(structure, ja, previous, output, report, review_context_path=review_context if review_context.exists() else None, review_queue_path=review_queue if review_queue.exists() else None, capture_index_path=capture_index if capture_index and capture_index.exists() else None)
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
    source_output = args.source_output or workspace / "intermediate" / f"{args.title}.source-faithful-ko.text-crosschecked-v1.srt"
    viewer_output = args.viewer_output or workspace / "intermediate" / f"{args.title}.viewer-natural-ko.text-crosschecked-v1.srt"
    report = args.report or workspace / "intermediate" / f"{args.title}.translation-application-v1.report.json"
    if args.dry_run:
        _emit({"status": "dry-run", "structure": str(structure), "decisions": str(decisions), "source_output": str(source_output), "viewer_output": str(viewer_output), "strict": args.strict}, args); return 0
    if not decisions.exists():
        _emit({"status": "fail", "error": f"번역 결정 파일이 없습니다: {decisions}"}, args); return 2
    if source_output.exists() or viewer_output.exists() or report.exists():
        _emit({"status": "fail", "error": "기존 번역 적용 결과를 덮어쓰지 않습니다.", "source_output": str(source_output), "viewer_output": str(viewer_output), "report": str(report)}, args); return 2
    try:
        result = apply_translation_decisions(structure, decisions, source_output, viewer_output, report, strict=args.strict)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    _emit(result, args); return 0 if result.get("status") != "fail" else 1


def cmd_init_translation_decisions(args: argparse.Namespace) -> int:
    root = _project_root(args)
    workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None)
    queue = args.queue or workspace / "intermediate" / f"{args.title}.translation-queue-v1.jsonl"
    output = args.output or workspace / "intermediate" / f"{args.title}.translation-decisions-template-v1.jsonl"
    if args.dry_run:
        _emit({"status": "dry-run", "queue": str(queue), "output": str(output)}, args); return 0
    if not queue.exists():
        _emit({"status": "fail", "error": f"translation queue가 없습니다: {queue}"}, args); return 2
    if output.exists():
        _emit({"status": "fail", "error": "기존 번역 결정 템플릿을 덮어쓰지 않습니다.", "output": str(output)}, args); return 2
    try:
        result = initialize_translation_decisions(queue, output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
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


def cmd_prepare_audio(args: argparse.Namespace) -> int:
    root = _project_root(args); workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None); input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        queue = resolve_role(input_root, "review_queue", args.review_queue, args.title, required=True)
        audio = resolve_role(input_root, "audio", args.audio, args.title, required=True)
    except DiscoveryError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
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
        result = run_asr(root, scenes, out, model=args.model, force_cpu=args.cpu, prompt=prompt if prompt.exists() else None, threshold=args.threshold, max_scenes=args.max_scenes, dry_run=args.dry_run)
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


def cmd_package(args: argparse.Namespace) -> int:
    root = _project_root(args); workspace = _workspace(root, args.title, str(args.workspace) if args.workspace else None); input_root = workspace / "inputs" if (workspace / "inputs").exists() else workspace
    try:
        reference = resolve_role(input_root, "structure", args.structure, args.title, required=True)
    except DiscoveryError as exc:
        _emit({"status": "fail", "error": str(exc)}, args); return 2
    output = args.output or workspace / "final"
    if args.dry_run:
        _emit({"status": "dry-run", "output": str(output), "stage": args.stage}, args); return 0
    try:
        result = package_title_outputs(args.title, reference, args.source_faithful.expanduser().resolve(), args.viewer_natural.expanduser().resolve(), output, stage=args.stage, version=args.version, japanese_path=args.ja, previous_path=args.previous_ko, photos_path=args.photos, scenes_path=args.scenes, asr_path=args.asr, translation_decisions_path=args.decisions, semantic_frames_path=args.semantic_frames, hypothesis_ledger_path=args.hypothesis_ledger, speaker_state_path=args.speaker_state, alignment_evidence_path=args.alignment_evidence, mqm_errors_path=args.mqm_errors, backtranslation_check_path=args.backtranslation_check, evaluation_summary_path=args.evaluation_summary, blind_review_pack_path=args.blind_review_pack, project_root=root, all_blocks_reviewed=args.all_blocks_reviewed, direct_human_listening=args.direct_human_listening, evidence_complete=args.evidence_complete)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="translation-forensics", description="일본어 자막 복원·번역·검증 통합 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="공용 자료와 실행 환경 검사"); _add_common(p); p.set_defaults(func=cmd_doctor)
    p = sub.add_parser("init-title", help="작품별 작업 디렉터리 생성"); _add_common(p); _add_title(p); p.set_defaults(func=cmd_init_title)
    p = sub.add_parser("inspect", help="입력 역할과 SRT 구조 검사"); _add_common(p); _add_title(p); _input_args(p); p.set_defaults(func=cmd_inspect)
    p = sub.add_parser("analyze", help="Subtitle Forensics 실행 또는 구조 기반 검토 큐 생성"); _add_common(p); _add_title(p); p.add_argument("--ja", type=Path); p.add_argument("--previous-ko", type=Path); p.add_argument("--output", type=Path); p.add_argument("--vendor-root", type=Path); p.set_defaults(func=cmd_analyze)
    p = sub.add_parser("build-korean-draft", help="일본어 구조에 맞춘 한국어 번역 초안 생성"); _add_common(p); _add_title(p); p.add_argument("--ja", type=Path); p.add_argument("--previous-ko", type=Path); p.add_argument("--output", type=Path); p.add_argument("--report", type=Path); p.set_defaults(func=cmd_build_korean_draft)
    p = sub.add_parser("build-translation-queue", help="블록별 의미 번역 결정 큐 생성"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--ja", type=Path); p.add_argument("--previous-ko", type=Path); p.add_argument("--photos", type=Path); p.add_argument("--review-context", type=Path); p.add_argument("--review-queue", type=Path); p.add_argument("--capture-index", type=Path); p.add_argument("--output", type=Path); p.add_argument("--report", type=Path); p.set_defaults(func=cmd_build_translation_queue)
    p = sub.add_parser("apply-translations", help="번역 결정 JSONL을 구조 고정 SRT 두 종으로 적용"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--decisions", required=True, type=Path); p.add_argument("--source-output", type=Path); p.add_argument("--viewer-output", type=Path); p.add_argument("--report", type=Path); p.add_argument("--strict", action="store_true", default=False); p.set_defaults(func=cmd_apply_translations)
    p = sub.add_parser("init-translation-decisions", help="번역 결정 템플릿 생성"); _add_common(p); _add_title(p); p.add_argument("--queue", type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_init_translation_decisions)
    p = sub.add_parser("merge-translation-decisions", help="검수 완료 블록을 결정 템플릿에 병합"); _add_common(p); p.add_argument("--base", required=True, type=Path); p.add_argument("--reviewed", required=True, type=Path); p.add_argument("--output", required=True, type=Path); p.set_defaults(func=cmd_merge_translation_decisions)
    p = sub.add_parser("init-forensic-records", help="의미 프레임·가설 원장을 명시적 미검수 상태로 초기화"); _add_common(p); _add_title(p); p.add_argument("--queue", type=Path); p.add_argument("--frames", type=Path); p.add_argument("--hypotheses", type=Path); p.set_defaults(func=cmd_init_forensic_records)
    p = sub.add_parser("validate-forensic-records", help="의미 프레임·가설 원장과 critical conflict escalation 검사"); _add_common(p); p.add_argument("--frames", required=True, type=Path); p.add_argument("--hypotheses", required=True, type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_validate_forensic_records)
    p = sub.add_parser("init-gold", help="정답을 만들지 않는 gold evaluation 디렉터리 골격 생성"); _add_common(p); p.set_defaults(func=cmd_init_gold)
    p = sub.add_parser("validate-mqm", help="subtitle MQM 오류 원장 형식과 critical 잔존 여부 검사"); _add_common(p); p.add_argument("--input", required=True, type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_validate_mqm)
    p = sub.add_parser("validate-evidence-artifact", help="reviewer-authored evidence artifact schema validation"); _add_common(p); p.add_argument("--kind", required=True, choices=("speaker-state", "alignment-evidence", "backtranslation-check")); p.add_argument("--input", required=True, type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_validate_evidence_artifact)
    p = sub.add_parser("prepare-audio", help="P1/P2 장면과 음성 클립 준비"); _add_common(p); _add_title(p); p.add_argument("--review-queue", type=Path); p.add_argument("--audio", type=Path); p.add_argument("--output", type=Path); p.add_argument("--bands", default="P1,P2"); p.add_argument("--padding", type=float, default=2.5); p.add_argument("--merge-gap", type=float, default=1.5); p.add_argument("--max-scene", type=float, default=45.0); p.add_argument("--no-audio", action="store_true"); p.add_argument("--force", action="store_true"); p.set_defaults(func=cmd_prepare_audio)
    p = sub.add_parser("run-asr", help="적응형 다중 ASR 실행"); _add_common(p); _add_title(p); p.add_argument("--scenes", type=Path); p.add_argument("--output", type=Path); p.add_argument("--prompt", type=Path); p.add_argument("--model", default="large-v3"); p.add_argument("--cpu", action="store_true"); p.add_argument("--threshold", type=float, default=0.82); p.add_argument("--max-scenes", type=int, default=0); p.set_defaults(func=cmd_run_asr)
    p = sub.add_parser("ingest-asr", help="ASR CSV/ZIP 검사 및 연결"); _add_common(p); _add_title(p); p.add_argument("--source", required=True, type=Path); p.add_argument("--output", type=Path); p.add_argument("--force", action="store_true"); p.set_defaults(func=cmd_ingest_asr)
    p = sub.add_parser("package-audio", help="완료된 음성 검토 작업을 work_audio ZIP으로 패키징"); _add_common(p); _add_title(p); p.add_argument("--work-dir", type=Path); p.add_argument("--output", type=Path); p.add_argument("--model", default="large-v3"); p.add_argument("--no-clips", action="store_true"); p.set_defaults(func=cmd_package_audio)
    p = sub.add_parser("build-review-context", help="장면 검토 패킷 생성"); _add_common(p); _add_title(p); p.add_argument("--ja", type=Path); p.add_argument("--previous-ko", type=Path); p.add_argument("--scenes", type=Path); p.add_argument("--asr", type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_build_review_context)
    p = sub.add_parser("validate", help="최종 SRT 구조·문자·가독성·회귀 검사"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--source-faithful", type=Path); p.add_argument("--viewer-natural", type=Path); p.add_argument("--output", type=Path); p.set_defaults(func=cmd_validate)
    p = sub.add_parser("package", help="새 버전으로 최종 산출물 패키징"); _add_common(p); _add_title(p); p.add_argument("--structure", type=Path); p.add_argument("--ja", type=Path, required=True); p.add_argument("--photos", type=Path); p.add_argument("--source-faithful", required=True, type=Path); p.add_argument("--viewer-natural", required=True, type=Path); p.add_argument("--previous-ko", type=Path); p.add_argument("--scenes", type=Path); p.add_argument("--asr", type=Path); p.add_argument("--decisions", type=Path); p.add_argument("--semantic-frames", type=Path); p.add_argument("--hypothesis-ledger", type=Path); p.add_argument("--speaker-state", type=Path); p.add_argument("--alignment-evidence", type=Path); p.add_argument("--mqm-errors", type=Path); p.add_argument("--backtranslation-check", type=Path); p.add_argument("--evaluation-summary", type=Path); p.add_argument("--blind-review-pack", type=Path); p.add_argument("--output", type=Path); p.add_argument("--stage", choices=STAGES, default="text-crosschecked"); p.add_argument("--version", type=int); p.add_argument("--all-blocks-reviewed", action="store_true"); p.add_argument("--direct-human-listening", action="store_true"); p.add_argument("--evidence-complete", action="store_true"); p.set_defaults(func=cmd_package)
    p = sub.add_parser("run", help="결정적 단계만 수행하고 의미 판정 전 중단"); _add_common(p); _add_title(p); _input_args(p); p.set_defaults(func=cmd_run)
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
