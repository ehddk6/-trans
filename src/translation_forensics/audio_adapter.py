from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from .asr_evidence import ASR_FIELDS, read_asr_candidates
from .discovery import AUDIO_EXTENSIONS, resolve_role
from .scenes import build_review_scenes, read_review_queue, write_review_scenes


def runner_dir(project_root: Path) -> Path:
    return project_root / "vendor" / "subtitle_audio_forensics_runner_v1"


def detect_device(force_cpu: bool = False) -> tuple[str, str]:
    if not force_cpu and shutil.which("nvidia-smi"):
        return "cuda", "float16"
    return "cpu", "int8"


def prepare_audio(project_root: Path, queue: Path, audio: Path, out_dir: Path, *, bands: str = "P1,P2", padding: float = 2.5, merge_gap: float = 1.5, max_scene: float = 45.0, include_audio: bool = True, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    script = runner_dir(project_root) / "prepare_review_pack.py"
    if not script.exists():
        raise RuntimeError(f"ASR 실행기 파일이 없습니다: {script}")
    if out_dir.exists() and any(out_dir.iterdir()) and not force and not dry_run:
        raise RuntimeError(f"중간 음성 작업 폴더가 이미 있어 덮어쓰지 않습니다: {out_dir}. --force를 명시하세요.")
    command = [sys.executable, str(script), "--queue", str(queue), "--audio", str(audio), "--out", str(out_dir), "--bands", bands, "--padding", str(padding), "--merge-gap", str(merge_gap), "--max-scene", str(max_scene)]
    if not include_audio:
        command.append("--no-audio")
    result: dict[str, Any] = {"command": command, "status": "dry-run" if dry_run else "ready"}
    if dry_run:
        return result
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    completed = subprocess.run(command, cwd=str(script.parent), capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, check=False)
    result.update({"returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:], "status": "completed" if completed.returncode == 0 else "failed"})
    return result


def run_asr(project_root: Path, scenes: Path, out_csv: Path, *, model: str = "large-v3", force_cpu: bool = False, prompt: Path | None = None, threshold: float = 0.82, max_scenes: int = 0, dry_run: bool = False, offline: bool = False) -> dict[str, Any]:
    scenes = scenes.expanduser().resolve()
    out_csv = out_csv.expanduser().resolve()
    if prompt:
        prompt = prompt.expanduser().resolve()
    script = runner_dir(project_root) / "run_asr_adaptive.py"
    if not script.exists():
        raise RuntimeError(f"ASR 실행기 파일이 없습니다: {script}")
    device, compute_type = detect_device(force_cpu)
    command = [sys.executable, str(script), "--scenes", str(scenes), "--model", model, "--device", device, "--compute-type", compute_type, "--out", str(out_csv), "--disagreement-threshold", str(threshold)]
    if prompt:
        command.extend(["--prompt", str(prompt)])
    if max_scenes:
        command.extend(["--max-scenes", str(max_scenes)])
    result: dict[str, Any] = {"command": command, "device": device, "compute_type": compute_type, "offline": offline, "status": "dry-run" if dry_run else "ready"}
    if dry_run:
        return result
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    completed = subprocess.run(command, cwd=str(script.parent), capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, check=False)
    result.update({"returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:], "status": "completed" if completed.returncode == 0 else "failed"})
    return result


def write_run_summary(work_dir: Path, *, title: str, model: str, device: str, compute_type: str, asr_path: Path, status: str, returncode: int | None = None) -> Path:
    summary = {
        "schema_version": "subtitle-audio-forensics/1",
        "status": status,
        "title": title,
        "work_dir": str(work_dir),
        "asr_candidates": str(asr_path),
        "model": model,
        "device": device,
        "compute_type": compute_type,
        "returncode": returncode,
        "direct_human_listening": False,
        "note": "동일 Whisper 모델의 여러 패스는 독립 증거가 아닌 후보 교차검토 자료입니다.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = work_dir / "run-summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def zip_work_audio(work_dir: Path, destination: Path, *, include_clips: bool = True) -> Path:
    if destination.exists():
        raise FileExistsError(f"기존 work_audio ZIP을 덮어쓰지 않습니다: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(work_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(work_dir)
            if not include_clips and relative.parts and relative.parts[0].lower() == "clips":
                continue
            archive.write(path, relative.as_posix())
    return destination


def _safe_zip_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    allowed = {"manifest.json", "review-scenes.csv", "asr-candidates.csv", "run-summary.json"}
    members = []
    for info in zf.infolist():
        name = Path(info.filename)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"ZIP 경로 탈출이 감지되었습니다: {info.filename}")
        if info.is_dir():
            continue
        if name.as_posix().split("/")[-1] in allowed or name.as_posix().startswith("clips/"):
            members.append(info)
    return members


def ingest_asr(source: Path, destination_dir: Path, *, force: bool = False) -> dict[str, Any]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".csv":
        rows = read_asr_candidates(source)
        destination = destination_dir / "asr-candidates.csv"
        if destination.exists() and not force and destination.read_bytes() != source.read_bytes():
            raise RuntimeError(f"기존 ASR 결과를 덮어쓰지 않습니다: {destination}")
        if not destination.exists() or force:
            shutil.copy2(source, destination)
        return {"status": "ingested", "rows": len(rows), "destination": str(destination), "source": str(source)}
    if source.suffix.lower() != ".zip":
        raise ValueError("ASR 입력은 CSV 또는 work_audio ZIP이어야 합니다.")
    with zipfile.ZipFile(source) as zf:
        members = _safe_zip_members(zf)
        by_name = {Path(info.filename).name: info for info in members}
        if "asr-candidates.csv" not in by_name:
            raise ValueError("work_audio ZIP에 asr-candidates.csv가 없습니다.")
        asr_bytes = zf.read(by_name["asr-candidates.csv"])
        temp = destination_dir / ".incoming-asr-candidates.csv"
        temp.write_bytes(asr_bytes)
        try:
            rows = read_asr_candidates(temp)
        finally:
            temp.unlink(missing_ok=True)
        destination = destination_dir / "asr-candidates.csv"
        if destination.exists() and not force and destination.read_bytes() != asr_bytes:
            raise RuntimeError(f"기존 ASR 결과를 덮어쓰지 않습니다: {destination}")
        if not destination.exists() or force:
            destination.write_bytes(asr_bytes)
        for name in ("manifest.json", "review-scenes.csv", "run-summary.json"):
            if name in by_name:
                target = destination_dir / name
                if not target.exists() or force:
                    target.write_bytes(zf.read(by_name[name]))
    return {"status": "ingested", "rows": len(rows), "destination": str(destination), "source": str(source)}
