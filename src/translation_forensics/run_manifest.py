from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import sha256_file, tool_versions


def _git(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(["git", *args], cwd=project_root, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def create_run_manifest(project_root: Path, output_path: Path, *, title: str, stage: str, inputs: list[Path], artifacts: list[Path] | None = None, prompt_manifest: Path | None = None, parent_run_id: str | None = None, run_id: str | None = None) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"기존 run manifest를 덮어쓰지 않습니다: {output_path}")
    missing = [str(path) for path in [*inputs, *(artifacts or [])] if not path.exists()]
    if missing:
        raise ValueError(f"run manifest 입력 또는 산출물이 없습니다: {', '.join(missing)}")
    now = datetime.now(timezone.utc).isoformat()
    prompt: dict[str, Any] | None = None
    if prompt_manifest:
        prompt = json.loads(prompt_manifest.read_text(encoding="utf-8"))
        prompt["manifest_sha256"] = sha256_file(prompt_manifest)
    value = {
        "schema_name": "translation-forensics/run-manifest", "schema_version": "1",
        "run_id": run_id or f"run-{uuid.uuid4()}", "parent_run_id": parent_run_id,
        "title_id": title, "stage": stage, "project_schema_version": "1",
        "code_commit": _git(project_root, "rev-parse", "HEAD"), "dirty_worktree": bool(_git(project_root, "status", "--porcelain")),
        "input_hashes": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in inputs],
        "artifact_hashes": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in artifacts or []],
        "prompt": prompt, "tool_versions": tool_versions(project_root),
        "verification_status": "not-validated", "started_at": now, "finished_at": now,
        "note": "이 manifest는 실행 계보 기록이며 의미 검증·사람 청취·최종 승격을 뜻하지 않습니다.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"status": "run-manifest-created", "output": str(output_path), "run_id": value["run_id"], "verification_status": "not-validated"}
