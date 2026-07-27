from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .forensic_model import read_jsonl
from .timeline import timeline_is_usable


def audit_workspace(project_root: Path, title: str, workspace: Path) -> dict[str, Any]:
    metadata, intermediate, inputs = workspace / "metadata", workspace / "intermediate", workspace / "inputs"
    checks: dict[str, dict[str, Any]] = {}
    checks["workspace_exists"] = {"ok": workspace.exists(), "path": str(workspace)}
    checks["structure_input"] = {"ok": bool(list(inputs.glob("*.structure.srt"))), "path": str(inputs)}
    checks["japanese_input"] = {"ok": bool(list(inputs.glob("*.ja.srt"))), "path": str(inputs)}
    media = [*inputs.glob("*.mp3"), *inputs.glob("*.wav"), *inputs.glob("*.m4a"), *inputs.glob("*.mp4")]
    checks["media_input"] = {"ok": bool(media), "paths": [str(path) for path in media]}
    timeline_paths = sorted(metadata.glob(f"{title}.timeline-validation-*.json"))
    if timeline_paths:
        try:
            timeline = json.loads(timeline_paths[-1].read_text(encoding="utf-8"))
            checks["timeline_validation"] = {"ok": timeline_is_usable(timeline), "status": timeline.get("status"), "path": str(timeline_paths[-1])}
            structure_path = Path(str(timeline.get("structure", {}).get("path", "")))
            media_path = Path(str((timeline.get("media") or {}).get("path", "")))
            if structure_path.exists():
                checks["structure_input"] = {"ok": True, "path": str(structure_path), "source": "timeline-validation"}
                if structure_path.name.lower().endswith(".ja.srt"):
                    checks["japanese_input"] = {"ok": True, "path": str(structure_path), "source": "timeline-validation"}
            if media_path.exists():
                checks["media_input"] = {"ok": True, "paths": [str(media_path)], "source": "timeline-validation"}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            checks["timeline_validation"] = {"ok": False, "error": str(exc)}
    else:
        checks["timeline_validation"] = {"ok": False, "status": "missing"}
    artifacts = []
    for path in sorted(intermediate.glob("*translation-decisions*.jsonl")):
        try:
            rows = read_jsonl(path)
            artifacts.append({"path": str(path), "records": len(rows), "approved_or_reviewed": sum(str(row.get("status")) in {"approved", "reviewed", "translated"} for row in rows)})
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    checks["translation_decisions"] = {"ok": bool(artifacts), "artifacts": artifacts}
    checks["gold_suite"] = {"ok": bool(list((project_root / "evaluation" / "gold" / "annotations").glob("*.gold-record.json"))), "path": str(project_root / "evaluation" / "gold" / "annotations")}
    checks["final_package"] = {"ok": bool(list((workspace / "final").glob("*.final-v*.srt"))), "path": str(workspace / "final")}
    missing = [name for name, value in checks.items() if not value.get("ok")]
    return {"schema_name": "translation-forensics/workspace-audit", "schema_version": "1", "title_id": title, "status": "ready-for-final-audit" if not missing else "partial", "checks": checks, "missing_requirements": missing, "note": "이 감사는 실제 사람 청취·골드·승인 근거를 생성하지 않습니다."}
