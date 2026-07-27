from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .forensic_model import read_jsonl


AUDIT_FIELDS = ["sample_id", "title_id", "block_number", "priority", "risk_features", "sampling_stratum", "random_seed", "selection_probability", "review_result", "mqm_error", "severity", "false_negative", "review_time_seconds"]


def sample_audit(queue_path: Path, output_path: Path, *, title: str, seed: int, rate: float, bands: set[str]) -> dict[str, Any]:
    if not 0 < rate <= 1:
        raise ValueError("audit rate는 0보다 크고 1 이하여야 합니다.")
    if output_path.exists():
        raise FileExistsError(f"기존 audit sample을 덮어쓰지 않습니다: {output_path}")
    queue = read_jsonl(queue_path)
    eligible = [row for row in queue if str(row.get("review_band", "P3")) in bands]
    rng = random.Random(seed)
    selected = [row for row in eligible if rng.random() < rate]
    if eligible and not selected:
        selected = [rng.choice(eligible)]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(selected, key=lambda value: int(value.get("block_number", 0))), 1):
        band = str(row.get("review_band", "P3"))
        risk = list(row.get("required_semantic_slots", [])) if isinstance(row.get("required_semantic_slots"), list) else []
        rows.append({
            "sample_id": f"A-{title}-{index:04d}", "title_id": title, "block_number": row.get("block_number"), "priority": band,
            "risk_features": ";".join(str(value) for value in risk), "sampling_stratum": band, "random_seed": seed,
            "selection_probability": rate, "review_result": "unreviewed", "mqm_error": "", "severity": "", "false_negative": "", "review_time_seconds": "",
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    return {"status": "audit-sample-created", "output": str(output_path), "eligible_blocks": len(eligible), "sampled_blocks": len(rows), "rate": rate, "random_seed": seed, "review_status": "not-reviewed"}


def summarize_audit(sample_path: Path, output_path: Path) -> dict[str, Any]:
    with sample_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = set(AUDIT_FIELDS)
    if not rows and not required:
        raise ValueError("empty audit CSV")
    if rows and required - set(rows[0]):
        raise ValueError(f"audit 필수 열 누락: {', '.join(sorted(required - set(rows[0])))}")
    reviewed = [row for row in rows if row.get("review_result") not in {"", "unreviewed"}]
    error_rows = [row for row in reviewed if row.get("mqm_error", "").strip()]
    critical = [row for row in error_rows if row.get("severity") == "critical"]
    summary = {
        "schema_name": "translation-forensics/audit-summary", "schema_version": "1",
        "created_at": datetime.now(timezone.utc).isoformat(), "sample": str(sample_path),
        "sampled_blocks": len(rows), "reviewed_blocks": len(reviewed), "unreviewed_blocks": len(rows) - len(reviewed),
        "error_findings": len(error_rows), "critical_findings": len(critical),
        "review_complete": bool(rows) and len(reviewed) == len(rows),
        "metrics_status": "not-demonstrated" if not rows or len(reviewed) != len(rows) else "descriptive-only",
        "note": "무작위 감사 표본만으로 전체 품질이나 우선순위 모델의 recall/precision을 주장하지 않습니다.",
    }
    if output_path.exists():
        raise FileExistsError(f"기존 audit summary를 덮어쓰지 않습니다: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary
