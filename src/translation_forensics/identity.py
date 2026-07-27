from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any, Iterable


ID_KINDS = {"title", "asset", "timeline", "scene", "block", "evidence", "hypothesis", "semantic-frame", "decision", "evaluation", "review", "terminology", "run"}


def _part(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return text.strip("-") or "unknown"


def stable_id(kind: str, title_id: str, local_key: object | None = None) -> str:
    if kind not in ID_KINDS:
        raise ValueError(f"지원하지 않는 ID 종류: {kind}")
    parts = [kind, _part(title_id)]
    if local_key is not None:
        parts.append(_part(local_key))
    return ":".join(parts)


def infer_title_id(path: Path) -> str:
    return _part(path.stem.split(".", 1)[0])


def attach_queue_identity(records: Iterable[dict[str, Any]], *, title_id: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        number = int(item["block_number"])
        item.update({
            "schema_name": "translation-forensics/translation-queue",
            "schema_version": "2",
            "title_id": title_id,
            "timeline_id": stable_id("timeline", title_id),
            "block_id": stable_id("block", title_id, number),
        })
        result.append(item)
    return result


def validate_identity_records(records: Iterable[dict[str, Any]], *, title_id: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    seen: set[str] = set()
    count = 0
    for index, row in enumerate(records, 1):
        count += 1
        value = row.get("title_id")
        block_id = row.get("block_id")
        if not isinstance(value, str) or not value:
            errors.append(f"record {index}: title_id가 없습니다.")
            continue
        if title_id and value != title_id:
            errors.append(f"record {index}: title_id 불일치")
        if not isinstance(block_id, str) or block_id != stable_id("block", value, row.get("block_number")):
            errors.append(f"record {index}: block_id가 안정 ID 규칙과 일치하지 않습니다.")
        if block_id in seen:
            errors.append(f"record {index}: block_id 중복: {block_id}")
        seen.add(str(block_id))
    return {"status": "pass" if not errors else "fail", "records": count, "errors": errors}


def migrate_jsonl_identity(input_path: Path, output_path: Path, *, title_id: str, kind: str) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"기존 마이그레이션 산출물을 덮어쓰지 않습니다: {output_path}")
    if kind not in {"queue", "decision", "semantic-frame", "hypothesis"}:
        raise ValueError("kind는 queue/decision/semantic-frame/hypothesis 중 하나여야 합니다.")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{line_number}행은 JSON 객체여야 합니다.")
        try:
            number = int(value["block_number"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{line_number}행 block_number가 잘못되었습니다.") from exc
        item = dict(value)
        item.update({"title_id": title_id, "block_id": stable_id("block", title_id, number)})
        if kind == "queue":
            item.update({"schema_name": "translation-forensics/translation-queue", "schema_version": "2", "timeline_id": stable_id("timeline", title_id)})
        if kind == "decision":
            item.update({"schema_name": "translation-forensics/translation-decision", "schema_version": "2", "decision_id": stable_id("decision", title_id, number)})
        if kind == "semantic-frame":
            item.update({"schema_name": "translation-forensics/semantic-frame", "schema_version": "2", "semantic_frame_id": stable_id("semantic-frame", title_id, number)})
        if kind == "hypothesis":
            key = item.get("hypothesis_id") or f"H{line_number}"
            item.update({"schema_name": "translation-forensics/hypothesis", "schema_version": "2", "hypothesis_id": stable_id("hypothesis", title_id, f"{number}-{key}")})
        rows.append(item)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
    return {"status": "migrated", "kind": kind, "title_id": title_id, "records": len(rows), "input": str(input_path), "output": str(output_path)}
