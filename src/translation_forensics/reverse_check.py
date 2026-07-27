from __future__ import annotations

import csv
from pathlib import Path

from .forensic_model import read_jsonl


FIELDS = ["block_number", "title_id", "source_japanese", "source_faithful_korean", "viewer_natural_korean", "reverse_japanese", "status", "difference_types", "review_status", "reviewer", "evidence_refs", "review_note"]
STATUSES = {"equivalent", "acceptable-ellipsis", "meaning-loss", "meaning-addition", "meaning-flip", "unresolved"}


def initialize_reverse_check(decisions_path: Path, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"기존 reverse semantic check를 덮어쓰지 않습니다: {output_path}")
    rows = []
    for item in read_jsonl(decisions_path):
        rows.append({
            "block_number": item.get("block_number"), "title_id": item.get("title_id", ""), "source_japanese": item.get("source_japanese", ""),
            "source_faithful_korean": item.get("source_faithful_korean", ""), "viewer_natural_korean": item.get("viewer_natural_korean", ""),
            "reverse_japanese": "", "status": "unresolved", "difference_types": "", "review_status": "unreviewed", "reviewer": "", "evidence_refs": "", "review_note": "",
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    return {"status": "reverse-check-template", "output": str(output_path), "blocks": len(rows), "review_status": "not-reviewed"}


def validate_reverse_check(decisions_path: Path, check_path: Path) -> dict[str, object]:
    expected = {int(item["block_number"]) for item in read_jsonl(decisions_path)}
    with check_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader); fields = set(reader.fieldnames or [])
    errors = [f"필수 열 누락: {', '.join(sorted(set(FIELDS) - fields))}"] if set(FIELDS) - fields else []
    actual: set[int] = set(); blocking: list[int] = []; unresolved: list[int] = []
    for index, row in enumerate(rows, 2):
        try:
            number = int(row.get("block_number", "")); actual.add(number)
        except ValueError:
            errors.append(f"{index}행: block_number가 잘못되었습니다."); continue
        if row.get("status") not in STATUSES:
            errors.append(f"{index}행: status가 잘못되었습니다.")
        if row.get("review_status") not in {"reviewed", "unresolved"}:
            errors.append(f"{index}행: review_status가 잘못되었습니다.")
        if row.get("status") in {"meaning-flip", "meaning-addition"}:
            blocking.append(number)
        if row.get("review_status") != "reviewed" or row.get("status") == "unresolved":
            unresolved.append(number)
    missing = sorted(expected - actual); extra = sorted(actual - expected)
    errors.extend(f"검사 누락 block {number}" for number in missing); errors.extend(f"구조에 없는 검사 block {number}" for number in extra)
    return {"status": "pass" if not errors and not blocking and not unresolved else "fail", "blocks": len(rows), "blocking_differences": blocking, "unresolved_blocks": unresolved, "errors": errors}
