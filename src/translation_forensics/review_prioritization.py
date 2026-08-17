from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .srt import SubtitleBlock, parse_srt
from .validation import validate_pair


PRIORITY_FIELDS = [
    "rank",
    "priority",
    "block_number",
    "timecode",
    "reason_codes",
    "reason_details",
    "decision_status",
    "evidence_strength",
    "evidence_ref_status",
    "uncertain_slots",
    "existing_review_band",
    "existing_forensics_risk",
    "required_evidence",
    "source_faithful_korean",
    "viewer_natural_korean",
    "next_action",
]
_PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL {line_number}번 레코드는 객체여야 합니다: {path}")
        rows.append(value)
    return rows


def _string_list(value: Any) -> tuple[list[str], bool]:
    if not isinstance(value, list):
        return [], False
    values = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return values, len(values) == len(value)


def _joined(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _by_block(records: list[dict[str, Any]], *, label: str, expected: set[int]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        try:
            number = int(record.get("block_number"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}의 block_number가 정수가 아닙니다: {record.get('block_number')!r}") from exc
        if number not in expected:
            raise ValueError(f"{label}에 구조 기준본에 없는 block이 있습니다: {number}")
        if number in result:
            raise ValueError(f"{label}에 중복 block이 있습니다: {number}")
        result[number] = record
    return result


def _forensics_rows(path: Path, expected: set[int]) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"block_number", "timecode", "review_band"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"기존 review queue 필수 열 누락: {', '.join(sorted(missing))}")
        rows = _by_block([dict(row) for row in reader], label="기존 review queue", expected=expected)
    return {number: {key: str(value or "") for key, value in row.items()} for number, row in rows.items()}


def _validation_issues(
    structure_path: Path,
    source_path: Path,
    viewer_path: Path,
    *,
    project_root: Path | None,
) -> dict[int, list[dict[str, str]]]:
    report = validate_pair(structure_path, source_path, viewer_path, project_root=project_root)
    if report.get("status") == "fail" and not (report.get("source_faithful") and report.get("viewer_natural")):
        raise ValueError("후보 SRT를 파싱할 수 없어 검수 우선순위 큐를 만들 수 없습니다. validate 결과를 먼저 해결하세요.")
    result: dict[int, list[dict[str, str]]] = {}
    for variant in ("source_faithful", "viewer_natural"):
        payload = report.get(variant, {})
        for issue in payload.get("issues", []):
            block = issue.get("block")
            if not isinstance(block, int):
                continue
            result.setdefault(block, []).append({
                "variant": variant,
                "code": str(issue.get("code", "validation_issue")),
                "severity": str(issue.get("severity", "warning")),
            })
    return result


def _add_reason(reasons: list[tuple[str, str, str]], priority: str, code: str, detail: str) -> None:
    if (priority, code, detail) not in reasons:
        reasons.append((priority, code, detail))


def build_uncertainty_review_queue(
    structure_path: Path,
    output_path: Path,
    *,
    decisions_path: Path | None = None,
    translation_queue_path: Path | None = None,
    forensics_queue_path: Path | None = None,
    source_path: Path | None = None,
    viewer_path: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Create a transparent, evidence-centred queue without inventing a score.

    Every row states the concrete signals that made a block review-worthy. It
    does not infer meaning or turn low evidence strength into a probability.
    """
    if output_path.exists():
        raise FileExistsError(f"기존 검수 우선순위 큐를 덮어쓰지 않습니다: {output_path}")
    if bool(source_path) != bool(viewer_path):
        raise ValueError("가독성 위험을 포함하려면 source-faithful와 viewer-natural을 함께 지정해야 합니다.")
    inputs = [decisions_path, translation_queue_path, forensics_queue_path, source_path, viewer_path]
    if not any(path and path.exists() for path in inputs):
        raise ValueError("결정 JSONL, translation queue, 기존 review queue 또는 두 후보 SRT 중 하나가 필요합니다.")

    structure, _, _ = parse_srt(structure_path)
    expected = {block.number for block in structure}
    decisions = _by_block(_read_jsonl(decisions_path), label="번역 결정", expected=expected) if decisions_path and decisions_path.exists() else {}
    translation_queue = _by_block(_read_jsonl(translation_queue_path), label="translation queue", expected=expected) if translation_queue_path and translation_queue_path.exists() else {}
    forensics = _forensics_rows(forensics_queue_path, expected) if forensics_queue_path and forensics_queue_path.exists() else {}
    validation = _validation_issues(structure_path, source_path, viewer_path, project_root=project_root) if source_path and viewer_path else {}

    rows: list[dict[str, str]] = []
    for block in structure:
        decision = decisions.get(block.number, {})
        queue_record = translation_queue.get(block.number, {})
        prior = forensics.get(block.number, {})
        reasons: list[tuple[str, str, str]] = []
        status = str(decision.get("status") or decision.get("source_status") or "").strip().lower()
        confidence = str(decision.get("confidence") or "").strip().lower()
        uncertain_slots = _joined(decision.get("uncertain_slots") or decision.get("inferred_slots"))
        risks = _joined(decision.get("risk_codes") or decision.get("abstention_reasons") or decision.get("competing_interpretations"))
        evidence_refs, refs_valid = _string_list(decision.get("evidence_refs")) if decision else ([], True)
        allowed_refs, queue_refs_valid = _string_list(queue_record.get("evidence_refs")) if queue_record else ([], True)

        if status in {"abstained", "unresolved", "untranslated", "hold"}:
            _add_reason(reasons, "critical", "decision-status", f"결정 상태: {status}")
        if decision and confidence in {"", "low", "unknown"}:
            _add_reason(reasons, "high", "low-evidence-strength", f"결정 evidence strength: {confidence or 'missing'}")
        if uncertain_slots:
            _add_reason(reasons, "high", "uncertain-slots", uncertain_slots)
        if risks:
            _add_reason(reasons, "high", "unresolved-risk-or-alternative", risks)
        if decision and (not refs_valid or not evidence_refs):
            _add_reason(reasons, "high", "missing-evidence-refs", "결정에 검증 가능한 evidence_refs가 없습니다")
        if translation_queue and (not queue_refs_valid or not allowed_refs):
            _add_reason(reasons, "high", "queue-evidence-contract", "translation queue의 evidence_refs 계약이 불완전합니다")
        if decision and translation_queue and evidence_refs and allowed_refs:
            unmapped = sorted(set(evidence_refs) - set(allowed_refs))
            if unmapped:
                _add_reason(reasons, "high", "unmapped-evidence-ref", ", ".join(unmapped))

        consistency_context = queue_record.get("consistency_context", {}) if isinstance(queue_record, dict) else {}
        if isinstance(consistency_context, dict):
            for conflict in consistency_context.get("conflicts", []):
                if not isinstance(conflict, dict):
                    continue
                entry_type = str(conflict.get("entry_type") or "consistency")
                entry_ids = ",".join(str(value) for value in conflict.get("entry_ids", []) if str(value))
                _add_reason(reasons, "high", "consistency-conflict", f"{entry_type}:{entry_ids or 'unidentified'}")
            for entry in consistency_context.get("unresolved_entries", []):
                if not isinstance(entry, dict):
                    continue
                entry_type = str(entry.get("entry_type") or "consistency")
                priority = "high" if entry_type in {"register", "address_term"} else "medium"
                _add_reason(reasons, priority, "consistency-unresolved", f"{entry_type}:{entry.get('consistency_id', '')}")

        band = prior.get("review_band", "")
        prior_confidence = prior.get("confidence", "").strip().lower()
        if band == "P1":
            _add_reason(reasons, "high", "forensics-p1", "기존 포렌식 큐의 P1 우선 검토")
        elif band == "P2":
            _add_reason(reasons, "medium", "forensics-p2", "기존 포렌식 큐의 P2 우선 검토")
        if prior_confidence == "low":
            _add_reason(reasons, "high", "forensics-low-confidence", "기존 포렌식 evidence strength가 low")
        elif prior_confidence == "medium":
            _add_reason(reasons, "medium", "forensics-medium-confidence", "기존 포렌식 evidence strength가 medium")
        if prior.get("uncertain_scope", "").strip():
            _add_reason(reasons, "high", "forensics-uncertain-scope", prior["uncertain_scope"].strip())

        for issue in validation.get(block.number, []):
            priority = "critical" if issue["severity"] == "error" else "medium"
            _add_reason(reasons, priority, f"{issue['variant']}:{issue['code']}", f"{issue['variant']} 자동 QA {issue['severity']}")

        if not reasons:
            continue
        priority = min((reason[0] for reason in reasons), key=lambda value: _PRIORITY_RANK[value])
        reason_codes = "; ".join(reason[1] for reason in reasons)
        reason_details = " | ".join(reason[2] for reason in reasons)
        next_action = "원음·일본어·문맥을 우선 확인하고 보류 또는 수정 결정을 기록"
        if any(code == "decision-status" for _, code, _ in reasons):
            next_action = "미확정 의미를 창작하지 말고 필요한 음성·문맥 근거를 확인"
        rows.append({
            "rank": "",
            "priority": priority,
            "block_number": str(block.number),
            "timecode": f"{block.start} --> {block.end}",
            "reason_codes": reason_codes,
            "reason_details": reason_details,
            "decision_status": status or "not-supplied",
            "evidence_strength": confidence or "not-supplied",
            "evidence_ref_status": "mapped" if decision and translation_queue and evidence_refs and set(evidence_refs) <= set(allowed_refs) else ("not-checked" if not decision or not translation_queue else "needs-review"),
            "uncertain_slots": uncertain_slots,
            "existing_review_band": band,
            "existing_forensics_risk": prior.get("forensics_risk", ""),
            "required_evidence": prior.get("required_evidence", ""),
            "source_faithful_korean": str(decision.get("source_faithful_korean") or decision.get("source_text") or ""),
            "viewer_natural_korean": str(decision.get("viewer_natural_korean") or decision.get("viewer_text") or ""),
            "next_action": next_action,
        })

    rows.sort(key=lambda row: (_PRIORITY_RANK[row["priority"]], int(row["block_number"])))
    for index, row in enumerate(rows, 1):
        row["rank"] = str(index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRIORITY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = {priority: sum(row["priority"] == priority for row in rows) for priority in _PRIORITY_RANK}
    return {
        "status": "review-queue-ready",
        "output": str(output_path),
        "queued_blocks": len(rows),
        "priority_counts": counts,
        "ranking_policy": "명시된 위험 이유의 최고 우선순위로 정렬; 숨은 단일 점수 없음",
        "final_promotion_allowed": False,
    }
