from __future__ import annotations

"""Reproducible engineering evaluation for the canonical evidence bridge."""

import hashlib
import json
from pathlib import Path
from typing import Any

from .automated_quality import audit_translation_decision
from .evidence_bridge import build_unit_source_evidence


def evaluate_source_evidence_contrasts(path: Path) -> dict[str, Any]:
    path = Path(path)
    project_root = Path(__file__).resolve().parents[2]
    try:
        suite_path = path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        suite_path = path.as_posix()
    suite = json.loads(path.read_text(encoding="utf-8"))
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("source-evidence contrast suite requires non-empty cases")

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    newly_queued_hazards = 0
    missed_hazards = 0
    clean_control_regressions = 0
    covered_reasons: set[str] = set()

    for index, raw in enumerate(cases, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"contrast case {index} must be an object")
        case_id = str(raw.get("case_id") or f"case-{index}")
        category = str(raw.get("category") or "")
        source = str(raw.get("source_japanese") or "")
        faithful = str(raw.get("source_faithful_korean") or "")
        natural = str(raw.get("viewer_natural_korean") or faithful)
        quality = str(raw.get("source_quality_status") or "trusted")
        evidence = build_unit_source_evidence(
            unit_id=case_id,
            start=0.0,
            end=1.0,
            source_text=source,
            asr_metrics=(
                raw.get("asr_metrics")
                if isinstance(raw.get("asr_metrics"), dict)
                else None
            ),
            source_quality_status=quality,
            evidence_ids=[f"fixture:{case_id}"],
        )
        common = {
            "unit_id": case_id,
            "source_japanese": source,
            "source_faithful_korean": faithful,
            "viewer_natural_korean": natural,
            "source_quality_status": quality,
            "confidence": "high",
        }
        baseline = audit_translation_decision(**common)
        bridged = audit_translation_decision(**common, source_evidence=evidence)
        risk_codes = list(evidence["asr_fusion"].get("risk_codes") or [])
        covered_reasons.update(bridged["warnings"])

        if category == "hazard":
            if (
                baseline["status"] == "passed"
                and bridged["status"] == "passed"
                and bridged["semantic_audit_recommended"]
            ):
                newly_queued_hazards += 1
            else:
                missed_hazards += 1
        elif category == "clean-control":
            if baseline["status"] != bridged["status"]:
                clean_control_regressions += 1
        else:
            errors.append(f"{case_id}: unsupported category {category!r}")

        expected_state = str(raw.get("expected_fusion_state") or "")
        expected_risks = sorted(str(value) for value in raw.get("expected_risk_codes", []))
        expected_bridge_status = str(raw.get("expected_bridge_status") or "")
        actual_state = str(evidence["asr_fusion"].get("state") or "")
        if actual_state != expected_state:
            errors.append(
                f"{case_id}: fusion state {actual_state!r} != {expected_state!r}"
            )
        if sorted(risk_codes) != expected_risks:
            errors.append(
                f"{case_id}: risk codes {sorted(risk_codes)!r} != {expected_risks!r}"
            )
        if bridged["status"] != expected_bridge_status:
            errors.append(
                f"{case_id}: bridge status {bridged['status']!r} != {expected_bridge_status!r}"
            )

        results.append(
            {
                "case_id": case_id,
                "category": category,
                "fusion_state": actual_state,
                "fusion_risk_codes": risk_codes,
                "baseline_status": baseline["status"],
                "bridge_status": bridged["status"],
                "bridge_reason_codes": [
                    reason
                    for reason in bridged["warnings"]
                    if reason.startswith("asr_")
                ],
            }
        )

    hazard_count = sum(row.get("category") == "hazard" for row in cases if isinstance(row, dict))
    clean_count = sum(
        row.get("category") == "clean-control" for row in cases if isinstance(row, dict)
    )
    status = (
        "pass"
        if not errors
        and newly_queued_hazards == hazard_count
        and missed_hazards == 0
        and clean_control_regressions == 0
        else "fail"
    )
    return {
        "schema_name": "translation-forensics/canonical-evidence-bridge-evaluation",
        "schema_version": "1",
        "status": status,
        "evaluation_kind": "synthetic-contrast-engineering-safety",
        "suite": suite_path,
        "suite_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "baseline_contract": "deterministic audit without source_evidence",
        "bridge_contract": (
            "same deterministic audit with canonical source_evidence; "
            "source conflicts are semantic-audit targets, not automatic rewrites"
        ),
        "case_count": len(cases),
        "hazard_count": hazard_count,
        "clean_control_count": clean_count,
        "newly_queued_hazards": newly_queued_hazards,
        "missed_hazards": missed_hazards,
        "clean_control_regressions": clean_control_regressions,
        "covered_asr_reason_codes": sorted(
            reason for reason in covered_reasons if reason.startswith("asr_")
        ),
        "results": results,
        "errors": errors,
        "human_quality_proven": False,
        "limits": [
            "hand-authored synthetic cases",
            "no model call",
            "no direct audio listening",
            "no adjudicated human gold",
        ],
        "claim_scope": (
            "deterministic source-evidence conflict targeting only; "
            "no human translation-quality claim"
        ),
    }


__all__ = ["evaluate_source_evidence_contrasts"]
