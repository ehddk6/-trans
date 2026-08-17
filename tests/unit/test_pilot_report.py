from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from translation_forensics.cli import main
from translation_forensics.pilot_report import PilotReportError, build_pilot_evaluation_report


ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, value: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8", newline="\n")
    return path


def _write_json(path: Path, value: object) -> Path:
    return _write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> Path:
    return _write(
        path,
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gates(status: str, *, passed: bool | None) -> dict[str, dict[str, object]]:
    return {
        "zero_new_critical_semantic_errors": {
            "status": status,
            "passed": passed,
            "required_count": 0,
            "observed_count": 0 if passed is not None else None,
            "reason": "zero required",
        },
        "critical_major_error_block_reduction": {
            "status": status,
            "passed": passed,
            "minimum_percent": 50.0,
            "observed_percent": 50.0 if passed is not None else None,
            "baseline_error_block_count": 2 if passed is not None else None,
            "improvement_error_block_count": 1 if passed is not None else None,
            "reason": "50 percent required",
        },
        "naturalness_balance": {
            "status": status,
            "passed": passed,
            "minimum_win_percent": 65.0,
            "maximum_loss_percent": 15.0,
            "observed_win_percent": 66.666667 if passed is not None else None,
            "observed_loss_percent": 0.0 if passed is not None else None,
            "win_count": 2 if passed is not None else None,
            "loss_count": 0 if passed is not None else None,
            "tie_count": 1 if passed is not None else None,
            "reason": "balance thresholds",
        },
    }


def _write_pack(path: Path, reviewer_slot: str, *, blocks: int, ready: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "awaiting-human-review" if ready else "blocked",
        "reviewer_slot": reviewer_slot,
        "block_count": blocks if ready else 0,
        "local_only": True,
        "external_delivery_performed": False,
        "candidate_provenance_hidden": True,
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest).encode("utf-8"))
    return path


def _inputs(tmp_path: Path, *, evaluated: bool = False, automated_passed: bool = True) -> dict[str, object]:
    blocks = 3
    baseline = _write(tmp_path / "baseline.srt", "baseline\n")
    source = _write(tmp_path / "candidate" / "source-faithful.srt", "source\n")
    viewer = _write(tmp_path / "candidate" / "viewer-natural.srt", "viewer\n")
    provenance = _write_json(tmp_path / "candidate" / "provenance.json", {
        "schema_name": "translation-forensics/pilot-candidate-provenance",
        "schema_version": "1",
        "title_id": "SSIS-908",
        "status": "candidate-generated",
        "block_count": blocks,
        "models": {
            "translation_decision": "gpt-5.6-terra",
            "independent_critique_repair": "gpt-5.6-sol",
        },
        "pipeline_order": [
            "terra-translation-decision",
            "sol-independent-critique-repair",
            "semantic-recheck",
        ],
        "inputs": {"frozen_baseline": {"sha256": _sha(baseline)}},
        "outputs": {
            "source_faithful": {"sha256": _sha(source)},
            "viewer_natural": {"sha256": _sha(viewer)},
        },
    })
    timeline = _write_json(tmp_path / "timeline" / "alignment.json", {
        "title_id": "SSIS-908",
        "expected_block_count": blocks,
        "status": "resolved" if evaluated else "unresolved",
        "clip_preparation_allowed": evaluated,
    })
    audio = _write_json(tmp_path / "audio-review" / "manifest.json", {
        "title_id": "SSIS-908",
        "expected_block_count": blocks,
        "block_count": blocks if evaluated else 0,
        "status": "ready" if evaluated else "blocked",
        "clip_preparation_allowed": evaluated,
    })
    packets = [
        _write_pack(
            tmp_path / "blind-review" / f"reviewer-{number}.pack.zip",
            f"reviewer-{number}",
            blocks=blocks,
            ready=evaluated,
        )
        for number in (1, 2)
    ]
    key = _write_json(tmp_path / "blind-review" / "internal-key.json", {
        "title_id": "SSIS-908",
        "expected_block_count": blocks,
        "status": "sealed" if evaluated else "blocked",
        "seal_status": "sealed-until-adjudication" if evaluated else "sealed-blocked-not-for-unblinding",
        "frozen_before_candidate_generation": evaluated,
    })
    adjudication = _write_jsonl(
        tmp_path / "adjudication" / "final-decisions.jsonl",
        [{"block_number": number} for number in range(1, blocks + 1)],
    )
    ledger = _write_jsonl(
        tmp_path / "results" / "error-ledger.jsonl",
        [{"block_number": number} for number in range(1, blocks + 1)],
    )
    metric_status = "pilot-evaluated" if evaluated else "awaiting-human-review"
    metric_result = "pass" if evaluated else None
    metrics = _write_json(tmp_path / "results" / "metrics.json", {
        "schema_name": "translation-forensics/pilot-metrics",
        "schema_version": "1",
        "title_id": "SSIS-908",
        "expected_block_count": blocks,
        "adjudicated_block_count": blocks if evaluated else 0,
        "blocking_blocks": [] if evaluated else [1, 2, 3],
        "unresolved_blocks": [],
        "status": metric_status,
        "result": metric_result,
        "gate_order": [
            "zero_new_critical_semantic_errors",
            "critical_major_error_block_reduction",
            "naturalness_balance",
        ],
        "gates": _gates("pass" if evaluated else "awaiting-human-review", passed=True if evaluated else None),
        "error_ledger_record_count": blocks,
        "error_ledger_sha256": _sha(ledger),
        "all_required_gates_passed": True if evaluated else None,
        "reason": "all gates passed" if evaluated else "two reviews are required",
    })
    automated = _write_json(tmp_path / "results" / "automated-validation.json", {
        "title_id": "SSIS-908",
        "expected_block_count": blocks,
        "status": "pass" if automated_passed else "fail",
        "automated_validation_passed": automated_passed,
        "failed_checks": [] if automated_passed else ["locked_srt_structure"],
    })
    attestations: list[Path] = []
    decisions: list[Path] = []
    if evaluated:
        decisions = [
            _write_jsonl(
                tmp_path / "reviewer-inputs" / f"reviewer-{number}.decisions.jsonl",
                [{"block_number": block} for block in range(1, blocks + 1)],
            )
            for number in (1, 2)
        ]
        attestations = [
            _write_json(tmp_path / "reviewer-inputs" / f"reviewer-{number}.attestation.json", {
                "title_id": "SSIS-908",
                "reviewer_id": f"listener-{number}",
                "pack_sha256": _sha(packets[number - 1]),
                "decisions_sha256": _sha(decisions[number - 1]),
                "japanese_listening_capable": True,
                "worked_independently": True,
                "direct_listening_all_blocks": True,
                "participated_in_baseline_or_candidate_production": False,
                "candidate_provenance_seen": False,
            })
            for number in (1, 2)
        ]
    return {
        "project_root": tmp_path,
        "baseline_path": baseline,
        "candidate_source_path": source,
        "candidate_viewer_path": viewer,
        "candidate_provenance_path": provenance,
        "timeline_alignment_path": timeline,
        "audio_review_manifest_path": audio,
        "blind_review_packet_paths": packets,
        "internal_key_path": key,
        "reviewer_attestation_paths": attestations,
        "reviewer_decision_paths": decisions,
        "adjudication_path": adjudication,
        "metrics_path": metrics,
        "error_ledger_path": ledger,
        "automated_validation_path": automated,
        "report_output_path": tmp_path / "PILOT_EVALUATION_REPORT.md",
        "expected_blocks": blocks,
        "expected_baseline_sha256": _sha(baseline),
    }


def test_awaiting_report_hashes_all_available_evidence_and_discloses_gaps(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    manifest, report = build_pilot_evaluation_report(**inputs)

    assert manifest["lifecycle_status"] == "awaiting-human-review"
    assert manifest["pilot_evaluated"] is False
    assert manifest["claim_boundary"]["final_promotion_allowed"] is False
    assert manifest["claim_boundary"]["human_reviewed_candidate_promotion_allowed"] is False
    assert manifest["reviewer_submissions"]["received_count"] == 0
    assert {item["code"] for item in manifest["unperformed_validations"]} == {
        "timeline-alignment-unresolved",
        "full-audio-review-packet-not-ready",
        "blind-review-packets-not-reviewable",
        "two-independent-reviews-not-returned",
        "full-adjudication-incomplete",
        "pilot-balance-contract-not-evaluable",
    }
    assert "다른 작품·13개 전체·시스템 전반" in report
    assert "final 또는 human-reviewed-candidate" in report
    assert manifest["generated_artifacts"]["pilot_evaluation_report"]["sha256"] == hashlib.sha256(
        report.encode("utf-8")
    ).hexdigest()


def test_pilot_pass_remains_scoped_and_cannot_be_promoted(tmp_path: Path) -> None:
    manifest, report = build_pilot_evaluation_report(**_inputs(tmp_path, evaluated=True))

    assert manifest["lifecycle_status"] == "pilot-evaluated/pass"
    assert manifest["human_reviewed"] is True
    assert manifest["reviewer_submissions"]["received_count"] == 2
    assert manifest["unperformed_validations"] == []
    assert manifest["claim_boundary"]["quality_improvement_claim_allowed"] is True
    assert manifest["claim_boundary"]["cross_title_generalization_allowed"] is False
    assert manifest["claim_boundary"]["final_promotion_allowed"] is False
    assert "SSIS-908 파일럿에만 한정" in report


def test_automated_failure_forces_fail_even_when_human_metrics_pass(tmp_path: Path) -> None:
    manifest, _ = build_pilot_evaluation_report(
        **_inputs(tmp_path, evaluated=True, automated_passed=False)
    )

    assert manifest["lifecycle_status"] == "pilot-evaluated/fail"
    assert manifest["claim_boundary"]["quality_improvement_claim_allowed"] is False
    assert "자동 검증 실패" in manifest["outcome_basis"]["reason"]


def test_candidate_provenance_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    Path(inputs["candidate_viewer_path"]).write_text("changed\n", encoding="utf-8", newline="\n")

    with pytest.raises(PilotReportError, match="viewer_natural hash mismatch"):
        build_pilot_evaluation_report(**inputs)


def test_cli_writes_schema_valid_report_and_refuses_overwrite(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    report_path = Path(inputs["report_output_path"])
    command = [
        "build-pilot-evaluation-report",
        "--project-root", str(tmp_path),
        "--baseline", str(inputs["baseline_path"]),
        "--candidate-source", str(inputs["candidate_source_path"]),
        "--candidate-viewer", str(inputs["candidate_viewer_path"]),
        "--candidate-provenance", str(inputs["candidate_provenance_path"]),
        "--timeline-alignment", str(inputs["timeline_alignment_path"]),
        "--audio-review-manifest", str(inputs["audio_review_manifest_path"]),
        "--blind-review-packet", str(inputs["blind_review_packet_paths"][0]),
        "--blind-review-packet", str(inputs["blind_review_packet_paths"][1]),
        "--internal-key", str(inputs["internal_key_path"]),
        "--adjudication", str(inputs["adjudication_path"]),
        "--metrics", str(inputs["metrics_path"]),
        "--error-ledger", str(inputs["error_ledger_path"]),
        "--automated-validation", str(inputs["automated_validation_path"]),
        "--expected-blocks", "3",
        "--expected-baseline-sha256", str(inputs["expected_baseline_sha256"]),
        "--manifest-output", str(manifest_path),
        "--report-output", str(report_path),
    ]

    assert main(command) == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "schemas" / "pilot-evaluation-manifest.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(manifest)
    assert not manifest_path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in manifest_path.read_bytes()
    assert b"\r\n" not in report_path.read_bytes()
    assert _sha(report_path) == manifest["generated_artifacts"]["pilot_evaluation_report"]["sha256"]
    assert main(command) == 2
