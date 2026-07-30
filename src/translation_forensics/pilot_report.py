from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from .manifest import sha256_file


SCHEMA_NAME = "translation-forensics/pilot-evaluation-manifest"
SCHEMA_VERSION = "1"
TITLE_ID = "SSIS-908"
BASELINE_SHA256 = "8653a42dc952152994c75e9d43265c49eddc1b7d581cf05d8012fd87e3c3e25b"
PIPELINE_ORDER = [
    "terra-translation-decision",
    "sol-independent-critique-repair",
    "semantic-recheck",
]
EVALUATION_ORDER = [
    "semantic_fidelity",
    "contextual_consistency",
    "natural_korean",
    "semantic_recheck",
]


class PilotReportError(ValueError):
    """Raised when the evidence cannot support an honest pilot report."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise PilotReportError(f"{label} must be a JSON object")
    return value


def _project_path(path: Path, project_root: Path) -> tuple[Path, str]:
    root = project_root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise PilotReportError(f"Pilot evidence must stay inside the project root: {path}") from exc
    if not resolved.is_file():
        raise PilotReportError(f"Pilot evidence is missing: {relative.as_posix()}")
    return resolved, relative.as_posix()


def _file_ref(path: Path, project_root: Path) -> dict[str, Any]:
    resolved, relative = _project_path(path, project_root)
    return {
        "path": relative,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _output_path(path: Path, project_root: Path) -> str:
    root = project_root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise PilotReportError(f"Pilot output must stay inside the project root: {path}") from exc


def _jsonl_count(path: Path) -> int:
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise PilotReportError(f"{path}:{line_number}: JSON object required")
        count += 1
    return count


def _blind_packet_ready(path: Path, expected_blocks: int) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    except (KeyError, OSError, UnicodeDecodeError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise PilotReportError(f"Blind review packet is invalid: {path}") from exc
    if not isinstance(manifest, Mapping):
        raise PilotReportError(f"Blind review packet manifest must be an object: {path}")
    return (
        manifest.get("status") == "awaiting-human-review"
        and manifest.get("block_count") == expected_blocks
        and manifest.get("local_only") is True
        and manifest.get("external_delivery_performed") is False
        and manifest.get("candidate_provenance_hidden") is True
    )


def _validate_reviewer_inputs(
    attestation_paths: list[Path],
    decision_paths: list[Path],
    packet_refs: list[Mapping[str, Any]],
    *,
    expected_blocks: int,
) -> int:
    reviewer_ids: set[str] = set()
    used_packet_hashes: set[str] = set()
    valid_packet_hashes = {str(ref["sha256"]) for ref in packet_refs}
    for attestation_path, decision_path in zip(attestation_paths, decision_paths):
        attestation = _read_json(attestation_path, "Reviewer attestation")
        reviewer_id = str(attestation.get("reviewer_id", "")).strip()
        if (
            attestation.get("title_id") != TITLE_ID
            or not reviewer_id
            or attestation.get("japanese_listening_capable") is not True
            or attestation.get("worked_independently") is not True
            or attestation.get("direct_listening_all_blocks") is not True
            or attestation.get("participated_in_baseline_or_candidate_production") is not False
            or attestation.get("candidate_provenance_seen") is not False
        ):
            raise PilotReportError("Reviewer attestation does not satisfy the independent listening contract")
        if reviewer_id in reviewer_ids:
            raise PilotReportError("Reviewer submissions must identify two different reviewers")
        reviewer_ids.add(reviewer_id)
        packet_hash = str(attestation.get("pack_sha256", ""))
        if packet_hash not in valid_packet_hashes or packet_hash in used_packet_hashes:
            raise PilotReportError("Reviewer attestation pack hash is missing, unknown, or duplicated")
        used_packet_hashes.add(packet_hash)
        if attestation.get("decisions_sha256") != sha256_file(decision_path):
            raise PilotReportError("Reviewer attestation decision hash mismatch")
        if _jsonl_count(decision_path) != expected_blocks:
            raise PilotReportError("Each reviewer decision file must cover every pilot block")
    return len(reviewer_ids)


def _require_identity(value: Mapping[str, Any], label: str, expected_blocks: int) -> None:
    if value.get("title_id") != TITLE_ID:
        raise PilotReportError(f"{label} is outside the SSIS-908 pilot")
    if value.get("expected_block_count", value.get("block_count")) != expected_blocks:
        raise PilotReportError(f"{label} does not cover exactly {expected_blocks} blocks")


def _validate_candidate_receipts(
    provenance: Mapping[str, Any],
    *,
    baseline_ref: Mapping[str, Any],
    source_ref: Mapping[str, Any],
    viewer_ref: Mapping[str, Any],
    expected_blocks: int,
) -> None:
    if provenance.get("schema_name") != "translation-forensics/pilot-candidate-provenance":
        raise PilotReportError("Candidate provenance schema is invalid")
    _require_identity(provenance, "Candidate provenance", expected_blocks)
    if provenance.get("pipeline_order") != PIPELINE_ORDER:
        raise PilotReportError("Candidate provenance has an invalid Terra/Sol/recheck order")
    models = provenance.get("models")
    if not isinstance(models, Mapping) or models.get("translation_decision") != "gpt-5.6-terra" or models.get(
        "independent_critique_repair"
    ) != "gpt-5.6-sol":
        raise PilotReportError("Candidate provenance does not identify the required Terra and Sol models")
    inputs = provenance.get("inputs")
    if not isinstance(inputs, Mapping) or not isinstance(inputs.get("frozen_baseline"), Mapping):
        raise PilotReportError("Candidate provenance lacks the frozen baseline receipt")
    if inputs["frozen_baseline"].get("sha256") != baseline_ref["sha256"]:
        raise PilotReportError("Candidate provenance baseline hash mismatch")
    outputs = provenance.get("outputs")
    if not isinstance(outputs, Mapping):
        raise PilotReportError("Candidate provenance lacks output receipts")
    expected_outputs = {"source_faithful": source_ref, "viewer_natural": viewer_ref}
    for role, actual_ref in expected_outputs.items():
        recorded = outputs.get(role)
        if not isinstance(recorded, Mapping) or recorded.get("sha256") != actual_ref["sha256"]:
            raise PilotReportError(f"Candidate provenance {role} hash mismatch")


def _validate_metrics(
    metrics: Mapping[str, Any],
    *,
    error_ledger_ref: Mapping[str, Any],
    expected_blocks: int,
) -> None:
    if metrics.get("schema_name") != "translation-forensics/pilot-metrics":
        raise PilotReportError("Pilot metrics schema is invalid")
    _require_identity(metrics, "Pilot metrics", expected_blocks)
    if metrics.get("error_ledger_record_count") != expected_blocks:
        raise PilotReportError("Pilot error ledger does not preserve the fixed block denominator")
    if metrics.get("error_ledger_sha256") != error_ledger_ref["sha256"]:
        raise PilotReportError("Pilot metrics/error-ledger hash mismatch")
    status = metrics.get("status")
    result = metrics.get("result")
    if status == "awaiting-human-review" and result is not None:
        raise PilotReportError("Awaiting metrics cannot contain a pilot result")
    if status == "pilot-evaluated" and result not in {"pass", "fail", "inconclusive"}:
        raise PilotReportError("Pilot-evaluated metrics require pass/fail/inconclusive")
    if status not in {"awaiting-human-review", "pilot-evaluated"}:
        raise PilotReportError("Pilot metrics status is invalid")
    if result == "pass":
        gates = metrics.get("gates")
        if (
            metrics.get("all_required_gates_passed") is not True
            or not isinstance(gates, Mapping)
            or not gates
            or any(not isinstance(gate, Mapping) or gate.get("passed") is not True for gate in gates.values())
        ):
            raise PilotReportError("Pilot pass requires every non-compensating gate to pass")


def _artifact_rows(manifest: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    candidate = manifest["candidate_release"]
    packet = manifest["review_packet"]
    submissions = manifest["reviewer_submissions"]
    rows: list[tuple[str, Mapping[str, Any]]] = [
        ("동결 기준본", manifest["frozen_baseline"]),
        ("후보 source-faithful", candidate["source_faithful"]),
        ("후보 viewer-natural", candidate["viewer_natural"]),
        ("후보 provenance", candidate["provenance"]),
        ("시간축 정렬", manifest["timeline_alignment"]["artifact"]),
        ("원음 검수 매니페스트", packet["audio_review_manifest"]),
        ("블라인드 내부 키", packet["internal_key"]),
        ("최종 조정 원장", manifest["adjudication_record"]["artifact"]),
        ("파일럿 지표", manifest["pilot_metrics"]["artifact"]),
        ("MQM 오류 원장", manifest["pilot_metrics"]["error_ledger"]),
        ("자동 검증", manifest["automated_validation"]["artifact"]),
    ]
    rows.extend(
        (f"블라인드 평가 패킷 {index}", ref)
        for index, ref in enumerate(packet["blind_review_packets"], 1)
    )
    rows.extend(
        (f"검수자 확인서 {index}", ref)
        for index, ref in enumerate(submissions["attestations"], 1)
    )
    rows.extend(
        (f"검수자 판정 {index}", ref)
        for index, ref in enumerate(submissions["decisions"], 1)
    )
    return rows


def _render_report(manifest: Mapping[str, Any]) -> str:
    result = manifest["result"] if manifest["result"] is not None else "미판정"
    basis = manifest["outcome_basis"]
    gates = manifest["pilot_metrics"]["gates"]
    lines = [
        "# SSIS-908 파일럿 평가 보고서",
        "",
        f"- 수명주기 상태: `{manifest['lifecycle_status']}`",
        f"- 현재 판정: `{result}`",
        f"- 대상: SSIS-908 전체 {manifest['expected_block_count']}블록",
        f"- 판정 근거: {basis['reason']}",
        "- 승격 경계: 이 산출물은 final 또는 human-reviewed-candidate로 승격할 수 없다.",
        "- 주장 범위: 결과는 SSIS-908 파일럿에만 한정되며 다른 작품·13개 전체·시스템 전반으로 일반화할 수 없다.",
        "",
        "## 비상쇄 판정 근거",
        "",
        "평가 순서는 의미·화행 → 맥락·일관성 → 자연스러움 → 의미 재검사이며, 뒤 단계 결과로 앞 단계 실패를 상쇄하지 않는다.",
        "",
        "| 게이트 | 상태 | 관측값 | 계약 |",
        "|---|---:|---:|---:|",
        (
            "| 새 critical 의미·화행 오류 | "
            f"{gates['zero_new_critical_semantic_errors']['status']} | "
            f"{gates['zero_new_critical_semantic_errors']['observed_count']} | 0건 |"
        ),
        (
            "| critical/major 의미·맥락 오류 블록 감소 | "
            f"{gates['critical_major_error_block_reduction']['status']} | "
            f"{gates['critical_major_error_block_reduction']['observed_percent']} | 50% 이상 |"
        ),
        (
            "| 자연스러움 승률 | "
            f"{gates['naturalness_balance']['status']} | "
            f"{gates['naturalness_balance']['observed_win_percent']} | 65% 이상 |"
        ),
        (
            "| 자연스러움 패배율 | "
            f"{gates['naturalness_balance']['status']} | "
            f"{gates['naturalness_balance']['observed_loss_percent']} | 15% 이하 |"
        ),
        (
            "| 자동 검증 | "
            f"{manifest['automated_validation']['status']} | "
            f"{manifest['automated_validation']['passed']} | 통과 필수 |"
        ),
        "",
        "pass는 모든 계약을 충족할 때만 가능하다. 새 critical 오류, 자동 검증 실패 또는 수치 미달은 fail이다. 기준본 오류 분모가 0이거나 합의·입력이 불완전하면 inconclusive이며, 사람 검수가 반환되지 않은 상태는 awaiting-human-review로 유지한다.",
        "",
        "## 해시된 증거",
        "",
        "| 산출물 | 경로 | SHA-256 |",
        "|---|---|---|",
    ]
    for label, ref in _artifact_rows(manifest):
        lines.append(f"| {label} | `{ref['path']}` | `{ref['sha256']}` |")

    lines.extend(["", "## 미수행 검증", ""])
    if manifest["unperformed_validations"]:
        for item in manifest["unperformed_validations"]:
            lines.append(f"- `{item['code']}`: {item['reason']} 재개 조건: {item['resume_condition']}")
    else:
        lines.append("- 없음. 파일럿 계약에 필요한 검증 입력이 모두 기록되었다.")

    lines.extend(["", "## 남은 위험", ""])
    for item in manifest["remaining_risks"]:
        lines.append(f"- `{item['code']}`: {item['detail']}")

    lines.extend(
        [
            "",
            "## 상태 및 주장 경계",
            "",
            f"- 사람 직접 청취 완료: `{manifest['human_reviewed']}`",
            f"- pilot-evaluated 상태: `{manifest['pilot_evaluated']}`",
            f"- SSIS-908 파일럿 개선 주장 허용: `{manifest['claim_boundary']['quality_improvement_claim_allowed']}`",
            "- final 승격 허용: `false`",
            "- human-reviewed-candidate 승격 허용: `false`",
            "- 외부 전송 수행: `false`",
            "",
        ]
    )
    return "\n".join(lines)


def build_pilot_evaluation_report(
    *,
    project_root: Path,
    baseline_path: Path,
    candidate_source_path: Path,
    candidate_viewer_path: Path,
    candidate_provenance_path: Path,
    timeline_alignment_path: Path,
    audio_review_manifest_path: Path,
    blind_review_packet_paths: Iterable[Path],
    internal_key_path: Path,
    reviewer_attestation_paths: Iterable[Path],
    reviewer_decision_paths: Iterable[Path],
    adjudication_path: Path,
    metrics_path: Path,
    error_ledger_path: Path,
    automated_validation_path: Path,
    report_output_path: Path,
    expected_blocks: int = 298,
    expected_baseline_sha256: str = BASELINE_SHA256,
) -> tuple[dict[str, Any], str]:
    """Build the hash-complete SSIS-908 pilot receipt and Markdown report."""
    if expected_blocks < 1:
        raise PilotReportError("expected_blocks must be positive")
    packet_paths = list(blind_review_packet_paths)
    attestation_paths = list(reviewer_attestation_paths)
    decision_paths = list(reviewer_decision_paths)
    if len(packet_paths) != 2:
        raise PilotReportError("Exactly two blind review packets are required")
    if len(attestation_paths) > 2 or len(decision_paths) > 2:
        raise PilotReportError("At most two reviewer submissions are allowed")
    if len(attestation_paths) != len(decision_paths):
        raise PilotReportError("Reviewer attestations and decisions must be supplied together")

    baseline_ref = _file_ref(baseline_path, project_root)
    if baseline_ref["sha256"] != expected_baseline_sha256:
        raise PilotReportError("Frozen baseline SHA-256 mismatch")
    source_ref = _file_ref(candidate_source_path, project_root)
    viewer_ref = _file_ref(candidate_viewer_path, project_root)
    provenance_ref = _file_ref(candidate_provenance_path, project_root)
    timeline_ref = _file_ref(timeline_alignment_path, project_root)
    audio_ref = _file_ref(audio_review_manifest_path, project_root)
    packet_refs = [_file_ref(path, project_root) for path in packet_paths]
    key_ref = _file_ref(internal_key_path, project_root)
    attestation_refs = [_file_ref(path, project_root) for path in attestation_paths]
    decision_refs = [_file_ref(path, project_root) for path in decision_paths]
    adjudication_ref = _file_ref(adjudication_path, project_root)
    metrics_ref = _file_ref(metrics_path, project_root)
    ledger_ref = _file_ref(error_ledger_path, project_root)
    automated_ref = _file_ref(automated_validation_path, project_root)

    provenance = _read_json(candidate_provenance_path, "Candidate provenance")
    timeline = _read_json(timeline_alignment_path, "Timeline alignment")
    audio_review = _read_json(audio_review_manifest_path, "Audio review manifest")
    internal_key = _read_json(internal_key_path, "Blind internal key")
    metrics = _read_json(metrics_path, "Pilot metrics")
    automated = _read_json(automated_validation_path, "Automated validation")
    _validate_candidate_receipts(
        provenance,
        baseline_ref=baseline_ref,
        source_ref=source_ref,
        viewer_ref=viewer_ref,
        expected_blocks=expected_blocks,
    )
    _validate_metrics(metrics, error_ledger_ref=ledger_ref, expected_blocks=expected_blocks)
    for value, label in (
        (timeline, "Timeline alignment"),
        (audio_review, "Audio review manifest"),
        (internal_key, "Blind internal key"),
        (automated, "Automated validation"),
    ):
        _require_identity(value, label, expected_blocks)
    if _jsonl_count(adjudication_path) != expected_blocks:
        raise PilotReportError("Adjudication record does not preserve the fixed block denominator")
    if _jsonl_count(error_ledger_path) != expected_blocks:
        raise PilotReportError("Error ledger does not preserve the fixed block denominator")
    received_reviewers = _validate_reviewer_inputs(
        attestation_paths,
        decision_paths,
        packet_refs,
        expected_blocks=expected_blocks,
    )
    packet_readiness = [_blind_packet_ready(path, expected_blocks) for path in packet_paths]
    if internal_key.get("frozen_baseline", {}).get("sha256") not in {None, baseline_ref["sha256"]}:
        raise PilotReportError("Blind internal key baseline hash mismatch")
    automated_baseline = automated.get("frozen_baseline")
    if isinstance(automated_baseline, Mapping) and automated_baseline.get("sha256") != baseline_ref["sha256"]:
        raise PilotReportError("Automated validation baseline hash mismatch")

    timeline_ready = timeline.get("status") == "resolved" and timeline.get("clip_preparation_allowed") is True
    audio_ready = (
        audio_review.get("status") == "ready"
        and audio_review.get("clip_preparation_allowed") is True
        and audio_review.get("block_count") == expected_blocks
    )
    key_frozen = (
        internal_key.get("status") == "sealed"
        and internal_key.get("seal_status") == "sealed-until-adjudication"
        and internal_key.get("frozen_before_candidate_generation") is True
    )
    packets_ready = all(packet_readiness)
    reviewer_inputs_complete = received_reviewers == 2
    adjudication_complete = metrics.get("adjudicated_block_count") == expected_blocks
    automated_passed = automated.get("status") == "pass" and automated.get("automated_validation_passed") is True
    integrity_ready = (
        timeline_ready
        and audio_ready
        and packets_ready
        and key_frozen
        and reviewer_inputs_complete
        and adjudication_complete
    )

    metric_status = metrics["status"]
    metric_result = metrics["result"]
    if not automated_passed:
        status, result = "pilot-evaluated", "fail"
        reason = "전체 자동 검증 실패는 사람 점수로 상쇄할 수 없으므로 파일럿은 fail이다."
    elif metric_status == "awaiting-human-review":
        status, result = "awaiting-human-review", None
        reason = str(metrics["reason"])
    elif metric_result == "pass" and not integrity_ready:
        status, result = "pilot-evaluated", "inconclusive"
        reason = "수치 게이트는 통과했지만 동결·시간축·검수 입력 또는 조정 완결성 증거가 부족하다."
    else:
        status, result = "pilot-evaluated", metric_result
        reason = str(metrics["reason"])

    unperformed: list[dict[str, str]] = []
    if not timeline_ready:
        unperformed.append({
            "code": "timeline-alignment-unresolved",
            "reason": "사람 확인 앵커와 오프셋 맵으로 298블록 시간축이 resolved가 되지 않았다.",
            "resume_condition": "새 파일럿 버전에서 사람 앵커·승인 offset map을 기록하고 clip_preparation_allowed=true를 확인한다.",
        })
    if not audio_ready:
        unperformed.append({
            "code": "full-audio-review-packet-not-ready",
            "reason": "298블록 원음·일본어·장면 문맥 패킷 생성과 확인을 수행하지 못했다.",
            "resume_condition": "resolved 시간축에서 298개 청취 클립과 문맥 레코드를 생성한다.",
        })
    if not packets_ready:
        unperformed.append({
            "code": "blind-review-packets-not-reviewable",
            "reason": "두 블라인드 패킷 중 하나 이상이 298블록 검수 가능 상태가 아니다.",
            "resume_condition": "resolved 원음 패킷과 후보 생성 전 동결 키로 새 버전의 패킷 두 벌을 생성한다.",
        })
    if not reviewer_inputs_complete:
        unperformed.append({
            "code": "two-independent-reviews-not-returned",
            "reason": f"독립 검수자 입력이 {received_reviewers}/2건만 기록되었다.",
            "resume_condition": "일본어 청취 가능한 두 검수자의 전수 판정과 독립성 확인서를 로컬로 회수한다.",
        })
    if not adjudication_complete:
        unperformed.append({
            "code": "full-adjudication-incomplete",
            "reason": f"완료된 최종 조정 블록은 {metrics.get('adjudicated_block_count', 0)}/{expected_blocks}개다.",
            "resume_condition": "두 검수자의 불일치를 합의 조정하고 미합의 항목은 unresolved로 기록한다.",
        })
    if metric_status == "awaiting-human-review":
        unperformed.append({
            "code": "pilot-balance-contract-not-evaluable",
            "reason": "새 critical, 오류 블록 감소율, 자연스러움 승·패율을 확정하지 못했다.",
            "resume_condition": "완결된 298블록 조정 원장으로 비상쇄 균형 계약을 다시 계산한다.",
        })

    risks = [
        {
            "code": "pilot-only-external-validity",
            "detail": "SSIS-908 단일 작품 파일럿은 다른 작품, 13개 전체 또는 시스템 전반의 품질 향상 근거가 아니다.",
        },
        {
            "code": "no-final-promotion",
            "detail": "pass인 경우에도 상태는 pilot-evaluated에 머물며 final 또는 human-reviewed-candidate로 승격할 수 없다.",
        },
    ]
    if not key_frozen:
        risks.append({
            "code": "pre-candidate-freeze-not-established",
            "detail": "평가 계약과 블라인드 내부 키가 후보 생성 전에 동결됐다는 증거가 없어 현재 버전의 평가 무결성이 제한된다.",
        })
    if result != "pass":
        risks.append({
            "code": "quality-improvement-not-established",
            "detail": "현재 증거로는 개선 후보가 동결 기준본보다 낫다고 주장할 수 없다.",
        })
    warning_counts = automated.get("non_blocking_warning_counts")
    if isinstance(warning_counts, Mapping) and any(isinstance(value, int) and value > 0 for value in warning_counts.values()):
        risks.append({
            "code": "automated-non-blocking-warnings",
            "detail": "자동 검증의 비차단 경고는 통과 판정을 의미 정확성 증거로 바꾸지 않으며 사람 검토가 필요하다.",
        })

    human_reviewed = reviewer_inputs_complete and adjudication_complete
    quality_claim_allowed = result == "pass" and integrity_ready and automated_passed
    manifest: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "title_id": TITLE_ID,
        "status": status,
        "result": result,
        "lifecycle_status": status if result is None else f"{status}/{result}",
        "pilot_evaluated": status == "pilot-evaluated",
        "human_reviewed": human_reviewed,
        "expected_block_count": expected_blocks,
        "frozen_baseline": baseline_ref,
        "candidate_release": {
            "status": provenance.get("status"),
            "source_faithful": source_ref,
            "viewer_natural": viewer_ref,
            "provenance": provenance_ref,
            "models": dict(provenance["models"]),
            "pipeline_order": list(provenance["pipeline_order"]),
        },
        "timeline_alignment": {
            "artifact": timeline_ref,
            "status": timeline.get("status"),
            "clip_preparation_allowed": timeline.get("clip_preparation_allowed") is True,
        },
        "review_packet": {
            "status": "ready" if timeline_ready and audio_ready and packets_ready and key_frozen else "blocked",
            "local_only": True,
            "external_delivery_performed": False,
            "audio_review_manifest": audio_ref,
            "blind_review_packets": packet_refs,
            "internal_key": key_ref,
            "key_frozen_before_candidate_generation": key_frozen,
        },
        "reviewer_submissions": {
            "expected_count": 2,
            "received_count": received_reviewers,
            "complete": reviewer_inputs_complete,
            "attestations": attestation_refs,
            "decisions": decision_refs,
        },
        "adjudication_record": {
            "artifact": adjudication_ref,
            "expected_block_count": expected_blocks,
            "adjudicated_block_count": int(metrics.get("adjudicated_block_count", 0)),
            "blocking_block_count": len(metrics.get("blocking_blocks", [])),
            "unresolved_block_count": len(metrics.get("unresolved_blocks", [])),
        },
        "pilot_metrics": {
            "artifact": metrics_ref,
            "error_ledger": ledger_ref,
            "status": metric_status,
            "result": metric_result,
            "gate_order": list(metrics.get("gate_order", [])),
            "gates": dict(metrics.get("gates", {})),
            "reason": str(metrics.get("reason", "")),
        },
        "automated_validation": {
            "artifact": automated_ref,
            "status": automated.get("status"),
            "passed": automated_passed,
            "failed_checks": list(automated.get("failed_checks", [])),
        },
        "outcome_basis": {
            "evaluation_order": EVALUATION_ORDER,
            "reason": reason,
            "pass_contract": {
                "new_critical_semantic_error_count": 0,
                "minimum_error_block_reduction_percent": 50.0,
                "minimum_naturalness_win_percent": 65.0,
                "maximum_naturalness_loss_percent": 15.0,
                "automated_validation_required": True,
                "all_298_blocks_required": True,
            },
            "fail_conditions": [
                "new critical semantic/speech-act error",
                "automated validation failure",
                "any numeric balance threshold missed",
            ],
            "inconclusive_conditions": [
                "baseline critical/major error block denominator is zero",
                "reviewer disagreement remains unresolved",
                "evaluation input or evidence is incomplete",
            ],
        },
        "claim_boundary": {
            "scope": "SSIS-908 pilot only",
            "quality_improvement_claim_allowed": quality_claim_allowed,
            "cross_title_generalization_allowed": False,
            "final_promotion_allowed": False,
            "human_reviewed_candidate_promotion_allowed": False,
            "human_final_allowed": False,
        },
        "unperformed_validations": unperformed,
        "remaining_risks": risks,
    }
    report = _render_report(manifest)
    report_bytes = report.encode("utf-8")
    report_relative = _output_path(report_output_path, project_root)
    manifest["generated_artifacts"] = {
        "pilot_evaluation_report": {
            "path": report_relative,
            "sha256": hashlib.sha256(report_bytes).hexdigest(),
            "size_bytes": len(report_bytes),
        }
    }
    return manifest, report


def write_pilot_evaluation_report(
    manifest_path: Path,
    report_path: Path,
    manifest: Mapping[str, Any],
    report: str,
    *,
    schema_path: Path,
) -> None:
    """Validate and write the two final pilot receipts without overwriting."""
    for path in (manifest_path, report_path):
        if path.exists():
            raise FileExistsError(f"Existing pilot evaluation artifact will not be overwritten: {path}")
    report_bytes = report.encode("utf-8")
    report_ref = manifest.get("generated_artifacts", {}).get("pilot_evaluation_report", {})
    if report_ref.get("sha256") != hashlib.sha256(report_bytes).hexdigest() or report_ref.get(
        "size_bytes"
    ) != len(report_bytes):
        raise PilotReportError("Report receipt does not match the rendered Markdown")
    schema = _read_json(schema_path, "Pilot evaluation manifest schema")
    Draft202012Validator.check_schema(schema)
    first_error = next(iter(Draft202012Validator(schema).iter_errors(dict(manifest))), None)
    if first_error is not None:
        location = "/".join(str(part) for part in first_error.absolute_path) or "<root>"
        raise PilotReportError(f"Pilot evaluation manifest schema violation at {location}: {first_error.message}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    manifest_path.write_text(
        json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
