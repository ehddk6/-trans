from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .manifest import sha256_file
from .srt import SubtitleBlock, compare_structure, parse_srt


KEY_SCHEMA_NAME = "translation-forensics/pilot-blind-internal-key"
PACK_SCHEMA_NAME = "translation-forensics/pilot-blind-review-pack"
SUBMISSION_SCHEMA_NAME = "translation-forensics/pilot-reviewer-submission"
DECISION_SCHEMA_NAME = "translation-forensics/pilot-reviewer-decision"
ADJUDICATION_SCHEMA_NAME = "translation-forensics/pilot-adjudication-record"
SCHEMA_VERSION = "1"
BASELINE_SHA256 = "8653a42dc952152994c75e9d43265c49eddc1b7d581cf05d8012fd87e3c3e25b"
REVIEWER_SLOTS = ("reviewer-1", "reviewer-2")
SEMANTIC_GATES = {"both-pass", "a-only", "b-only", "both-fail", "unresolved", "nonverbal"}
CONTEXT_GATES = SEMANTIC_GATES
NATURALNESS_CHOICES = {"candidate-a", "candidate-b", "tie", "unresolved", "nonverbal"}
MQM_DIMENSIONS = {"semantic_fidelity", "contextual_consistency", "natural_korean"}
MQM_SEVERITIES = {"critical", "major", "minor"}
EVALUATION_ORDER = [
    "semantic_fidelity",
    "contextual_consistency",
    "natural_korean",
    "semantic_recheck",
]
MQM_SEVERITY_DEFINITIONS = {
    "critical": "의미·화행·극성·주체·대상·강도를 뒤집거나 필수 내용을 심각하게 왜곡해 장면 이해를 무너뜨리는 오류",
    "major": "핵심 의미나 문맥을 유의미하게 손상하지만 장면 전체를 완전히 뒤집지는 않는 오류",
    "minor": "핵심 의미·문맥을 보존하면서 자연스러움이나 표현 품질을 낮추는 국소 오류",
}


class PilotBlindReviewError(ValueError):
    """Raised when the blind human-review contract is incomplete or inconsistent."""


class PilotBlindReviewBlocked(PilotBlindReviewError):
    """Raised when the timeline/audio hard gate prevents review packet creation."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise PilotBlindReviewError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise PilotBlindReviewError(f"{path}:{line_number}: JSON object required")
        values.append(value)
    return values


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
        newline="\n",
    )


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.expanduser().resolve()
    root = project_root.expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.name


def _file_ref(path: Path, project_root: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": _portable_path(resolved, project_root),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _ensure_targets_absent(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Pilot review artifacts will not be overwritten: {', '.join(existing)}")


def _validate_pair(
    structure_path: Path,
    baseline_path: Path,
    candidate_path: Path,
    *,
    expected_blocks: int,
    expected_baseline_sha256: str,
) -> tuple[list[SubtitleBlock], list[SubtitleBlock], list[SubtitleBlock]]:
    if sha256_file(baseline_path) != expected_baseline_sha256:
        raise PilotBlindReviewError("Frozen baseline SHA-256 mismatch")
    structure, _, _ = parse_srt(structure_path)
    baseline, _, _ = parse_srt(baseline_path)
    candidate, _, _ = parse_srt(candidate_path)
    if len(structure) != expected_blocks or len(baseline) != expected_blocks or len(candidate) != expected_blocks:
        raise PilotBlindReviewError(f"Blind pilot requires exactly {expected_blocks} blocks")
    if [block.number for block in structure] != list(range(1, expected_blocks + 1)):
        raise PilotBlindReviewError("Locked block numbers must be consecutive from 1")
    if not compare_structure(structure, baseline)["pass"]:
        raise PilotBlindReviewError("Frozen baseline does not match locked structure")
    if not compare_structure(structure, candidate)["pass"]:
        raise PilotBlindReviewError("Improvement candidate does not match locked structure")
    return structure, baseline, candidate


def _validate_candidate_provenance(candidate_path: Path, provenance_path: Path, expected_blocks: int) -> dict[str, Any]:
    provenance = _read_json(provenance_path)
    if provenance.get("schema_name") != "translation-forensics/pilot-candidate-provenance":
        raise PilotBlindReviewError("Invalid pilot candidate provenance")
    if provenance.get("title_id") != "SSIS-908" or provenance.get("block_count") != expected_blocks:
        raise PilotBlindReviewError("Pilot candidate provenance identity/coverage mismatch")
    if provenance.get("status") != "candidate-generated" or provenance.get("pilot_evaluated") is not False:
        raise PilotBlindReviewError("Candidate provenance is not in the pre-evaluation state")
    output_ref = provenance.get("outputs", {}).get("viewer_natural", {})
    if output_ref.get("sha256") != sha256_file(candidate_path):
        raise PilotBlindReviewError("Candidate provenance viewer-natural hash mismatch")
    if provenance.get("human_reviewed") is not False or provenance.get("final_promotion_allowed") is not False:
        raise PilotBlindReviewError("Candidate provenance crossed a forbidden human/final boundary")
    return provenance


def _mapping_for_seed(title_id: str, random_seed: str, expected_blocks: int) -> list[dict[str, Any]]:
    mapping: list[dict[str, Any]] = []
    for number in range(1, expected_blocks + 1):
        digest = hashlib.sha256(f"{title_id}\0{random_seed}\0{number}".encode("utf-8")).digest()
        a_variant, b_variant = ("baseline", "improvement") if digest[0] % 2 == 0 else ("improvement", "baseline")
        mapping.append({
            "block_number": number,
            "candidate_a_variant": a_variant,
            "candidate_b_variant": b_variant,
        })
    return mapping


def _key_commitment_payload(key: dict[str, Any]) -> dict[str, Any]:
    return {
        "title_id": key.get("title_id"),
        "expected_block_count": key.get("expected_block_count"),
        "randomization_algorithm": key.get("randomization_algorithm"),
        "randomization_seed": key.get("randomization_seed"),
        "frozen_baseline_sha256": key.get("frozen_baseline", {}).get("sha256"),
        "evaluation_contract_sha256": key.get("evaluation_contract", {}).get("sha256"),
        "mqm_schema_sha256": key.get("mqm_schema", {}).get("sha256"),
        "candidate_mapping": key.get("candidate_mapping"),
    }


def _validate_key(key: dict[str, Any], *, expected_blocks: int, expected_baseline_sha256: str) -> None:
    if key.get("schema_name") != KEY_SCHEMA_NAME or key.get("schema_version") != SCHEMA_VERSION:
        raise PilotBlindReviewError("Invalid pilot blind internal key schema")
    if key.get("title_id") != "SSIS-908" or key.get("expected_block_count") != expected_blocks:
        raise PilotBlindReviewError("Internal key identity/coverage mismatch")
    if key.get("status") != "sealed" or key.get("seal_status") != "sealed-until-adjudication":
        raise PilotBlindReviewError("Internal key is not sealed")
    if key.get("frozen_before_candidate_generation") is not True:
        raise PilotBlindReviewError("Internal key was not frozen before candidate generation")
    if key.get("frozen_baseline", {}).get("sha256") != expected_baseline_sha256:
        raise PilotBlindReviewError("Internal key baseline hash mismatch")
    mapping = key.get("candidate_mapping")
    if not isinstance(mapping, list) or len(mapping) != expected_blocks:
        raise PilotBlindReviewError("Internal key mapping coverage mismatch")
    if [row.get("block_number") for row in mapping] != list(range(1, expected_blocks + 1)):
        raise PilotBlindReviewError("Internal key mapping block order mismatch")
    for row in mapping:
        if {row.get("candidate_a_variant"), row.get("candidate_b_variant")} != {"baseline", "improvement"}:
            raise PilotBlindReviewError("Internal key mapping is not a baseline/improvement permutation")
    if key.get("key_commitment_sha256") != _canonical_sha256(_key_commitment_payload(key)):
        raise PilotBlindReviewError("Internal key commitment mismatch")


def initialize_pilot_blind_internal_key(
    *,
    title_id: str,
    baseline_path: Path,
    candidate_expected_path: Path,
    evaluation_contract_path: Path,
    mqm_schema_path: Path,
    output_path: Path,
    project_root: Path,
    random_seed: str,
    expected_blocks: int = 298,
    expected_baseline_sha256: str = BASELINE_SHA256,
) -> dict[str, Any]:
    """Freeze the randomization and evaluation contract before candidate creation."""
    if title_id != "SSIS-908":
        raise PilotBlindReviewError("This frozen pilot contract is limited to SSIS-908")
    if output_path.exists():
        raise FileExistsError(f"Existing internal key will not be overwritten: {output_path}")
    if candidate_expected_path.exists():
        raise PilotBlindReviewError("Internal key must be frozen before the improvement candidate exists")
    if not random_seed.strip():
        raise PilotBlindReviewError("A non-empty frozen randomization seed is required")
    baseline, _, _ = parse_srt(baseline_path)
    if len(baseline) != expected_blocks or sha256_file(baseline_path) != expected_baseline_sha256:
        raise PilotBlindReviewError("Frozen baseline count or SHA-256 mismatch")
    mapping = _mapping_for_seed(title_id, random_seed, expected_blocks)
    key: dict[str, Any] = {
        "schema_name": KEY_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "title_id": title_id,
        "status": "sealed",
        "seal_status": "sealed-until-adjudication",
        "frozen_before_candidate_generation": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_block_count": expected_blocks,
        "randomization_algorithm": "sha256-first-byte-parity-v1",
        "randomization_seed": random_seed,
        "frozen_baseline": _file_ref(baseline_path, project_root),
        "candidate_expected_path": _portable_path(candidate_expected_path, project_root),
        "evaluation_contract": _file_ref(evaluation_contract_path, project_root),
        "evaluation_order": EVALUATION_ORDER,
        "mqm_schema": _file_ref(mqm_schema_path, project_root),
        "mqm_severity_definitions": MQM_SEVERITY_DEFINITIONS,
        "candidate_mapping": mapping,
        "reviewer_visible": False,
        "external_delivery_performed": False,
        "unseal_condition": "two complete independent submissions exist and disagreement adjudication has started",
    }
    key["key_commitment_sha256"] = _canonical_sha256(_key_commitment_payload(key))
    _write_json(output_path, key)
    return key


def _safe_manifest_path(manifest_path: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise PilotBlindReviewBlocked(f"Audio review {label} path is missing")
    base = manifest_path.parent.resolve()
    target = (base / raw).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise PilotBlindReviewBlocked(f"Audio review {label} path escapes its packet") from exc
    if not target.is_file():
        raise PilotBlindReviewBlocked(f"Audio review {label} file is missing")
    return target


def _load_audio_records(audio_manifest_path: Path, expected_blocks: int) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    manifest = _read_json(audio_manifest_path)
    ready = (
        manifest.get("status") == "ready"
        and manifest.get("clip_preparation_allowed") is True
        and manifest.get("block_count") == expected_blocks
        and manifest.get("all_blocks_have_audio") is True
        and manifest.get("all_blocks_have_japanese") is True
        and manifest.get("all_blocks_have_scene_context") is True
    )
    if not ready:
        raise PilotBlindReviewBlocked("Timeline/audio review gate is not ready for 298-block blind evaluation")
    records_ref = manifest.get("records")
    if not isinstance(records_ref, dict):
        raise PilotBlindReviewBlocked("Audio review records reference is missing")
    records_path = _safe_manifest_path(audio_manifest_path, records_ref.get("path"), "records")
    if records_ref.get("sha256") != sha256_file(records_path):
        raise PilotBlindReviewBlocked("Audio review records hash mismatch")
    records = _read_jsonl(records_path)
    if len(records) != expected_blocks or [row.get("block_number") for row in records] != list(range(1, expected_blocks + 1)):
        raise PilotBlindReviewBlocked("Audio review block coverage mismatch")
    for row in records:
        audio = row.get("audio")
        if not isinstance(audio, dict):
            raise PilotBlindReviewBlocked(f"Block {row.get('block_number')}: audio reference is missing")
        clip = _safe_manifest_path(audio_manifest_path, audio.get("path"), "clip")
        if audio.get("sha256") != sha256_file(clip) or not str(row.get("japanese", "")).strip() or not row.get("scene_context"):
            raise PilotBlindReviewBlocked(f"Block {row.get('block_number')}: audio/Japanese/context evidence is incomplete")
    return manifest, records, audio_manifest_path.parent.resolve()


def _zip_write(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def _review_template(pack_id: str, block_number: int) -> dict[str, Any]:
    return {
        "schema_name": DECISION_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "pack_id": pack_id,
        "review_id": f"SSIS-908-{block_number:04d}",
        "block_number": block_number,
        "decision_status": "unresolved",
        "direct_listening_completed": False,
        "semantic_gate": "unresolved",
        "context_gate": "unresolved",
        "naturalness_preference": "unresolved",
        "nonverbal": None,
        "mqm_errors": [],
        "review_note": "",
    }


def _pack_readme() -> str:
    return (
        "# SSIS-908 blind listening review\n\n"
        "후보 출처를 추측하거나 내부 키를 열지 마십시오. 각 블록의 원음을 직접 듣고 일본어·장면 문맥을 확인한 뒤 "
        "semantic_fidelity → contextual_consistency → natural_korean 순서로 판정하십시오. "
        "자연스러움은 앞선 의미·문맥 실패를 상쇄하지 않습니다. submission.template.jsonl의 모든 행을 작성하고 "
        "별도 독립성 확인서와 함께 로컬로 반환하십시오. 시스템은 이 패킷을 외부로 전송하지 않았습니다.\n"
    )


def _build_one_pack(
    *,
    reviewer_slot: str,
    output_path: Path,
    structure: list[SubtitleBlock],
    baseline: list[SubtitleBlock],
    candidate: list[SubtitleBlock],
    audio_records: list[dict[str, Any]],
    audio_root: Path,
    key: dict[str, Any],
    audio_manifest_path: Path,
) -> dict[str, Any]:
    mapping = {int(row["block_number"]): row for row in key["candidate_mapping"]}
    rows: list[dict[str, Any]] = []
    templates: list[dict[str, Any]] = []
    audio_members: list[tuple[str, Path]] = []
    pack_id = f"SSIS-908-pilot-{reviewer_slot}-v1"
    for locked, base, improved, audio_record in zip(structure, baseline, candidate, audio_records):
        assignment = mapping[locked.number]
        variants = {"baseline": base.text, "improvement": improved.text}
        clip_source = (audio_root / audio_record["audio"]["path"]).resolve()
        clip_member = f"audio/block-{locked.number:04d}{clip_source.suffix.lower()}"
        rows.append({
            "review_id": f"SSIS-908-{locked.number:04d}",
            "block_number": locked.number,
            "timecode": f"{locked.start} --> {locked.end}",
            "japanese": audio_record["japanese"],
            "scene_context": audio_record["scene_context"],
            "audio": {
                "path": clip_member,
                "sha256": sha256_file(clip_source),
                "size_bytes": clip_source.stat().st_size,
            },
            "candidate_a": variants[assignment["candidate_a_variant"]],
            "candidate_b": variants[assignment["candidate_b_variant"]],
        })
        templates.append(_review_template(pack_id, locked.number))
        audio_members.append((clip_member, clip_source))

    review_bytes = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    template_bytes = b"".join(_canonical_bytes(row) + b"\n" for row in templates)
    manifest = {
        "schema_name": PACK_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "title_id": "SSIS-908",
        "pack_id": pack_id,
        "reviewer_slot": reviewer_slot,
        "status": "awaiting-human-review",
        "local_only": True,
        "external_delivery_performed": False,
        "candidate_provenance_hidden": True,
        "randomization_seed_hidden": True,
        "same_blind_layout_for_both_reviewers": True,
        "expected_block_count": len(rows),
        "block_count": len(rows),
        "direct_listening_required_for_every_block": True,
        "evaluation_order": EVALUATION_ORDER,
        "mqm_severity_definitions": MQM_SEVERITY_DEFINITIONS,
        "randomization_commitment_sha256": key["key_commitment_sha256"],
        "blind_rows_sha256": hashlib.sha256(review_bytes).hexdigest(),
        "audio_manifest_sha256": sha256_file(audio_manifest_path),
        "human_reviewed": False,
        "pilot_evaluated": False,
        "final_promotion_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "x") as archive:
        _zip_write(archive, "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        _zip_write(archive, "review.jsonl", review_bytes)
        _zip_write(archive, "submission.template.jsonl", template_bytes)
        _zip_write(archive, "README.md", _pack_readme().encode("utf-8"))
        for member, source in audio_members:
            _zip_write(archive, member, source.read_bytes())
    return manifest


def build_pilot_blind_review_packets(
    *,
    title_id: str,
    structure_path: Path,
    baseline_path: Path,
    candidate_path: Path,
    candidate_provenance_path: Path,
    audio_manifest_path: Path,
    internal_key_path: Path,
    output_dir: Path,
    expected_blocks: int = 298,
    expected_baseline_sha256: str = BASELINE_SHA256,
) -> dict[str, Any]:
    """Create two local-only packs after timeline/audio and pre-frozen-key gates pass."""
    if title_id != "SSIS-908":
        raise PilotBlindReviewError("This frozen pilot contract is limited to SSIS-908")
    targets = [output_dir / "reviewer-1.pack.zip", output_dir / "reviewer-2.pack.zip"]
    _ensure_targets_absent(targets)
    structure, baseline, candidate = _validate_pair(
        structure_path,
        baseline_path,
        candidate_path,
        expected_blocks=expected_blocks,
        expected_baseline_sha256=expected_baseline_sha256,
    )
    _validate_candidate_provenance(candidate_path, candidate_provenance_path, expected_blocks)
    key = _read_json(internal_key_path)
    _validate_key(key, expected_blocks=expected_blocks, expected_baseline_sha256=expected_baseline_sha256)
    audio_manifest, audio_records, audio_root = _load_audio_records(audio_manifest_path, expected_blocks)
    if audio_manifest.get("title_id") != title_id:
        raise PilotBlindReviewBlocked("Audio review title identity mismatch")
    manifests = [
        _build_one_pack(
            reviewer_slot=slot,
            output_path=target,
            structure=structure,
            baseline=baseline,
            candidate=candidate,
            audio_records=audio_records,
            audio_root=audio_root,
            key=key,
            audio_manifest_path=audio_manifest_path,
        )
        for slot, target in zip(REVIEWER_SLOTS, targets)
    ]
    return {
        "status": "awaiting-human-review",
        "title_id": title_id,
        "block_count": expected_blocks,
        "packs": [str(path) for path in targets],
        "internal_key": str(internal_key_path),
        "randomization_commitment_sha256": key["key_commitment_sha256"],
        "same_blind_layout": manifests[0]["blind_rows_sha256"] == manifests[1]["blind_rows_sha256"],
        "local_only": True,
        "external_delivery_performed": False,
        "human_reviewed": False,
        "pilot_evaluated": False,
        "final_promotion_allowed": False,
    }


def _blocked_pack_manifest(reviewer_slot: str, expected_blocks: int, reason: str) -> dict[str, Any]:
    return {
        "schema_name": PACK_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "title_id": "SSIS-908",
        "pack_id": f"SSIS-908-pilot-{reviewer_slot}-v1",
        "reviewer_slot": reviewer_slot,
        "status": "blocked",
        "local_only": True,
        "external_delivery_performed": False,
        "candidate_provenance_hidden": True,
        "randomization_seed_hidden": True,
        "same_blind_layout_for_both_reviewers": True,
        "expected_block_count": expected_blocks,
        "block_count": 0,
        "direct_listening_required_for_every_block": True,
        "evaluation_order": EVALUATION_ORDER,
        "mqm_severity_definitions": MQM_SEVERITY_DEFINITIONS,
        "human_reviewed": False,
        "pilot_evaluated": False,
        "final_promotion_allowed": False,
        "errors": [reason],
        "resume_condition": "resolve timeline alignment, build all audio clips, then create a new version with a pre-frozen key",
    }


def write_blocked_pilot_review_artifacts(
    *,
    title_id: str,
    structure_path: Path,
    baseline_path: Path,
    candidate_path: Path,
    candidate_provenance_path: Path,
    audio_manifest_path: Path,
    evaluation_contract_path: Path,
    mqm_schema_path: Path,
    blind_review_dir: Path,
    internal_key_path: Path,
    adjudication_output_path: Path,
    project_root: Path,
    random_seed: str,
    reason: str,
    expected_blocks: int = 298,
    expected_baseline_sha256: str = BASELINE_SHA256,
) -> dict[str, Any]:
    """Materialize honest stop records without creating review rows or human claims."""
    if title_id != "SSIS-908":
        raise PilotBlindReviewError("This frozen pilot contract is limited to SSIS-908")
    targets = [
        blind_review_dir / "reviewer-1.pack.zip",
        blind_review_dir / "reviewer-2.pack.zip",
        internal_key_path,
        adjudication_output_path,
    ]
    _ensure_targets_absent(targets)
    structure, _, _ = _validate_pair(
        structure_path,
        baseline_path,
        candidate_path,
        expected_blocks=expected_blocks,
        expected_baseline_sha256=expected_baseline_sha256,
    )
    _validate_candidate_provenance(candidate_path, candidate_provenance_path, expected_blocks)
    if not reason.strip():
        raise PilotBlindReviewError("Blocked review artifacts require a concrete reason")
    mapping = _mapping_for_seed(title_id, random_seed, expected_blocks)
    blocked_key: dict[str, Any] = {
        "schema_name": KEY_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "title_id": title_id,
        "status": "blocked",
        "seal_status": "sealed-blocked-not-for-unblinding",
        "frozen_before_candidate_generation": False,
        "expected_block_count": expected_blocks,
        "randomization_algorithm": "sha256-first-byte-parity-v1",
        "randomization_seed": random_seed,
        "frozen_baseline": _file_ref(baseline_path, project_root),
        "evaluation_contract": _file_ref(evaluation_contract_path, project_root),
        "evaluation_order": EVALUATION_ORDER,
        "mqm_schema": _file_ref(mqm_schema_path, project_root),
        "mqm_severity_definitions": MQM_SEVERITY_DEFINITIONS,
        "candidate_mapping": mapping,
        "reviewer_visible": False,
        "external_delivery_performed": False,
        "errors": [reason],
        "resume_condition": "create a new pilot version only after timeline/audio readiness and pre-candidate key freeze",
    }
    blocked_key["key_commitment_sha256"] = _canonical_sha256(_key_commitment_payload(blocked_key))
    blind_review_dir.mkdir(parents=True, exist_ok=True)
    for slot in REVIEWER_SLOTS:
        target = blind_review_dir / f"{slot}.pack.zip"
        manifest = _blocked_pack_manifest(slot, expected_blocks, reason)
        with zipfile.ZipFile(target, "x") as archive:
            _zip_write(archive, "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            _zip_write(
                archive,
                "README.md",
                ("# Blocked blind review pack\n\n이 파일은 검수 패킷이 아닙니다. " + reason + "\n").encode("utf-8"),
            )
    _write_json(internal_key_path, blocked_key)
    unresolved = [{
        "schema_name": ADJUDICATION_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "title_id": title_id,
        "block_number": block.number,
        "timecode": f"{block.start} --> {block.end}",
        "adjudication_status": "unresolved",
        "reviewer_ids": [],
        "direct_listening_completed_by": [],
        "semantic_gate": "unresolved",
        "context_gate": "unresolved",
        "naturalness_outcome": "unresolved",
        "nonverbal": None,
        "mqm_errors": [],
        "consensus_recorded": False,
        "internal_key_unsealed_for_adjudication": False,
        "human_reviewed": False,
        "reason": reason,
    } for block in structure]
    _write_jsonl(adjudication_output_path, unresolved)
    return {
        "status": "blocked",
        "title_id": title_id,
        "expected_block_count": expected_blocks,
        "reviewable_blocks": 0,
        "unresolved_blocks": expected_blocks,
        "packs": [str(blind_review_dir / f"{slot}.pack.zip") for slot in REVIEWER_SLOTS],
        "internal_key": str(internal_key_path),
        "adjudication_output": str(adjudication_output_path),
        "human_reviewed": False,
        "pilot_evaluated": False,
        "final_promotion_allowed": False,
        "errors": [reason],
    }


def _read_pack(pack_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with zipfile.ZipFile(pack_path) as archive:
        names = set(archive.namelist())
        if not {"manifest.json", "review.jsonl"}.issubset(names):
            raise PilotBlindReviewError("Reviewer pack is blocked or incomplete")
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        rows = [json.loads(line) for line in archive.read("review.jsonl").decode("utf-8").splitlines() if line.strip()]
    if manifest.get("schema_name") != PACK_SCHEMA_NAME or manifest.get("status") != "awaiting-human-review":
        raise PilotBlindReviewError("Reviewer pack is not ready for submissions")
    review_bytes = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    if manifest.get("blind_rows_sha256") != hashlib.sha256(review_bytes).hexdigest():
        raise PilotBlindReviewError("Reviewer pack blind rows hash mismatch")
    return manifest, rows


def _validate_mqm_errors(errors_value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(errors_value, list):
        raise PilotBlindReviewError(f"{label}: mqm_errors must be an array")
    normalized: list[dict[str, Any]] = []
    for index, error in enumerate(errors_value, 1):
        if not isinstance(error, dict):
            raise PilotBlindReviewError(f"{label}: MQM error {index} must be an object")
        if error.get("candidate") not in {"candidate-a", "candidate-b"}:
            raise PilotBlindReviewError(f"{label}: MQM error {index} candidate is invalid")
        if error.get("dimension") not in MQM_DIMENSIONS or error.get("severity") not in MQM_SEVERITIES:
            raise PilotBlindReviewError(f"{label}: MQM error {index} dimension/severity is invalid")
        if not str(error.get("error_type", "")).strip() or not str(error.get("detail", "")).strip():
            raise PilotBlindReviewError(f"{label}: MQM error {index} lacks type/detail")
        evidence_refs = error.get("evidence_refs")
        if not isinstance(evidence_refs, list) or not evidence_refs or not all(str(item).strip() for item in evidence_refs):
            raise PilotBlindReviewError(f"{label}: MQM error {index} lacks evidence_refs")
        normalized.append({
            "candidate": error["candidate"],
            "dimension": error["dimension"],
            "severity": error["severity"],
            "error_type": str(error["error_type"]).strip(),
            "detail": str(error["detail"]).strip(),
            "evidence_refs": sorted(str(item).strip() for item in evidence_refs),
        })
    return sorted(normalized, key=lambda row: _canonical_bytes(row))


def _validate_decision(row: dict[str, Any], expected_pack_id: str, expected_review_id: str, block_number: int) -> dict[str, Any]:
    label = f"Block {block_number}"
    if row.get("schema_name") != DECISION_SCHEMA_NAME or row.get("schema_version") != SCHEMA_VERSION:
        raise PilotBlindReviewError(f"{label}: invalid reviewer decision schema")
    if row.get("pack_id") != expected_pack_id or row.get("review_id") != expected_review_id or row.get("block_number") != block_number:
        raise PilotBlindReviewError(f"{label}: reviewer decision identity mismatch")
    if row.get("decision_status") not in {"completed", "unresolved"}:
        raise PilotBlindReviewError(f"{label}: decision_status must be completed or unresolved")
    if row.get("direct_listening_completed") is not True:
        raise PilotBlindReviewError(f"{label}: direct listening is required")
    if row.get("semantic_gate") not in SEMANTIC_GATES or row.get("context_gate") not in CONTEXT_GATES:
        raise PilotBlindReviewError(f"{label}: semantic/context gate is invalid")
    if row.get("naturalness_preference") not in NATURALNESS_CHOICES:
        raise PilotBlindReviewError(f"{label}: naturalness preference is invalid")
    if row.get("nonverbal") not in {True, False}:
        raise PilotBlindReviewError(f"{label}: nonverbal must be explicitly true or false")
    if row["nonverbal"]:
        if (row["semantic_gate"], row["context_gate"], row["naturalness_preference"]) != ("nonverbal", "nonverbal", "nonverbal"):
            raise PilotBlindReviewError(f"{label}: nonverbal decisions must use nonverbal for every quality field")
    elif "nonverbal" in {row["semantic_gate"], row["context_gate"], row["naturalness_preference"]}:
        raise PilotBlindReviewError(f"{label}: verbal decisions cannot use a nonverbal quality field")
    if row["decision_status"] == "completed" and "unresolved" in {
        row["semantic_gate"], row["context_gate"], row["naturalness_preference"]
    }:
        raise PilotBlindReviewError(f"{label}: completed decision retains an unresolved field")
    if row["decision_status"] == "unresolved" and "unresolved" not in {
        row["semantic_gate"], row["context_gate"], row["naturalness_preference"]
    }:
        raise PilotBlindReviewError(f"{label}: unresolved decision must identify an unresolved field")
    return {
        "review_id": expected_review_id,
        "block_number": block_number,
        "decision_status": row["decision_status"],
        "direct_listening_completed": True,
        "semantic_gate": row["semantic_gate"],
        "context_gate": row["context_gate"],
        "naturalness_preference": row["naturalness_preference"],
        "nonverbal": row["nonverbal"],
        "mqm_errors": _validate_mqm_errors(row.get("mqm_errors"), label),
        "review_note": str(row.get("review_note", "")).strip(),
    }


def validate_pilot_reviewer_submission(
    *,
    pack_path: Path,
    attestation_path: Path,
    decisions_path: Path,
    expected_blocks: int = 298,
) -> dict[str, Any]:
    manifest, review_rows = _read_pack(pack_path)
    if manifest.get("block_count") != expected_blocks or len(review_rows) != expected_blocks:
        raise PilotBlindReviewError("Reviewer pack coverage mismatch")
    attestation = _read_json(attestation_path)
    if attestation.get("schema_name") != SUBMISSION_SCHEMA_NAME or attestation.get("schema_version") != SCHEMA_VERSION:
        raise PilotBlindReviewError("Invalid reviewer submission attestation schema")
    if attestation.get("pack_id") != manifest.get("pack_id") or attestation.get("reviewer_slot") != manifest.get("reviewer_slot"):
        raise PilotBlindReviewError("Reviewer submission pack/slot mismatch")
    if attestation.get("pack_sha256") != sha256_file(pack_path) or attestation.get("decisions_sha256") != sha256_file(decisions_path):
        raise PilotBlindReviewError("Reviewer submission artifact hash mismatch")
    if not str(attestation.get("reviewer_id", "")).strip() or not str(attestation.get("completed_at_utc", "")).strip():
        raise PilotBlindReviewError("Reviewer identity/completion timestamp is required")
    required_true = (
        "japanese_listening_capable",
        "worked_independently",
        "direct_listening_all_blocks",
    )
    if any(attestation.get(field) is not True for field in required_true):
        raise PilotBlindReviewError("Reviewer capability, independence, and full direct listening must be attested")
    required_false = (
        "participated_in_baseline_or_candidate_production",
        "candidate_provenance_seen",
    )
    if any(attestation.get(field) is not False for field in required_false):
        raise PilotBlindReviewError("Reviewer must not produce candidates or see candidate provenance")
    decisions = _read_jsonl(decisions_path)
    by_review_id: dict[str, dict[str, Any]] = {}
    for row in decisions:
        review_id = str(row.get("review_id", ""))
        if review_id in by_review_id:
            raise PilotBlindReviewError(f"Duplicate reviewer decision: {review_id}")
        by_review_id[review_id] = row
    expected_ids = [str(row["review_id"]) for row in review_rows]
    if set(by_review_id) != set(expected_ids) or len(decisions) != expected_blocks:
        raise PilotBlindReviewError("Reviewer decision coverage mismatch")
    normalized = [
        _validate_decision(by_review_id[review_id], str(manifest["pack_id"]), review_id, block_number)
        for block_number, review_id in enumerate(expected_ids, 1)
    ]
    return {
        "status": "complete",
        "reviewer_id": str(attestation["reviewer_id"]).strip(),
        "reviewer_slot": manifest["reviewer_slot"],
        "pack_id": manifest["pack_id"],
        "pack_sha256": sha256_file(pack_path),
        "blind_rows_sha256": manifest["blind_rows_sha256"],
        "randomization_commitment_sha256": manifest["randomization_commitment_sha256"],
        "reviewed_blocks": len(normalized),
        "unresolved_blocks": sum(row["decision_status"] == "unresolved" for row in normalized),
        "attestation": attestation,
        "decisions": normalized,
    }


def _decision_comparison(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_status": row["decision_status"],
        "semantic_gate": row["semantic_gate"],
        "context_gate": row["context_gate"],
        "naturalness_preference": row["naturalness_preference"],
        "nonverbal": row["nonverbal"],
        "mqm_errors": row["mqm_errors"],
    }


def _validate_consensus(
    row: dict[str, Any],
    *,
    block_number: int,
    reviewer_ids: set[str],
) -> dict[str, Any] | None:
    if set(row.get("recorded_by", [])) != reviewer_ids or set(row.get("relistened_by", [])) != reviewer_ids:
        raise PilotBlindReviewError(f"Block {block_number}: consensus must be recorded and relistened by both reviewers")
    if row.get("agreement_reached") is not True:
        return None
    if not str(row.get("reason", "")).strip():
        raise PilotBlindReviewError(f"Block {block_number}: consensus reason is required")
    candidate = {
        "review_id": f"SSIS-908-{block_number:04d}",
        "block_number": block_number,
        "decision_status": row.get("decision_status"),
        "direct_listening_completed": True,
        "semantic_gate": row.get("semantic_gate"),
        "context_gate": row.get("context_gate"),
        "naturalness_preference": row.get("naturalness_preference"),
        "nonverbal": row.get("nonverbal"),
        "mqm_errors": row.get("mqm_errors"),
        "review_note": str(row.get("reason", "")).strip(),
    }
    validated = _validate_decision(
        {"schema_name": DECISION_SCHEMA_NAME, "schema_version": SCHEMA_VERSION, "pack_id": "consensus", **candidate},
        "consensus",
        candidate["review_id"],
        block_number,
    )
    return validated


def _unblind_gate(value: str, assignment: dict[str, Any]) -> str:
    if value == "a-only":
        return f"{assignment['candidate_a_variant']}-only"
    if value == "b-only":
        return f"{assignment['candidate_b_variant']}-only"
    return value


def _unblind_naturalness(value: str, assignment: dict[str, Any]) -> str:
    if value == "candidate-a":
        return str(assignment["candidate_a_variant"])
    if value == "candidate-b":
        return str(assignment["candidate_b_variant"])
    return value


def _unblind_errors(errors: list[dict[str, Any]], assignment: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for error in errors:
        variant = assignment["candidate_a_variant"] if error["candidate"] == "candidate-a" else assignment["candidate_b_variant"]
        result.append({**error, "candidate": variant})
    return result


def adjudicate_pilot_reviews(
    *,
    reviewer_1_pack_path: Path,
    reviewer_1_attestation_path: Path,
    reviewer_1_decisions_path: Path,
    reviewer_2_pack_path: Path,
    reviewer_2_attestation_path: Path,
    reviewer_2_decisions_path: Path,
    internal_key_path: Path,
    consensus_path: Path,
    output_path: Path,
    expected_blocks: int = 298,
    expected_baseline_sha256: str = BASELINE_SHA256,
) -> dict[str, Any]:
    """Validate two independent full submissions and unblind only for adjudication output."""
    if output_path.exists():
        raise FileExistsError(f"Existing adjudication output will not be overwritten: {output_path}")
    first = validate_pilot_reviewer_submission(
        pack_path=reviewer_1_pack_path,
        attestation_path=reviewer_1_attestation_path,
        decisions_path=reviewer_1_decisions_path,
        expected_blocks=expected_blocks,
    )
    second = validate_pilot_reviewer_submission(
        pack_path=reviewer_2_pack_path,
        attestation_path=reviewer_2_attestation_path,
        decisions_path=reviewer_2_decisions_path,
        expected_blocks=expected_blocks,
    )
    if {first["reviewer_slot"], second["reviewer_slot"]} != set(REVIEWER_SLOTS):
        raise PilotBlindReviewError("Exactly reviewer-1 and reviewer-2 submissions are required")
    reviewer_ids = {first["reviewer_id"], second["reviewer_id"]}
    if len(reviewer_ids) != 2:
        raise PilotBlindReviewError("Two distinct independent reviewer identities are required")
    if first["blind_rows_sha256"] != second["blind_rows_sha256"]:
        raise PilotBlindReviewError("Reviewer packs do not share the same blind A/B layout")
    if first["randomization_commitment_sha256"] != second["randomization_commitment_sha256"]:
        raise PilotBlindReviewError("Reviewer packs use different randomization commitments")
    key = _read_json(internal_key_path)
    _validate_key(key, expected_blocks=expected_blocks, expected_baseline_sha256=expected_baseline_sha256)
    if key["key_commitment_sha256"] != first["randomization_commitment_sha256"]:
        raise PilotBlindReviewError("Sealed key does not match reviewer packs")

    raw_consensus = _read_jsonl(consensus_path) if consensus_path.exists() else []
    consensus_by_block: dict[int, dict[str, Any]] = {}
    for row in raw_consensus:
        try:
            number = int(row.get("block_number"))
        except (TypeError, ValueError) as exc:
            raise PilotBlindReviewError("Consensus block_number must be an integer") from exc
        if number in consensus_by_block or not 1 <= number <= expected_blocks:
            raise PilotBlindReviewError(f"Invalid or duplicate consensus record for block {number}")
        consensus_by_block[number] = row

    mapping = {int(row["block_number"]): row for row in key["candidate_mapping"]}
    final_rows: list[dict[str, Any]] = []
    disagreement_blocks: list[int] = []
    unresolved_blocks: list[int] = []
    for number, (left, right) in enumerate(zip(first["decisions"], second["decisions"]), 1):
        same = _decision_comparison(left) == _decision_comparison(right)
        needs_consensus = not same or left["decision_status"] == "unresolved"
        selected: dict[str, Any] | None = left if not needs_consensus else None
        status = "agreed" if selected is not None else "unresolved"
        consensus_recorded = False
        reason = "independent reviewer agreement"
        if needs_consensus:
            disagreement_blocks.append(number)
            consensus = consensus_by_block.get(number)
            if consensus is not None:
                selected = _validate_consensus(consensus, block_number=number, reviewer_ids=reviewer_ids)
                consensus_recorded = selected is not None
                if selected is not None:
                    status = "adjudicated"
                    reason = str(consensus["reason"]).strip()
            if selected is None:
                unresolved_blocks.append(number)
                selected = {
                    "semantic_gate": "unresolved",
                    "context_gate": "unresolved",
                    "naturalness_preference": "unresolved",
                    "nonverbal": None,
                    "mqm_errors": [],
                }

        assignment = mapping[number]
        nonverbal_agreed = (
            selected.get("nonverbal") is True
            and ((not needs_consensus and left["nonverbal"] is True and right["nonverbal"] is True) or consensus_recorded)
        )
        if selected.get("nonverbal") is True and not nonverbal_agreed:
            status = "unresolved"
            unresolved_blocks.append(number)
            consensus_recorded = False
        final_rows.append({
            "schema_name": ADJUDICATION_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "title_id": "SSIS-908",
            "block_number": number,
            "adjudication_status": status,
            "reviewer_ids": sorted(reviewer_ids),
            "direct_listening_completed_by": sorted(reviewer_ids),
            "semantic_gate": _unblind_gate(str(selected["semantic_gate"]), assignment),
            "context_gate": _unblind_gate(str(selected["context_gate"]), assignment),
            "naturalness_outcome": _unblind_naturalness(str(selected["naturalness_preference"]), assignment),
            "nonverbal": selected.get("nonverbal"),
            "mqm_errors": _unblind_errors(list(selected.get("mqm_errors", [])), assignment),
            "consensus_recorded": consensus_recorded,
            "internal_key_unsealed_for_adjudication": True,
            "human_reviewed": status in {"agreed", "adjudicated"},
            "reason": reason if status != "unresolved" else "reviewer disagreement remains unresolved",
        })
    unexpected_consensus = sorted(set(consensus_by_block) - set(disagreement_blocks))
    if unexpected_consensus:
        raise PilotBlindReviewError(f"Consensus records exist for blocks without disagreement: {unexpected_consensus[:10]}")
    unresolved_blocks = sorted(set(unresolved_blocks))
    _write_jsonl(output_path, final_rows)
    return {
        "status": "adjudicated" if not unresolved_blocks else "awaiting-adjudication",
        "title_id": "SSIS-908",
        "reviewed_blocks": expected_blocks,
        "reviewer_ids": sorted(reviewer_ids),
        "disagreement_blocks": disagreement_blocks,
        "consensus_records": sum(row["consensus_recorded"] for row in final_rows),
        "unresolved_blocks": unresolved_blocks,
        "adjudication_complete": not unresolved_blocks,
        "internal_key_unsealed_for_adjudication": True,
        "output": str(output_path),
        "pilot_evaluated": False,
        "final_promotion_allowed": False,
    }
