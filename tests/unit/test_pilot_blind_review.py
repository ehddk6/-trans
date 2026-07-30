from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from translation_forensics.cli import main
from translation_forensics.manifest import sha256_file
from translation_forensics.pilot_blind_review import (
    PilotBlindReviewError,
    adjudicate_pilot_reviews,
    build_pilot_blind_review_packets,
    initialize_pilot_blind_internal_key,
    validate_pilot_reviewer_submission,
)
from translation_forensics.srt import seconds_to_timecode


ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _write_json(path: Path, value: object) -> Path:
    return _write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> Path:
    return _write(path, "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values))


def _srt(text_prefix: str, blocks: int = 3) -> str:
    parts = []
    for number in range(1, blocks + 1):
        start = seconds_to_timecode(float(number - 1))
        end = seconds_to_timecode(float(number))
        parts.append(f"{number}\n{start} --> {end}\n{text_prefix} {number}")
    return "\n\n".join(parts) + "\n"


def _ready_inputs(tmp_path: Path, *, blocks: int = 3) -> dict[str, Path | str]:
    structure = _write(tmp_path / "locked.srt", _srt("日本語", blocks))
    baseline = _write(tmp_path / "baseline.srt", _srt("기준 문장", blocks))
    candidate = tmp_path / "candidate" / "viewer-natural.srt"
    evaluation_contract = _write(tmp_path / "evaluation-contract.yaml", "seed_id: frozen-pilot\n")
    mqm_schema = _write(tmp_path / "mqm.json", "{}\n")
    internal_key = tmp_path / "blind-review" / "internal-key.json"
    baseline_sha = hashlib.sha256(baseline.read_bytes()).hexdigest()
    initialize_pilot_blind_internal_key(
        title_id="SSIS-908",
        baseline_path=baseline,
        candidate_expected_path=candidate,
        evaluation_contract_path=evaluation_contract,
        mqm_schema_path=mqm_schema,
        output_path=internal_key,
        project_root=tmp_path,
        random_seed="frozen-seed-908",
        expected_blocks=blocks,
        expected_baseline_sha256=baseline_sha,
    )
    _write(candidate, _srt("개선 문장", blocks))
    provenance = _write_json(tmp_path / "candidate" / "provenance.json", {
        "schema_name": "translation-forensics/pilot-candidate-provenance",
        "schema_version": "1",
        "title_id": "SSIS-908",
        "status": "candidate-generated",
        "block_count": blocks,
        "human_reviewed": False,
        "pilot_evaluated": False,
        "final_promotion_allowed": False,
        "outputs": {"viewer_natural": {"sha256": sha256_file(candidate)}},
    })

    audio_dir = tmp_path / "audio-review"
    clips = audio_dir / "clips"
    clips.mkdir(parents=True)
    audio_records: list[dict[str, object]] = []
    for number in range(1, blocks + 1):
        clip = clips / f"block-{number:04d}.wav"
        clip.write_bytes(f"audio-{number}".encode("ascii"))
        audio_records.append({
            "block_number": number,
            "japanese": f"日本語 {number}",
            "scene_context": [{"block_number": number, "japanese": f"日本語 {number}", "is_target": True}],
            "audio": {
                "path": f"clips/block-{number:04d}.wav",
                "sha256": sha256_file(clip),
                "size_bytes": clip.stat().st_size,
            },
        })
    records = _write_jsonl(audio_dir / "blocks.jsonl", audio_records)
    audio_manifest = _write_json(audio_dir / "manifest.json", {
        "schema_name": "translation-forensics/pilot-audio-review-manifest",
        "schema_version": "1",
        "title_id": "SSIS-908",
        "status": "ready",
        "clip_preparation_allowed": True,
        "block_count": blocks,
        "all_blocks_have_audio": True,
        "all_blocks_have_japanese": True,
        "all_blocks_have_scene_context": True,
        "records": {"path": "blocks.jsonl", "sha256": sha256_file(records)},
    })
    return {
        "structure": structure,
        "baseline": baseline,
        "baseline_sha": baseline_sha,
        "candidate": candidate,
        "provenance": provenance,
        "evaluation_contract": evaluation_contract,
        "mqm_schema": mqm_schema,
        "internal_key": internal_key,
        "audio_manifest": audio_manifest,
    }


def _build_packs(tmp_path: Path, inputs: dict[str, Path | str], *, blocks: int = 3) -> Path:
    output = tmp_path / "packs"
    result = build_pilot_blind_review_packets(
        title_id="SSIS-908",
        structure_path=Path(inputs["structure"]),
        baseline_path=Path(inputs["baseline"]),
        candidate_path=Path(inputs["candidate"]),
        candidate_provenance_path=Path(inputs["provenance"]),
        audio_manifest_path=Path(inputs["audio_manifest"]),
        internal_key_path=Path(inputs["internal_key"]),
        output_dir=output,
        expected_blocks=blocks,
        expected_baseline_sha256=str(inputs["baseline_sha"]),
    )
    assert result["status"] == "awaiting-human-review"
    assert result["same_blind_layout"] is True
    return output


def _completed_decisions(pack: Path, destination: Path, *, naturalness: str = "tie") -> Path:
    with zipfile.ZipFile(pack) as archive:
        templates = [json.loads(line) for line in archive.read("submission.template.jsonl").decode("utf-8").splitlines()]
    for row in templates:
        row.update({
            "decision_status": "completed",
            "direct_listening_completed": True,
            "semantic_gate": "both-pass",
            "context_gate": "both-pass",
            "naturalness_preference": naturalness,
            "nonverbal": False,
            "mqm_errors": [],
            "review_note": "원음과 문맥을 직접 확인함",
        })
    return _write_jsonl(destination, templates)


def _attestation(pack: Path, decisions: Path, destination: Path, reviewer_id: str) -> Path:
    with zipfile.ZipFile(pack) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    return _write_json(destination, {
        "schema_name": "translation-forensics/pilot-reviewer-submission",
        "schema_version": "1",
        "title_id": "SSIS-908",
        "pack_id": manifest["pack_id"],
        "reviewer_slot": manifest["reviewer_slot"],
        "reviewer_id": reviewer_id,
        "pack_sha256": sha256_file(pack),
        "decisions_sha256": sha256_file(decisions),
        "completed_at_utc": "2026-07-28T00:00:00+00:00",
        "japanese_listening_capable": True,
        "worked_independently": True,
        "direct_listening_all_blocks": True,
        "participated_in_baseline_or_candidate_production": False,
        "candidate_provenance_seen": False,
    })


def test_blind_packets_have_same_hidden_layout_audio_and_no_internal_key(tmp_path: Path) -> None:
    inputs = _ready_inputs(tmp_path)
    packs = _build_packs(tmp_path, inputs)
    first = packs / "reviewer-1.pack.zip"
    second = packs / "reviewer-2.pack.zip"

    review_payloads = []
    for pack in (first, second):
        with zipfile.ZipFile(pack) as archive:
            names = set(archive.namelist())
            assert "internal-key.json" not in names
            assert {f"audio/block-{number:04d}.wav" for number in range(1, 4)}.issubset(names)
            manifest = json.loads(archive.read("manifest.json"))
            rows = archive.read("review.jsonl")
            review_payloads.append(rows)
            assert manifest["block_count"] == 3
            assert manifest["candidate_provenance_hidden"] is True
            assert manifest["randomization_seed_hidden"] is True
            assert b'"baseline"' not in rows
            assert b'"improvement"' not in rows
    assert review_payloads[0] == review_payloads[1]
    key = json.loads(Path(inputs["internal_key"]).read_text(encoding="utf-8"))
    assert key["seal_status"] == "sealed-until-adjudication"
    assert key["frozen_before_candidate_generation"] is True


def test_two_independent_full_submissions_adjudicate_all_blocks(tmp_path: Path) -> None:
    inputs = _ready_inputs(tmp_path)
    packs = _build_packs(tmp_path, inputs)
    pack_1 = packs / "reviewer-1.pack.zip"
    pack_2 = packs / "reviewer-2.pack.zip"
    decisions_1 = _completed_decisions(pack_1, tmp_path / "reviewer-1.jsonl")
    decisions_2 = _completed_decisions(pack_2, tmp_path / "reviewer-2.jsonl")
    attestation_1 = _attestation(pack_1, decisions_1, tmp_path / "reviewer-1.attestation.json", "listener-a")
    attestation_2 = _attestation(pack_2, decisions_2, tmp_path / "reviewer-2.attestation.json", "listener-b")

    validated = validate_pilot_reviewer_submission(
        pack_path=pack_1,
        attestation_path=attestation_1,
        decisions_path=decisions_1,
        expected_blocks=3,
    )
    assert validated["reviewed_blocks"] == 3
    assert validated["unresolved_blocks"] == 0

    consensus = _write(tmp_path / "consensus.jsonl", "")
    output = tmp_path / "final-decisions.jsonl"
    result = adjudicate_pilot_reviews(
        reviewer_1_pack_path=pack_1,
        reviewer_1_attestation_path=attestation_1,
        reviewer_1_decisions_path=decisions_1,
        reviewer_2_pack_path=pack_2,
        reviewer_2_attestation_path=attestation_2,
        reviewer_2_decisions_path=decisions_2,
        internal_key_path=Path(inputs["internal_key"]),
        consensus_path=consensus,
        output_path=output,
        expected_blocks=3,
        expected_baseline_sha256=str(inputs["baseline_sha"]),
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["adjudication_complete"] is True
    assert result["reviewer_ids"] == ["listener-a", "listener-b"]
    assert len(rows) == 3
    assert all(row["adjudication_status"] == "agreed" for row in rows)
    assert all(row["direct_listening_completed_by"] == ["listener-a", "listener-b"] for row in rows)
    assert all(row["naturalness_outcome"] == "tie" for row in rows)


def test_missing_listening_duplicate_reviewer_and_unresolved_disagreement_are_blocked(tmp_path: Path) -> None:
    inputs = _ready_inputs(tmp_path)
    packs = _build_packs(tmp_path, inputs)
    pack_1 = packs / "reviewer-1.pack.zip"
    pack_2 = packs / "reviewer-2.pack.zip"
    decisions_1 = _completed_decisions(pack_1, tmp_path / "reviewer-1.jsonl")
    first_rows = [json.loads(line) for line in decisions_1.read_text(encoding="utf-8").splitlines()]
    first_rows[0]["direct_listening_completed"] = False
    _write_jsonl(decisions_1, first_rows)
    attestation_1 = _attestation(pack_1, decisions_1, tmp_path / "reviewer-1.attestation.json", "same-listener")
    with pytest.raises(PilotBlindReviewError, match="direct listening"):
        validate_pilot_reviewer_submission(
            pack_path=pack_1,
            attestation_path=attestation_1,
            decisions_path=decisions_1,
            expected_blocks=3,
        )

    decisions_1 = _completed_decisions(pack_1, decisions_1)
    attestation_1 = _attestation(pack_1, decisions_1, attestation_1, "same-listener")
    decisions_2 = _completed_decisions(pack_2, tmp_path / "reviewer-2.jsonl", naturalness="candidate-a")
    attestation_2 = _attestation(pack_2, decisions_2, tmp_path / "reviewer-2.attestation.json", "same-listener")
    consensus = _write(tmp_path / "consensus.jsonl", "")
    with pytest.raises(PilotBlindReviewError, match="distinct independent"):
        adjudicate_pilot_reviews(
            reviewer_1_pack_path=pack_1,
            reviewer_1_attestation_path=attestation_1,
            reviewer_1_decisions_path=decisions_1,
            reviewer_2_pack_path=pack_2,
            reviewer_2_attestation_path=attestation_2,
            reviewer_2_decisions_path=decisions_2,
            internal_key_path=Path(inputs["internal_key"]),
            consensus_path=consensus,
            output_path=tmp_path / "duplicate-reviewer.jsonl",
            expected_blocks=3,
            expected_baseline_sha256=str(inputs["baseline_sha"]),
        )

    attestation_2 = _attestation(pack_2, decisions_2, attestation_2, "other-listener")
    result = adjudicate_pilot_reviews(
        reviewer_1_pack_path=pack_1,
        reviewer_1_attestation_path=attestation_1,
        reviewer_1_decisions_path=decisions_1,
        reviewer_2_pack_path=pack_2,
        reviewer_2_attestation_path=attestation_2,
        reviewer_2_decisions_path=decisions_2,
        internal_key_path=Path(inputs["internal_key"]),
        consensus_path=consensus,
        output_path=tmp_path / "unresolved.jsonl",
        expected_blocks=3,
        expected_baseline_sha256=str(inputs["baseline_sha"]),
    )
    assert result["adjudication_complete"] is False
    assert result["unresolved_blocks"] == [1, 2, 3]

    consensus_rows = [{
        "block_number": number,
        "recorded_by": ["same-listener", "other-listener"],
        "relistened_by": ["same-listener", "other-listener"],
        "agreement_reached": True,
        "decision_status": "completed",
        "semantic_gate": "both-pass",
        "context_gate": "both-pass",
        "naturalness_preference": "tie",
        "nonverbal": False,
        "mqm_errors": [],
        "reason": "두 검수자가 공동 재청취 후 합의함",
    } for number in range(1, 4)]
    _write_jsonl(consensus, consensus_rows)
    resolved = adjudicate_pilot_reviews(
        reviewer_1_pack_path=pack_1,
        reviewer_1_attestation_path=attestation_1,
        reviewer_1_decisions_path=decisions_1,
        reviewer_2_pack_path=pack_2,
        reviewer_2_attestation_path=attestation_2,
        reviewer_2_decisions_path=decisions_2,
        internal_key_path=Path(inputs["internal_key"]),
        consensus_path=consensus,
        output_path=tmp_path / "resolved-by-consensus.jsonl",
        expected_blocks=3,
        expected_baseline_sha256=str(inputs["baseline_sha"]),
    )
    assert resolved["adjudication_complete"] is True
    assert resolved["consensus_records"] == 3


def test_key_freeze_and_audio_gate_create_only_explicit_block_records(tmp_path: Path) -> None:
    structure = _write(tmp_path / "locked.srt", _srt("日本語"))
    baseline = _write(tmp_path / "baseline.srt", _srt("기준 문장"))
    candidate = _write(tmp_path / "candidate" / "viewer-natural.srt", _srt("개선 문장"))
    baseline_sha = sha256_file(baseline)
    evaluation_contract = _write(tmp_path / "evaluation-contract.yaml", "seed_id: frozen-pilot\n")
    mqm_schema = _write(tmp_path / "mqm.json", "{}\n")
    with pytest.raises(PilotBlindReviewError, match="before the improvement candidate exists"):
        initialize_pilot_blind_internal_key(
            title_id="SSIS-908",
            baseline_path=baseline,
            candidate_expected_path=candidate,
            evaluation_contract_path=evaluation_contract,
            mqm_schema_path=mqm_schema,
            output_path=tmp_path / "too-late-key.json",
            project_root=tmp_path,
            random_seed="too-late",
            expected_blocks=3,
            expected_baseline_sha256=baseline_sha,
        )

    provenance = _write_json(tmp_path / "candidate" / "provenance.json", {
        "schema_name": "translation-forensics/pilot-candidate-provenance",
        "schema_version": "1",
        "title_id": "SSIS-908",
        "status": "candidate-generated",
        "block_count": 3,
        "human_reviewed": False,
        "pilot_evaluated": False,
        "final_promotion_allowed": False,
        "outputs": {"viewer_natural": {"sha256": sha256_file(candidate)}},
    })
    audio_manifest = _write_json(tmp_path / "audio-review" / "manifest.json", {
        "schema_name": "translation-forensics/pilot-audio-review-manifest",
        "schema_version": "1",
        "title_id": "SSIS-908",
        "status": "blocked",
        "clip_preparation_allowed": False,
        "block_count": 0,
        "errors": ["human anchors and approved offset map are missing"],
    })
    blind_dir = tmp_path / "pilot" / "blind-review"
    final_decisions = tmp_path / "pilot" / "adjudication" / "final-decisions.jsonl"
    internal_key = blind_dir / "internal-key.json"
    assert main([
        "build-pilot-blind-review",
        "--project-root", str(tmp_path),
        "--title", "SSIS-908",
        "--structure", str(structure),
        "--baseline", str(baseline),
        "--candidate", str(candidate),
        "--candidate-provenance", str(provenance),
        "--audio-review-manifest", str(audio_manifest),
        "--evaluation-contract", str(evaluation_contract),
        "--mqm-schema", str(mqm_schema),
        "--internal-key", str(internal_key),
        "--random-seed", "blocked-seed",
        "--expected-blocks", "3",
        "--expected-baseline-sha256", baseline_sha,
        "--adjudication-output", str(final_decisions),
        "--output", str(blind_dir),
    ]) == 0
    for slot in ("reviewer-1", "reviewer-2"):
        with zipfile.ZipFile(blind_dir / f"{slot}.pack.zip") as archive:
            assert set(archive.namelist()) == {"manifest.json", "README.md"}
            assert json.loads(archive.read("manifest.json"))["status"] == "blocked"
    key = json.loads(internal_key.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in final_decisions.read_text(encoding="utf-8").splitlines()]
    assert key["status"] == "blocked"
    assert key["frozen_before_candidate_generation"] is False
    assert len(rows) == 3
    assert all(row["adjudication_status"] == "unresolved" and row["human_reviewed"] is False for row in rows)
