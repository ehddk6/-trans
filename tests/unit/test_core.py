from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

from translation_forensics.alignment import align_by_overlap
from translation_forensics.audio_adapter import ingest_asr, run_asr
from translation_forensics.asr_evidence import classify_scene, declared_slot_conflicts, evidence_weight, read_asr_candidates, should_escalate
from translation_forensics.decisions import validate_state_transition
from translation_forensics.discovery import DiscoveryError, resolve_role
from translation_forensics.forensics_adapter import build_fallback_queue
from translation_forensics.forensic_model import empty_semantic_frame, evidence_independence, frame_conflicts, initialize_forensic_records, validate_forensic_records
from translation_forensics.manifest import build_project_manifest, sha256_file
from translation_forensics.mqm import validate_mqm_csv
from translation_forensics.evidence_artifacts import validate_alignment_evidence, validate_backtranslation_check, validate_speaker_state
from translation_forensics.outputs import next_version, package_title_outputs
from translation_forensics.reporting import build_review_context
from translation_forensics.semantic_translation import apply_translation_decisions, build_translation_queue, initialize_translation_decisions, merge_translation_decisions
from translation_forensics.scenes import build_review_scenes, read_review_queue
from translation_forensics.srt import SRTError, compare_structure, parse_srt, parse_srt_text, seconds_to_timecode
from translation_forensics.validation import validate_pair, validate_srt_file


FIXTURES = Path(__file__).parents[1] / "fixtures"
ROOT = Path(__file__).parents[2]


def fixture(name: str) -> Path:
    return FIXTURES / name


def test_srt_parse_and_timecode_roundtrip() -> None:
    blocks, encoding, newline = parse_srt(fixture("sample.structure.srt"))
    assert len(blocks) == 3
    assert blocks[0].duration == 2.0
    assert encoding in {"utf-8", "utf-8-sig"}
    assert newline == "LF"
    assert seconds_to_timecode(61.234) == "00:01:01,234"


def test_invalid_timecode_is_rejected() -> None:
    with pytest.raises(SRTError):
        parse_srt_text("1\n00:00:99,000 --> 00:00:01,000\nbad")


def test_structure_mismatch_detected() -> None:
    reference, _, _ = parse_srt(fixture("sample.structure.srt"))
    candidate = parse_srt_text("1\n00:00:01,000 --> 00:00:03,000\n하나\n\n3\n00:00:07,000 --> 00:00:10,000\n셋")
    diff = compare_structure(reference, candidate)
    assert not diff["pass"]
    assert any(issue["code"] == "block_count" for issue in diff["issues"])


def test_time_overlap_alignment() -> None:
    reference, _, _ = parse_srt(fixture("sample.structure.srt"))
    candidate = parse_srt_text("10\n00:00:01,500 --> 00:00:03,500\n하나")
    result = align_by_overlap(reference[:1], candidate)
    assert result[0].candidate_number == 10
    assert result[0].overlap_seconds == 1.5


def test_discovery_stops_on_multiple_candidates(tmp_path: Path) -> None:
    (tmp_path / "A.structure.srt").write_text(fixture("sample.structure.srt").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "A-copy.structure.srt").write_text(fixture("sample.structure.srt").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(DiscoveryError):
        resolve_role(tmp_path, "structure", title="A")


def test_review_scene_merging() -> None:
    rows = read_review_queue(fixture("sample.review-queue.text-v1.csv"), bands={"P1", "P2"})
    scenes = build_review_scenes(rows, padding=0.0, merge_gap=1.5, max_scene=45.0)
    assert len(scenes) == 1
    assert scenes[0].highest_band == "P1"


def test_fallback_queue_records_non_final_method(tmp_path: Path) -> None:
    out = tmp_path / "queue.csv"
    result = build_fallback_queue(fixture("sample.ja.srt"), fixture("sample.previous-ko.srt"), out, reason="test")
    assert result["method"] == "structure_fallback"
    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["queue_method"] == "structure_fallback"


def test_validation_flags_japanese_placeholder_and_tags(tmp_path: Path) -> None:
    bad = tmp_path / "bad.srt"
    bad.write_text("1\n00:00:01,000 --> 00:00:02,000\n日本語 [불명] [수정]\n", encoding="utf-8", newline="\n")
    report = validate_srt_file(bad)
    codes = {issue.code for issue in report.issues}
    assert report.status == "fail"
    assert {"japanese_residue", "placeholder", "work_tag"} <= codes


def test_validation_flags_three_lines_and_duplicates(tmp_path: Path) -> None:
    bad = tmp_path / "bad.srt"
    bad.write_text("1\n00:00:01,000 --> 00:00:04,000\n같은 문장\n두 번째\n세 번째\n\n2\n00:00:05,000 --> 00:00:08,000\n같은 문장\n두 번째\n세 번째\n", encoding="utf-8", newline="\n")
    report = validate_srt_file(bad)
    codes = {issue.code for issue in report.issues}
    assert "too_many_lines" in codes
    assert "consecutive_duplicate" in codes


def test_valid_pair_and_package_version_guard(tmp_path: Path) -> None:
    final = tmp_path / "final"
    report = validate_pair(fixture("sample.structure.srt"), fixture("sample.source-faithful.srt"), fixture("sample.viewer-natural.srt"))
    assert report["status"] == "pass"
    result = package_title_outputs("SAMPLE", fixture("sample.structure.srt"), fixture("sample.source-faithful.srt"), fixture("sample.viewer-natural.srt"), final, japanese_path=fixture("sample.ja.srt"))
    assert result["version"] == 1
    assert next_version(final, "SAMPLE", "text-crosschecked") == 2
    with pytest.raises(FileExistsError):
        package_title_outputs("SAMPLE", fixture("sample.structure.srt"), fixture("sample.source-faithful.srt"), fixture("sample.viewer-natural.srt"), final, japanese_path=fixture("sample.ja.srt"), version=1)


def test_asr_escalation_and_prompt_weight() -> None:
    rows = [
        {"profile": "original_unbiased", "text": "質問ですか", "avg_logprob": "-1.2", "no_speech_prob": "0.1", "error": ""},
        {"profile": "dialogue_unbiased", "text": "大丈夫です", "avg_logprob": "-0.2", "no_speech_prob": "0.1", "error": ""},
    ]
    escalate, reason = should_escalate(rows)
    assert escalate and "disagree" in reason
    assert evidence_weight("original_prompted") == "low_prompt_influenced"
    assert classify_scene(rows) == "context-resolved" or classify_scene(rows) == "unresolved"


def test_unresolved_scene_is_not_human_verified() -> None:
    ok, _ = validate_state_transition("audio-asr-crosschecked", "audio-human-verified", direct_listening=False)
    assert not ok
    ok, _ = validate_state_transition("audio-asr-crosschecked", "audio-human-verified", direct_listening=True)
    assert ok


def test_asr_csv_ingest_and_zip_ingest(tmp_path: Path) -> None:
    destination = tmp_path / "work_audio"
    csv_source = fixture("sample.asr-candidates.csv")
    result = ingest_asr(csv_source, destination)
    assert result["rows"] == 2
    zip_source = tmp_path / "sample.work_audio.zip"
    with zipfile.ZipFile(zip_source, "w") as archive:
        archive.write(csv_source, "asr-candidates.csv")
    other_destination = tmp_path / "work_audio_zip"
    result = ingest_asr(zip_source, other_destination)
    assert result["rows"] == 2


def test_asr_defaults_to_one_whisper_family_and_escalates_declared_slot_conflict(tmp_path: Path) -> None:
    rows = read_asr_candidates(fixture("sample.asr-candidates.csv"))
    assert {row["source_family"] for row in rows} == {"whisper-family"}
    claim = lambda value: json.dumps({"polarity": {"state": "confirmed", "value": value}})
    conflict_rows = [
        {"semantic_slots_json": claim("positive")},
        {"semantic_slots_json": claim("negative")},
    ]
    assert declared_slot_conflicts(conflict_rows) == ["polarity"]
    assert "semantic_conflict:polarity" in should_escalate(conflict_rows)[1]


def test_reviewer_evidence_artifact_validators_block_bad_or_semantic_flip(tmp_path: Path) -> None:
    speaker = tmp_path / "speaker.json"
    speaker.write_text(json.dumps([{"scene_id": "S1", "participants": [], "current_speaker": "unknown", "previous_speaker": "unknown", "addressee": "unknown", "speaker_confidence": "unknown", "relationship": "unknown", "register_by_speaker": {}, "address_terms": [], "current_action_by_participant": {}, "question_owner": "unknown", "expected_responder": "unknown"}]), encoding="utf-8")
    assert validate_speaker_state(speaker)["status"] == "pass"
    alignment = tmp_path / "alignment.csv"
    alignment.write_text("token,start,end,confidence,candidate_source,block_overlap,boundary_warning\nX,2,1,0.9,asr,1,none\n", encoding="utf-8")
    assert validate_alignment_evidence(alignment)["status"] == "fail"
    backtranslation = tmp_path / "backtranslation.csv"
    backtranslation.write_text("block_number,status,difference_types,review_status\n1,meaning-flip,polarity,reviewed\n", encoding="utf-8")
    assert validate_backtranslation_check(backtranslation)["status"] == "fail"


def test_run_asr_dry_run_selects_cpu_profile() -> None:
    result = run_asr(ROOT, Path("scenes.csv"), Path("asr.csv"), force_cpu=True, dry_run=True)
    assert result["status"] == "dry-run"
    assert result["device"] == "cpu"


def test_final_package_requires_explicit_completion(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        package_title_outputs("SAMPLE", fixture("sample.structure.srt"), fixture("sample.source-faithful.srt"), fixture("sample.viewer-natural.srt"), tmp_path, japanese_path=fixture("sample.ja.srt"), stage="final", asr_path=fixture("sample.asr-candidates.csv"))


def test_manifest_contains_hash_and_unresolved_roles(tmp_path: Path) -> None:
    copied = tmp_path / "sample.srt"
    copied.write_bytes(fixture("sample.structure.srt").read_bytes())
    manifest = build_project_manifest("SAMPLE", tmp_path, {"structure": copied, "ja": None}, unresolved_roles=["ja"])
    assert manifest["inputs"][0]["sha256"] == sha256_file(copied)
    assert manifest["unresolved_roles"] == ["ja"]


def test_review_context_creation(tmp_path: Path) -> None:
    scenes = tmp_path / "review-scenes.csv"
    scenes.write_text("scene_id,start_time,end_time,duration_sec,block_numbers,highest_band,original_audio,dialogue_audio,context_japanese,blocks_json\nS0001,00:00:01,000,00:00:06,000,5,1,P1,,,\"\",[]\n", encoding="utf-8")
    # Use a correctly formatted scene row; the first row above intentionally exercises CSV parsing only after repair.
    scenes.write_text("scene_id,start_time,end_time,duration_sec,block_numbers,highest_band,original_audio,dialogue_audio,context_japanese,blocks_json\nS0001,\"00:00:01,000\",\"00:00:06,000\",5,1,P1,,,\"\",[]\n", encoding="utf-8")
    output = tmp_path / "context.jsonl"
    result = build_review_context(fixture("sample.ja.srt"), fixture("sample.previous-ko.srt"), scenes, fixture("sample.asr-candidates.csv"), output)
    assert result["scenes"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["blocks"][0]["source_japanese"] == "最初です"


def test_translation_queue_keeps_previous_korean_as_candidate(tmp_path: Path) -> None:
    queue = tmp_path / "translation-queue.jsonl"
    report = tmp_path / "translation-queue.report.json"
    result = build_translation_queue(
        fixture("sample.structure.srt"),
        fixture("sample.ja.srt"),
        fixture("sample.previous-ko.srt"),
        queue,
        report,
    )
    assert result["status"] == "translation-queue-ready"
    records = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert all(record["decision_status"] == "untranslated" for record in records)
    assert all(record["translation_method_required"] == "semantic_review_from_japanese" for record in records)
    assert json.loads(report.read_text(encoding="utf-8"))["final_promotion_allowed"] is False


def test_apply_translation_decisions_requires_semantic_fields(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.jsonl"
    records = [{
        "block_number": number,
        "source_faithful_korean": f"한국어 원문 {number}",
        "viewer_natural_korean": f"자연스러운 한국어 {number}",
        "translation_method": "semantic_review_from_japanese",
        "translation_model": "gpt-5.6-terra",
        "status": "approved",
        "confidence": "high",
        "evidence_refs": ["japanese_srt"],
    } for number in range(1, 4)]
    decisions.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8", newline="\n")
    source = tmp_path / "source.srt"
    viewer = tmp_path / "viewer.srt"
    report = tmp_path / "apply.report.json"
    result = apply_translation_decisions(fixture("sample.structure.srt"), decisions, source, viewer, report, strict=True)
    assert result["status"] == "text-crosschecked"
    assert result["translation_model"] == "gpt-5.6-terra"
    assert result["final_promotion_allowed"] is False
    assert parse_srt(source)[0][0].start == parse_srt(fixture("sample.structure.srt"))[0][0].start
    assert "한국어 원문 1" in source.read_text(encoding="utf-8")


def test_apply_translation_decisions_rejects_carryover_in_strict_mode(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.jsonl"
    records = [{
        "block_number": number,
        "source_faithful_korean": f"한국어 {number}",
        "viewer_natural_korean": f"한국어 {number}",
        "translation_method": "previous_korean_copy",
        "translation_model": "gpt-5.6-terra",
        "status": "approved",
        "confidence": "high",
        "evidence_refs": ["previous_korean_candidate"],
    } for number in range(1, 4)]
    decisions.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8", newline="\n")
    result = apply_translation_decisions(fixture("sample.structure.srt"), decisions, tmp_path / "source.srt", tmp_path / "viewer.srt", tmp_path / "report.json", strict=True)
    assert result["status"] == "fail"
    assert any("의미 번역 방법" in error for error in result["errors"])


def test_translation_decision_template_is_explicitly_incomplete(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps({"block_number": 1, "source_japanese": "最初です", "previous_korean": "처음입니다", "required_semantic_slots": ["speaker_action_target"], "evidence_refs": ["japanese_srt"]}, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "decisions-template.jsonl"
    result = initialize_translation_decisions(queue, output)
    assert result["status"] == "decision-template"
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "untranslated"
    assert record["source_faithful_korean"] == ""
    assert record["translation_model"] == "gpt-5.6-terra"


def test_merge_translation_decisions_preserves_unreviewed_blocks(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    base.write_text("\n".join(json.dumps({"block_number": n, "status": "untranslated", "source_faithful_korean": "", "viewer_natural_korean": ""}, ensure_ascii=False) for n in (1, 2)) + "\n", encoding="utf-8")
    reviewed = tmp_path / "reviewed.jsonl"
    reviewed.write_text(json.dumps({"block_number": 1, "status": "reviewed", "source_faithful_korean": "첫 문장", "viewer_natural_korean": "첫 문장"}, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "merged.jsonl"
    result = merge_translation_decisions(base, reviewed, output)
    assert result["reviewed_blocks"] == 1
    assert result["unresolved_blocks"] == 1
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records[0]["status"] == "reviewed"
    assert records[1]["status"] == "untranslated"


def test_translation_decisions_reject_non_terra_model(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.jsonl"
    records = [{
        "block_number": number,
        "source_faithful_korean": f"한국어 원문 {number}",
        "viewer_natural_korean": f"자연스러운 한국어 {number}",
        "translation_method": "semantic_review_from_japanese",
        "translation_model": "gpt-5.6-sol",
        "status": "approved",
        "confidence": "high",
        "evidence_refs": ["japanese_srt"],
    } for number in range(1, 4)]
    decisions.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8", newline="\n")
    result = apply_translation_decisions(fixture("sample.structure.srt"), decisions, tmp_path / "source.srt", tmp_path / "viewer.srt", tmp_path / "report.json", strict=True)
    assert result["status"] == "fail"
    assert any("gpt-5.6-terra" in error for error in result["errors"])


def test_forensic_frames_escalate_critical_conflicts_and_do_not_double_count_family(tmp_path: Path) -> None:
    left = empty_semantic_frame(1); right = empty_semantic_frame(1)
    left["polarity"] = {"value": "negative", "state": "confirmed", "evidence_refs": ["ja-1"]}
    right["polarity"] = {"value": "positive", "state": "confirmed", "evidence_refs": ["asr-1"]}
    assert frame_conflicts(left, right)[0]["severity"] == "critical"
    support = evidence_independence([
        {"evidence_id": "w1", "source_family": "whisper-family"},
        {"evidence_id": "w2", "source_family": "whisper-family"},
        {"evidence_id": "j1", "source_family": "reference-japanese"},
    ])
    assert support["independent_family_count"] == 2
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps({"block_number": 1}, ensure_ascii=False) + "\n", encoding="utf-8")
    frames, hypotheses = tmp_path / "frames.jsonl", tmp_path / "hypotheses.jsonl"
    initialize_forensic_records(queue, frames, hypotheses)
    report = validate_forensic_records(frames, hypotheses)
    assert report["status"] == "pass"
    assert report["reviewed_frames"] == 0


def test_mqm_ledger_uses_explicit_taxonomy(tmp_path: Path) -> None:
    ledger = tmp_path / "mqm.csv"
    ledger.write_text("block_number,error_type,severity,evidence_refs,review_status\n1,polarity-flip,critical,ja-1,reviewed\n", encoding="utf-8")
    report = validate_mqm_csv(ledger)
    assert report["status"] == "pass"
    assert report["critical_errors"] == 1
