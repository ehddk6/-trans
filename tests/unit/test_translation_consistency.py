from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from translation_forensics.consistency import validate_consistency_ledger
from translation_forensics.manifest import file_record
from translation_forensics.outputs import package_title_outputs
from translation_forensics.review_prioritization import build_uncertainty_review_queue
from translation_forensics.semantic_translation import apply_translation_decisions, build_translation_queue
from translation_forensics.srt import parse_srt
from translation_forensics.translation_quality import validate_quality_regression_suite


ROOT = Path(__file__).parents[2]


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")
    return path


def _srt(texts: list[str]) -> str:
    blocks = []
    for number, text in enumerate(texts, 1):
        start = number * 2 - 1
        blocks.append(f"{number}\n00:00:{start:02d},000 --> 00:00:{start + 1:02d},000\n{text}")
    return "\n\n".join(blocks) + "\n"


def _ledger_entry(identifier: str, *, entry_type: str, scope: str, key: str, korean: str, speaker: str = "", addressee: str = "") -> dict[str, object]:
    return {
        "consistency_id": identifier,
        "title_id": "SAMPLE",
        "entry_type": entry_type,
        "scope": scope,
        "key": key,
        "source_japanese": key,
        "korean": korean,
        "status": "confirmed",
        "evidence_refs": ["japanese_srt"],
        "speaker_id": speaker,
        "addressee_id": addressee,
        "scene_ids": [],
        "start_block": None,
        "end_block": None,
        "last_changed_block": 1,
        "note": "synthetic",
    }


def _decision(number: int, *, conflicts: list[str]) -> dict[str, object]:
    return {
        "block_number": number,
        "source_faithful_korean": f"원문 번역 {number}",
        "viewer_natural_korean": f"자연 번역 {number}",
        "translation_method": "semantic_review_from_japanese",
        "translation_model": "gpt-5.6-terra",
        "status": "reviewed",
        "confidence": "high",
        "evidence_refs": ["japanese_srt"],
        "consistency_refs": [],
        "consistency_conflicts": conflicts,
        "preserved_meaning": ["register"],
        "review_required_reasons": ["consistency-conflict"] if conflicts else [],
    }


def test_consistency_ledger_flows_into_queue_review_and_strict_application(tmp_path: Path) -> None:
    structure = _write(tmp_path / "structure.srt", _srt(["美咲です", "分かりました", "またね"]))
    japanese = _write(tmp_path / "ja.srt", _srt(["美咲です", "分かりました", "またね"]))
    before = structure.read_bytes()
    ledger = tmp_path / "consistency.jsonl"
    entries = [
        _ledger_entry("C-NAME-MISAKI", entry_type="proper_noun", scope="title", key="美咲", korean="미사키"),
        _ledger_entry("C-REG-A-1", entry_type="register", scope="speaker", key="A-register", korean="해요체", speaker="A"),
        _ledger_entry("C-REG-A-2", entry_type="register", scope="speaker", key="A-register", korean="해체", speaker="A"),
    ]
    ledger.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries), encoding="utf-8", newline="\n")
    ledger_report = validate_consistency_ledger(ledger)
    assert ledger_report["status"] == "pass"
    assert ledger_report["conflict_count"] == 1
    terminology = _write(tmp_path / "terminology.jsonl", json.dumps({
        "terminology_id": "T-BODYTEMP", "japanese": "体温", "approved_korean": "체온", "scope": "title",
        "approval_status": "approved", "evidence_refs": ["japanese_srt"], "approved_by": "reviewer",
    }, ensure_ascii=False) + "\n")

    speaker_state = _write(tmp_path / "speaker.json", json.dumps([{
        "scene_id": "S1", "participants": ["A", "B"], "current_speaker": "A", "previous_speaker": "unknown",
        "addressee": "B", "identity_evidence_refs": ["japanese_srt"], "relationship": "colleagues",
        "speaker_confidence": "confirmed", "register_by_speaker": {"A": "polite"}, "address_terms": [],
        "current_action_by_participant": {}, "question_owner": "unknown", "expected_responder": "unknown", "review_status": "reviewed",
    }], ensure_ascii=False))
    context = _write(tmp_path / "context.jsonl", json.dumps({"scene_id": "S1", "blocks": [{"block_number": 1}], "asr_candidates": []}, ensure_ascii=False) + "\n")
    queue = tmp_path / "queue.jsonl"
    queue_report = tmp_path / "queue-report.json"
    build_translation_queue(
        structure, japanese, None, queue, queue_report,
        review_context_path=context, consistency_ledger_path=ledger, terminology_path=terminology, speaker_state_path=speaker_state,
        title_id="SAMPLE",
    )
    rows = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
    first_context = rows[0]["consistency_context"]
    assert [item["consistency_id"] for item in first_context["applied_entries"]] == ["C-NAME-MISAKI", "terminology:T-BODYTEMP"]
    assert first_context["conflicts"][0]["entry_ids"] == ["C-REG-A-1", "C-REG-A-2"]
    assert first_context["ledger_sha256"] == hashlib.sha256(ledger.read_bytes()).hexdigest()
    assert json.loads(queue_report.read_text(encoding="utf-8"))["consistency_ledger"]["conflict_count"] == 1
    assert first_context["terminology_sha256"] == hashlib.sha256(terminology.read_bytes()).hexdigest()

    decisions = [_decision(1, conflicts=[]), _decision(2, conflicts=[]), _decision(3, conflicts=[])]
    decisions_path = tmp_path / "decisions.jsonl"
    decisions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions), encoding="utf-8", newline="\n")
    rejected = apply_translation_decisions(structure, decisions_path, tmp_path / "rejected-source.srt", tmp_path / "rejected-viewer.srt", tmp_path / "rejected-report.json", strict=True, translation_queue_path=queue)
    assert rejected["status"] == "fail"
    assert any("consistency conflict" in error for error in rejected["errors"])

    decisions[0]["consistency_conflicts"] = ["C-REG-A-1", "C-REG-A-2"]
    decisions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions), encoding="utf-8", newline="\n")
    source, viewer, report = tmp_path / "source.srt", tmp_path / "viewer.srt", tmp_path / "apply.json"
    applied = apply_translation_decisions(structure, decisions_path, source, viewer, report, strict=True, translation_queue_path=queue)
    assert applied["status"] == "text-crosschecked"
    assert structure.read_bytes() == before

    priority = tmp_path / "priority.csv"
    queue_result = build_uncertainty_review_queue(structure, priority, decisions_path=decisions_path, translation_queue_path=queue, source_path=source, viewer_path=viewer)
    assert queue_result["priority_counts"]["high"] == 1
    with priority.open(encoding="utf-8", newline="") as handle:
        priority_rows = list(csv.DictReader(handle))
    assert priority_rows[0]["block_number"] == "1"
    assert "consistency-conflict" in priority_rows[0]["reason_codes"]

    package = package_title_outputs("SAMPLE", structure, source, viewer, tmp_path / "final", japanese_path=japanese, translation_decisions_path=decisions_path, translation_queue_path=queue)
    manifest = json.loads(Path(package["files"]["run_manifest"]).read_text(encoding="utf-8"))
    assert any(item["sha256"] == hashlib.sha256(queue.read_bytes()).hexdigest() for item in manifest["input_hashes"])
    repeated = package_title_outputs("SAMPLE", structure, source, viewer, tmp_path / "final", japanese_path=japanese, translation_decisions_path=decisions_path, translation_queue_path=queue)
    assert repeated["version"] == 2
    assert repeated["files"]["source"] != package["files"]["source"]


def test_quality_regression_suite_covers_synthetic_translation_failures() -> None:
    report = validate_quality_regression_suite(ROOT / "tests" / "fixtures" / "translation-quality-regressions.json")
    assert report["status"] == "pass"
    assert report["cases"] == 15
    assert all(not row["passing_codes"] for row in report["results"])


def test_bom_input_is_recorded_and_read_while_generated_outputs_remain_bom_free(tmp_path: Path) -> None:
    raw = _srt(["最初です"])
    structure = tmp_path / "structure.srt"
    japanese = tmp_path / "ja.srt"
    structure.write_bytes(b"\xef\xbb\xbf" + raw.encode("utf-8"))
    japanese.write_bytes(b"\xef\xbb\xbf" + raw.encode("utf-8"))
    assert parse_srt(structure)[1] == "utf-8-sig"
    assert file_record(structure, role="structure")["encoding"] == "utf-8-sig"
    queue = tmp_path / "queue.jsonl"
    build_translation_queue(structure, japanese, None, queue, tmp_path / "queue-report.json", title_id="SAMPLE")
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(json.dumps(_decision(1, conflicts=[]), ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    source, viewer = tmp_path / "source.srt", tmp_path / "viewer.srt"
    result = apply_translation_decisions(structure, decisions, source, viewer, tmp_path / "apply.json", strict=True, translation_queue_path=queue)
    assert result["status"] == "text-crosschecked"
    assert not source.read_bytes().startswith(b"\xef\xbb\xbf")
    assert not viewer.read_bytes().startswith(b"\xef\xbb\xbf")
