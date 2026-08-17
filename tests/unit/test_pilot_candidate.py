from __future__ import annotations

import json
from pathlib import Path

import pytest

from translation_forensics.cli import main
from translation_forensics.pilot_candidate import (
    SEMANTIC_RECHECK_KEYS,
    PilotCandidateError,
    build_pilot_candidate,
    canonical_record_sha256,
    validate_pilot_candidate,
    write_pilot_sol_reviews,
)
from translation_forensics.srt import compare_structure, parse_srt, seconds_to_timecode


ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    return _write(path, "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records))


def _srt(blocks: int) -> str:
    parts = []
    for number in range(1, blocks + 1):
        start = seconds_to_timecode(float(number - 1))
        end = seconds_to_timecode(float(number))
        parts.append(f"{number}\n{start} --> {end}\nはい")
    return "\n\n".join(parts) + "\n"


def _records(blocks: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    decisions: list[dict[str, object]] = []
    reviews: list[dict[str, object]] = []
    for number in range(1, blocks + 1):
        decision: dict[str, object] = {
            "schema_name": "translation-forensics/pilot-terra-decision",
            "schema_version": "1",
            "title_id": "SSIS-908",
            "block_number": number,
            "translation_model": "gpt-5.6-terra",
            "source_faithful_korean": "네",
            "viewer_natural_korean": "네",
            "confidence": "high",
            "evidence_refs": [f"japanese-srt:{number}"],
            "risk_codes": [],
            "reason": "일본어 긍정 응답",
        }
        decisions.append(decision)
        reviews.append({
            "schema_name": "translation-forensics/pilot-sol-review",
            "schema_version": "1",
            "title_id": "SSIS-908",
            "block_number": number,
            "reviewer_model": "gpt-5.6-sol",
            "independent_from_terra_creation": True,
            "terra_decision_sha256": canonical_record_sha256(decision),
            "verdict": "accept",
            "critical_slot_conflicts": [],
            "unsupported_additions": [],
            "risk_codes": [],
            "resolved_issues": [],
            "semantic_recheck": {key: True for key in sorted(SEMANTIC_RECHECK_KEYS)},
            "reason": "의미와 화행을 보존함",
        })
    return decisions, reviews


def _inputs(tmp_path: Path) -> dict[str, Path | str]:
    structure = _write(tmp_path / "locked.srt", _srt(298))
    baseline = _write(tmp_path / "baseline.srt", structure.read_text(encoding="utf-8"))
    decisions, reviews = _records(298)
    return {
        "structure": structure,
        "baseline": baseline,
        "baseline_sha256": __import__("hashlib").sha256(baseline.read_bytes()).hexdigest(),
        "decisions": _write_jsonl(tmp_path / "terra.jsonl", decisions),
        "reviews": _write_jsonl(tmp_path / "sol.jsonl", reviews),
        "terra_prompt": _write(tmp_path / "terra.md", "terra prompt\n"),
        "sol_prompt": _write(tmp_path / "sol.md", "sol prompt\n"),
    }


def _build(tmp_path: Path, inputs: dict[str, Path | str], output: Path) -> dict[str, object]:
    return build_pilot_candidate(
        title_id="SSIS-908",
        structure_path=inputs["structure"],
        baseline_path=inputs["baseline"],
        terra_decisions_path=inputs["decisions"],
        sol_reviews_path=inputs["reviews"],
        terra_prompt_path=inputs["terra_prompt"],
        sol_prompt_path=inputs["sol_prompt"],
        provenance_schema_path=ROOT / "schemas" / "pilot-candidate-provenance.schema.json",
        output_dir=output,
        project_root=tmp_path,
        expected_baseline_sha256=inputs["baseline_sha256"],
    )


def test_full_pilot_candidate_preserves_structure_and_lineage(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "candidate"
    provenance = _build(tmp_path, inputs, output)

    source, source_encoding, source_newline = parse_srt(output / "source-faithful.srt")
    viewer, viewer_encoding, viewer_newline = parse_srt(output / "viewer-natural.srt")
    locked, _, _ = parse_srt(inputs["structure"])
    assert len(source) == len(viewer) == 298
    assert compare_structure(locked, source)["pass"] is True
    assert compare_structure(locked, viewer)["pass"] is True
    assert (source_encoding, source_newline, viewer_encoding, viewer_newline) == ("utf-8", "LF", "utf-8", "LF")
    assert provenance["pipeline_order"] == [
        "terra-translation-decision",
        "sol-independent-critique-repair",
        "semantic-recheck",
    ]
    assert provenance["human_reviewed"] is False
    assert provenance["pilot_evaluated"] is False
    assert validate_pilot_candidate(
        output_dir=output,
        structure_path=inputs["structure"],
        baseline_path=inputs["baseline"],
        provenance_schema_path=ROOT / "schemas" / "pilot-candidate-provenance.schema.json",
        project_root=tmp_path,
        expected_baseline_sha256=inputs["baseline_sha256"],
    )["status"] == "pass"
    assert main([
        "validate-pilot-candidate",
        "--project-root", str(tmp_path),
        "--title", "SSIS-908",
        "--candidate", str(output),
        "--structure", str(inputs["structure"]),
        "--baseline", str(inputs["baseline"]),
        "--provenance-schema", str(ROOT / "schemas" / "pilot-candidate-provenance.schema.json"),
        "--expected-baseline-sha256", str(inputs["baseline_sha256"]),
    ]) == 0
    with pytest.raises(FileExistsError):
        _build(tmp_path, inputs, output)


def test_missing_or_quarantined_sol_review_blocks_generation(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    reviews = [json.loads(line) for line in Path(inputs["reviews"]).read_text(encoding="utf-8").splitlines()]
    _write_jsonl(Path(inputs["reviews"]), reviews[:-1])
    output = tmp_path / "missing-review"
    with pytest.raises(PilotCandidateError, match="coverage mismatch"):
        _build(tmp_path, inputs, output)
    assert not (output / "source-faithful.srt").exists()

    decisions, reviews = _records(298)
    reviews[0]["verdict"] = "quarantine"
    _write_jsonl(Path(inputs["reviews"]), reviews)
    with pytest.raises(PilotCandidateError, match="quarantined"):
        _build(tmp_path, inputs, tmp_path / "quarantined")


def test_candidate_validation_detects_output_tampering(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "candidate"
    _build(tmp_path, inputs, output)
    viewer = output / "viewer-natural.srt"
    viewer.write_text(viewer.read_text(encoding="utf-8").replace("네", "아니요", 1), encoding="utf-8", newline="\n")
    result = validate_pilot_candidate(
        output_dir=output,
        structure_path=inputs["structure"],
        baseline_path=inputs["baseline"],
        provenance_schema_path=ROOT / "schemas" / "pilot-candidate-provenance.schema.json",
        project_root=tmp_path,
        expected_baseline_sha256=inputs["baseline_sha256"],
    )
    assert result["status"] == "fail"
    assert "Provenance viewer_natural output hash mismatch" in result["errors"]


def test_sol_review_plan_requires_explicit_full_coverage(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    plan = {
        "schema_name": "translation-forensics/pilot-sol-review-plan",
        "schema_version": "1",
        "title_id": "SSIS-908",
        "reviewer_model": "gpt-5.6-sol",
        "independent_from_terra_creation": True,
        "reviewed_block_numbers": list(range(1, 299)),
        "accepted_review_reason": "독립 비평 결과 의미 슬롯을 보존함",
        "semantic_recheck": {key: True for key in sorted(SEMANTIC_RECHECK_KEYS)},
        "repairs": [{
            "block_number": 1,
            "repaired_source_faithful_korean": "예",
            "repaired_viewer_natural_korean": "응",
            "critical_slot_conflicts": ["register"],
            "unsupported_additions": [],
            "risk_codes": [],
            "resolved_issues": ["register"],
            "reason": "대화체 높임 수준을 바로잡음",
        }],
    }
    plan_path = _write(tmp_path / "plan.json", json.dumps(plan, ensure_ascii=False))
    output = tmp_path / "recorded-sol.jsonl"
    result = write_pilot_sol_reviews(
        terra_decisions_path=inputs["decisions"],
        review_plan_path=plan_path,
        output_path=output,
    )
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["reviewed_blocks"] == len(records) == 298
    assert records[0]["verdict"] == "repair"
    assert records[1]["verdict"] == "accept"

    plan["reviewed_block_numbers"] = list(range(1, 298))
    incomplete = _write(tmp_path / "incomplete-plan.json", json.dumps(plan, ensure_ascii=False))
    with pytest.raises(PilotCandidateError, match="explicitly enumerate"):
        write_pilot_sol_reviews(
            terra_decisions_path=inputs["decisions"],
            review_plan_path=incomplete,
            output_path=tmp_path / "incomplete-sol.jsonl",
        )
