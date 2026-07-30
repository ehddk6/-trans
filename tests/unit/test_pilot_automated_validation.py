from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from translation_forensics.cli import main
from translation_forensics.pilot_automated_validation import validate_pilot_automated
from translation_forensics.pilot_candidate import (
    SEMANTIC_RECHECK_KEYS,
    build_pilot_candidate,
    canonical_record_sha256,
)
from translation_forensics.srt import seconds_to_timecode


ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    return _write(path, "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))


def _srt(text: str, *, blocks: int = 298) -> str:
    rows = []
    for number in range(1, blocks + 1):
        start = float((number - 1) * 3)
        rows.append(
            f"{number}\n{seconds_to_timecode(start)} --> {seconds_to_timecode(start + 2)}\n{text} {number}"
        )
    return "\n\n".join(rows) + "\n"


def _candidate(
    tmp_path: Path,
    *,
    long_viewer: bool = False,
    control_character: bool = False,
) -> dict[str, Path | str]:
    structure = _write(tmp_path / "structure.srt", _srt("日本語"))
    baseline = _write(tmp_path / "baseline.srt", _srt("기준"))
    decisions: list[dict[str, object]] = []
    reviews: list[dict[str, object]] = []
    for number in range(1, 299):
        decision: dict[str, object] = {
            "schema_name": "translation-forensics/pilot-terra-decision",
            "schema_version": "1",
            "title_id": "SSIS-908",
            "block_number": number,
            "translation_model": "gpt-5.6-terra",
            "source_faithful_korean": f"한국어 {number}",
            "viewer_natural_korean": (
                "가" * 43
                if long_viewer and number == 1
                else "제어\x00문자"
                if control_character and number == 1
                else f"자연어 {number}"
            ),
            "confidence": "high",
            "evidence_refs": [f"japanese-srt:{number}"],
            "risk_codes": [],
            "reason": "일본어 근거 범위에서 의미를 복원했다.",
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
            "review_plan_sha256": "a" * 64,
            "verdict": "accept",
            "critical_slot_conflicts": [],
            "unsupported_additions": [],
            "risk_codes": [],
            "resolved_issues": [],
            "semantic_recheck": {key: True for key in sorted(SEMANTIC_RECHECK_KEYS)},
            "reason": "독립 비평과 의미 재검사를 완료했다.",
        })
    terra = _write_jsonl(tmp_path / "terra.jsonl", decisions)
    sol = _write_jsonl(tmp_path / "sol.jsonl", reviews)
    output = tmp_path / "candidate"
    baseline_sha256 = hashlib.sha256(baseline.read_bytes()).hexdigest()
    build_pilot_candidate(
        title_id="SSIS-908",
        structure_path=structure,
        baseline_path=baseline,
        terra_decisions_path=terra,
        sol_reviews_path=sol,
        terra_prompt_path=ROOT / "prompts" / "terra-semantic-translation-v1.md",
        sol_prompt_path=ROOT / "prompts" / "autonomous-subtitle-critic-v1.md",
        provenance_schema_path=ROOT / "schemas" / "pilot-candidate-provenance.schema.json",
        output_dir=output,
        project_root=tmp_path,
        expected_baseline_sha256=baseline_sha256,
    )
    return {
        "structure": structure,
        "baseline": baseline,
        "candidate": output,
        "baseline_sha256": baseline_sha256,
    }


def _validate(tmp_path: Path, inputs: dict[str, Path | str]) -> dict[str, object]:
    return validate_pilot_automated(
        title_id="SSIS-908",
        structure_path=Path(inputs["structure"]),
        baseline_path=Path(inputs["baseline"]),
        candidate_dir=Path(inputs["candidate"]),
        provenance_schema_path=ROOT / "schemas" / "pilot-candidate-provenance.schema.json",
        terra_manifest_path=ROOT / "prompts" / "terra-semantic-translation-v1.manifest.json",
        autonomous_manifest_path=ROOT / "prompts" / "autonomous-subtitle-decision-v1.manifest.json",
        regression_suite_path=ROOT / "tests" / "fixtures" / "translation-quality-regressions.json",
        report_schema_path=ROOT / "schemas" / "pilot-automated-validation.schema.json",
        project_root=tmp_path,
        expected_baseline_sha256=str(inputs["baseline_sha256"]),
    )


def test_integrated_automated_validation_passes_all_locked_gates(tmp_path: Path) -> None:
    inputs = _candidate(tmp_path)
    report = _validate(tmp_path, inputs)

    assert report["status"] == "pass"
    assert report["automated_validation_passed"] is True
    assert report["failed_checks"] == []
    assert set(report["checks"]) == {
        "locked_srt_structure",
        "character_safety",
        "readability",
        "provenance",
        "evidence_references",
        "prompt_contracts",
        "synthetic_quality_regressions",
    }
    assert all(gate["status"] == "pass" for gate in report["checks"].values())
    assert report["checks"]["evidence_references"]["checked_refs"] == 298
    assert report["checks"]["synthetic_quality_regressions"]["cases"] == 19
    schema = json.loads(
        (ROOT / "schemas" / "pilot-automated-validation.schema.json").read_text(encoding="utf-8")
    )
    assert list(Draft202012Validator(schema).iter_errors(report)) == []

    output = tmp_path / "results" / "automated-validation.json"
    assert main([
        "validate-pilot-automated",
        "--project-root", str(tmp_path),
        "--title", "SSIS-908",
        "--structure", str(inputs["structure"]),
        "--baseline", str(inputs["baseline"]),
        "--candidate", str(inputs["candidate"]),
        "--provenance-schema", str(ROOT / "schemas" / "pilot-candidate-provenance.schema.json"),
        "--terra-manifest", str(ROOT / "prompts" / "terra-semantic-translation-v1.manifest.json"),
        "--autonomous-manifest", str(ROOT / "prompts" / "autonomous-subtitle-decision-v1.manifest.json"),
        "--regressions", str(ROOT / "tests" / "fixtures" / "translation-quality-regressions.json"),
        "--report-schema", str(ROOT / "schemas" / "pilot-automated-validation.schema.json"),
        "--expected-baseline-sha256", str(inputs["baseline_sha256"]),
        "--output", str(output),
    ]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"


def test_length_and_cps_violations_are_hard_failures(tmp_path: Path) -> None:
    inputs = _candidate(tmp_path, long_viewer=True)
    report = _validate(tmp_path, inputs)

    assert report["status"] == "fail"
    assert report["checks"]["provenance"]["status"] == "pass"
    assert report["checks"]["readability"]["status"] == "fail"
    assert {item["code"] for item in report["checks"]["readability"]["violations"]} == {
        "high_cps",
        "long_line",
    }


def test_control_character_is_a_character_safety_failure(tmp_path: Path) -> None:
    inputs = _candidate(tmp_path, control_character=True)
    report = _validate(tmp_path, inputs)

    assert report["status"] == "fail"
    assert report["checks"]["provenance"]["status"] == "pass"
    assert report["checks"]["character_safety"]["status"] == "fail"
    assert {item["code"] for item in report["checks"]["character_safety"]["violations"]} == {
        "control_character"
    }
