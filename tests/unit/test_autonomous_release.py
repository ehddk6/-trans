from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from translation_forensics.autonomous_release import prove_autonomous_claim, run_autonomous_release, validate_autonomous_regressions, validate_autonomous_release
from translation_forensics.cli import main
from translation_forensics.outputs import STAGES
from translation_forensics.prompt_contract import validate_prompt_contract
from translation_forensics.srt import parse_srt


ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, value: str) -> Path:
    path.write_text(value, encoding="utf-8", newline="\n")
    return path


def _srt(lines: list[str]) -> str:
    return "\n\n".join(f"{index}\n00:00:0{index},000 --> 00:00:0{index + 1},000\n{text}" for index, text in enumerate(lines, 1)) + "\n"


class FakeAutonomousProvider:
    def __init__(self) -> None:
        self.budget = SimpleNamespace(spent_usd=0.0)
        self.calls = 0

    def generate_decisions(self, *, title_id, prompt, payload, schema):
        self.calls += 1
        evidence = payload["evidence"]
        rows = []
        for item in evidence:
            number = item["block_number"]
            japanese = item["japanese"]["text"]
            if number == 1:
                rows.append({
                    "title_id": title_id, "block_number": number,
                    "source_faithful_korean": "안녕하세요.", "viewer_natural_korean": "안녕하세요.",
                    "source_status": "accepted", "viewer_status": "supported", "confidence": "high",
                    "evidence_refs": [f"japanese-srt:{number}"], "semantic_slots": self._slots(action="인사"),
                    "inferred_slots": [], "competing_interpretations": [], "risk_codes": [], "reason": japanese,
                })
            else:
                rows.append({
                    "title_id": title_id, "block_number": number,
                    "source_faithful_korean": "", "viewer_natural_korean": "…",
                    "source_status": "abstained", "viewer_status": "unrecoverable", "confidence": "low",
                    "evidence_refs": [f"japanese-srt:{number}"], "semantic_slots": self._slots(),
                    "inferred_slots": [], "competing_interpretations": [], "risk_codes": ["corrupt-japanese"], "reason": "손상",
                })
        return {"results": rows}, self._call(title_id, "translation-generation", "gpt-5.6-terra")

    def critique_decisions(self, *, title_id, prompt, payload, schema):
        reviews = []
        for number in payload["requested_block_numbers"]:
            reviews.append({
                "title_id": title_id, "block_number": number, "verdict": "quarantine",
                "critical_slot_conflicts": [], "unsupported_additions": [], "repair_instructions": "", "reason": "근거 부족",
            })
        return {"reviews": reviews}, self._call(title_id, "semantic-critique", "gpt-5.6-sol")

    def transcribe_audio(self, *, title_id, scene_id, audio_path):
        raise AssertionError("cloud ASR should not run without scenes")

    @staticmethod
    def _call(title_id: str, kind: str, model: str) -> dict[str, object]:
        return {
            "schema_name": "translation-forensics/model-call-manifest",
            "schema_version": "1",
            "title_id": title_id,
            "call_kind": kind,
            "provider": "openai",
            "model": model,
            "request_id": "fake-request",
            "request_sha256": "0" * 64,
            "response_sha256": "1" * 64,
            "evidence_sha256": "2" * 64,
            "model_settings_sha256": "3" * 64,
            "environment_sha256": "4" * 64,
            "environment": {},
            "cache_key": "5" * 64,
            "cache_hit": False,
            "cost_usd": 0.0,
            "external_transfer": False,
            "store": False,
            "tools_enabled": False,
        }

    @staticmethod
    def _slots(**values) -> dict[str, object]:
        keys = (
            "question", "polarity", "refusal_permission", "command_strength",
            "speaker", "actor", "action", "target", "location", "tense_aspect",
            "direction", "intensity",
        )
        return {key: values.get(key) for key in keys}


def test_autonomous_release_dual_output_and_claim_boundary(tmp_path: Path) -> None:
    structure = _write(tmp_path / "structure.srt", _srt(["こんにちは", "????"]))
    japanese = _write(tmp_path / "ja.srt", _srt(["こんにちは", "????"]))
    output = tmp_path / "release"
    result = run_autonomous_release(
        title_id="SAMPLE",
        structure_path=structure,
        japanese_path=japanese,
        output_dir=output,
        provider=FakeAutonomousProvider(),
        decision_prompt_path=ROOT / "prompts" / "autonomous-subtitle-decision-v1.md",
        critic_prompt_path=ROOT / "prompts" / "autonomous-subtitle-critic-v1.md",
        decision_schema_path=ROOT / "schemas" / "autonomous-decision-response.schema.json",
        critique_schema_path=ROOT / "schemas" / "autonomous-critique-response.schema.json",
    )
    assert result["status"] == "autonomous-release-packaged"
    source, _, _ = parse_srt(next(output.glob("*.source-faithful-ko.*.srt")))
    viewer, _, _ = parse_srt(next(output.glob("*.viewer-complete-ko.*.srt")))
    assert [block.text for block in source] == ["안녕하세요.", "…"]
    assert [block.text for block in viewer] == ["안녕하세요.", "…"]
    validation = validate_autonomous_release(output)
    assert validation["status"] == "pass"
    proof = prove_autonomous_claim(output)
    assert proof["human_reference_equality"] == "unidentifiable"
    assert proof["100_percent_equal"] is False
    assert proof["human_final_allowed"] is False
    assert proof["final_promotion_allowed"] is False
    viewer_path = next(output.glob("*.viewer-complete-ko.*.srt"))
    viewer_path.write_text(viewer_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert validate_autonomous_release(output)["status"] == "fail"


def test_autonomous_prompt_contract_has_six_failure_modes() -> None:
    result = validate_prompt_contract(ROOT / "prompts" / "autonomous-subtitle-decision-v1.manifest.json")
    assert result["status"] == "pass"
    assert result["test_case_count"] == 6


def test_autonomous_cli_dry_run_finds_workspace_inputs(tmp_path: Path) -> None:
    structure = _write(tmp_path / "structure.srt", _srt(["こんにちは"]))
    japanese = _write(tmp_path / "ja.srt", _srt(["こんにちは"]))
    assert main([
        "run-autonomous-release", "--title", "SAMPLE", "--structure", str(structure),
        "--ja", str(japanese), "--output", str(tmp_path / "release"), "--dry-run", "--json",
    ]) == 0
    assert not (tmp_path / "release").exists()


def test_autonomous_cli_live_mode_preflight_blocks_without_key(tmp_path: Path, monkeypatch) -> None:
    structure = _write(tmp_path / "structure.srt", _srt(["こんにちは"]))
    japanese = _write(tmp_path / "ja.srt", _srt(["こんにちは"]))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main([
        "run-autonomous-release", "--title", "SAMPLE", "--structure", str(structure),
        "--ja", str(japanese), "--output", str(tmp_path / "release"), "--allow-network",
        "--max-cost-usd", "1", "--cache-dir", str(tmp_path / "cache"), "--json",
    ]) == 2
    assert not (tmp_path / "release").exists()


def test_autonomous_release_is_not_a_human_final_stage() -> None:
    assert "autonomous-release" not in STAGES
    assert "final" in STAGES


def test_known_regression_case_enforces_abstention() -> None:
    decisions = [{
        "block_number": 978,
        "source_status": "abstained",
        "viewer_status": "best_effort",
        "risk_codes": ["context-conflict"],
    }]
    critiques = [{"block_number": 978, "attempt": 0}]
    suite = ROOT / "regressions" / "autonomous-release-v1.json"
    passed = validate_autonomous_regressions(title_id="JUQ-778", decisions=decisions, critiques=critiques, suite_path=suite)
    assert passed["status"] == "pass"
    decisions[0]["source_status"] = "accepted"
    failed = validate_autonomous_regressions(title_id="JUQ-778", decisions=decisions, critiques=critiques, suite_path=suite)
    assert failed["status"] == "fail"


def test_autonomous_record_schemas_are_valid_json() -> None:
    names = [
        "autonomous-evidence-bundle.schema.json",
        "autonomous-decision.schema.json",
        "autonomous-critique.schema.json",
        "model-call-manifest.schema.json",
        "autonomous-proof.schema.json",
    ]
    for name in names:
        schema = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
        assert schema["$id"].endswith("/v1")
