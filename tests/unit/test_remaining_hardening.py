from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest

from translation_forensics.autonomous_release import _model_calls_traceable, validate_autonomous_release
from translation_forensics.closed_world import _candidate_family, _decide_block
from translation_forensics.openai_provider import BudgetExceededError, BudgetTracker, OpenAIProvider
from translation_forensics.srt import SubtitleBlock


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _valid_call_record() -> dict[str, object]:
    digest = "a" * 64
    return {
        "schema_name": "translation-forensics/model-call-manifest",
        "schema_version": "1",
        "call_kind": "translation-generation",
        "provider": "openai",
        "model": "gpt-5.6-terra",
        "request_id": "request-1",
        "request_sha256": digest,
        "response_sha256": digest,
        "evidence_sha256": digest,
        "model_settings_sha256": digest,
        "environment": {},
        "environment_sha256": digest,
        "cache_key": digest,
        "cache_hit": False,
        "cost_usd": 0.01,
        "external_transfer": True,
        "store": False,
        "tools_enabled": False,
    }


def test_empty_model_call_manifest_is_not_traceable() -> None:
    assert _model_calls_traceable([]) is False
    assert _model_calls_traceable([_valid_call_record()]) is True


def test_autonomous_validator_rejects_manifest_path_escape(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    _write(package / "structure.srt", "1\n00:00:01,000 --> 00:00:02,000\n一\n")
    _write(package / "autonomous-decisions.jsonl", "")
    _write(package / "autonomous-critiques.jsonl", "")
    _write(package / "autonomous-evidence.jsonl", "")
    _write(package / "model-call-manifest.jsonl", "")
    _write(package / "uncertainty-map.jsonl", "")
    _write(package / "title-memory.json", "{}\n")
    _write(package / "machine-alignment.json", "{}\n")
    _write(package / "evidence-graph.json", "{}\n")
    _write(package / "qa-report.json", '{"status":"fail"}\n')
    _write(
        package / "autonomous-release-report.json",
        json.dumps(
            {
                "title_id": "TITLE",
                "release_kind": "autonomous-release",
                "status": "autonomous-release-invalid",
                "model_calls": 0,
                "estimated_cost_usd": 0.0,
                "all_model_calls_traceable": False,
                "human_final_allowed": False,
                "final_promotion_allowed": False,
            }
        )
        + "\n",
    )
    _write(
        package / "autonomous-proof.json",
        json.dumps(
            {
                "human_reference_equality": "unidentifiable",
                "100_percent_equal": False,
                "all_model_calls_traceable": False,
                "human_final_allowed": False,
                "final_promotion_allowed": False,
            }
        )
        + "\n",
    )
    _write(
        package / "autonomous-release-manifest.json",
        json.dumps(
            {
                "title_id": "TITLE",
                "outputs": [{"path": "../outside.txt", "sha256": "0" * 64}],
                "final_promotion_allowed": False,
            }
        )
        + "\n",
    )

    result = validate_autonomous_release(package)
    assert result["status"] == "fail"
    assert any("unsafe manifest output path" in error for error in result["errors"])




def test_environment_record_hashes_executable_bytes() -> None:
    record = OpenAIProvider._environment_record()
    executable = Path(sys.executable)
    if executable.is_file():
        expected = hashlib.sha256(executable.read_bytes()).hexdigest()
        assert record["executable_sha256"] == expected

def test_budget_commit_records_actual_charge_before_raising() -> None:
    budget = BudgetTracker(maximum_usd=0.10)
    budget.reserve(0.05)
    with pytest.raises(BudgetExceededError):
        budget.commit(0.05, 0.20)
    assert budget.spent_usd == pytest.approx(0.20)
    assert budget.reserved_usd == pytest.approx(0.0)


class _InvalidStructuredResponse:
    output_text = "{invalid-json"
    usage = {"input_tokens": 1_000, "output_tokens": 100}
    id = "response-1"


class _Responses:
    def create(self, **_: object) -> _InvalidStructuredResponse:
        return _InvalidStructuredResponse()


class _Client:
    responses = _Responses()


def test_paid_invalid_structured_response_is_charged(tmp_path: Path) -> None:
    budget = BudgetTracker(maximum_usd=10.0)
    provider = OpenAIProvider(
        cache_dir=tmp_path / "cache",
        budget=budget,
        allow_network=True,
        api_key="test-key",
        client=_Client(),
        max_retries=0,
    )
    with pytest.raises(json.JSONDecodeError):
        provider.generate_decisions(
            title_id="TITLE",
            prompt="prompt",
            payload={"evidence": []},
            schema={"type": "object"},
        )
    assert budget.spent_usd > 0
    assert budget.reserved_usd == pytest.approx(0.0)


def _load_codex_bridge_module() -> object:
    path = Path(__file__).resolve().parents[2] / "tools" / "codex_autonomous_release.py"
    spec = importlib.util.spec_from_file_location("codex_autonomous_release_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_codex_finalize_detects_prepare_input_change(tmp_path: Path) -> None:
    module = _load_codex_bridge_module()
    source = _write(tmp_path / "source.srt", "original\n")
    info = {
        "prepared_inputs": [
            {"role": "structure", "path": str(source), "sha256": module._sha256(source)}
        ]
    }
    source.write_text("changed\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="prepare 이후 입력이 변경"):
        module._verify_prepared_inputs(info)


def test_unprovenanced_candidates_are_one_family_and_cannot_self_confirm(tmp_path: Path) -> None:
    first = tmp_path / "model-a.source-faithful.srt"
    second = tmp_path / "renamed.viewer-natural.srt"
    family_a = _candidate_family(first, previous_path=None, provenance=None)
    family_b = _candidate_family(second, previous_path=None, provenance=None)
    assert family_a == family_b == "unprovenanced-korean-candidates"

    block = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "ありがとう", 1.0, 2.0)
    decision, _, _ = _decide_block(
        block,
        "ありがとう",
        [
            {"path": str(first), "family": family_a, "role": "source-faithful", "text": "고마워"},
            {"path": str(second), "family": family_b, "role": "viewer-natural", "text": "고마워"},
        ],
        [],
    )
    assert decision["status"] == "abstained"
    assert "insufficient-generated-candidate-families" in decision["abstention_reasons"]


def test_verified_source_family_controls_candidate_identity(tmp_path: Path) -> None:
    path = tmp_path / "candidate.srt"
    first = _candidate_family(path, previous_path=None, provenance={"source_family": "model-a/run-1"})
    second = _candidate_family(path, previous_path=None, provenance={"source_family": "model-a/run-1"})
    other = _candidate_family(path, previous_path=None, provenance={"source_family": "model-b/run-2"})
    assert first == second
    assert first != other
