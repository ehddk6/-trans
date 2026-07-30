from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from translation_forensics.codex_exec_provider import (
    CodexExecError,
    CodexExecProvider,
    CodexUsageLimitError,
    validate_call_receipt,
)


SCHEMA = {
    "type": "object",
    "required": ["value"],
    "properties": {"value": {"type": "string"}},
    "additionalProperties": False,
}


def fake_runner(command, **kwargs):
    if "--version" in command:
        return subprocess.CompletedProcess(command, 0, "codex-cli 0.test\n", "")
    output_path = Path(command[command.index("--output-last-message") + 1])
    output_path.write_text(json.dumps({"value": "ok"}), encoding="utf-8")
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-test"}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": '{"value":"ok"}'}}),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 4}}),
        ]
    )
    assert "OPENAI_API_KEY" not in kwargs["env"]
    assert "--ephemeral" in command and "--ignore-user-config" in command and "--ignore-rules" in command
    return subprocess.CompletedProcess(command, 0, stdout, "warning")


def test_codex_exec_receipt_proves_completed_requested_model(tmp_path, monkeypatch):
    monkeypatch.setattr("translation_forensics.codex_exec_provider.shutil.which", lambda _: "codex")
    provider = CodexExecProvider(cache_dir=tmp_path / "cache", runner=fake_runner)
    response, receipt = provider.run_structured(
        role="meaning-frame-sol",
        title_id="SAMPLE",
        call_id="scene-1.sol",
        prompt="Review independently.",
        payload={"scene": 1},
        schema=SCHEMA,
        resume=False,
    )
    assert response == {"value": "ok"}
    assert receipt["requested_model"] == "gpt-5.6-sol"
    assert receipt["reasoning_effort"] == "xhigh"
    assert receipt["thread_id"] == "thread-test"
    assert receipt["model_call_verified"] is True
    assert receipt["actual_server_model_reported_by_cli"] is False
    assert validate_call_receipt(receipt) == []


def test_codex_exec_cache_keeps_original_execution_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr("translation_forensics.codex_exec_provider.shutil.which", lambda _: "codex")
    provider = CodexExecProvider(cache_dir=tmp_path / "cache", runner=fake_runner)
    first = provider.run_structured(
        role="translation-terra",
        title_id="SAMPLE",
        call_id="scene-1.terra",
        prompt="Translate.",
        payload={"scene": 1},
        schema=SCHEMA,
        resume=True,
    )
    second = provider.run_structured(
        role="translation-terra",
        title_id="SAMPLE",
        call_id="scene-1.terra",
        prompt="Translate.",
        payload={"scene": 1},
        schema=SCHEMA,
        resume=True,
    )
    assert first[1]["cache_hit"] is False
    assert second[1]["cache_hit"] is True
    assert second[1]["thread_id"] == first[1]["thread_id"]


def test_receipt_tampered_model_fails_validation():
    receipt = {
        "role": "meaning-frame-sol",
        "requested_model": "gpt-5.6-terra",
        "reasoning_effort": "xhigh",
        "ephemeral": True,
        "isolated_temporary_directory": True,
        "requested_model_verified_by_cli_invocation": True,
        "model_call_verified": True,
        "api_key_used": False,
        "sandbox": "read-only",
        "exit_code": 0,
        "thread_id": "t",
        "codex_cli_version": "v",
        "request_sha256": "a",
        "prompt_sha256": "b",
        "evidence_sha256": "c",
        "schema_sha256": "d",
        "response_sha256": "e",
    }
    assert any("model mismatch" in error for error in validate_call_receipt(receipt))


def test_unknown_role_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("translation_forensics.codex_exec_provider.shutil.which", lambda _: "codex")
    provider = CodexExecProvider(cache_dir=tmp_path, runner=fake_runner)
    with pytest.raises(CodexExecError, match="Unsupported"):
        provider.run_structured(
            role="fallback-sol-to-terra",
            title_id="SAMPLE",
            call_id="bad",
            prompt="x",
            payload={},
            schema=SCHEMA,
        )


def test_usage_limit_is_typed_and_preserves_retry_time(tmp_path, monkeypatch):
    monkeypatch.setattr("translation_forensics.codex_exec_provider.shutil.which", lambda _: "codex")

    def usage_limited_runner(command, **kwargs):
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, "codex-cli 0.test\n", "")
        stdout = "ERROR: You've hit your usage limit. To get more access now, try again at Aug 4th, 2026 4:49 PM."
        return subprocess.CompletedProcess(command, 1, stdout, "")

    provider = CodexExecProvider(cache_dir=tmp_path, runner=usage_limited_runner)
    with pytest.raises(CodexUsageLimitError) as caught:
        provider.run_structured(
            role="translation-terra",
            title_id="ADN-622",
            call_id="scene-0004.translation.terra",
            prompt="Translate.",
            payload={"scene": 4},
            schema=SCHEMA,
            resume=False,
        )
    assert caught.value.role == "translation-terra"
    assert caught.value.call_id == "scene-0004.translation.terra"
    assert caught.value.retry_after == "Aug 4th, 2026 4:49 PM"
