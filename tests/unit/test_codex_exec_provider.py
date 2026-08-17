from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from translation_forensics.codex_exec_provider import (
    CodexExecError,
    CodexExecProvider,
    CodexTimeoutError,
    CodexUsageLimitError,
    _run_with_process_tree_timeout,
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


@pytest.mark.parametrize("corrupt_name", ["response.json", "receipt.json"])
def test_codex_exec_malformed_cache_is_a_miss(tmp_path, monkeypatch, corrupt_name):
    monkeypatch.setattr("translation_forensics.codex_exec_provider.shutil.which", lambda _: "codex")
    calls = 0

    def counting_runner(command, **kwargs):
        nonlocal calls
        if "--version" not in command:
            calls += 1
        return fake_runner(command, **kwargs)

    provider = CodexExecProvider(cache_dir=tmp_path / "cache", runner=counting_runner)
    first = provider.run_structured(
        role="translation-terra",
        title_id="SAMPLE",
        call_id="scene-1.terra",
        prompt="Translate.",
        payload={"scene": 1},
        schema=SCHEMA,
        resume=True,
    )
    cache_file = tmp_path / "cache" / first[1]["request_sha256"] / corrupt_name
    cache_file.write_text("{truncated", encoding="utf-8")

    second = provider.run_structured(
        role="translation-terra",
        title_id="SAMPLE",
        call_id="scene-1.terra",
        prompt="Translate.",
        payload={"scene": 1},
        schema=SCHEMA,
        resume=True,
    )

    assert calls == 2
    assert second[1]["cache_hit"] is False
    assert json.loads(cache_file.read_text(encoding="utf-8"))
    assert not list(cache_file.parent.glob(".*.tmp"))


def test_codex_exec_cache_rejects_wrong_request_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr("translation_forensics.codex_exec_provider.shutil.which", lambda _: "codex")
    calls = 0

    def counting_runner(command, **kwargs):
        nonlocal calls
        if "--version" not in command:
            calls += 1
        return fake_runner(command, **kwargs)

    provider = CodexExecProvider(cache_dir=tmp_path / "cache", runner=counting_runner)
    first = provider.run_structured(
        role="translation-terra",
        title_id="SAMPLE",
        call_id="scene-1.terra",
        prompt="Translate.",
        payload={"scene": 1},
        schema=SCHEMA,
        resume=True,
    )
    receipt_path = tmp_path / "cache" / first[1]["request_sha256"] / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["request_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    second = provider.run_structured(
        role="translation-terra",
        title_id="SAMPLE",
        call_id="scene-1.terra",
        prompt="Translate.",
        payload={"scene": 1},
        schema=SCHEMA,
        resume=True,
    )

    assert calls == 2
    assert second[1]["cache_hit"] is False


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


def test_timeout_is_typed_and_preserves_call_context(tmp_path, monkeypatch):
    monkeypatch.setattr("translation_forensics.codex_exec_provider.shutil.which", lambda _: "codex")

    def timeout_runner(command, **kwargs):
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, "codex-cli 0.test\n", "")
        raise subprocess.TimeoutExpired(command, timeout=kwargs["timeout"])

    provider = CodexExecProvider(cache_dir=tmp_path, runner=timeout_runner, timeout_seconds=7)
    with pytest.raises(CodexTimeoutError) as caught:
        provider.run_structured(
            role="translation-audit-sol",
            title_id="ADN-622",
            call_id="batch-0001.translation-audit.sol",
            prompt="Audit.",
            payload={"batch": 1},
            schema=SCHEMA,
            resume=False,
        )

    assert caught.value.role == "translation-audit-sol"
    assert caught.value.call_id == "batch-0001.translation-audit.sol"
    assert caught.value.timeout_seconds == 7


@pytest.mark.skipif(os.name != "nt", reason="Windows taskkill process-tree contract")
def test_default_runner_terminates_descendant_process_tree_on_timeout(tmp_path):
    marker = tmp_path / "child.pid"
    child_code = "import time; time.sleep(60)"
    parent_code = (
        "import pathlib,subprocess,sys,time;"
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid));"
        "time.sleep(60)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        _run_with_process_tree_timeout(
            [sys.executable, "-c", parent_code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1,
        )
    child_pid = int(marker.read_text(encoding="utf-8"))
    still_running = True
    for _ in range(20):
        listing = subprocess.run(
            ["tasklist", "/FI", f"PID eq {child_pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        ).stdout
        still_running = f'"{child_pid}"' in listing
        if not still_running:
            break
        time.sleep(0.1)
    assert still_running is False


def test_image_attachment_requires_explicit_transfer_and_is_receipted(tmp_path, monkeypatch):
    monkeypatch.setattr("translation_forensics.codex_exec_provider.shutil.which", lambda _: "codex")
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg-test")
    provider = CodexExecProvider(cache_dir=tmp_path / "cache", runner=fake_runner)

    with pytest.raises(CodexExecError, match="allow_image_transfer"):
        provider.run_structured(
            role="critique-sol",
            title_id="SAMPLE",
            call_id="unit-1.visual.sol",
            prompt="Review the supplied frame.",
            payload={"unit_id": "unit-1"},
            schema=SCHEMA,
            image_paths=[image],
        )

    response, receipt = provider.run_structured(
        role="critique-sol",
        title_id="SAMPLE",
        call_id="unit-1.visual.sol",
        prompt="Review the supplied frame.",
        payload={"unit_id": "unit-1"},
        schema=SCHEMA,
        image_paths=[image],
        allow_image_transfer=True,
        resume=False,
    )
    assert response == {"value": "ok"}
    assert receipt["external_transfer"] is True
    assert receipt["pixel_external_transfer_count"] == 1
    assert receipt["image_attachments"][0]["name"] == "frame.jpg"
    assert len(receipt["image_attachments"][0]["sha256"]) == 64
    assert validate_call_receipt(receipt, expected_role="critique-sol") == []


def test_image_attachment_limit_is_three(tmp_path, monkeypatch):
    monkeypatch.setattr("translation_forensics.codex_exec_provider.shutil.which", lambda _: "codex")
    images = []
    for index in range(4):
        image = tmp_path / f"frame-{index}.jpg"
        image.write_bytes(bytes([index]))
        images.append(image)
    provider = CodexExecProvider(cache_dir=tmp_path / "cache", runner=fake_runner)
    with pytest.raises(CodexExecError, match="At most 3"):
        provider.run_structured(
            role="critique-sol",
            title_id="SAMPLE",
            call_id="unit-1.visual.sol",
            prompt="Review.",
            payload={},
            schema=SCHEMA,
            image_paths=images,
            allow_image_transfer=True,
        )


def test_image_cache_identity_uses_content_not_staging_path(tmp_path, monkeypatch):
    monkeypatch.setattr("translation_forensics.codex_exec_provider.shutil.which", lambda _: "codex")
    first_image = tmp_path / "run-a" / "frame.jpg"
    second_image = tmp_path / "run-b" / "frame.jpg"
    first_image.parent.mkdir()
    second_image.parent.mkdir()
    first_image.write_bytes(b"same-pixels")
    second_image.write_bytes(b"same-pixels")
    calls = 0

    def counting_runner(command, **kwargs):
        nonlocal calls
        if "--version" not in command:
            calls += 1
        return fake_runner(command, **kwargs)

    provider = CodexExecProvider(
        cache_dir=tmp_path / "cache", runner=counting_runner
    )
    common = dict(
        role="critique-sol",
        title_id="SAMPLE",
        call_id="unit-1.visual.sol",
        prompt="Review.",
        payload={"unit_id": "unit-1"},
        schema=SCHEMA,
        resume=True,
        allow_image_transfer=True,
    )
    provider.run_structured(**common, image_paths=[first_image])
    _, replay = provider.run_structured(**common, image_paths=[second_image])
    assert calls == 1
    assert replay["cache_hit"] is True
    assert replay["cache_replay_pixel_external_transfer_count"] == 0
    assert replay["image_attachments"][0]["path"] == str(second_image.resolve())
