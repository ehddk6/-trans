from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator


TERRA_MODEL = "gpt-5.6-terra"
SOL_MODEL = "gpt-5.6-sol"
ROLE_POLICY = {
    "meaning-frame-terra": (TERRA_MODEL, "high"),
    "meaning-frame-sol": (SOL_MODEL, "xhigh"),
    "translation-terra": (TERRA_MODEL, "high"),
    "critique-sol": (SOL_MODEL, "xhigh"),
    "repair-terra": (TERRA_MODEL, "high"),
    "proxy-evaluator-terra": (TERRA_MODEL, "high"),
    "proxy-evaluator-sol": (SOL_MODEL, "xhigh"),
}


class CodexExecError(RuntimeError):
    pass


class CodexUsageLimitError(CodexExecError):
    """Codex service quota prevented a requested model call from starting."""

    def __init__(self, message: str, *, role: str, call_id: str, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.role = role
        self.call_id = call_id
        self.retry_after = retry_after


def _usage_limit_retry_after(output: str) -> str | None:
    match = re.search(r"try again at\s+(.+?\b(?:AM|PM))\b", output, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"try again at\s+([^\r\n}\"]+)", output, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).strip().rstrip(".")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_events(stdout: str) -> tuple[str, dict[str, int], str]:
    thread_id = ""
    usage: dict[str, int] = {}
    last_message = ""
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "thread.started":
            thread_id = str(event.get("thread_id") or "")
        elif event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = {
                key: int(event["usage"].get(key, 0) or 0)
                for key in (
                    "input_tokens",
                    "cached_input_tokens",
                    "cache_write_input_tokens",
                    "output_tokens",
                    "reasoning_output_tokens",
                )
            }
        elif event.get("type") == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                last_message = str(item.get("text") or "")
    return thread_id, usage, last_message


class CodexExecProvider:
    """Structured Codex CLI adapter with fail-closed, hash-linked receipts.

    Each live call runs in a fresh temporary directory, uses the requested
    model explicitly, disables project/user instructions, removes API-key
    environment variables, and records the CLI event stream needed to prove
    that a turn actually completed.  The CLI currently reports the requested
    model through the invocation, not as a server-returned model identifier;
    the receipt keeps that distinction explicit.
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        codex_executable: str = "codex",
        timeout_seconds: int = 600,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.cache_dir = cache_dir.expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.codex_executable = codex_executable
        self.timeout_seconds = max(1, int(timeout_seconds))
        self._runner = runner
        self._version: str | None = None

    def _cli_version(self) -> str:
        if self._version is not None:
            return self._version
        executable = shutil.which(self.codex_executable)
        if not executable:
            raise CodexExecError(f"Codex CLI executable not found: {self.codex_executable}")
        completed = self._runner(
            [executable, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=30,
        )
        if completed.returncode != 0:
            raise CodexExecError(f"Codex CLI version check failed: {completed.stderr.strip()[:400]}")
        self._version = completed.stdout.strip() or "unknown"
        return self._version

    def preflight(self) -> dict[str, Any]:
        errors: list[str] = []
        try:
            version = self._cli_version()
        except CodexExecError as exc:
            version = "unavailable"
            errors.append(str(exc))
        return {
            "status": "pass" if not errors else "blocked",
            "provider": "codex-cli",
            "codex_cli_version": version,
            "api_key_required": False,
            "allowed_models": [TERRA_MODEL, SOL_MODEL],
            "errors": errors,
        }

    def _cache_paths(self, request_sha256: str) -> tuple[Path, Path]:
        root = self.cache_dir / request_sha256
        return root / "response.json", root / "receipt.json"

    @staticmethod
    def _read_cache(response_path: Path, receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if not response_path.is_file() or not receipt_path.is_file():
            return None
        response = json.loads(response_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(response, dict) or not isinstance(receipt, dict):
            return None
        if receipt.get("status") != "succeeded" or receipt.get("model_call_verified") is not True:
            return None
        if receipt.get("response_sha256") != sha256_json(response):
            return None
        return response, {**receipt, "cache_hit": True}

    def run_structured(
        self,
        *,
        role: str,
        title_id: str,
        call_id: str,
        prompt: str,
        payload: dict[str, Any],
        schema: dict[str, Any],
        resume: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if role not in ROLE_POLICY:
            raise CodexExecError(f"Unsupported Codex quality role: {role}")
        model, reasoning_effort = ROLE_POLICY[role]
        cli_version = self._cli_version()
        request_contract = {
            "schema_name": "translation-forensics/codex-exec-request",
            "schema_version": "1",
            "title_id": title_id,
            "call_id": call_id,
            "role": role,
            "requested_model": model,
            "reasoning_effort": reasoning_effort,
            "codex_cli_version": cli_version,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "evidence_sha256": sha256_json(payload),
            "schema_sha256": sha256_json(schema),
            "sandbox": "read-only",
            "ephemeral": True,
            "ignore_user_config": True,
            "ignore_project_instructions": True,
            "api_key_used": False,
        }
        request_sha256 = sha256_json(request_contract)
        response_cache, receipt_cache = self._cache_paths(request_sha256)
        if resume:
            cached = self._read_cache(response_cache, receipt_cache)
            if cached is not None:
                return cached

        started_at = _utc_now()
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="translation-forensics-codex-") as temp_name:
            temp_dir = Path(temp_name)
            schema_path = temp_dir / "response.schema.json"
            output_path = temp_dir / "response.json"
            schema_path.write_text(
                json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            input_text = (
                prompt.rstrip()
                + "\n\nReturn only one JSON object that conforms to the supplied output schema."
                + " Do not use tools and do not inspect the filesystem.\n\nINPUT_JSON:\n"
                + canonical_json(payload)
            )
            executable = shutil.which(self.codex_executable)
            if not executable:
                raise CodexExecError(f"Codex CLI executable not found: {self.codex_executable}")
            command = [
                executable,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "-s",
                "read-only",
                "-C",
                str(temp_dir),
                "-m",
                model,
                "-c",
                f'model_reasoning_effort="{reasoning_effort}"',
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "--json",
                "-",
            ]
            env = os.environ.copy()
            env.pop("OPENAI_API_KEY", None)
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            try:
                completed = self._runner(
                    command,
                    input=input_text,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                    timeout=self.timeout_seconds,
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                raise CodexExecError(f"Codex call timed out for {call_id} after {self.timeout_seconds}s") from exc

            thread_id, usage, event_message = _parse_events(completed.stdout)
            raw_response = output_path.read_text(encoding="utf-8").strip() if output_path.is_file() else event_message.strip()
            if completed.returncode != 0:
                failure_output = f"{completed.stdout}\n{completed.stderr}"
                if "usage limit" in failure_output.lower():
                    retry_after = _usage_limit_retry_after(failure_output)
                    suffix = f"; retry after {retry_after}" if retry_after else ""
                    raise CodexUsageLimitError(
                        f"Codex usage limit stopped {role} call {call_id}{suffix}",
                        role=role,
                        call_id=call_id,
                        retry_after=retry_after,
                    )
                raise CodexExecError(
                    f"Codex {role} call failed for {call_id} with exit {completed.returncode}: "
                    f"stdout={completed.stdout.strip()[-1200:]} stderr={completed.stderr.strip()[-800:]}"
                )
            if not thread_id or not usage:
                raise CodexExecError(f"Codex {role} call lacks thread or usage completion evidence for {call_id}")
            try:
                response = json.loads(raw_response)
            except json.JSONDecodeError as exc:
                raise CodexExecError(f"Codex {role} returned invalid JSON for {call_id}") from exc
            if not isinstance(response, dict):
                raise CodexExecError(f"Codex {role} response must be a JSON object for {call_id}")
            schema_error = next(iter(Draft202012Validator(schema).iter_errors(response)), None)
            if schema_error is not None:
                raise CodexExecError(f"Codex {role} schema error for {call_id}: {schema_error.message}")

        receipt = {
            "schema_name": "translation-forensics/codex-model-call-receipt",
            "schema_version": "1",
            "status": "succeeded",
            "title_id": title_id,
            "call_id": call_id,
            "role": role,
            "provider": "codex-cli",
            "execution_mode": "codex-exec-ephemeral",
            "requested_model": model,
            "reasoning_effort": reasoning_effort,
            "codex_cli_version": cli_version,
            "thread_id": thread_id,
            "request_sha256": request_sha256,
            "prompt_sha256": request_contract["prompt_sha256"],
            "evidence_sha256": request_contract["evidence_sha256"],
            "schema_sha256": request_contract["schema_sha256"],
            "response_sha256": sha256_json(response),
            "input_tokens": usage.get("input_tokens", 0),
            "cached_input_tokens": usage.get("cached_input_tokens", 0),
            "cache_write_input_tokens": usage.get("cache_write_input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "reasoning_output_tokens": usage.get("reasoning_output_tokens", 0),
            "exit_code": 0,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "sandbox": "read-only",
            "ephemeral": True,
            "isolated_temporary_directory": True,
            "api_key_used": False,
            "authentication": "codex-login",
            "requested_model_verified_by_cli_invocation": True,
            "actual_server_model_reported_by_cli": False,
            "model_call_verified": True,
            "cache_hit": False,
            "stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
        }
        response_cache.parent.mkdir(parents=True, exist_ok=True)
        response_cache.write_text(
            json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        receipt_cache.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        return response, receipt


def validate_call_receipt(receipt: dict[str, Any], *, expected_role: str | None = None) -> list[str]:
    errors: list[str] = []
    role = str(receipt.get("role") or "")
    if expected_role is not None and role != expected_role:
        errors.append(f"role mismatch: expected {expected_role}, got {role or '<empty>'}")
    if role not in ROLE_POLICY:
        errors.append(f"unsupported role: {role or '<empty>'}")
        return errors
    model, effort = ROLE_POLICY[role]
    if receipt.get("requested_model") != model:
        errors.append(f"model mismatch for {role}: {receipt.get('requested_model')!r}")
    if receipt.get("reasoning_effort") != effort:
        errors.append(f"reasoning effort mismatch for {role}: {receipt.get('reasoning_effort')!r}")
    required_true = (
        "ephemeral",
        "isolated_temporary_directory",
        "requested_model_verified_by_cli_invocation",
        "model_call_verified",
    )
    for field in required_true:
        if receipt.get(field) is not True:
            errors.append(f"{field} must be true")
    if receipt.get("api_key_used") is not False:
        errors.append("api_key_used must be false")
    if receipt.get("sandbox") != "read-only" or receipt.get("exit_code") != 0:
        errors.append("Codex sandbox/exit contract failed")
    for field in (
        "thread_id",
        "codex_cli_version",
        "request_sha256",
        "prompt_sha256",
        "evidence_sha256",
        "schema_sha256",
        "response_sha256",
    ):
        if not str(receipt.get(field) or "").strip():
            errors.append(f"missing {field}")
    return errors
