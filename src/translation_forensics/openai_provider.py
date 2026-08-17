from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


TEXT_MODEL_PRICING_PER_MILLION = {
    "gpt-5.6-terra": {"input": 2.50, "output": 15.00},
    "gpt-5.6-sol": {"input": 5.00, "output": 30.00},
}
AUDIO_MODEL_PRICING_PER_MILLION = {
    "gpt-4o-transcribe": {"input": 2.50, "output": 10.00},
}


class BudgetExceededError(RuntimeError):
    pass


class ProviderUnavailableError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _object_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass
class BudgetTracker:
    maximum_usd: float
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reserve(self, amount_usd: float) -> None:
        amount_usd = max(0.0, float(amount_usd))
        with self._lock:
            if self.spent_usd + self.reserved_usd + amount_usd > self.maximum_usd + 1e-9:
                raise BudgetExceededError(
                    f"OpenAI 비용 한도를 초과할 수 있어 호출을 중단합니다: "
                    f"spent={self.spent_usd:.6f}, reserved={self.reserved_usd:.6f}, "
                    f"next={amount_usd:.6f}, max={self.maximum_usd:.6f}"
                )
            self.reserved_usd += amount_usd

    def commit(self, reserved_usd: float, actual_usd: float) -> None:
        reserved_usd = max(0.0, float(reserved_usd))
        actual_usd = max(0.0, float(actual_usd))
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - reserved_usd)
            self.spent_usd += actual_usd
            if self.spent_usd + self.reserved_usd > self.maximum_usd + 1e-9:
                raise BudgetExceededError(
                    "Actual OpenAI cost exceeded the configured cap after the provider response: "
                    f"spent={self.spent_usd:.6f}, reserved={self.reserved_usd:.6f}, "
                    f"max={self.maximum_usd:.6f}"
                )

    def release(self, reserved_usd: float) -> None:
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - max(0.0, reserved_usd))

    def restore_spent(self, amount_usd: float) -> None:
        """Restore charges from a checkpoint so resumes keep one cost cap."""
        amount_usd = max(0.0, float(amount_usd))
        with self._lock:
            if self.spent_usd + self.reserved_usd + amount_usd > self.maximum_usd + 1e-9:
                raise BudgetExceededError(
                    "Restored checkpoint cost exceeds --max-cost-usd: "
                    f"spent={self.spent_usd:.6f}, checkpoint={amount_usd:.6f}, max={self.maximum_usd:.6f}"
                )
            self.spent_usd += amount_usd


class OpenAIProvider:
    """Responses/Transcription API adapter with cache, retry and cost provenance.

    The OpenAI package is imported lazily so local structural tests do not need
    the optional cloud dependency.  Live calls are impossible unless network
    use, an API key and a positive budget were all supplied explicitly.
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        budget: BudgetTracker,
        allow_network: bool,
        api_key: str | None = None,
        max_retries: int = 2,
        request_timeout_seconds: float = 600.0,
        client: Any | None = None,
    ) -> None:
        self.cache_dir = cache_dir.expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.budget = budget
        self.allow_network = bool(allow_network)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.max_retries = max(0, int(max_retries))
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self._client = client
        self._client_lock = threading.Lock()
        self._cache_lock = threading.Lock()

    @staticmethod
    def _environment_record() -> dict[str, str]:
        try:
            import openai

            sdk_version = str(getattr(openai, "__version__", "unknown"))
        except ImportError:
            sdk_version = "not-installed"
        executable = Path(sys.executable)
        try:
            executable_sha256 = sha256_bytes(executable.read_bytes())
        except OSError:
            executable_sha256 = sha256_bytes(str(executable).encode("utf-8"))
        return {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "openai_sdk": sdk_version,
            "executable_sha256": executable_sha256,
        }

    def preflight(self) -> dict[str, Any]:
        errors: list[str] = []
        if not self.allow_network:
            errors.append("--allow-network가 필요합니다.")
        if not self.api_key:
            errors.append("OPENAI_API_KEY가 없습니다.")
        if self.budget.maximum_usd <= 0:
            errors.append("--max-cost-usd는 0보다 커야 합니다.")
        sdk_available = True
        if self._client is None:
            try:
                import openai  # noqa: F401
            except ImportError:
                sdk_available = False
                errors.append("선택 의존성 openai가 설치되지 않았습니다: pip install -e .[cloud]")
        return {
            "status": "pass" if not errors else "blocked",
            "allow_network": self.allow_network,
            "api_key_present": bool(self.api_key),
            "sdk_available": sdk_available,
            "max_cost_usd": self.budget.maximum_usd,
            "errors": errors,
        }

    def _client_instance(self) -> Any:
        report = self.preflight()
        if report["status"] != "pass":
            raise ProviderUnavailableError("; ".join(report["errors"]))
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    from openai import OpenAI

                    self._client = OpenAI(
                        api_key=self.api_key, timeout=self.request_timeout_seconds
                    )
        return self._client

    @staticmethod
    def _estimate_text_reserve(model: str, request: dict[str, Any]) -> float:
        pricing = TEXT_MODEL_PRICING_PER_MILLION.get(model, {"input": 10.0, "output": 30.0})
        # UTF-8 byte length is a deliberately conservative token upper bound for
        # Japanese/Korean payloads.  The live request also caps model output.
        input_tokens = max(1, len(canonical_json(request).encode("utf-8")))
        output_tokens = 8_000
        return round((input_tokens * pricing["input"] * 1.25 + output_tokens * pricing["output"]) / 1_000_000, 6)

    @staticmethod
    def _actual_text_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = TEXT_MODEL_PRICING_PER_MILLION.get(model, {"input": 10.0, "output": 30.0})
        return round((input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000, 6)

    @staticmethod
    def _actual_audio_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = AUDIO_MODEL_PRICING_PER_MILLION[model]
        return round((input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000, 6)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _structured_call(
        self,
        *,
        call_kind: str,
        title_id: str,
        model: str,
        reasoning_effort: str,
        prompt: str,
        payload: dict[str, Any],
        schema: dict[str, Any],
        schema_name: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request_fingerprint = {
            "call_kind": call_kind,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "payload_sha256": sha256_json(payload),
            "schema_sha256": sha256_json(schema),
            "store": False,
            "tools": [],
            "max_output_tokens": 8_000,
        }
        cache_key = sha256_json(request_fingerprint)
        cache_path = self._cache_path(cache_key)
        with self._cache_lock:
            if cache_path.exists():
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    response_payload = cached["response_payload"]
                    cached_record = cached["call_record"]
                    schema_error = next(
                        iter(Draft202012Validator(schema).iter_errors(response_payload)),
                        None,
                    )
                    if (
                        not isinstance(response_payload, dict)
                        or not isinstance(cached_record, dict)
                        or cached_record.get("request_sha256") != sha256_json(request_fingerprint)
                        or cached_record.get("response_sha256") != sha256_json(response_payload)
                        or schema_error is not None
                    ):
                        raise ValueError("cached OpenAI response contract mismatch")
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    pass
                else:
                    record = {
                        **cached_record,
                        "cache_hit": True,
                        "cost_usd": 0.0,
                        "original_cost_usd": float(cached_record.get("cost_usd", 0.0) or 0.0),
                        "external_transfer": False,
                        "cache_origin_external_transfer": bool(cached_record.get("external_transfer", False)),
                    }
                    return response_payload, record

        client = self._client_instance()
        reserve = self._estimate_text_reserve(model, {"prompt": prompt, "payload": payload, "schema": schema})
        self.budget.reserve(reserve)
        started = time.time()
        response: Any | None = None
        settled = False
        try:
            last_error: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.responses.create(
                        model=model,
                        reasoning={"effort": reasoning_effort},
                        input=[
                            {"role": "developer", "content": prompt},
                            {"role": "user", "content": canonical_json(payload)},
                        ],
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": schema_name,
                                "schema": schema,
                                "strict": True,
                            }
                        },
                        tools=[],
                        store=False,
                        max_output_tokens=8_000,
                        timeout=self.request_timeout_seconds,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= self.max_retries:
                        raise
                    time.sleep(0.25 * (2**attempt))
            if response is None:
                raise ProviderUnavailableError(f"OpenAI 응답이 없습니다: {last_error}")

            usage = _object_value(response, "usage", {}) or {}
            input_tokens = int(_object_value(usage, "input_tokens", 0) or 0)
            output_tokens = int(_object_value(usage, "output_tokens", 0) or 0)
            if input_tokens or output_tokens:
                actual_cost = self._actual_text_cost(model, input_tokens, output_tokens)
                cost_basis = "provider-token-usage"
            else:
                actual_cost = reserve
                cost_basis = "conservative-reservation-no-usage"

            output_text = str(_object_value(response, "output_text", "")).strip()
            if not output_text:
                raise ValueError("OpenAI structured response output_text가 비어 있습니다.")
            response_payload = json.loads(output_text)
            schema_error = next(
                iter(Draft202012Validator(schema).iter_errors(response_payload)), None
            )
            if schema_error is not None:
                raise ValueError(
                    f"OpenAI structured response schema error: {schema_error.message}"
                )
            settled = True
            self.budget.commit(reserve, actual_cost)
            record = {
                "schema_name": "translation-forensics/model-call-manifest",
                "schema_version": "1",
                "title_id": title_id,
                "call_kind": call_kind,
                "provider": "openai",
                "model": model,
                "reasoning_effort": reasoning_effort,
                "request_id": str(_object_value(response, "_request_id", "") or _object_value(response, "id", "") or "unavailable"),
                "request_sha256": sha256_json(request_fingerprint),
                "response_sha256": sha256_json(response_payload),
                "prompt_sha256": request_fingerprint["prompt_sha256"],
                "evidence_sha256": request_fingerprint["payload_sha256"],
                "schema_sha256": request_fingerprint["schema_sha256"],
                "model_settings_sha256": sha256_json({
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "store": False,
                    "tools": [],
                    "max_output_tokens": 8_000,
                    "timeout_seconds": self.request_timeout_seconds,
                }),
                "cache_key": cache_key,
                "cache_hit": False,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": actual_cost,
                "cost_basis": cost_basis,
                "elapsed_seconds": round(time.time() - started, 3),
                "external_transfer": True,
                "store": False,
                "tools_enabled": False,
                "environment": self._environment_record(),
            }
            record["environment_sha256"] = sha256_json(record["environment"])
            raw = response.model_dump() if hasattr(response, "model_dump") else {"output_text": output_text}
            cache_value = {"response_payload": response_payload, "call_record": record, "raw_response": raw}
            with self._cache_lock:
                temporary = cache_path.with_name(
                    f".{cache_path.name}.{uuid.uuid4().hex}.tmp"
                )
                try:
                    temporary.write_text(
                        json.dumps(cache_value, ensure_ascii=False, indent=2, default=str) + "\n",
                        encoding="utf-8",
                        newline="\n",
                    )
                    os.replace(temporary, cache_path)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            return response_payload, record
        except Exception as exc:
            if not settled:
                if response is None:
                    self.budget.release(reserve)
                else:
                    usage = _object_value(response, "usage", {}) or {}
                    input_tokens = int(_object_value(usage, "input_tokens", 0) or 0)
                    output_tokens = int(_object_value(usage, "output_tokens", 0) or 0)
                    actual_cost = (
                        self._actual_text_cost(model, input_tokens, output_tokens)
                        if input_tokens or output_tokens
                        else reserve
                    )
                    settled = True
                    try:
                        self.budget.commit(reserve, actual_cost)
                    except BudgetExceededError as budget_error:
                        raise budget_error from exc
            raise

    def generate_decisions(
        self,
        *,
        title_id: str,
        prompt: str,
        payload: dict[str, Any],
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._structured_call(
            call_kind="translation-generation",
            title_id=title_id,
            model="gpt-5.6-terra",
            reasoning_effort="medium",
            prompt=prompt,
            payload=payload,
            schema=schema,
            schema_name="autonomous_subtitle_decisions",
        )

    def critique_decisions(
        self,
        *,
        title_id: str,
        prompt: str,
        payload: dict[str, Any],
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._structured_call(
            call_kind="semantic-critique",
            title_id=title_id,
            model="gpt-5.6-sol",
            reasoning_effort="high",
            prompt=prompt,
            payload=payload,
            schema=schema,
            schema_name="autonomous_subtitle_critiques",
        )

    def transcribe_audio(self, *, title_id: str, scene_id: str, audio_path: Path) -> tuple[str, dict[str, Any]]:
        audio_path = audio_path.expanduser().resolve()
        fingerprint = {
            "call_kind": "cloud-asr",
            "model": "gpt-4o-transcribe",
            "audio_sha256": sha256_bytes(audio_path.read_bytes()),
            "language": "ja",
        }
        cache_key = sha256_json(fingerprint)
        cache_path = self._cache_path(cache_key)
        with self._cache_lock:
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                original = cached["call_record"]
                return str(cached["text"]), {
                    **original,
                    "cache_hit": True,
                    "cost_usd": 0.0,
                    "original_cost_usd": float(original.get("cost_usd", 0.0) or 0.0),
                    "external_transfer": False,
                    "cache_origin_external_transfer": bool(original.get("external_transfer", False)),
                }
        client = self._client_instance()
        reserve = 0.25
        self.budget.reserve(reserve)
        started = time.time()
        response: Any | None = None
        settled = False
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    with audio_path.open("rb") as handle:
                        response = client.audio.transcriptions.create(
                            model="gpt-4o-transcribe",
                            file=handle,
                            language="ja",
                            response_format="json",
                        )
                    break
                except Exception:
                    if attempt >= self.max_retries:
                        raise
                    time.sleep(0.25 * (2**attempt))
            text = str(_object_value(response, "text", "")).strip()
            usage = _object_value(response, "usage", {}) or {}
            input_tokens = int(_object_value(usage, "input_tokens", 0) or 0)
            output_tokens = int(_object_value(usage, "output_tokens", 0) or 0)
            if input_tokens or output_tokens:
                actual = self._actual_audio_cost("gpt-4o-transcribe", input_tokens, output_tokens)
                cost_basis = "provider-token-usage"
            else:
                actual = reserve
                cost_basis = "conservative-per-clip-reservation"
            if not text:
                raise ValueError("OpenAI transcription response text가 비어 있습니다.")
            settled = True
            self.budget.commit(reserve, actual)
            record = {
                "schema_name": "translation-forensics/model-call-manifest",
                "schema_version": "1",
                "title_id": title_id,
                "scene_id": scene_id,
                "call_kind": "cloud-asr",
                "provider": "openai",
                "model": "gpt-4o-transcribe",
                "request_id": str(_object_value(response, "_request_id", "") or _object_value(response, "id", "") or "unavailable"),
                "request_sha256": sha256_json(fingerprint),
                "response_sha256": sha256_bytes(text.encode("utf-8")),
                "evidence_sha256": fingerprint["audio_sha256"],
                "model_settings_sha256": sha256_json({"model": "gpt-4o-transcribe", "language": "ja"}),
                "cache_key": cache_key,
                "cache_hit": False,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": actual,
                "cost_basis": cost_basis,
                "elapsed_seconds": round(time.time() - started, 3),
                "external_transfer": True,
                "store": False,
                "tools_enabled": False,
                "environment": self._environment_record(),
            }
            record["environment_sha256"] = sha256_json(record["environment"])
            with self._cache_lock:
                cache_path.write_text(
                    json.dumps({"text": text, "call_record": record}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
            return text, record
        except Exception as exc:
            if not settled:
                if response is None:
                    self.budget.release(reserve)
                else:
                    usage = _object_value(response, "usage", {}) or {}
                    input_tokens = int(_object_value(usage, "input_tokens", 0) or 0)
                    output_tokens = int(_object_value(usage, "output_tokens", 0) or 0)
                    actual = (
                        self._actual_audio_cost("gpt-4o-transcribe", input_tokens, output_tokens)
                        if input_tokens or output_tokens
                        else reserve
                    )
                    settled = True
                    try:
                        self.budget.commit(reserve, actual)
                    except BudgetExceededError as budget_error:
                        raise budget_error from exc
            raise
