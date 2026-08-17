from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from translation_forensics.openai_provider import BudgetExceededError, BudgetTracker, OpenAIProvider


SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array"}},
    "required": ["results"],
    "additionalProperties": False,
}


class FakeResponses:
    def __init__(self, *, fail_once: bool = False) -> None:
        self.calls = 0
        self.fail_once = fail_once
        self.last_kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.fail_once and self.calls == 1:
            raise RuntimeError("transient")
        return SimpleNamespace(
            id="resp-test",
            output_text=json.dumps({"results": []}),
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


class FakeClient:
    def __init__(self, *, fail_once: bool = False) -> None:
        self.responses = FakeResponses(fail_once=fail_once)


def test_structured_provider_retries_and_replays_cache(tmp_path: Path) -> None:
    client = FakeClient(fail_once=True)
    budget = BudgetTracker(10.0)
    provider = OpenAIProvider(cache_dir=tmp_path / "cache", budget=budget, allow_network=True, api_key="test", client=client)
    payload, first = provider.generate_decisions(title_id="T", prompt="prompt", payload={"x": 1}, schema=SCHEMA)
    replay, second = provider.generate_decisions(title_id="T", prompt="prompt", payload={"x": 1}, schema=SCHEMA)
    assert payload == replay == {"results": []}
    assert client.responses.calls == 2
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["cost_usd"] == 0.0
    assert second["external_transfer"] is False
    assert second["cache_origin_external_transfer"] is True
    assert client.responses.last_kwargs["max_output_tokens"] == 8_000
    assert client.responses.last_kwargs["store"] is False
    assert client.responses.last_kwargs["tools"] == []
    assert client.responses.last_kwargs["timeout"] == 600.0
    assert len(first["environment_sha256"]) == 64
    assert budget.spent_usd > 0


def test_structured_provider_rejects_schema_invalid_response_and_charges_call(tmp_path: Path) -> None:
    class InvalidResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                id="invalid",
                output_text=json.dumps({"wrong": []}),
                usage=SimpleNamespace(input_tokens=10, output_tokens=2),
            )

    client = SimpleNamespace(responses=InvalidResponses())
    budget = BudgetTracker(10.0)
    provider = OpenAIProvider(
        cache_dir=tmp_path / "cache",
        budget=budget,
        allow_network=True,
        api_key="test",
        client=client,
        max_retries=0,
    )
    with pytest.raises(ValueError, match="schema error"):
        provider.generate_decisions(
            title_id="T", prompt="prompt", payload={"x": 1}, schema=SCHEMA
        )
    assert budget.spent_usd > 0
    assert budget.reserved_usd == 0


def test_structured_provider_treats_malformed_cache_as_miss(tmp_path: Path) -> None:
    client = FakeClient()
    provider = OpenAIProvider(
        cache_dir=tmp_path / "cache",
        budget=BudgetTracker(10.0),
        allow_network=True,
        api_key="test",
        client=client,
    )
    provider.generate_decisions(
        title_id="T", prompt="prompt", payload={"x": 1}, schema=SCHEMA
    )
    cache_file = next((tmp_path / "cache").glob("*.json"))
    cache_file.write_text("{truncated", encoding="utf-8")
    provider.generate_decisions(
        title_id="T", prompt="prompt", payload={"x": 1}, schema=SCHEMA
    )
    assert client.responses.calls == 2
    assert json.loads(cache_file.read_text(encoding="utf-8"))


def test_resume_cost_restoration_obeys_same_budget() -> None:
    budget = BudgetTracker(1.0)
    budget.restore_spent(0.75)
    with pytest.raises(BudgetExceededError):
        budget.restore_spent(0.26)


def test_provider_blocks_call_before_budget_overrun(tmp_path: Path) -> None:
    provider = OpenAIProvider(cache_dir=tmp_path / "cache", budget=BudgetTracker(0.000001), allow_network=True, api_key="test", client=FakeClient())
    with pytest.raises(BudgetExceededError):
        provider.generate_decisions(title_id="T", prompt="prompt", payload={"x": 1}, schema=SCHEMA)


def test_provider_preflight_requires_network_key_and_budget(tmp_path: Path) -> None:
    provider = OpenAIProvider(cache_dir=tmp_path / "cache", budget=BudgetTracker(0), allow_network=False, api_key="", client=FakeClient())
    report = provider.preflight()
    assert report["status"] == "blocked"
    assert report["api_key_present"] is False
    assert len(report["errors"]) == 3
