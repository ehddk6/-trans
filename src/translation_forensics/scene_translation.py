"""The ``scene_v2`` structured translation pipeline.

This module deliberately owns orchestration, not scene detection or subtitle
writing.  Keeping it separate makes the legacy block pipeline an independent
execution path and, importantly, makes the information boundary between the
natural-dialogue and source-faithful passes testable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from jsonschema import Draft202012Validator

from .scene_segmentation import should_use_targeted_visual
from .scene_models import DialogueScene, validate_scene_partition


ARCHITECTURE = "scene_v2"
ARCHITECTURE_STATUS = "experimental-unbenchmarked"
BENCHMARK_STATUS = "not-run"

_CONTRACT_STEMS = (
    "scene-semantic-reconstruction-v1",
    "scene-visual-semantic-observation-v1",
    "scene-dialogue-realization-v1",
    "scene-subtitle-segmentation-v1",
    "scene-source-faithful-v1",
    "scene-semantic-drift-critic-v1",
    "korean-dialogue-critic-v1",
    "scene-dialogue-repair-v1",
)
_CONTRACT_KEYS = {
    "semantic": _CONTRACT_STEMS[0],
    "visual_observation": _CONTRACT_STEMS[1],
    "dialogue": _CONTRACT_STEMS[2],
    "segmentation": _CONTRACT_STEMS[3],
    "source_faithful": _CONTRACT_STEMS[4],
    "semantic_critic": _CONTRACT_STEMS[5],
    "dialogue_critic": _CONTRACT_STEMS[6],
    "repair": _CONTRACT_STEMS[7],
}


class SceneTranslationError(ValueError):
    """Raised for an invalid scene-v2 model or projection contract."""


class StructuredProvider(Protocol):
    def run_structured(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class SceneContract:
    stem: str
    role: str
    model: str
    prompt: str
    schema: Mapping[str, Any]
    prompt_sha256: str
    schema_sha256: str


@dataclass(frozen=True, slots=True)
class SceneTranslationResult:
    decisions: list[dict[str, Any]]
    artifacts: dict[str, list[dict[str, Any]]]
    receipts: list[dict[str, Any]]
    cache_identity: str
    qa: dict[str, Any]
    architecture_status: str = ARCHITECTURE_STATUS
    benchmark_status: str = BENCHMARK_STATUS


def run_scene_translation_v2(
    provider: StructuredProvider,
    *,
    title_id: str,
    units: Sequence[object],
    scenes: Sequence[object],
    prompts_dir: Path | None = None,
    schemas_dir: Path | None = None,
    resume: bool = True,
    repair_attempts: int = 1,
    cache_context: Mapping[str, Any] | None = None,
    visual_policy: str = "off",
    visual_context_by_unit: Mapping[str, Mapping[str, Any]] | None = None,
    visual_observer: Callable[..., object] | None = None,
    contracts: Mapping[str, SceneContract | Mapping[str, Any]] | None = None,
) -> SceneTranslationResult:
    """Run separated scene-v2 passes and return package-ready unit decisions.

    ``contracts`` is a dependency-injection hook for offline tests.  Normal
    callers load prompt manifests and output schemas from the repository; a
    missing or hash-mismatched contract fails before any model invocation.
    """

    if not title_id.strip():
        raise SceneTranslationError("title_id is required")
    if repair_attempts < 0:
        raise SceneTranslationError("repair_attempts must be non-negative")
    if visual_policy not in {"off", "metadata", "targeted"}:
        raise SceneTranslationError("visual_policy must be off, metadata, or targeted")

    normalized_units = [_unit_record(unit) for unit in units]
    _validate_units(normalized_units)
    if scenes and all(isinstance(scene, DialogueScene) for scene in scenes):
        # The domain-model validator also checks source order and timing, which
        # cannot be recovered from a generic serialized scene mapping alone.
        validate_scene_partition(units, scenes)
    normalized_scenes = [_scene_record(scene) for scene in scenes]
    scene_units = _validate_scene_coverage(normalized_units, normalized_scenes)
    loaded_contracts = _load_contracts(
        prompts_dir=prompts_dir, schemas_dir=schemas_dir, supplied=contracts
    )
    cache_identity = _cache_identity(
        title_id=title_id,
        units=normalized_units,
        scenes=normalized_scenes,
        contracts=loaded_contracts,
        repair_attempts=repair_attempts,
        visual_policy=visual_policy,
        cache_context=cache_context,
    )

    receipts: list[dict[str, Any]] = []
    semantic_frames: list[dict[str, Any]] = []
    realizations: list[dict[str, Any]] = []
    cue_projections: list[dict[str, Any]] = []
    source_faithful: list[dict[str, Any]] = []
    semantic_audits: list[dict[str, Any]] = []
    dialogue_audits: list[dict[str, Any]] = []
    repair_history: list[dict[str, Any]] = []
    visual_observations: list[dict[str, Any]] = []
    natural_by_unit: dict[str, str] = {}
    faithful_by_unit: dict[str, str] = {}

    visual_context_by_unit = visual_context_by_unit or {}
    for scene in normalized_scenes:
        scene_id = str(scene["scene_id"])
        scene_source_units = [unit for unit in normalized_units if unit["unit_id"] in scene_units[scene_id]]
        scene_packet = {**scene, "units": scene_source_units}

        semantic_payload = {
            "title_id": title_id,
            "translation_architecture": ARCHITECTURE,
            "scene": scene_packet,
            "visual_policy": visual_policy,
            "visual_context": _scene_visual_metadata(scene_units[scene_id], visual_context_by_unit),
        }
        semantic_response, semantic_receipt = _call(
            provider, loaded_contracts["semantic"], title_id=title_id,
            call_id=f"{scene_id}.semantic-reconstruction", payload=semantic_payload,
            resume=resume,
            image_paths=[],
        )
        receipts.append(semantic_receipt)
        frames = _coverage_rows(
            semantic_response, ("semantic_frames", "frames"), scene_units[scene_id],
            call_id=str(semantic_receipt["call_id"]), label="semantic frames",
        )
        for frame in frames:
            frame["scene_id"] = scene_id
        observations, visual_receipts = _observe_targeted_visuals(
            provider, contract=loaded_contracts["visual_observation"], title_id=title_id,
            visual_policy=visual_policy, scene_id=scene_id, frames=frames,
            visual_context=visual_context_by_unit, observer=visual_observer, resume=resume,
        )
        receipts.extend(visual_receipts)
        for observation in observations:
            unit_id = str(observation["unit_id"])
            next(frame for frame in frames if frame["unit_id"] == unit_id).setdefault(
                "visual_observations", []
            ).append(observation)
        visual_observations.extend(observations)
        semantic_frames.extend(frames)

        # Natural dialogue sees source Japanese and semantic reconstruction, never
        # the independent source-faithful Korean output (which is not generated yet).
        dialogue_payload = {
            "title_id": title_id,
            "translation_architecture": ARCHITECTURE,
            "scene": scene_packet,
            "semantic_frames": frames,
            "style_memory": _style_memory(cache_context, scene_id),
        }
        _assert_no_source_faithful(dialogue_payload, where="dialogue realization")
        dialogue_response, dialogue_receipt = _call(
            provider, loaded_contracts["dialogue"], title_id=title_id,
            call_id=f"{scene_id}.dialogue-realization", payload=dialogue_payload, resume=resume,
        )
        receipts.append(dialogue_receipt)
        scene_realizations = _turn_rows(dialogue_response, scene_units[scene_id], call_id=str(dialogue_receipt["call_id"]))
        for realization in scene_realizations:
            realization["scene_id"] = scene_id
        realizations.extend(scene_realizations)

        segmentation_payload = {"title_id": title_id, "translation_architecture": ARCHITECTURE, "scene": scene_packet, "semantic_frames": frames, "realizations": scene_realizations}
        _assert_no_source_faithful(segmentation_payload, where="subtitle segmentation")
        segmentation_response, segmentation_receipt = _call(provider, loaded_contracts["segmentation"], title_id=title_id, call_id=f"{scene_id}.subtitle-segmentation", payload=segmentation_payload, resume=resume)
        receipts.append(segmentation_receipt)
        scene_projections = _coverage_rows(segmentation_response, ("projections",), scene_units[scene_id], call_id=str(segmentation_receipt["call_id"]), label="subtitle projections", text_keys=("viewer_natural_korean", "text"))
        for projection in scene_projections:
            projection["scene_id"] = scene_id
            natural_by_unit[projection["unit_id"]] = _text(projection, ("viewer_natural_korean", "text"))
        cue_projections.extend(scene_projections)

        # This source-faithful pass is intentionally independent of Korean dialogue.
        faithful_payload = {
            "title_id": title_id,
            "translation_architecture": ARCHITECTURE,
            "scene": scene_packet,
            "semantic_frames": frames,
        }
        source_response, source_receipt = _call(
            provider, loaded_contracts["source_faithful"], title_id=title_id,
            call_id=f"{scene_id}.source-faithful", payload=faithful_payload, resume=resume,
        )
        receipts.append(source_receipt)
        faithful_rows = _coverage_rows(
            source_response, ("translations", "source_faithful", "source_faithful_translations"),
            scene_units[scene_id], call_id=str(source_receipt["call_id"]),
            label="source-faithful translations",
            text_keys=("source_faithful_korean", "text"),
        )
        for row in faithful_rows:
            faithful_by_unit[row["unit_id"]] = _text(row, ("source_faithful_korean", "text"))
            row["scene_id"] = scene_id
        source_faithful.extend(faithful_rows)

    projected = _project(normalized_units, natural_by_unit, faithful_by_unit, normalized_scenes, cue_projections)
    audit_scope = str(dict(cache_context or {}).get("semantic_audit_scope", "all"))
    semantic_audits, dialogue_audits = _run_critics(
        provider, contracts=loaded_contracts, title_id=title_id, scenes=normalized_scenes,
        scene_units=scene_units, units=normalized_units, frames=semantic_frames,
        projected=projected, resume=resume, receipts=receipts, audit_scope=audit_scope,
    )

    for attempt in range(1, repair_attempts + 1):
        affected = _affected_by_scene(semantic_audits, dialogue_audits)
        if not affected:
            break
        changed = False
        for scene in normalized_scenes:
            scene_id = str(scene["scene_id"])
            affected_ids = affected.get(scene_id, [])
            if not affected_ids:
                continue
            before = {unit_id: natural_by_unit[unit_id] for unit_id in affected_ids}
            scene_source_units = [unit for unit in normalized_units if unit["unit_id"] in scene_units[scene_id]]
            repair_payload = {
                "title_id": title_id,
                "translation_architecture": ARCHITECTURE,
                "scene": {**scene, "units": scene_source_units},
                "semantic_frames": _scene_rows(semantic_frames, scene_id),
                "current_viewer_natural": [
                    {"unit_id": unit_id, "viewer_natural_korean": natural_by_unit[unit_id]}
                    for unit_id in scene_units[scene_id]
                ],
                "semantic_audit": _scene_rows(semantic_audits, scene_id),
                "dialogue_audit": _scene_rows(dialogue_audits, scene_id),
                "affected_unit_ids": affected_ids,
            }
            _assert_no_source_faithful(repair_payload, where="dialogue repair")
            response, receipt = _call(
                provider, loaded_contracts["repair"], title_id=title_id,
                call_id=f"{scene_id}.dialogue-repair.{attempt}", payload=repair_payload,
                resume=resume,
            )
            receipts.append(receipt)
            repaired = _repair_rows(response, affected_ids, call_id=str(receipt["call_id"]))
            for row in repaired:
                natural_by_unit[row["unit_id"]] = _text(row, ("viewer_natural_korean", "text"))
            changed = True
            repair_history.append({
                "scene_id": scene_id, "attempt": attempt, "affected_unit_ids": affected_ids,
                "before": before, "after": {unit_id: natural_by_unit[unit_id] for unit_id in affected_ids},
                "repair_call_id": receipt["call_id"], "status": "applied",
            })

        if not changed:
            break
        cue_projections = _resegment_repaired_scenes(provider, contracts=loaded_contracts, title_id=title_id, scenes=normalized_scenes, scene_units=scene_units, units=normalized_units, frames=semantic_frames, natural=natural_by_unit, existing=cue_projections, resume=resume, receipts=receipts, attempt=attempt, affected=affected)
        projected = _project(normalized_units, natural_by_unit, faithful_by_unit, normalized_scenes, cue_projections)
        after_semantic, after_dialogue = _run_critics(
            provider, contracts=loaded_contracts, title_id=title_id, scenes=normalized_scenes,
            scene_units=scene_units, units=normalized_units, frames=semantic_frames,
            projected=projected, resume=resume, receipts=receipts, audit_scope=audit_scope,
            suffix=f"repair-{attempt}",
        )
        regressions = _semantic_regressions(semantic_audits, after_semantic)
        if regressions:
            for history in repair_history:
                if history["attempt"] == attempt and history["scene_id"] in regressions:
                    for unit_id, text in history["before"].items():
                        natural_by_unit[unit_id] = text
                    history["status"] = "rolled_back_semantic_regression"
                    history["semantic_regression_unit_ids"] = regressions[history["scene_id"]]
            cue_projections = _resegment_repaired_scenes(provider, contracts=loaded_contracts, title_id=title_id, scenes=normalized_scenes, scene_units=scene_units, units=normalized_units, frames=semantic_frames, natural=natural_by_unit, existing=cue_projections, resume=resume, receipts=receipts, attempt=attempt, affected=regressions)
            projected = _project(normalized_units, natural_by_unit, faithful_by_unit, normalized_scenes, cue_projections)
            semantic_audits, dialogue_audits = _run_critics(
                provider, contracts=loaded_contracts, title_id=title_id, scenes=normalized_scenes,
                scene_units=scene_units, units=normalized_units, frames=semantic_frames,
                projected=projected, resume=resume, receipts=receipts, audit_scope=audit_scope,
                suffix=f"repair-{attempt}-rollback",
            )
            break
        semantic_audits, dialogue_audits = after_semantic, after_dialogue

    decisions = _decisions(normalized_units, projected, semantic_frames, semantic_audits, dialogue_audits, repair_history)
    artifacts = {
        "translation-input/dialogue_scenes.jsonl": normalized_scenes,
        "semantic_frames.jsonl": semantic_frames,
        "scene_realizations.jsonl": realizations,
        "scene_cue_projection.jsonl": projected,
        "scene_source_faithful.jsonl": source_faithful,
        "semantic_drift_audit.jsonl": semantic_audits,
        "korean_dialogue_audit.jsonl": dialogue_audits,
        "repair_history.jsonl": repair_history,
    }
    if visual_observations:
        artifacts["visual_semantic_observations.jsonl"] = visual_observations
    qa = {
        "translation_architecture": ARCHITECTURE, "architecture_status": ARCHITECTURE_STATUS,
        "benchmark_status": BENCHMARK_STATUS, "unit_coverage": len(decisions),
        "semantic_issues": sum(_is_issue(row) for row in semantic_audits),
        "dialogue_issues": sum(_is_issue(row) for row in dialogue_audits),
        "repair_count": len(repair_history),
    }
    return SceneTranslationResult(decisions, artifacts, receipts, cache_identity, qa)


def _load_contracts(*, prompts_dir: Path | None, schemas_dir: Path | None, supplied: Mapping[str, SceneContract | Mapping[str, Any]] | None) -> dict[str, SceneContract]:
    if supplied is not None:
        missing = sorted(set(_CONTRACT_KEYS) - set(supplied))
        if missing:
            raise SceneTranslationError(f"missing injected scene contracts: {missing}")
        return {key: _coerce_contract(key, supplied[key]) for key in _CONTRACT_KEYS}
    root = Path(__file__).resolve().parents[2]
    prompts = Path(prompts_dir) if prompts_dir is not None else root / "prompts"
    schemas = Path(schemas_dir) if schemas_dir is not None else root / "schemas"
    result: dict[str, SceneContract] = {}
    for key, stem in _CONTRACT_KEYS.items():
        manifest_path = prompts / f"{stem}.manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing scene prompt manifest: {manifest_path}")
        manifest = _read_object(manifest_path)
        prompt_name = str(manifest.get("prompt_file") or f"{stem}.md")
        schema_name = str(manifest.get("output_schema_file") or f"{stem}.schema.json")
        prompt_path = prompts / prompt_name
        schema_path = schemas / schema_name
        if not prompt_path.is_file():
            raise FileNotFoundError(f"missing scene prompt: {prompt_path}")
        if not schema_path.is_file():
            raise FileNotFoundError(f"missing scene output schema: {schema_path}")
        prompt = prompt_path.read_text(encoding="utf-8")
        schema_text = schema_path.read_text(encoding="utf-8")
        schema = _read_object(schema_path)
        prompt_hash = _sha_text(prompt)
        schema_hash = _sha_text(schema_text)
        if manifest.get("prompt_sha256") not in {None, prompt_hash}:
            raise SceneTranslationError(f"scene prompt hash mismatch: {manifest_path}")
        if manifest.get("output_schema_sha256") not in {None, schema_hash}:
            raise SceneTranslationError(f"scene schema hash mismatch: {manifest_path}")
        result[key] = SceneContract(stem, str(manifest.get("role") or stem), str(manifest.get("model") or "unspecified"), prompt, schema, prompt_hash, schema_hash)
    return result


def _coerce_contract(key: str, value: SceneContract | Mapping[str, Any]) -> SceneContract:
    if isinstance(value, SceneContract):
        return value
    if not isinstance(value, Mapping):
        raise SceneTranslationError(f"invalid injected scene contract: {key}")
    prompt = str(value.get("prompt") or "")
    schema = value.get("schema")
    if not prompt or not isinstance(schema, Mapping):
        raise SceneTranslationError(f"injected scene contract requires prompt and schema: {key}")
    return SceneContract(str(value.get("stem") or _CONTRACT_KEYS[key]), str(value.get("role") or _CONTRACT_KEYS[key]), str(value.get("model") or "fake"), prompt, dict(schema), str(value.get("prompt_sha256") or _sha(prompt.encode("utf-8"))), str(value.get("schema_sha256") or _sha_json(schema)))


def _call(provider: StructuredProvider, contract: SceneContract, *, title_id: str, call_id: str, payload: dict[str, Any], resume: bool, image_paths: list[Path] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call a contract and turn a provider receipt into a complete local receipt."""
    kwargs: dict[str, Any] = {"role": contract.role, "title_id": title_id, "call_id": call_id, "prompt": contract.prompt, "payload": payload, "schema": dict(contract.schema), "resume": resume}
    if image_paths is not None:
        kwargs["image_paths"] = image_paths
        kwargs["allow_image_transfer"] = bool(image_paths)
    response, provider_receipt = provider.run_structured(**kwargs)
    _validate_response(response, contract.schema, call_id=call_id)
    receipt = dict(provider_receipt or {})
    receipt.update({"call_id": call_id, "role": contract.role, "model": receipt.get("model", contract.model), "prompt_sha256": contract.prompt_sha256, "schema_sha256": contract.schema_sha256, "source_sha256": _sha_json(payload), "cache_hit": bool(receipt.get("cache_hit", False)), "external_transfer": bool(receipt.get("pixel_external_transfer") or receipt.get("pixel_external_transfer_count") or image_paths)})
    return dict(response), receipt


def _validate_response(value: object, schema: Mapping[str, Any], *, call_id: str) -> None:
    if not isinstance(value, Mapping):
        raise SceneTranslationError(f"structured response must be an object: {call_id}")
    error = next(iter(Draft202012Validator(dict(schema)).iter_errors(value)), None)
    if error is not None:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise SceneTranslationError(f"structured response schema error for {call_id} at {location}: {error.message}")


def _coverage_rows(response: Mapping[str, Any], keys: tuple[str, ...], expected: Sequence[object], *, call_id: str, label: str, text_keys: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    rows = next((response.get(key) for key in keys if isinstance(response.get(key), list)), None)
    if rows is None or not all(isinstance(row, Mapping) for row in rows):
        raise SceneTranslationError(f"{label} response lacks a record array: {call_id}")
    expected_ids = [str(item["unit_id"]) if isinstance(item, Mapping) else str(item) for item in expected]
    ids = [str(row.get("unit_id") or "") for row in rows]
    missing = sorted(set(expected_ids) - set(ids)); extra = sorted(set(ids) - set(expected_ids)); duplicate = sorted({value for value in ids if ids.count(value) > 1})
    if not all(ids) or missing or extra or duplicate:
        raise SceneTranslationError(f"{label} coverage mismatch for {call_id}: missing={missing}, extra={extra}, duplicate={duplicate}")
    copied = [dict(row) for row in rows]
    for row in copied:
        if text_keys and not _text(row, text_keys).strip():
            raise SceneTranslationError(f"empty {label} text for {row['unit_id']}: {call_id}")
    return copied


def _run_critics(provider: StructuredProvider, *, contracts: Mapping[str, SceneContract], title_id: str, scenes: Sequence[Mapping[str, Any]], scene_units: Mapping[str, list[str]], units: Sequence[Mapping[str, Any]], frames: Sequence[Mapping[str, Any]], projected: Sequence[Mapping[str, Any]], resume: bool, receipts: list[dict[str, Any]], suffix: str = "initial", audit_scope: str = "all") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    semantic_rows: list[dict[str, Any]] = []; dialogue_rows: list[dict[str, Any]] = []
    for scene in scenes:
        scene_id = str(scene["scene_id"]); ids = scene_units[scene_id]
        source_units = [unit for unit in units if unit["unit_id"] in ids]
        semantic_payload = {"title_id": title_id, "scene": {**scene, "units": source_units}, "semantic_audit_scope": audit_scope, "semantic_frames": _scene_rows(frames, scene_id), "cue_projection": _scene_rows(projected, scene_id)}
        response, receipt = _call(provider, contracts["semantic_critic"], title_id=title_id, call_id=f"{scene_id}.semantic-critic.{suffix}", payload=semantic_payload, resume=resume)
        receipts.append(receipt)
        current = _critic_rows(response, ids, call_id=str(receipt["call_id"]), label="semantic audits")
        for row in current: row["scene_id"] = scene_id; row["critic"] = "semantic"
        semantic_rows.extend(current)
        dialogue_payload = {"title_id": title_id, "scene": {**scene, "units": source_units}, "semantic_audit_scope": audit_scope, "semantic_frames": _scene_rows(frames, scene_id), "cue_projection": _natural_projection(_scene_rows(projected, scene_id))}
        _assert_no_source_faithful(dialogue_payload, where="dialogue critic")
        response, receipt = _call(provider, contracts["dialogue_critic"], title_id=title_id, call_id=f"{scene_id}.dialogue-critic.{suffix}", payload=dialogue_payload, resume=resume)
        receipts.append(receipt)
        current = _critic_rows(response, ids, call_id=str(receipt["call_id"]), label="dialogue audits")
        for row in current: row["scene_id"] = scene_id; row["critic"] = "dialogue"
        dialogue_rows.extend(current)
    return semantic_rows, dialogue_rows


def _project(units: Sequence[Mapping[str, Any]], natural: Mapping[str, str], faithful: Mapping[str, str], scenes: Sequence[Mapping[str, Any]], projection_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scene_by_unit = {unit_id: str(scene["scene_id"]) for scene in scenes for unit_id in scene["unit_ids"]}
    rows = []
    for unit in units:
        unit_id = str(unit["unit_id"])
        natural_text = natural.get(unit_id, "").strip(); faithful_text = faithful.get(unit_id, "").strip()
        if not natural_text or not faithful_text:
            raise SceneTranslationError(f"projection lacks translated text for {unit_id}")
        if _contains_japanese(natural_text) or _contains_japanese(faithful_text):
            raise SceneTranslationError(f"projection contains Japanese residual for {unit_id}")
        base = next((dict(row) for row in projection_rows if str(row.get("unit_id")) == unit_id), {})
        rows.append({**base, "scene_id": scene_by_unit[unit_id], "unit_id": unit_id, "viewer_natural_korean": natural_text, "source_faithful_korean": faithful_text})
    return rows


def _turn_rows(response: Mapping[str, Any], expected_ids: Sequence[str], *, call_id: str) -> list[dict[str, Any]]:
    rows = response.get("realizations")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise SceneTranslationError(f"dialogue realizations response lacks a record array: {call_id}")
    referenced = [str(unit_id) for row in rows for unit_id in row.get("source_unit_ids", [])]
    missing = sorted(set(expected_ids) - set(referenced)); extra = sorted(set(referenced) - set(expected_ids))
    if missing or extra or len(referenced) != len(set(referenced)):
        raise SceneTranslationError(f"dialogue realization coverage mismatch for {call_id}: missing={missing}, extra={extra}")
    for row in rows:
        if not str(row.get("korean") or row.get("viewer_natural_korean") or "").strip():
            raise SceneTranslationError(f"empty dialogue realization text: {call_id}")
    return [dict(row) for row in rows]


def _repair_rows(response: Mapping[str, Any], expected_ids: Sequence[str], *, call_id: str) -> list[dict[str, Any]]:
    """Expand the repair contract's turn/span output to affected source units."""
    if isinstance(response.get("repair"), list):
        expanded: list[dict[str, Any]] = []
        for row in response["repair"]:
            if not isinstance(row, Mapping) or not isinstance(row.get("affected_unit_ids"), list):
                raise SceneTranslationError(f"targeted repair has invalid affected_unit_ids: {call_id}")
            text = str(row.get("korean") or "").strip()
            if not text:
                raise SceneTranslationError(f"targeted repair has empty Korean text: {call_id}")
            expanded.extend({"unit_id": str(unit_id), "viewer_natural_korean": text, "repair_turn_id": row.get("turn_id")} for unit_id in row["affected_unit_ids"])
        return _coverage_rows({"repairs": expanded}, ("repairs",), expected_ids, call_id=call_id, label="targeted repairs", text_keys=("viewer_natural_korean",))
    return _coverage_rows(response, ("repairs", "translations", "realizations"), expected_ids, call_id=call_id, label="targeted repairs", text_keys=("viewer_natural_korean", "text"))


def _critic_rows(response: Mapping[str, Any], expected_ids: Sequence[str], *, call_id: str, label: str) -> list[dict[str, Any]]:
    """Expand span-based critic output to deterministic, unit-level statuses."""
    raw = response.get("audits")
    if not isinstance(raw, list) or not all(isinstance(row, Mapping) for row in raw):
        raise SceneTranslationError(f"{label} response lacks a record array: {call_id}")
    result = [{"unit_id": unit_id, "verdict": "pass"} for unit_id in expected_ids]
    by_id = {row["unit_id"]: row for row in result}
    for audit in raw:
        affected = audit.get("unit_ids", [audit.get("unit_id")])
        if not isinstance(affected, list) or not affected:
            raise SceneTranslationError(f"{label} lacks affected unit ids: {call_id}")
        for unit_id in affected:
            unit_id = str(unit_id)
            if unit_id not in by_id:
                raise SceneTranslationError(f"{label} contains unexpected unit_id {unit_id}: {call_id}")
            if _is_issue(audit):
                by_id[unit_id] = {**dict(audit), "unit_id": unit_id}
    return [by_id[unit_id] for unit_id in expected_ids]


def _resegment_repaired_scenes(provider: StructuredProvider, *, contracts: Mapping[str, SceneContract], title_id: str, scenes: Sequence[Mapping[str, Any]], scene_units: Mapping[str, list[str]], units: Sequence[Mapping[str, Any]], frames: Sequence[Mapping[str, Any]], natural: Mapping[str, str], existing: Sequence[Mapping[str, Any]], resume: bool, receipts: list[dict[str, Any]], attempt: int, affected: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    result = [dict(row) for row in existing]
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        if scene_id not in affected:
            continue
        ids = scene_units[scene_id]
        source_units = [unit for unit in units if unit["unit_id"] in ids]
        payload = {"title_id": title_id, "scene": {**scene, "units": source_units}, "semantic_frames": _scene_rows(frames, scene_id), "realizations": [{"turn_id": f"repair-{unit_id}", "source_unit_ids": [unit_id], "korean": natural[unit_id]} for unit_id in ids]}
        _assert_no_source_faithful(payload, where="repair subtitle segmentation")
        response, receipt = _call(provider, contracts["segmentation"], title_id=title_id, call_id=f"{scene_id}.subtitle-segmentation.repair-{attempt}", payload=payload, resume=resume)
        receipts.append(receipt)
        replacement = _coverage_rows(response, ("projections",), ids, call_id=str(receipt["call_id"]), label="repair subtitle projections", text_keys=("viewer_natural_korean", "text"))
        for row in replacement:
            row["scene_id"] = scene_id
            natural[str(row["unit_id"])] = _text(row, ("viewer_natural_korean", "text"))
        result = [row for row in result if str(row.get("scene_id")) != scene_id] + replacement
    return result


def _decisions(units: Sequence[Mapping[str, Any]], projected: Sequence[Mapping[str, Any]], frames: Sequence[Mapping[str, Any]], semantic: Sequence[Mapping[str, Any]], dialogue: Sequence[Mapping[str, Any]], history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projection = {str(row["unit_id"]): row for row in projected}; frame = {str(row["unit_id"]): row for row in frames}; sem = {str(row["unit_id"]): row for row in semantic}; dia = {str(row["unit_id"]): row for row in dialogue}
    repairs = {str(unit_id): 0 for unit_id in projection}
    for row in history:
        if row.get("status") == "applied":
            for unit_id in row.get("affected_unit_ids", []): repairs[str(unit_id)] += 1
    result = []
    for unit in units:
        unit_id = str(unit["unit_id"]); cue = projection[unit_id]
        result.append({"unit_id": unit_id, "viewer_complete_ko": cue["viewer_natural_korean"], "viewer_natural_korean": cue["viewer_natural_korean"], "source_faithful_korean": cue["source_faithful_korean"], "evidence_ids": list(unit.get("evidence_ids", [])), "translation_architecture": ARCHITECTURE, "scene_id": cue["scene_id"], "semantic_frame_ref": unit_id, "dialogue_turn_refs": [unit_id], "cue_projection_ref": unit_id, "semantic_drift_status": _status(sem[unit_id]), "korean_dialogue_status": _status(dia[unit_id]), "repair_attempts": repairs[unit_id], "alignment_refs": [unit_id], "semantic_slots": frame[unit_id].get("semantic_slots", frame[unit_id].get("slots", {}))})
    return result


def _affected_by_scene(*audits: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for rows in audits:
        for row in rows:
            if _is_issue(row): result.setdefault(str(row["scene_id"]), []).append(str(row["unit_id"]))
    return {scene_id: list(dict.fromkeys(ids)) for scene_id, ids in result.items()}


def _semantic_regressions(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    previous = {str(row["unit_id"]): _is_issue(row) for row in before}; result: dict[str, list[str]] = {}
    for row in after:
        if _is_issue(row) and not previous.get(str(row["unit_id"]), False): result.setdefault(str(row["scene_id"]), []).append(str(row["unit_id"]))
    return result


def _unit_record(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping): return dict(value)
    json_method = getattr(value, "json", None)
    if callable(json_method): return dict(json_method())
    raise SceneTranslationError("scene_v2 unit must be a mapping or TranslationUnit-like object")


def _scene_record(value: object) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, Mapping) else dict(value.json()) if callable(getattr(value, "json", None)) else None
    if raw is None: raise SceneTranslationError("scene_v2 scene must be a mapping or Scene-like object")
    unit_values = raw.get("unit_ids", raw.get("units", raw.get("translation_unit_ids", [])))
    raw["scene_id"] = str(raw.get("scene_id") or "")
    raw["unit_ids"] = [str(item.get("unit_id")) if isinstance(item, Mapping) else str(item) for item in unit_values]
    return raw


def _validate_units(units: Sequence[Mapping[str, Any]]) -> None:
    ids = [str(unit.get("unit_id") or "") for unit in units]
    if not ids or not all(ids) or len(set(ids)) != len(ids): raise SceneTranslationError("scene_v2 requires unique, non-empty unit_id values")


def _validate_scene_coverage(units: Sequence[Mapping[str, Any]], scenes: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    expected = [str(unit["unit_id"]) for unit in units]; seen: list[str] = []; result: dict[str, list[str]] = {}
    for scene in scenes:
        scene_id = str(scene.get("scene_id") or ""); ids = list(scene.get("unit_ids") or [])
        if not scene_id or not ids or scene_id in result: raise SceneTranslationError("every scene requires a unique scene_id and at least one unit")
        result[scene_id] = ids; seen.extend(ids)
    if set(seen) != set(expected) or len(seen) != len(set(seen)):
        raise SceneTranslationError("scene coverage must contain every input unit exactly once")
    return result


def _cache_identity(*, title_id: str, units: Sequence[Mapping[str, Any]], scenes: Sequence[Mapping[str, Any]], contracts: Mapping[str, SceneContract], repair_attempts: int, visual_policy: str, cache_context: Mapping[str, Any] | None) -> str:
    return _sha_json({"translation_architecture": ARCHITECTURE, "title_id": title_id, "units": units, "scenes": scenes, "prompt_schema_hashes": {key: {"prompt": contract.prompt_sha256, "schema": contract.schema_sha256} for key, contract in contracts.items()}, "repair_attempts": repair_attempts, "visual_policy": visual_policy, "cache_context": dict(cache_context or {})})


def _scene_visual_metadata(ids: Sequence[str], context: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in dict(context.get(unit_id, {})).items() if key not in {"image_paths", "paths", "frames"}} | {"unit_id": unit_id} for unit_id in ids if unit_id in context]


def _targeted_images(policy: str, ids: Sequence[str], context: Mapping[str, Mapping[str, Any]]) -> list[Path]:
    if policy != "targeted": return []
    result: list[Path] = []
    for unit_id in ids:
        record = context.get(unit_id, {})
        if not record.get("visual_eligible", False): continue
        paths = record.get("image_paths", record.get("paths", []))
        for path in paths:
            resolved = Path(path)
            if resolved not in result: result.append(resolved)
    return result[:3]


def _observe_targeted_visuals(provider: StructuredProvider, *, contract: SceneContract, title_id: str, visual_policy: str, scene_id: str, frames: Sequence[Mapping[str, Any]], visual_context: Mapping[str, Mapping[str, Any]], observer: Callable[..., object] | None, resume: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Produce source-bound visual evidence only for explicitly eligible units.

    It intentionally has no Korean text field.  The dialogue pass can consume the
    resulting semantic evidence, but cannot treat an observation as spoken text.
    """
    if visual_policy != "targeted":
        return [], []
    result: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for frame in frames:
        unit_id = str(frame["unit_id"])
        context = dict(visual_context.get(unit_id, {}))
        try:
            eligible = should_use_targeted_visual(frame)
        except (TypeError, ValueError):
            eligible = False
        if not eligible:
            continue
        image_paths = _verified_image_paths(context)
        if not image_paths:
            continue
        if observer is not None:
            value = observer(scene_id=scene_id, semantic_frame=dict(frame), visual_context=context, image_paths=image_paths)
            observation = value[0] if isinstance(value, tuple) else value
            if not isinstance(observation, Mapping):
                raise SceneTranslationError(f"visual observer returned a non-object for {unit_id}")
            record = dict(observation)
            record.setdefault("observation_provenance", "test-observer")
        else:
            payload = {"title_id": title_id, "translation_architecture": ARCHITECTURE, "scene_id": scene_id, "unit_id": unit_id, "scene": {"scene_id": scene_id, "units": [{"unit_id": unit_id}]}, "semantic_frame": dict(frame), "visual_context": _scene_visual_metadata([unit_id], {unit_id: context})}
            response, receipt = _call(provider, contract, title_id=title_id, call_id=f"{scene_id}.{unit_id}.visual-semantic-observation", payload=payload, resume=resume, image_paths=image_paths)
            receipts.append(receipt)
            rows = response.get("observations")
            if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
                raise SceneTranslationError(f"visual observation coverage mismatch for {receipt['call_id']}")
            record = dict(rows[0])
            record["observation_provenance"] = receipt["call_id"]
        if str(record.get("unit_id") or unit_id) != unit_id:
            raise SceneTranslationError(f"visual observation unit mismatch for {unit_id}")
        record["unit_id"] = unit_id
        record["scene_id"] = scene_id
        forbidden = {"viewer_natural_korean", "source_faithful_korean", "korean_text"} & set(record)
        if forbidden:
            raise SceneTranslationError(f"visual observation may not contain Korean translation fields: {sorted(forbidden)}")
        allowed = {"speaker", "addressee", "deictic_location", "deictic_referent", "on_screen_text", "scene_continuity"}
        observed_slots = record.get("observed_slots", {})
        if isinstance(observed_slots, Mapping):
            slot_names = set(observed_slots)
        elif isinstance(observed_slots, list) and all(isinstance(item, Mapping) for item in observed_slots):
            slot_names = {str(item.get("slot") or "") for item in observed_slots}
            if any(not str(item.get("observation") or "").strip() for item in observed_slots):
                raise SceneTranslationError(f"visual observation has empty slot evidence: {unit_id}")
        else:
            raise SceneTranslationError(f"visual observation requires observed_slots: {unit_id}")
        if not slot_names <= allowed:
            raise SceneTranslationError(f"visual observation contains a non-visual slot: {unit_id}")
        result.append(record)
    return result, receipts


def _verified_image_paths(context: Mapping[str, Any]) -> list[Path]:
    paths = [Path(path) for path in context.get("image_paths", context.get("paths", []))]
    verified = [path for path in paths if path.is_file()]
    expected_hashes = context.get("image_sha256", {})
    if isinstance(expected_hashes, Mapping):
        verified = [path for path in verified if not expected_hashes.get(str(path)) or _sha(path.read_bytes()) == str(expected_hashes[str(path)])]
    return verified[:3]


def _scene_rows(rows: Sequence[Mapping[str, Any]], scene_id: str) -> list[dict[str, Any]]: return [dict(row) for row in rows if str(row.get("scene_id")) == scene_id]
def _natural_projection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]: return [{"unit_id": row["unit_id"], "scene_id": row["scene_id"], "viewer_natural_korean": row["viewer_natural_korean"]} for row in rows]
def _style_memory(context: Mapping[str, Any] | None, scene_id: str) -> Any: return dict(context or {}).get("style_memory_by_scene", {}).get(scene_id, [])
def _text(row: Mapping[str, Any], keys: Sequence[str]) -> str: return next((str(row.get(key) or "") for key in keys if str(row.get(key) or "").strip()), "")
def _is_issue(row: Mapping[str, Any]) -> bool:
    verdict = str(row.get("verdict", row.get("status", "pass"))).lower()
    if verdict in {"issue", "fail", "repair", "major", "critical"}:
        return True
    # korean-dialogue-critic-v1 is intentionally a style-only issue list: it
    # reports category/severity without a pass/issue verdict.  Synthetic
    # per-unit defaults above contain neither field and therefore remain pass.
    return "category" in row and str(row.get("severity") or "").lower() in {"minor", "major"}
def _status(row: Mapping[str, Any]) -> str: return "issue" if _is_issue(row) else str(row.get("verdict", row.get("status", "pass")))
def _contains_japanese(text: str) -> bool: return any("\u3040" <= character <= "\u30ff" or "\u3400" <= character <= "\u9fff" for character in text)
def _assert_no_source_faithful(payload: Mapping[str, Any], *, where: str) -> None:
    if "source_faithful" in json.dumps(payload, ensure_ascii=False).lower(): raise SceneTranslationError(f"source-faithful Korean leaked into {where} payload")
def _read_object(path: Path) -> dict[str, Any]:
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error: raise SceneTranslationError(f"invalid JSON contract: {path}") from error
    if not isinstance(value, dict): raise SceneTranslationError(f"JSON contract must be an object: {path}")
    return value
def _sha(value: bytes) -> str: return hashlib.sha256(value).hexdigest()
def _sha_text(value: str) -> str: return _sha(value.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))
def _sha_json(value: Any) -> str: return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


__all__ = ["ARCHITECTURE", "ARCHITECTURE_STATUS", "BENCHMARK_STATUS", "SceneContract", "SceneTranslationError", "SceneTranslationResult", "run_scene_translation_v2"]
