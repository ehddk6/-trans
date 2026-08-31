from __future__ import annotations

"""Local-only evaluation contracts for the experimental ``scene_v2`` path.

This module deliberately does not call an external translation service.  An
external baseline is an audited user-supplied input, not something this project
may synthesize.  Review records are treated as submitted data: validating them
does not assert that a human review occurred.
"""

import hashlib
import json
import math
import random
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator


SCHEMA_VERSION = "1"
SYSTEM_LABELS = ("block_v1", "external_baseline", "scene_v2")
CANDIDATE_CODES = ("candidate-a", "candidate-b", "candidate-c")
SEVERITIES = ("critical", "major", "minor")
NATURALNESS_CHOICES = frozenset((*CANDIDATE_CODES, "tie", "unresolved"))
ABLATION_CONDITIONS = ("A", "B", "C", "D", "E")
VISUAL_METRICS = (
    "semantic_accuracy", "speaker_addressee", "deictic_referent", "on_screen_text",
    "scene_continuity", "unsupported_visual_addition", "hallucination",
    "korean_naturalness", "cost", "latency", "pixel_transfer_count",
)


class SceneEvaluationError(ValueError):
    """Raised when a scene benchmark artifact violates its local contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Existing evaluation artifact will not be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl_to_zip(archive: zipfile.ZipFile, name: str, rows: Iterable[Mapping[str, Any]]) -> None:
    archive.writestr(name, "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise SceneEvaluationError(f"JSON object required: {path}")
    return value


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(parsed, dict):
        parsed = parsed.get("records", [parsed])
    if not isinstance(parsed, list) or not all(isinstance(row, dict) for row in parsed):
        raise SceneEvaluationError(f"JSON or JSONL objects required: {path}")
    return list(parsed)


def _schema_path(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "schemas" / name


def _validate_schema(value: Mapping[str, Any], schema_name: str) -> None:
    schema = _read_json(_schema_path(schema_name))
    error = next(iter(Draft202012Validator(schema).iter_errors(dict(value))), None)
    if error:
        path = "/".join(str(part) for part in error.absolute_path)
        raise SceneEvaluationError(f"{schema_name} validation failed at {path or '<root>'}: {error.message}")


def _normalized_output_rows(value: Mapping[str, Any], *, fallback_scene_id: str | None = None) -> list[dict[str, str]]:
    scene_id = str(value.get("scene_id") or fallback_scene_id or "").strip()
    outputs = value.get("outputs")
    if not scene_id:
        raise SceneEvaluationError("Candidate output is missing scene_id")
    if isinstance(outputs, str):
        outputs = [{"korean": outputs}]
    if not isinstance(outputs, list) or not outputs:
        raise SceneEvaluationError(f"{scene_id}: outputs must contain one or more Korean strings")
    normalized: list[dict[str, str]] = []
    for index, output in enumerate(outputs, 1):
        if isinstance(output, str):
            output = {"korean": output}
        if not isinstance(output, Mapping):
            raise SceneEvaluationError(f"{scene_id}: output {index} must be an object")
        korean = str(output.get("korean", output.get("text", ""))).strip()
        if not korean:
            raise SceneEvaluationError(f"{scene_id}: output {index} is missing korean text")
        unit_id = str(output.get("unit_id", index))
        normalized.append({"unit_id": unit_id, "korean": korean})
    return normalized


def _candidate_scenes(path: Path, *, expected_label: str) -> dict[str, str]:
    rows = _read_json_or_jsonl(path)
    scene_outputs: dict[str, list[dict[str, str]]] = {}
    seen_units: set[tuple[str, str]] = set()
    for row in rows:
        # Native artifacts are normally one projected cue per row; external
        # baseline records carry a complete scene in ``outputs``.
        if "outputs" not in row:
            native_text = row.get("korean", row.get("viewer_natural_korean", row.get("text")))
            if native_text is not None:
                row = dict(row)
                row["outputs"] = [{"unit_id": row.get("unit_id", "1"), "korean": native_text}]
        if "outputs" not in row:
            raise SceneEvaluationError(f"{expected_label} candidate record has no Korean output")
        outputs = _normalized_output_rows(row)
        scene_id = str(row["scene_id"])
        for output in outputs:
            identity = (scene_id, output["unit_id"])
            if identity in seen_units:
                raise SceneEvaluationError(f"Duplicate {expected_label} output for {scene_id}/{output['unit_id']}")
            seen_units.add(identity)
            scene_outputs.setdefault(scene_id, []).append(output)
    scenes = {scene_id: "\n".join(item["korean"] for item in outputs) for scene_id, outputs in scene_outputs.items()}
    if not scenes:
        raise SceneEvaluationError(f"{expected_label} candidate has no scenes: {path}")
    return scenes


def initialize_scene_benchmark(root: Path, *, benchmark_id: str = "scene-v2", scene_ids: Sequence[str] = ()) -> dict[str, Any]:
    """Create an empty benchmark manifest without asserting an evaluation exists."""
    unique_scene_ids = list(dict.fromkeys(str(value).strip() for value in scene_ids if str(value).strip()))
    if len(unique_scene_ids) != len(scene_ids):
        raise SceneEvaluationError("scene_ids must be non-empty and unique")
    output = Path(root) / "evaluation" / "scene-benchmarks" / benchmark_id / "manifest.json"
    manifest = {
        "schema_name": "translation-forensics/scene-benchmark-manifest",
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "created_at": _utc_now(),
        "scene_ids": unique_scene_ids,
        "candidate_systems": {"candidate-a": "block_v1", "candidate-b": "external_baseline", "candidate-c": "scene_v2"},
        "external_baseline_status": "not-supplied",
        "evaluation_status": "not-evaluated",
        "human_evaluation_claimed": False,
        "note": "This initializes local evaluation bookkeeping only; it does not create a baseline, reviews, or a quality claim.",
    }
    _validate_schema(manifest, "scene-benchmark-manifest-v1.schema.json")
    _write_json(output, manifest)
    return {"status": "scene-benchmark-initialized", "manifest": str(output), "evaluation_status": "not-evaluated"}


def validate_external_baseline(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a user-provided baseline record.  This function never calls Gemini."""
    _validate_schema(value, "external-translation-baseline-v1.schema.json")
    outputs = _normalized_output_rows(value)
    return {"status": "pass", "scene_id": value["scene_id"], "outputs": len(outputs), "external_call_performed": False}


def ingest_external_baseline(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Copy validated, user-provided baseline records into a local benchmark input."""
    rows = _read_json_or_jsonl(Path(input_path))
    if not rows:
        raise SceneEvaluationError("External baseline input is empty")
    scene_ids: set[str] = set()
    for row in rows:
        validate_external_baseline(row)
        scene_id = str(row["scene_id"])
        if scene_id in scene_ids:
            raise SceneEvaluationError(f"Duplicate external baseline scene_id: {scene_id}")
        scene_ids.add(scene_id)
    document = {
        "schema_name": "translation-forensics/external-baseline-ingest",
        "schema_version": SCHEMA_VERSION,
        "ingested_at": _utc_now(),
        "input_sha256": _sha256_file(Path(input_path)),
        "external_call_performed": False,
        "records": rows,
    }
    _write_json(Path(output_path), document)
    return {"status": "external-baseline-ingested", "output": str(output_path), "records": len(rows), "external_call_performed": False}


def _blind_pack_paths(output_path: Path, key_path: Path | None) -> tuple[Path, Path]:
    output = Path(output_path)
    key = Path(key_path) if key_path else output.with_suffix(".internal-key.json")
    if output.exists() or key.exists():
        existing = output if output.exists() else key
        raise FileExistsError(f"Existing blind artifact will not be overwritten: {existing}")
    return output, key


def build_scene_blind_review_pack(
    *,
    block_v1_path: Path,
    scene_v2_path: Path,
    output_path: Path,
    random_seed: int,
    external_baseline_path: Path | None = None,
    key_path: Path | None = None,
    semantic_context_path: Path | None = None,
) -> dict[str, Any]:
    """Create a scene-wise blinded A/B/C pack and a separate sealed mapping key."""
    output, key = _blind_pack_paths(Path(output_path), key_path)
    systems: list[tuple[str, Path]] = [("block_v1", Path(block_v1_path))]
    if external_baseline_path is not None:
        systems.append(("external_baseline", Path(external_baseline_path)))
    systems.append(("scene_v2", Path(scene_v2_path)))
    scenes_by_system = {label: _candidate_scenes(path, expected_label=label) for label, path in systems}
    required_scene_ids = set(scenes_by_system["block_v1"])
    if set(scenes_by_system["scene_v2"]) != required_scene_ids:
        raise SceneEvaluationError("block_v1 and scene_v2 must cover exactly the same scene_ids")
    if "external_baseline" in scenes_by_system and set(scenes_by_system["external_baseline"]) != required_scene_ids:
        raise SceneEvaluationError("external baseline must cover exactly the same scene_ids as block_v1")
    contexts: dict[str, Any] = {}
    if semantic_context_path:
        for row in _read_json_or_jsonl(Path(semantic_context_path)):
            scene_id = str(row.get("scene_id", ""))
            if scene_id:
                contexts[scene_id] = row.get("japanese", row.get("source", ""))

    labels = [label for label, _ in systems]
    code_by_label = dict(zip(labels, CANDIDATE_CODES))
    rng = random.Random(random_seed)
    semantic_rows: list[dict[str, Any]] = []
    naturalness_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for scene_id in sorted(required_scene_ids):
        ordered_labels = labels[:]
        rng.shuffle(ordered_labels)
        codes = CANDIDATE_CODES[: len(ordered_labels)]
        assignment = dict(zip(codes, ordered_labels))
        candidate_text = {code: scenes_by_system[label][scene_id] for code, label in assignment.items()}
        semantic_rows.append({
            "scene_id": scene_id,
            "japanese_context": contexts.get(scene_id, ""),
            "candidates": candidate_text,
            "semantic_errors": [],
            "review_status": "not-reviewed",
        })
        naturalness_rows.append({
            "scene_id": scene_id,
            "candidates": candidate_text,
            "naturalness_choice": "",
            "readability_choice": "",
            "reason_tags": [],
            "review_status": "not-reviewed",
        })
        mapping_rows.append({"scene_id": scene_id, "candidate_mapping": assignment})
    pack_manifest = {
        "schema_name": "translation-forensics/scene-blind-review-pack",
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "scene_count": len(required_scene_ids),
        "candidate_codes": list(CANDIDATE_CODES[: len(labels)]),
        "candidate_identity_hidden": True,
        "random_seed_hidden": True,
        "review_status": "awaiting-review-submission",
        "human_evaluation_claimed": False,
    }
    _validate_schema(pack_manifest, "scene-blind-review-pack-v1.schema.json")
    key_document = {
        "schema_name": "translation-forensics/scene-blind-internal-key",
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "pack_sha256": None,
        "random_seed": random_seed,
        "release_condition": "A complete valid review submission must be validated before this key may be read for a summary.",
        "candidate_mapping": mapping_rows,
        "reviewer_visible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(pack_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        _write_jsonl_to_zip(archive, "semantic-panel.jsonl", semantic_rows)
        _write_jsonl_to_zip(archive, "naturalness-panel.jsonl", naturalness_rows)
        archive.writestr("README.md", "# Scene blind review\n\n후보 출처·모델·프롬프트 정보를 보지 말고 semantic panel과 naturalness panel을 별도로 작성하세요. 이 패키지는 실제 사람 평가가 완료됐다는 주장이 아닙니다.\n")
    key_document["pack_sha256"] = _sha256_file(output)
    _validate_schema(key_document, "scene-blind-internal-key-v1.schema.json")
    _write_json(key, key_document)
    return {"status": "scene-blind-pack-created", "output": str(output), "internal_key": str(key), "scenes": len(required_scene_ids), "external_baseline_included": "external_baseline" in scenes_by_system, "review_status": "not-reviewed"}


def _read_pack(pack_path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    with zipfile.ZipFile(pack_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        panels = {
            name: [json.loads(line) for line in archive.read(name).decode("utf-8").splitlines() if line.strip()]
            for name in ("semantic-panel.jsonl", "naturalness-panel.jsonl")
        }
    _validate_schema(manifest, "scene-blind-review-pack-v1.schema.json")
    return manifest, panels


def _review_rows(path: Path) -> list[dict[str, Any]]:
    value = _read_json(Path(path))
    rows = value.get("reviews")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise SceneEvaluationError("Review submission must contain an object list in reviews")
    return rows


def validate_scene_review(pack_path: Path, review_path: Path) -> dict[str, Any]:
    """Validate panels independently, without reading the blinded mapping key."""
    manifest, panels = _read_pack(Path(pack_path))
    rows = _review_rows(Path(review_path))
    expected = {str(row["scene_id"]) for row in panels["semantic-panel.jsonl"]}
    candidate_codes = set(manifest["candidate_codes"])
    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        scene_id = str(row.get("scene_id", ""))
        if scene_id in seen:
            errors.append(f"duplicate review scene_id: {scene_id}")
        seen.add(scene_id)
        semantic = row.get("semantic_review")
        natural = row.get("naturalness_review")
        if not isinstance(semantic, Mapping) or not isinstance(natural, Mapping):
            errors.append(f"{scene_id}: semantic_review and naturalness_review are required")
            continue
        semantic_errors = semantic.get("errors", [])
        if not isinstance(semantic_errors, list):
            errors.append(f"{scene_id}: semantic errors must be a list")
            continue
        critical_candidates: set[str] = set()
        for issue in semantic_errors:
            if not isinstance(issue, Mapping) or issue.get("candidate") not in candidate_codes:
                errors.append(f"{scene_id}: semantic issue candidate is invalid")
                continue
            if issue.get("severity") not in SEVERITIES:
                errors.append(f"{scene_id}: semantic issue severity is invalid")
            if not str(issue.get("category", "")).strip():
                errors.append(f"{scene_id}: semantic issue category is required")
            if issue.get("severity") == "critical" and issue.get("source_uncertainty") is not True:
                critical_candidates.add(str(issue["candidate"]))
        choice = natural.get("choice")
        readability_choice = natural.get("readability_choice", "tie")
        if choice not in NATURALNESS_CHOICES or (choice in candidate_codes and choice not in candidate_codes):
            errors.append(f"{scene_id}: naturalness choice is invalid")
        if readability_choice not in NATURALNESS_CHOICES:
            errors.append(f"{scene_id}: readability choice is invalid")
        if choice in critical_candidates:
            errors.append(f"{scene_id}: critical semantic-error candidate cannot win naturalness")
        if not isinstance(natural.get("reason_tags", []), list):
            errors.append(f"{scene_id}: naturalness reason_tags must be a list")
    if seen != expected:
        errors.append("Review scene_id set does not match the blind pack")
    if len(rows) != len(expected):
        errors.append("Review must have exactly one result for every packed scene")
    result = {
        "status": "pass" if not errors else "fail",
        "pack": str(pack_path),
        "review": str(review_path),
        "reviewed_scenes": len(rows),
        "expected_scenes": len(expected),
        "key_read": False,
        "evaluation_status": "review-submission-validated" if not errors else "not-evaluated",
        "human_evaluation_claimed": False,
        "errors": errors,
    }
    return result


def _wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if trials == 0:
        return None, None
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt((rate * (1 - rate) + z * z / (4 * trials)) / trials) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _pairwise_metrics(rows: list[dict[str, Any]], candidate: str, comparator: str) -> dict[str, Any]:
    qualified: list[dict[str, Any]] = []
    candidate_critical = candidate_major = comparator_critical = comparator_major = 0
    candidate_semantic_error_scenes = 0
    translationese_scenes = 0
    dialogue_inconsistency_scenes = 0
    for row in rows:
        issues = row["semantic_review"].get("errors", [])
        severe = {(issue.get("candidate"), issue.get("severity")) for issue in issues if issue.get("source_uncertainty") is not True}
        candidate_has_semantic_error = any(
            issue.get("candidate") == candidate
            for issue in issues
            if issue.get("source_uncertainty") is not True
        )
        candidate_semantic_error_scenes += candidate_has_semantic_error
        candidate_bad = (candidate, "critical") in severe
        comparator_bad = (comparator, "critical") in severe
        candidate_critical += candidate_bad
        comparator_critical += comparator_bad
        candidate_major += (candidate, "major") in severe
        comparator_major += (comparator, "major") in severe
        reason_tags = {
            str(tag).strip().casefold()
            for tag in row["naturalness_review"].get("reason_tags", [])
        }
        translationese_scenes += any("translationese" in tag for tag in reason_tags)
        dialogue_inconsistency_scenes += bool(
            {"dialogue_inconsistency", "scene_coherence", "response_mismatch"}
            & reason_tags
        )
        if not candidate_bad and not comparator_bad:
            qualified.append(row)
    wins = sum(row["naturalness_review"].get("choice") == candidate for row in qualified)
    losses = sum(row["naturalness_review"].get("choice") == comparator for row in qualified)
    ties = sum(row["naturalness_review"].get("choice") == "tie" for row in qualified)
    readability_wins = sum(
        row["naturalness_review"].get("readability_choice") == candidate
        for row in qualified
    )
    readability_losses = sum(
        row["naturalness_review"].get("readability_choice") == comparator
        for row in qualified
    )
    readability_ties = sum(
        row["naturalness_review"].get("readability_choice") == "tie"
        for row in qualified
    )
    lower, upper = _wilson_interval(wins, wins + losses) if wins + losses else (None, None)
    return {
        "comparison": f"{candidate}-vs-{comparator}", "scene_unit": "scene", "qualified_scenes": len(qualified),
        "decisive_scenes": wins + losses, "wins": wins, "losses": losses, "ties": ties,
        "qualified_naturalness_win_rate": wins / (wins + losses) if wins + losses else None,
        "naturalness_loss_rate": losses / len(qualified) if qualified else None,
        "qualified_naturalness_win_rate_wilson_95": {"lower": lower, "upper": upper},
        "candidate_critical_error_rate": candidate_critical / len(rows) if rows else None,
        "comparator_critical_error_rate": comparator_critical / len(rows) if rows else None,
        "candidate_major_error_rate": candidate_major / len(rows) if rows else None,
        "comparator_major_error_rate": comparator_major / len(rows) if rows else None,
        "semantic_error_rate": candidate_semantic_error_scenes / len(rows) if rows else None,
        "critical_error_rate": candidate_critical / len(rows) if rows else None,
        "major_error_rate": candidate_major / len(rows) if rows else None,
        "tie_rate": ties / len(qualified) if qualified else None,
        "subtitle_readability_preference": {
            "wins": readability_wins,
            "losses": readability_losses,
            "ties": readability_ties,
            "win_rate": readability_wins / (readability_wins + readability_losses)
            if readability_wins + readability_losses
            else None,
        },
        "translationese_reason_rate": translationese_scenes / len(rows) if rows else None,
        "dialogue_inconsistency_rate": dialogue_inconsistency_scenes / len(rows) if rows else None,
    }


def summarize_scene_benchmark(pack_path: Path, review_path: Path, key_path: Path, output_path: Path) -> dict[str, Any]:
    """Deblind only after a complete review validates; incomplete data is not evaluated."""
    validation = validate_scene_review(pack_path, review_path)
    if validation["status"] != "pass":
        result = {
            "schema_name": "translation-forensics/scene-benchmark-summary", "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(), "evaluation_status": "not-evaluated", "human_evaluation_claimed": False,
            "key_read_after_review_validation": False, "review_validation": validation, "comparisons": [],
            "note": "No blinded key was read because the review submission is incomplete or invalid.",
        }
        _validate_schema(result, "scene-benchmark-summary-v1.schema.json")
        _write_json(Path(output_path), result)
        return result
    # Intentionally read the key only after validation.  Keep this ordering when changing this function.
    key = _read_json(Path(key_path))
    if key.get("pack_sha256") != _sha256_file(Path(pack_path)):
        raise SceneEvaluationError("Blind internal key does not belong to this pack")
    _validate_schema(key, "scene-blind-internal-key-v1.schema.json")
    _, panels = _read_pack(Path(pack_path))
    reviews = _review_rows(Path(review_path))
    mapping_by_scene = {row["scene_id"]: row["candidate_mapping"] for row in key["candidate_mapping"]}
    unblinded: list[dict[str, Any]] = []
    for review in reviews:
        mapping = mapping_by_scene[str(review["scene_id"])]
        remapped = json.loads(json.dumps(review))
        for issue in remapped["semantic_review"].get("errors", []):
            issue["candidate"] = mapping[issue["candidate"]]
        for field in ("choice", "readability_choice"):
            value = remapped["naturalness_review"].get(field)
            if value in mapping:
                remapped["naturalness_review"][field] = mapping[value]
        unblinded.append(remapped)
    comparisons = [_pairwise_metrics(unblinded, "scene_v2", "block_v1")]
    if "external_baseline" in set().union(*(set(item["candidate_mapping"].values()) for item in key["candidate_mapping"])):
        comparisons.append(_pairwise_metrics(unblinded, "scene_v2", "external_baseline"))
    result = {
        "schema_name": "translation-forensics/scene-benchmark-summary", "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(), "evaluation_status": "evaluated-from-submitted-review-records",
        "human_evaluation_claimed": False, "key_read_after_review_validation": True,
        "review_validation": validation, "scenes": len(unblinded), "comparisons": comparisons,
        "note": "Metrics are scene-level and non-compensating. Submitted review records are not proof that a human evaluation occurred.",
    }
    _validate_schema(result, "scene-benchmark-summary-v1.schema.json")
    _write_json(Path(output_path), result)
    return result


def build_visual_ablation_manifest(
    output_path: Path,
    *,
    experiment_id: str,
    scene_ids: Sequence[str],
    allow_negative_control: bool = False,
    production_output: bool = False,
) -> dict[str, Any]:
    """Build a no-transfer experiment plan; condition E is research-only and never production."""
    ids = list(dict.fromkeys(str(value).strip() for value in scene_ids if str(value).strip()))
    if not ids or len(ids) != len(scene_ids):
        raise SceneEvaluationError("scene_ids must be non-empty and unique")
    conditions = list(ABLATION_CONDITIONS[:4])
    if allow_negative_control:
        conditions.append("E")
    if production_output and "E" in conditions:
        raise SceneEvaluationError("Negative-control condition E is forbidden for production output")
    manifest = {
        "schema_name": "translation-forensics/visual-ablation-manifest", "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id, "created_at": _utc_now(), "scene_ids": ids,
        "conditions": conditions, "negative_control_enabled": allow_negative_control,
        "production_output": production_output, "external_pixel_transfer_opt_in": False,
        "metrics": list(VISUAL_METRICS), "evaluation_status": "not-evaluated",
        "condition_definitions": {
            "A": "text only", "B": "text plus metadata", "C": "text plus targeted image",
            "D": "text plus broader scene visual context", "E": "shuffled or incongruent image negative control",
        },
    }
    _validate_schema(manifest, "visual-ablation-manifest-v1.schema.json")
    _write_json(Path(output_path), manifest)
    return manifest


def validate_visual_ablation_result(path: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    result = _read_json(Path(path))
    _validate_schema(result, "visual-ablation-result-v1.schema.json")
    errors: list[str] = []
    if result.get("condition") == "E" and result.get("production_output") is True:
        errors.append("Negative-control condition E cannot be used for production output")
    if manifest_path is not None:
        manifest = _read_json(Path(manifest_path))
        _validate_schema(manifest, "visual-ablation-manifest-v1.schema.json")
        if result.get("experiment_id") != manifest.get("experiment_id"):
            errors.append("Result experiment_id does not match manifest")
        if result.get("condition") not in manifest.get("conditions", []):
            errors.append("Result condition is not enabled by manifest")
        if result.get("scene_id") not in manifest.get("scene_ids", []):
            errors.append("Result scene_id is not in manifest")
        if result.get("pixel_transfer_count", 0) and not manifest.get("external_pixel_transfer_opt_in"):
            errors.append("Pixel transfer result is not allowed without explicit opt-in")
    return {"status": "pass" if not errors else "fail", "result": str(path), "errors": errors, "human_evaluation_claimed": False}


def summarize_visual_ablation(manifest_path: Path, result_paths: Sequence[Path], output_path: Path) -> dict[str, Any]:
    manifest = _read_json(Path(manifest_path))
    _validate_schema(manifest, "visual-ablation-manifest-v1.schema.json")
    reports = [validate_visual_ablation_result(Path(path), Path(manifest_path)) for path in result_paths]
    if not result_paths or any(report["status"] != "pass" for report in reports):
        summary = {
            "schema_name": "translation-forensics/visual-ablation-summary", "schema_version": SCHEMA_VERSION,
            "experiment_id": manifest["experiment_id"], "evaluation_status": "not-evaluated", "human_evaluation_claimed": False,
            "results": len(result_paths), "condition_summary": {}, "errors": [error for report in reports for error in report["errors"]] or ["No validated ablation results supplied"],
        }
    else:
        records = [_read_json(Path(path)) for path in result_paths]
        condition_summary: dict[str, dict[str, float | int]] = {}
        for condition in manifest["conditions"]:
            rows = [row for row in records if row["condition"] == condition]
            if not rows:
                continue
            condition_summary[condition] = {metric: sum(float(row["metrics"][metric]) for row in rows) / len(rows) for metric in VISUAL_METRICS}
            condition_summary[condition]["scenes"] = len(rows)
        summary = {
            "schema_name": "translation-forensics/visual-ablation-summary", "schema_version": SCHEMA_VERSION,
            "experiment_id": manifest["experiment_id"], "evaluation_status": "summarized-from-submitted-results",
            "human_evaluation_claimed": False, "results": len(records), "condition_summary": condition_summary, "errors": [],
            "note": "This summary does not infer that images help. Targeted visual benefit requires the pre-registered comparison conditions.",
        }
    _validate_schema(summary, "visual-ablation-summary-v1.schema.json")
    _write_json(Path(output_path), summary)
    return summary
