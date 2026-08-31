from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from translation_forensics.scene_evaluation import (
    SceneEvaluationError,
    build_scene_blind_review_pack,
    build_visual_ablation_manifest,
    ingest_external_baseline,
    initialize_scene_benchmark,
    summarize_scene_benchmark,
    summarize_visual_ablation,
    validate_external_baseline,
    validate_scene_review,
    validate_visual_ablation_result,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _native_candidate(path: Path, prefix: str) -> None:
    _write_jsonl(path, [
        {"scene_id": "scene-001", "unit_id": "u1", "korean": f"{prefix} 첫 장면"},
        {"scene_id": "scene-002", "unit_id": "u2", "korean": f"{prefix} 둘째 장면"},
    ])


def _external_record(scene_id: str, korean: str) -> dict[str, object]:
    return {
        "schema_version": "1", "scene_id": scene_id, "system_label": "user-baseline",
        "service_label": "Gemini", "model_label": "user-recorded", "generation_date": "2026-08-31",
        "source_hash": "a" * 64, "prompt_hash": "b" * 64,
        "text_context_condition": "scene text", "visual_condition": "none",
        "outputs": [{"unit_id": "u", "korean": korean}], "user_supplied": True,
    }


def _review_rows() -> dict[str, object]:
    return {"reviews": [
        {"scene_id": "scene-001", "semantic_review": {"errors": []}, "naturalness_review": {"choice": "candidate-a", "readability_choice": "tie", "reason_tags": ["dialogue-flow"]}},
        {"scene_id": "scene-002", "semantic_review": {"errors": [{"candidate": "candidate-b", "category": "polarity", "severity": "critical", "source_uncertainty": False}]}, "naturalness_review": {"choice": "candidate-a", "readability_choice": "candidate-a", "reason_tags": ["register"]}},
    ]}


def test_external_baseline_is_user_supplied_validation_only(tmp_path: Path) -> None:
    record = _external_record("scene-001", "외부 후보")
    assert validate_external_baseline(record)["external_call_performed"] is False
    source = tmp_path / "external.jsonl"
    _write_jsonl(source, [record])
    output = tmp_path / "ingested.json"
    report = ingest_external_baseline(source, output)
    assert report["records"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["external_call_performed"] is False
    record["user_supplied"] = False
    with pytest.raises(SceneEvaluationError):
        validate_external_baseline(record)


def test_benchmark_initializes_as_not_evaluated(tmp_path: Path) -> None:
    report = initialize_scene_benchmark(tmp_path, benchmark_id="fixture", scene_ids=["scene-001"])
    manifest = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))
    assert manifest["evaluation_status"] == "not-evaluated"
    assert manifest["human_evaluation_claimed"] is False


def test_blind_pack_hides_identity_and_separates_key(tmp_path: Path) -> None:
    block, scene, external = (tmp_path / name for name in ("block.jsonl", "scene.jsonl", "external.jsonl"))
    _native_candidate(block, "legacy")
    _native_candidate(scene, "scene-v2")
    _write_jsonl(external, [_external_record("scene-001", "외부 첫 장면"), _external_record("scene-002", "외부 둘째 장면")])
    pack = tmp_path / "blind.zip"
    result = build_scene_blind_review_pack(block_v1_path=block, scene_v2_path=scene, external_baseline_path=external, output_path=pack, random_seed=101)
    assert Path(result["internal_key"]).exists()
    with zipfile.ZipFile(pack) as archive:
        packed = "\n".join(archive.read(name).decode("utf-8") for name in archive.namelist())
    assert "block_v1" not in packed
    assert "scene_v2" not in packed
    assert "external_baseline" not in packed
    assert "candidate-a" in packed and "candidate-b" in packed and "candidate-c" in packed


def test_blind_pack_accepts_scene_v2_cue_projection_rows(tmp_path: Path) -> None:
    block, scene = tmp_path / "block.jsonl", tmp_path / "scene-projection.jsonl"
    _native_candidate(block, "legacy")
    _write_jsonl(scene, [
        {"scene_id": "scene-001", "unit_id": "u1", "text": "장면 첫 대사"},
        {"scene_id": "scene-001", "unit_id": "u2", "text": "장면 다음 대사"},
        {"scene_id": "scene-002", "unit_id": "u3", "viewer_natural_korean": "둘째 장면"},
    ])
    report = build_scene_blind_review_pack(block_v1_path=block, scene_v2_path=scene, output_path=tmp_path / "blind.zip", random_seed=13)
    assert report["scenes"] == 2


def test_invalid_review_does_not_read_or_need_key(tmp_path: Path) -> None:
    block, scene = tmp_path / "block.jsonl", tmp_path / "scene.jsonl"
    _native_candidate(block, "legacy")
    _native_candidate(scene, "scene")
    pack = tmp_path / "blind.zip"
    build_scene_blind_review_pack(block_v1_path=block, scene_v2_path=scene, output_path=pack, random_seed=7)
    review = tmp_path / "incomplete-review.json"
    review.write_text(json.dumps({"reviews": []}), encoding="utf-8")
    validation = validate_scene_review(pack, review)
    assert validation["status"] == "fail"
    summary = summarize_scene_benchmark(pack, review, tmp_path / "missing-key.json", tmp_path / "summary.json")
    assert summary["evaluation_status"] == "not-evaluated"
    assert summary["key_read_after_review_validation"] is False


def test_scene_summary_reports_scene_metrics_and_wilson_without_human_claim(tmp_path: Path) -> None:
    block, scene = tmp_path / "block.jsonl", tmp_path / "scene.jsonl"
    _native_candidate(block, "legacy")
    _native_candidate(scene, "scene")
    pack = tmp_path / "blind.zip"
    build = build_scene_blind_review_pack(block_v1_path=block, scene_v2_path=scene, output_path=pack, random_seed=1)
    key = json.loads(Path(build["internal_key"]).read_text(encoding="utf-8"))
    mapping = {row["scene_id"]: row["candidate_mapping"] for row in key["candidate_mapping"]}
    reviews = _review_rows()
    # Make scene_v2 win after unblinding, independent of its visible code per scene.
    for row in reviews["reviews"]:
        row["naturalness_review"]["choice"] = next(code for code, label in mapping[row["scene_id"]].items() if label == "scene_v2")
    review = tmp_path / "review.json"
    review.write_text(json.dumps(reviews), encoding="utf-8")
    assert validate_scene_review(pack, review)["status"] == "pass"
    summary = summarize_scene_benchmark(pack, review, Path(build["internal_key"]), tmp_path / "summary.json")
    comparison = summary["comparisons"][0]
    assert comparison["scene_unit"] == "scene"
    assert comparison["qualified_naturalness_win_rate_wilson_95"]["lower"] is not None
    assert comparison["semantic_error_rate"] is not None
    assert comparison["subtitle_readability_preference"]["wins"] >= 0
    assert comparison["translationese_reason_rate"] is not None
    assert comparison["dialogue_inconsistency_rate"] is not None
    assert summary["human_evaluation_claimed"] is False


def _ablation_result(*, condition: str = "C", production_output: bool = False) -> dict[str, object]:
    metrics = {
        "semantic_accuracy": 0.8, "speaker_addressee": 0.8, "deictic_referent": 0.8,
        "on_screen_text": 0.8, "scene_continuity": 0.8, "unsupported_visual_addition": 0.0,
        "hallucination": 0.0, "korean_naturalness": 0.8, "cost": 1.0, "latency": 2.0,
        "pixel_transfer_count": 0.0,
    }
    return {"schema_name": "translation-forensics/visual-ablation-result", "schema_version": "1", "experiment_id": "visual-fixture", "scene_id": "scene-001", "condition": condition, "production_output": production_output, "pixel_transfer_count": 0, "metrics": metrics}


def test_visual_ablation_rejects_production_negative_control_and_summarizes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = build_visual_ablation_manifest(manifest_path, experiment_id="visual-fixture", scene_ids=["scene-001"], allow_negative_control=True)
    assert manifest["conditions"][-1] == "E"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_ablation_result(condition="E", production_output=True)), encoding="utf-8")
    with pytest.raises(SceneEvaluationError):
        validate_visual_ablation_result(bad)
    result = tmp_path / "result.json"
    result.write_text(json.dumps(_ablation_result()), encoding="utf-8")
    assert validate_visual_ablation_result(result, manifest_path)["status"] == "pass"
    summary = summarize_visual_ablation(manifest_path, [result], tmp_path / "summary.json")
    assert summary["evaluation_status"] == "summarized-from-submitted-results"
    assert summary["human_evaluation_claimed"] is False
