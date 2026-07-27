from __future__ import annotations

import json
import hashlib
import random
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .srt import compare_structure, parse_srt


GOLD_DIRECTORIES = ("train", "development", "release-test", "sealed-test", "annotations", "adjudications", "manifests")
GOLD_REQUIRED_FIELDS = {
    "gold_id", "title_id", "scene_id", "block_ids", "audio_range",
    "human_listened", "listener_1", "listener_2", "initial_transcript_1",
    "initial_transcript_2", "adjudicator", "adjudicated_japanese",
    "semantic_frame", "acceptable_korean_range", "forbidden_interpretations",
    "final_gold_status", "split",
}
GOLD_SPLITS = {"train", "development", "release-test", "sealed-test", "locked-test"}
BLIND_DECISIONS = {"approve-a", "approve-b", "tie", "hold", "reject", "needs-human-listening", "needs-more-context", "nonverbal", "unresolved"}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize_gold_layout(root: Path) -> dict[str, Any]:
    """Create an answer-free gold layout once; never replace existing files."""
    gold = root / "evaluation" / "gold"
    manifest = gold / "manifest.json"
    if manifest.exists():
        raise FileExistsError(f"기존 gold manifest를 덮어쓰지 않습니다: {manifest}")
    for name in GOLD_DIRECTORIES:
        (gold / name).mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": "1.0",
        "status": "empty-no-gold-answers",
        "required_scene_fields": [
            "audio_directly_checked", "japanese_transcript", "semantic_frame",
            "allowed_korean_range", "forbidden_interpretations", "acceptable_answers",
            "error_types", "severity", "scene_context",
        ],
        "split_names": ["train", "development", "release-test", "sealed-test"],
        "sealed_test_policy": "sealed-test 정답은 주요 구조 변경 또는 공식 검증 때만 사용하며, 결과를 본 뒤 같은 버전을 다시 개발하지 않습니다.",
        "note": "이 파일은 평가 골격이며 정답·평가 완료·품질 개선을 의미하지 않습니다.",
    }
    manifest.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"status": "gold-layout-initialized", "gold_root": str(gold), "manifest": str(manifest)}


def migrate_gold_layout(root: Path) -> dict[str, Any]:
    """Upgrade a legacy empty gold layout without replacing answers or records."""
    gold = root / "evaluation" / "gold"
    manifest = gold / "manifest.json"
    if not manifest.exists():
        return initialize_gold_layout(root)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("gold manifest는 JSON 객체여야 합니다.")
    for name in GOLD_DIRECTORIES:
        (gold / name).mkdir(parents=True, exist_ok=True)
    previous_version = value.get("schema_version", "unknown")
    value.update({
        "schema_version": "2.0",
        "split_names": ["train", "development", "release-test", "sealed-test"],
        "sealed_test_policy": "sealed-test 정답은 주요 구조 변경 또는 공식 검증 때만 사용하며, 결과를 본 뒤 같은 버전을 다시 개발하지 않습니다.",
    })
    value.setdefault("migration_history", []).append({"from_schema_version": previous_version, "to_schema_version": "2.0", "action": "created split directories without rewriting gold records"})
    manifest.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"status": "gold-layout-migrated", "gold_root": str(gold), "manifest": str(manifest), "previous_schema_version": previous_version}


def initialize_gold_record(root: Path, *, gold_id: str, title_id: str, scene_id: str, block_ids: list[int], split: str) -> dict[str, Any]:
    """Create an explicitly incomplete, reviewer-owned gold annotation template."""
    if not gold_id.strip() or not title_id.strip() or not scene_id.strip() or not block_ids:
        raise ValueError("gold_id, title_id, scene_id, block_ids가 필요합니다.")
    if split not in GOLD_SPLITS:
        raise ValueError(f"gold split이 잘못되었습니다: {split}")
    gold = root / "evaluation" / "gold"
    output = gold / "annotations" / f"{gold_id}.gold-record.json"
    if output.exists():
        raise FileExistsError(f"기존 gold 레코드를 덮어쓰지 않습니다: {output}")
    record = {
        "schema_name": "translation-forensics/gold-record",
        "schema_version": "1",
        "gold_id": gold_id,
        "title_id": title_id,
        "scene_id": scene_id,
        "block_ids": block_ids,
        "audio_range": {"start_seconds": None, "end_seconds": None},
        "video_checked": False,
        "human_listened": False,
        "listener_1": "",
        "listener_2": "",
        "initial_transcript_1": "",
        "initial_transcript_2": "",
        "disagreement_type": "",
        "adjudicator": "",
        "adjudicated_japanese": "",
        "semantic_frame": {},
        "acceptable_korean_range": [],
        "forbidden_interpretations": [],
        "representative_wrong_answers": [],
        "mqm_errors": [],
        "difficulty": "",
        "context_dependency": "",
        "split": split,
        "final_gold_status": "unresolved-gold",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "이 템플릿은 정답이나 직접 청취 완료를 의미하지 않습니다.",
    }
    _write_json(output, record)
    return {"status": "gold-record-template", "output": str(output), "gold_id": gold_id, "final_gold_status": "unresolved-gold"}


def validate_gold_record(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if not isinstance(value, dict):
        return {"status": "fail", "record": str(path), "errors": ["gold record는 JSON 객체여야 합니다."]}
    missing = sorted(GOLD_REQUIRED_FIELDS - set(value))
    if missing:
        errors.append(f"필수 필드 누락: {', '.join(missing)}")
    if value.get("split") not in GOLD_SPLITS:
        errors.append("split은 train/development/locked-test 중 하나여야 합니다.")
    if not isinstance(value.get("block_ids"), list) or not value.get("block_ids") or not all(isinstance(item, int) and item > 0 for item in value.get("block_ids", [])):
        errors.append("block_ids는 하나 이상의 양의 정수여야 합니다.")
    if value.get("final_gold_status") == "adjudicated":
        if value.get("human_listened") is not True:
            errors.append("adjudicated gold에는 human_listened=true가 필요합니다.")
        for key in ("listener_1", "listener_2", "initial_transcript_1", "initial_transcript_2", "adjudicator", "adjudicated_japanese"):
            if not str(value.get(key, "")).strip():
                errors.append(f"adjudicated gold에는 {key}가 필요합니다.")
        if not isinstance(value.get("acceptable_korean_range"), list) or not value.get("acceptable_korean_range"):
            errors.append("adjudicated gold에는 acceptable_korean_range가 필요합니다.")
    elif value.get("final_gold_status") not in {"unresolved-gold", "rejected"}:
        errors.append("final_gold_status는 unresolved-gold/rejected/adjudicated 중 하나여야 합니다.")
    return {"status": "pass" if not errors else "fail", "record": str(path), "gold_id": value.get("gold_id"), "final_gold_status": value.get("final_gold_status"), "errors": errors}


def validate_gold_suite(root: Path) -> dict[str, Any]:
    gold = root / "evaluation" / "gold"
    annotation_dir = gold / "annotations"
    if not annotation_dir.exists():
        return {"status": "fail", "records": 0, "errors": ["gold annotations 디렉터리가 없습니다."], "title_split_conflicts": []}
    reports = [validate_gold_record(path) for path in sorted(annotation_dir.glob("*.gold-record.json"))]
    title_splits: dict[str, set[str]] = {}
    for path in annotation_dir.glob("*.gold-record.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        title_splits.setdefault(str(value.get("title_id", "")), set()).add(str(value.get("split", "")))
    conflicts = [{"title_id": title, "splits": sorted(splits)} for title, splits in title_splits.items() if len(splits) > 1]
    errors = [error for report in reports for error in report["errors"]]
    if conflicts:
        errors.append("같은 title_id가 여러 gold split에 포함됩니다.")
    adjudicated = sum(report.get("final_gold_status") == "adjudicated" and report["status"] == "pass" for report in reports)
    sealed = 0
    for path in annotation_dir.glob("*.gold-record.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("final_gold_status") == "adjudicated" and value.get("split") in {"sealed-test", "locked-test"}:
            sealed += 1
    return {"status": "pass" if not errors else "fail", "records": len(reports), "adjudicated_records": adjudicated, "sealed_test_records": sealed, "locked_test_records": sealed, "errors": errors, "title_split_conflicts": conflicts, "note": "이 검증은 골드 품질을 자동 확정하지 않으며 작품 간 분할 누출만 차단합니다."}


def build_blind_review_pack(
    title: str,
    structure_path: Path,
    source_path: Path,
    viewer_path: Path,
    output_path: Path,
    *,
    random_seed: int,
) -> dict[str, Any]:
    """Create a local-only A/B pack; it deliberately excludes candidate provenance."""
    if output_path.exists():
        raise FileExistsError(f"기존 blind review pack을 덮어쓰지 않습니다: {output_path}")
    structure, _, _ = parse_srt(structure_path)
    source, _, _ = parse_srt(source_path)
    viewer, _, _ = parse_srt(viewer_path)
    if not compare_structure(structure, source)["pass"] or not compare_structure(structure, viewer)["pass"]:
        raise ValueError("blind review pack에는 구조 기준본과 동일한 두 SRT가 필요합니다.")
    rng = random.Random(random_seed)
    rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for reference, faithful, natural in zip(structure, source, viewer):
        items = [("source-faithful", faithful.text), ("viewer-natural", natural.text)]
        rng.shuffle(items)
        rows.append({
            "review_id": f"BR-{title}-{reference.number:04d}",
            "block_number": reference.number,
            "timecode": f"{reference.start} --> {reference.end}",
            "candidate_a": items[0][1],
            "candidate_b": items[1][1],
            "decision": "",
            "mqm_errors": [],
            "review_note": "",
        })
        key_rows.append({"review_id": f"BR-{title}-{reference.number:04d}", "candidate_a_variant": items[0][0], "candidate_b_variant": items[1][0]})
    pack_manifest = {
        "schema_name": "translation-forensics/blind-review-pack",
        "schema_version": "1",
        "title_id": title,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "reviewer_visible_fields": ["review_id", "block_number", "timecode", "candidate_a", "candidate_b", "decision", "mqm_errors", "review_note"],
        "provenance_hidden": True,
        "source_hashes": {"structure": _sha256(structure_path), "source_faithful": _sha256(source_path), "viewer_natural": _sha256(viewer_path)},
        "block_count": len(rows),
        "status": "review-template-not-reviewed",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    key_path = output_path.with_suffix(".internal-key.json")
    if key_path.exists():
        raise FileExistsError(f"기존 blind review key를 덮어쓰지 않습니다: {key_path}")
    with zipfile.ZipFile(output_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("review.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        archive.writestr("README.md", "# Blind review pack\n\n후보의 출처·모델·기존 결과를 보지 말고 각 줄의 decision과 MQM 오류만 기록하세요. 이 패키지는 사람 검토 완료나 품질 개선을 의미하지 않습니다.\n")
        archive.writestr("manifest.json", json.dumps(pack_manifest, ensure_ascii=False, indent=2) + "\n")
    _write_json(key_path, {"status": "internal-unblinding-key", "pack_sha256": _sha256(output_path), "candidate_mapping": key_rows})
    return {"status": "blind-review-pack-created", "output": str(output_path), "internal_key": str(key_path), "blocks": len(rows), "random_seed": random_seed, "review_status": "not-reviewed"}


def summarize_blind_review(pack_path: Path, reviewed_jsonl_path: Path, key_path: Path, output_path: Path) -> dict[str, Any]:
    """Deblind completed human decisions; unreviewed templates never score."""
    with zipfile.ZipFile(pack_path) as archive:
        pack = json.loads(archive.read("manifest.json"))
        expected = {row["review_id"] for row in (json.loads(line) for line in archive.read("review.jsonl").decode("utf-8").splitlines() if line.strip())}
    key = json.loads(key_path.read_text(encoding="utf-8"))
    if key.get("pack_sha256") != _sha256(pack_path):
        raise ValueError("internal key가 blind review pack과 일치하지 않습니다.")
    reviewed = [json.loads(line) for line in reviewed_jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = {str(row.get("review_id", "")) for row in reviewed}
    errors = []
    if seen != expected:
        errors.append("review_id 집합이 blind pack과 일치하지 않습니다.")
    for row in reviewed:
        if row.get("decision") not in BLIND_DECISIONS:
            errors.append(f"{row.get('review_id', '?')}: 유효하지 않거나 미입력인 decision")
    mapping = {row["review_id"]: row for row in key.get("candidate_mapping", [])}
    counts = {decision: sum(row.get("decision") == decision for row in reviewed) for decision in sorted(BLIND_DECISIONS)}
    selected = {"source-faithful": 0, "viewer-natural": 0}
    for row in reviewed:
        decision = row.get("decision")
        map_row = mapping.get(row.get("review_id"), {})
        if decision == "approve-a":
            selected[str(map_row.get("candidate_a_variant"))] = selected.get(str(map_row.get("candidate_a_variant")), 0) + 1
        if decision == "approve-b":
            selected[str(map_row.get("candidate_b_variant"))] = selected.get(str(map_row.get("candidate_b_variant")), 0) + 1
    result = {
        "schema_name": "translation-forensics/blind-review-summary", "schema_version": "1", "created_at": datetime.now(timezone.utc).isoformat(),
        "pack": str(pack_path), "reviewed_input": str(reviewed_jsonl_path), "title_id": pack.get("title_id"),
        "reviewed_blocks": len(reviewed), "review_complete": not errors, "decision_counts": counts, "selected_variant_counts": selected,
        "evaluation_status": "human-reviewed" if not errors else "not-demonstrated", "errors": errors,
        "note": "선호 집계는 정확도·품질 개선 또는 전체 작품 성능을 뜻하지 않습니다.",
    }
    if output_path.exists():
        raise FileExistsError(f"기존 blind review summary를 덮어쓰지 않습니다: {output_path}")
    _write_json(output_path, result)
    return result


def validate_evaluation_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if value.get("evaluation_status") != "human-reviewed":
        errors.append("evaluation summary는 human-reviewed 상태여야 합니다.")
    if value.get("review_complete") is not True:
        errors.append("evaluation summary에 완료된 사람 검수가 필요합니다.")
    if not isinstance(value.get("reviewed_blocks"), int) or value.get("reviewed_blocks", 0) < 1:
        errors.append("evaluation summary에 1개 이상의 검수 블록이 필요합니다.")
    return {"status": "pass" if not errors else "fail", "evaluation_summary": str(path), "errors": errors}
