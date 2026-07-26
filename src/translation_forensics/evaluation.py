from __future__ import annotations

import json
from pathlib import Path
from typing import Any


GOLD_DIRECTORIES = ("scenes", "annotations", "adjudications", "splits")


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
        "split_names": ["train", "development", "locked-test"],
        "locked_test_policy": "locked-test 정답은 모델 학습·프롬프트 개발에 사용하지 않습니다.",
        "note": "이 파일은 평가 골격이며 정답·평가 완료·품질 개선을 의미하지 않습니다.",
    }
    manifest.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"status": "gold-layout-initialized", "gold_root": str(gold), "manifest": str(manifest)}
