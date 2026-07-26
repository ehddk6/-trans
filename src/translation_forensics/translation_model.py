"""Fixed model policy for Korean semantic translation artifacts.

This does not configure ASR, Subtitle Forensics, or legacy external
machine-draft helpers. It guards only semantic translation decisions and
their SRT-application path.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_TRANSLATION_MODEL = "gpt-5.6-terra"
ALLOWED_TRANSLATION_MODELS = (DEFAULT_TRANSLATION_MODEL,)


def _validate_config() -> None:
    path = Path(__file__).resolve().parents[2] / "config" / "translation-model.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"번역 모델 설정을 읽을 수 없습니다: {path}") from exc
    if config.get("default_translation_model") != DEFAULT_TRANSLATION_MODEL or tuple(config.get("allowed_translation_models", [])) != ALLOWED_TRANSLATION_MODELS:
        raise ValueError("번역 모델 설정은 gpt-5.6-terra 하나로 고정되어야 합니다.")


def resolve_translation_model(value: object | None = None) -> str:
    """Return the sole approved Korean-translation model or reject the value."""
    _validate_config()
    model = DEFAULT_TRANSLATION_MODEL if value is None else str(value).strip()
    if model not in ALLOWED_TRANSLATION_MODELS:
        allowed = ", ".join(ALLOWED_TRANSLATION_MODELS)
        raise ValueError(f"한국어 의미 번역 모델은 {allowed}로 고정되어 있습니다: {model!r}")
    return model
