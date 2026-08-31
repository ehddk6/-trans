from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .codex_exec_provider import (
    CodexExecProvider,
    CodexTimeoutError,
    CodexUsageLimitError,
    validate_call_receipt,
)
from .integrated_pipeline import (
    BundleBlockedError,
    GateResult,
    JapaneseSubtitleBundle,
    TranslationInputArtifacts,
    TranslationUnit,
    UnitTranslation,
    build_cache_identity,
    build_translation_inputs,
    evaluate_bundle,
    load_review_decisions,
    load_translation_units,
    normalize_whitespace,
    package_dual_outputs,
    promote_staged_run,
    stage_run,
)
from .integrated_translation import (
    audit_translations_with_sol,
    review_translation_with_visuals,
    translate_units_with_terra,
)
from .automated_quality import audit_translation_decisions
from .srt import has_japanese, parse_srt
from .visual_context import (
    ALLOWED_VISUAL_SLOTS,
    VisualFrame,
    build_visual_context_record,
    index_legacy_captures,
    prepare_model_attachments,
    sha256_file,
    title_code_matches,
)


PROCESS_SCHEMA_VERSION = "3"
_HASHED_RUN_ARTIFACTS = (
    "input_manifest.json",
    "capture_index.jsonl",
    "japanese_bundle.json",
    "translation-input/translation_ja.srt",
    "translation-input/translation_units_ja.jsonl",
    "translation_decisions.jsonl",
    "translation_draft_decisions.jsonl",
    "automated_quality.jsonl",
    "model_call_receipts.jsonl",
    "translation_draft_receipts.jsonl",
    "visual_context.jsonl",
    "translation-stage.json",
    "quality-stage.json",
    "packaging-stage.json",
    "review_queue.jsonl",
    "review_decisions.jsonl",
    "outputs/viewer_complete_ko.srt",
    "outputs/source_faithful_ko.srt",
    "outputs/viewer_natural_ko.srt",
    "outputs/translation_qa.json",
    "qa_report.json",
)
_REQUIRED_RUN_ARTIFACTS = (*_HASHED_RUN_ARTIFACTS, "run_manifest.json")
_DEICTIC_RE = re.compile(r"(?:これ|それ|あれ|ここ|そこ|あそこ|こっち|そっち|どこ|どれ)")
_ADDRESSEE_RE = re.compile(r"(?:あなた|あんた|君|お前|先生|さん|ちゃん|くん|誰)")
_SCREEN_TEXT_RE = re.compile(r"(?:画面|表示|書いて|書かれて|読んで|名前|文字)")
_CONTINUITY_RE = re.compile(r"(?:さっき|今度|まだ|もう|続き|前|後で|次)")
_NEGATION_RE = re.compile(r"(?:ない|ません|ぬ|いや|駄目|だめ|やめ)")


@dataclass(frozen=True, slots=True)
class ProcessTitleConfig:
    project_root: Path
    title_id: str
    media: Path
    reference_ja: Path | None = None
    reference_ja_approved: bool = False
    japanese_bundle: Path | None = None
    legacy_captures: Path | None = None
    translation_policy: str = "dual"
    translation_architecture: str = "block_v1"
    visual_policy: str = "targeted"
    max_visual_units: int = 20
    max_frames_per_unit: int = 3
    auto_capture_frames: bool = True
    quality_policy: str = "automated"
    translation_batch_size: int = 40
    model_batch_workers: int = 1
    qwen_root: Path | None = None
    review_decisions: Path | None = None
    resume: bool = False
    codex_timeout_seconds: int = 600
    audit_attempt: int = 0
    scene_gap_threshold_seconds: float = 4.0
    scene_max_units: int = 24
    scene_max_source_characters: int = 12000
    naturalness_repair_attempts: int = 1
    semantic_audit_scope: str = "all"
    dialogue_memory_policy: str = "confirmed-only"
    output_root: Path | None = None

    def validate(self) -> None:
        if not self.title_id.strip():
            raise ValueError("title_id is required")
        if self.translation_policy != "dual":
            raise ValueError("translation_policy must be dual")
        if self.translation_architecture not in {"block_v1", "scene_v2"}:
            raise ValueError("translation_architecture must be block_v1 or scene_v2")
        if self.visual_policy not in {"off", "metadata", "targeted"}:
            raise ValueError("visual_policy must be off, metadata, or targeted")
        if not 0 <= self.max_visual_units:
            raise ValueError("max_visual_units must be non-negative")
        if not 0 <= self.max_frames_per_unit <= 3:
            raise ValueError("max_frames_per_unit must be between 0 and 3")
        if not isinstance(self.auto_capture_frames, bool):
            raise ValueError("auto_capture_frames must be a boolean")
        if self.quality_policy not in {"legacy", "automated"}:
            raise ValueError("quality_policy must be legacy or automated")
        if self.quality_policy == "automated" and self.review_decisions is not None:
            raise ValueError(
                "review_decisions cannot be combined with the automated quality policy"
            )
        if self.translation_batch_size < 1:
            raise ValueError("translation_batch_size must be positive")
        if not 1 <= self.model_batch_workers <= 8:
            raise ValueError("model_batch_workers must be between 1 and 8")
        if self.audit_attempt < 0:
            raise ValueError("audit_attempt must be non-negative")
        if self.translation_architecture == "scene_v2":
            if not math.isfinite(self.scene_gap_threshold_seconds) or self.scene_gap_threshold_seconds <= 0:
                raise ValueError("scene_gap_threshold_seconds must be positive")
            if self.scene_max_units < 1:
                raise ValueError("scene_max_units must be positive")
            if self.scene_max_source_characters < 1:
                raise ValueError("scene_max_source_characters must be positive")
            if not 0 <= self.naturalness_repair_attempts <= 3:
                raise ValueError("naturalness_repair_attempts must be between 0 and 3")
            if self.semantic_audit_scope not in {"all", "targeted"}:
                raise ValueError("semantic_audit_scope must be all or targeted")
            if self.dialogue_memory_policy not in {
                "off",
                "confirmed-only",
                "provisional-style-only",
            }:
                raise ValueError(
                    "dialogue_memory_policy must be off, confirmed-only, or provisional-style-only"
                )
        if self.reference_ja is not None and not self.reference_ja_approved:
            raise ValueError("--reference-ja requires --reference-ja-approved")
        if self.reference_ja_approved and self.reference_ja is None:
            raise ValueError("--reference-ja-approved requires --reference-ja")
        if self.reference_ja is not None and self.japanese_bundle is not None:
            raise ValueError("reference_ja and japanese_bundle are mutually exclusive")


def stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_subtitle_backend(
    *, reference_ja: Path | None, reference_ja_approved: bool, japanese_bundle: Path | None
) -> str:
    if japanese_bundle is not None:
        return "existing-bundle"
    if reference_ja is not None:
        if not reference_ja_approved:
            raise ValueError("a Japanese reference cannot be authoritative without explicit approval")
        return "reference"
    return "ensemble"


def ensure_normalized_audio(media: Path, cache_root: Path, media_sha256: str) -> tuple[Path, dict[str, Any]]:
    """Create one persistent 16 kHz mono PCM cache shared by Whisper and Qwen."""

    media = Path(media).resolve()
    root = Path(cache_root).resolve() / "normalized-audio" / media_sha256
    audio_path = root / "audio-16k-mono-pcm-s16le.wav"
    manifest_path = root / "manifest.json"
    identity = {
        "schema_name": "translation-forensics/normalized-audio-cache",
        "schema_version": "1",
        "media_sha256": media_sha256,
        "sample_rate": 16000,
        "channels": 1,
        "codec": "pcm_s16le",
    }
    if audio_path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        if (
            manifest.get("status") == "complete"
            and manifest.get("identity") == identity
            and manifest.get("audio_sha256") == stream_sha256(audio_path)
        ):
            return audio_path, {**manifest, "cache_hit": True}
    if audio_path.exists() or manifest_path.exists():
        raise ValueError(f"stale normalized-audio cache requires manual inspection: {root}")
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{uuid.uuid4().hex}.partial.wav"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(media),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            check=True,
            timeout=7_200,
        )
        os.replace(temporary, audio_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {
        "status": "complete",
        "identity": identity,
        "source_path": str(media),
        "audio_path": str(audio_path),
        "audio_sha256": stream_sha256(audio_path),
        "created_at": _utc_now(),
        "cache_hit": False,
    }
    _write_json_atomic(manifest_path, manifest)
    return audio_path, manifest


def process_title(
    config: ProcessTitleConfig,
    *,
    provider: CodexExecProvider | Any | None = None,
    subtitle_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config.validate()
    project_root = Path(config.project_root).expanduser().resolve()
    media = Path(config.media).expanduser().resolve()
    if not media.is_file():
        raise FileNotFoundError(media)
    if not title_code_matches(config.title_id, media.name):
        raise ValueError(f"media path does not contain the exact title code {config.title_id}: {media}")
    reference = Path(config.reference_ja).expanduser().resolve() if config.reference_ja else None
    if reference is not None and not reference.is_file():
        raise FileNotFoundError(reference)
    if reference is not None and not _path_contains_exact_title(config.title_id, reference):
        raise ValueError(
            f"reference path does not contain the exact title code {config.title_id}: {reference}"
        )
    existing_bundle = Path(config.japanese_bundle).expanduser().resolve() if config.japanese_bundle else None
    if existing_bundle is not None and not existing_bundle.is_dir():
        raise NotADirectoryError(existing_bundle)
    if existing_bundle is not None and not _path_contains_exact_title(config.title_id, existing_bundle):
        raise ValueError(
            f"Japanese bundle path does not contain the exact title code {config.title_id}: {existing_bundle}"
        )
    captures = Path(config.legacy_captures).expanduser().resolve() if config.legacy_captures else None
    if captures is not None:
        if not captures.is_dir():
            raise NotADirectoryError(captures)
        if not _path_contains_exact_title(config.title_id, captures):
            raise ValueError(f"capture path does not contain the exact title code {config.title_id}: {captures}")
    review_decisions_path = (
        Path(config.review_decisions).expanduser().resolve() if config.review_decisions else None
    )
    if review_decisions_path is not None and not review_decisions_path.is_file():
        raise FileNotFoundError(review_decisions_path)
    if review_decisions_path is not None:
        load_review_decisions(review_decisions_path)
    review_decision_rows = _read_jsonl(review_decisions_path) if review_decisions_path else []
    review_decisions_sha256 = hashlib.sha256(
        json.dumps(
            review_decision_rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    media_sha256, media_hash_receipt = _resolve_media_sha256(
        project_root, config.title_id.upper(), media
    )
    reference_sha256 = stream_sha256(reference) if reference else None
    indexed_legacy_frames = index_legacy_captures(captures) if captures else []
    capture_inventory = _capture_inventory(captures, indexed_legacy_frames)
    existing_bundle_sha256 = _existing_bundle_identity(existing_bundle) if existing_bundle else None
    backend = choose_subtitle_backend(
        reference_ja=reference,
        reference_ja_approved=config.reference_ja_approved,
        japanese_bundle=existing_bundle,
    )
    code_version = _code_version(config.translation_architecture)
    cache_identity = build_cache_identity(
        code_version=code_version,
        schema_version=PROCESS_SCHEMA_VERSION,
        media_sha256=media_sha256,
        options={
            "title_id": config.title_id.upper(),
            "backend": backend,
            "reference_sha256": reference_sha256,
            "japanese_bundle": str(existing_bundle) if existing_bundle else None,
            "japanese_bundle_sha256": existing_bundle_sha256,
            "capture_inventory_sha256": (
                capture_inventory.get("aggregate_sha256") if capture_inventory else None
            ),
            "translation_policy": config.translation_policy,
            "translation_architecture": config.translation_architecture,
            "visual_policy": config.visual_policy,
            "max_visual_units": config.max_visual_units,
            "max_frames_per_unit": config.max_frames_per_unit,
            "auto_capture_frames": config.auto_capture_frames,
            "quality_policy": config.quality_policy,
            "translation_batch_size": config.translation_batch_size,
            "model_batch_workers": config.model_batch_workers,
            "audit_attempt": config.audit_attempt,
            "qwen_root": str(Path(config.qwen_root).expanduser().resolve()) if config.qwen_root else None,
            "review_decisions_sha256": review_decisions_sha256,
            "scene_v2": (
                {
                    "gap_threshold_seconds": config.scene_gap_threshold_seconds,
                    "max_units": config.scene_max_units,
                    "max_source_characters": config.scene_max_source_characters,
                    "naturalness_repair_attempts": config.naturalness_repair_attempts,
                    "semantic_audit_scope": config.semantic_audit_scope,
                    "dialogue_memory_policy": config.dialogue_memory_policy,
                }
                if config.translation_architecture == "scene_v2"
                else None
            ),
        },
    )
    run_id = f"{config.title_id.upper()}-{cache_identity['identity_sha256'][:16]}"
    workspace = project_root / "workspaces" / config.title_id.upper()
    output_root = Path(config.output_root).expanduser().resolve() if config.output_root else workspace / "integrated"
    final_dir = output_root / run_id
    if final_dir.is_dir():
        if not config.resume:
            raise FileExistsError(f"integrated run already exists: {final_dir}")
        manifest = _read_json(final_dir / "run_manifest.json")
        if manifest.get("cache_identity") != cache_identity:
            raise ValueError("existing integrated run has a different cache identity")
        _verify_promoted_run(final_dir, manifest)
        recovered = stage_run(
            output_root, run_id, cache_identity=cache_identity, resume=True
        )
        if recovered != final_dir:
            raise ValueError("promoted run recovery returned an unexpected path")
        _bundle_from_manifest(
            _read_json(final_dir / "japanese_bundle.json"),
            base_dir=final_dir,
            expected_media_sha256=media_sha256,
        )
        return {**manifest, "cache_hit": True, "run_dir": str(final_dir)}

    stage = stage_run(output_root, run_id, cache_identity=cache_identity, resume=config.resume)
    input_manifest = {
        "schema_name": "translation-forensics/integrated-inputs",
        "schema_version": PROCESS_SCHEMA_VERSION,
        "title_id": config.title_id.upper(),
        "media": {
            **_file_record(media, known_sha256=media_sha256),
            "sha256_receipt": media_hash_receipt,
        },
        "reference_ja": _file_record(reference, known_sha256=reference_sha256) if reference else None,
        "reference_ja_approved": bool(config.reference_ja_approved),
        "legacy_captures": capture_inventory,
        "translation_architecture": config.translation_architecture,
        "visual_capture": {
            "mode": "legacy_or_auto_generated",
            "auto_capture_frames": config.auto_capture_frames,
            "max_frames_per_unit": config.max_frames_per_unit,
            "scope": "selected_visual_units_only",
        },
        "quality_policy": config.quality_policy,
        "audit_attempt": config.audit_attempt,
        "prompt_contracts": _prompt_contract_inventory(config.translation_architecture),
        "review_decisions": (
            _file_record(review_decisions_path) if review_decisions_path else None
        ),
        "review_decisions_semantic_sha256": review_decisions_sha256,
        "source_bundle": str(existing_bundle) if existing_bundle else None,
        "cache_identity": cache_identity,
    }
    _write_once_or_match(stage / "input_manifest.json", input_manifest, resume=config.resume)
    capture_index_path = stage / "capture_index.jsonl"
    # When no legacy directory is supplied, defer creating the index until
    # generated frames (if any) are available.  This keeps resume atomic while
    # allowing a photo-less title to acquire its own evidence frames.
    if captures is not None:
        _write_jsonl_once_or_match(
            capture_index_path,
            (
                {
                    "path": str(frame.path),
                    "timestamp_seconds": frame.timestamp_seconds,
                    "sha256": frame.sha256,
                    "source": "legacy",
                }
                for frame in indexed_legacy_frames
            ),
            resume=config.resume,
        )
    _write_jsonl_once_or_match(
        stage / "review_decisions.jsonl", review_decision_rows, resume=config.resume
    )

    bundle_manifest_path = stage / "japanese_bundle.json"
    if bundle_manifest_path.is_file():
        if not config.resume:
            raise FileExistsError(bundle_manifest_path)
        bundle = _bundle_from_manifest(
            _read_json(bundle_manifest_path),
            base_dir=stage,
            expected_media_sha256=media_sha256,
        )
    else:
        if existing_bundle is not None:
            bundle = load_japanese_bundle(
                existing_bundle, media_sha256=media_sha256, expected_media=media
            )
            audio_cache_receipt = None
        else:
            if subtitle_runner is None:
                from subtitle_pipeline.pipeline import run_pipeline as subtitle_runner
            qwen_runtime = None
            normalized_audio = None
            audio_cache_receipt = None
            if backend == "ensemble":
                from subtitle_pipeline.qwen import QwenRuntime

                qwen_runtime = QwenRuntime.discover(config.qwen_root) if config.qwen_root else QwenRuntime.discover()
                qwen_runtime.validate()
                normalized_audio, audio_cache_receipt = ensure_normalized_audio(
                    media, project_root / ".cache" / "integrated", media_sha256
                )
            subtitle_root = stage / "japanese"
            subtitle_runner(
                media,
                subtitle_root,
                model_name="large-v3-turbo",
                language="ja",
                profile_name="viewer_ja",
                normalization="layout",
                word_timestamps=True,
                vad=False,
                beam_size=5,
                temperature=0.0,
                condition_on_previous_text=False,
                backend=backend,
                qwen_runtime=qwen_runtime,
                reference_srt=reference,
                review_samples=30,
                reuse_qwen_cache=True,
                qwen_audit_samples=4,
                normalized_audio_path=normalized_audio,
                media_sha256=media_sha256,
            )
            bundle_dir = subtitle_root / ("reference_primary" if backend == "reference" else "ensemble_qwen_whisper")
            bundle = load_japanese_bundle(
                bundle_dir, media_sha256=media_sha256, expected_media=media
            )
        bundle_manifest = _bundle_manifest(bundle, run_root=stage)
        bundle_manifest["audio_cache_receipt"] = audio_cache_receipt
        _write_json_atomic(bundle_manifest_path, bundle_manifest)

    disposition = evaluate_bundle(bundle)
    if not disposition.complete_draft_allowed:
        raise BundleBlockedError("Japanese bundle cannot produce a draft: " + ", ".join(disposition.reasons))
    transcript_path = Path(bundle.artifacts["transcript_ja.jsonl"])
    translation_input_dir = stage / "translation-input"
    if (translation_input_dir / "translation_units_ja.jsonl").is_file():
        if not config.resume:
            raise FileExistsError(translation_input_dir)
        inputs = _load_translation_inputs(translation_input_dir)
    else:
        inputs = build_translation_inputs(transcript_path, translation_input_dir, bundle=bundle)

    source_repeat = _maximum_consecutive_text(unit.text_raw for unit in inputs.units)
    if source_repeat[0] >= 100:
        raise BundleBlockedError(
            f"Japanese source contains {source_repeat[0]} consecutive identical units: {source_repeat[1]!r}"
        )

    if provider is None:
        provider = CodexExecProvider(
            cache_dir=project_root / ".cache" / "integrated" / "codex",
            timeout_seconds=config.codex_timeout_seconds,
        )
        preflight = provider.preflight()
        if preflight.get("status") != "pass":
            raise RuntimeError("Codex provider preflight failed: " + ", ".join(preflight.get("errors", [])))
    if config.translation_architecture == "scene_v2":
        return _process_scene_v2(
            config=config,
            provider=provider,
            stage=stage,
            output_root=output_root,
            run_id=run_id,
            cache_identity=cache_identity,
            inputs=inputs,
            bundle=bundle,
            source_repeat=source_repeat,
            indexed_legacy_frames=indexed_legacy_frames,
        )
    decisions_path = stage / "translation_decisions.jsonl"
    automated_quality_path = stage / "automated_quality.jsonl"
    receipts_path = stage / "model_call_receipts.jsonl"
    translation_checkpoint_path = stage / "translation-stage.json"
    translation_stage_artifacts = (
        "translation_draft_decisions.jsonl",
        "translation_draft_receipts.jsonl",
        "visual_context.jsonl",
    )
    if _checkpoint_is_valid(
        stage, translation_checkpoint_path, translation_stage_artifacts
    ):
        decisions = _read_jsonl(stage / "translation_draft_decisions.jsonl")
        receipts = _read_jsonl(stage / "translation_draft_receipts.jsonl")
    else:
        model_units = [_unit_for_model(unit) for unit in inputs.units]
        decisions, receipts = translate_units_with_terra(
            provider,
            title_id=config.title_id.upper(),
            units=model_units,
            resume=config.resume,
            batch_size=config.translation_batch_size,
            max_batch_workers=config.model_batch_workers,
        )

        base_visual_records: dict[str, dict[str, Any]] = {}
        visual_receipts: list[dict[str, Any]] = []
        if config.visual_policy != "off" and config.max_visual_units and config.max_frames_per_unit:
            indexed_frames = indexed_legacy_frames
            candidates = _visual_candidate_ids(decisions, inputs.units, config.max_visual_units)
            frames_by_unit: dict[str, list[Path]] = {}
            units_by_id = {unit.unit_id: unit for unit in inputs.units}
            decisions_by_id = {str(row["unit_id"]): row for row in decisions}
            for unit_id in candidates:
                unit = units_by_id[unit_id]
                decision = decisions_by_id[unit_id]
                requested_slots = _visual_slots(
                    unit.text_raw, decision.get("uncertain_slots", [])
                )
                available = indexed_frames
                if not available and config.auto_capture_frames:
                    available = _extract_unit_frames(
                        media,
                        stage / "generated-frames" / unit_id,
                        unit,
                        config.max_frames_per_unit,
                    )
                record = build_visual_context_record(
                    title_id=config.title_id.upper(),
                    unit_id=unit_id,
                    start_seconds=unit.start,
                    end_seconds=unit.end,
                    frames=available,
                    transfer_mode="targeted" if config.visual_policy == "targeted" else "metadata-only",
                    max_frames=config.max_frames_per_unit,
                )
                record["unit_selection_reasons"] = _visual_selection_reasons(
                    decision, unit, requested_slots
                )
                record["requested_visual_slots"] = sorted(requested_slots)
                base_visual_records[unit_id] = record
                if config.visual_policy == "targeted":
                    attachments = prepare_model_attachments(record)
                    frames_by_unit[unit_id] = [
                        Path(row["path"]) for row in attachments
                    ]
                else:
                    frames_by_unit[unit_id] = []
            if config.visual_policy == "targeted" and frames_by_unit:
                before = {row["unit_id"]: dict(row) for row in decisions}
                decisions, visual_receipts, reviewed = review_translation_with_visuals(
                    provider,
                    title_id=config.title_id.upper(),
                    decisions=decisions,
                    units_by_id={unit_id: _unit_for_model(unit) for unit_id, unit in units_by_id.items()},
                    frames_by_unit=frames_by_unit,
                    resume=config.resume,
                )
                for review in reviewed:
                    unit_id = review["unit_id"]
                    base = base_visual_records[unit_id]
                    base["visual_slots"] = {
                        slot: review.get("allowed_visual_slots", {}).get(slot)
                        for slot in ALLOWED_VISUAL_SLOTS
                    }
                    receipt = review["model_call_receipt"]
                    cache_replay = bool(receipt.get("cache_hit"))
                    attachment_hashes = [
                        item.get("sha256") for item in receipt.get("image_attachments", [])
                    ]
                    base["external_transfer_receipt"] = {
                        "status": "cache-replay" if cache_replay else receipt.get("status", "succeeded"),
                        "external_transfer": False if cache_replay else receipt.get("external_transfer", True),
                        "pixel_transfer_count": 0 if cache_replay else receipt.get("pixel_external_transfer_count", len(base["frames"])),
                        "transferred_frame_sha256": [] if cache_replay else attachment_hashes,
                        "cache_origin_frame_sha256": attachment_hashes if cache_replay else [],
                        "provider": receipt.get("provider", "codex-cli"),
                        "request_id": receipt.get("request_sha256") or receipt.get("call_id"),
                    }
                    base["critical_visual_impact"] = review["critical_visual_impact"]
                    base["verdict"] = review["verdict"]
                blocked_reviews = [
                    review
                    for review in reviewed
                    if review.get("model_call_receipt", {}).get("status") == "blocked"
                ]
                if blocked_reviews:
                    reviewed_ids = {str(review["unit_id"]) for review in reviewed}
                    blocked_receipt = blocked_reviews[0]["model_call_receipt"]
                    decisions_by_id = {str(row["unit_id"]): row for row in decisions}
                    for unit_id, base in base_visual_records.items():
                        if unit_id in reviewed_ids:
                            continue
                        decision = decisions_by_id[unit_id]
                        decision["review_required_reasons"] = list(
                            dict.fromkeys(
                                [
                                    *decision.get("review_required_reasons", []),
                                    "visual_review_not_sent_after_usage_limit",
                                ]
                            )
                        )
                        base["external_transfer_receipt"] = {
                            "status": "not-sent-usage-limit",
                            "external_transfer": False,
                            "pixel_transfer_count": 0,
                            "transferred_frame_sha256": [],
                            "provider": "codex-cli",
                            "request_id": None,
                            "retry_after": blocked_receipt.get("retry_after"),
                        }
                        base["verdict"] = "machine-uncertain-not-sent"
                _write_json_atomic(
                    stage / "visual_ab_report.json",
                    _visual_ab_report(before, {row["unit_id"]: row for row in decisions}, reviewed),
                )
            receipts.extend(visual_receipts)
        _make_generated_frame_paths_portable(
            stage, base_visual_records.values(), receipts
        )
        _write_jsonl_atomic(stage / "visual_context.jsonl", base_visual_records.values())
        _write_jsonl_atomic(decisions_path, decisions)
        _write_jsonl_atomic(receipts_path, receipts)
        _write_jsonl_atomic(stage / "translation_draft_decisions.jsonl", decisions)
        _write_jsonl_atomic(stage / "translation_draft_receipts.jsonl", receipts)
        _write_checkpoint(
            stage, translation_checkpoint_path, translation_stage_artifacts
        )

    visual_context_rows = _read_jsonl(stage / "visual_context.jsonl")
    _ensure_capture_index(stage, visual_context_rows, resume=config.resume)
    automated_quality_records: list[dict[str, Any]] = []
    quality_checkpoint_path = stage / "quality-stage.json"
    quality_stage_artifacts = (
        "translation_decisions.jsonl",
        "automated_quality.jsonl",
        "model_call_receipts.jsonl",
    )
    if config.quality_policy == "automated":
        if _checkpoint_is_valid(
            stage, quality_checkpoint_path, quality_stage_artifacts
        ):
            decisions = _read_jsonl(decisions_path)
            automated_quality_records = _read_jsonl(automated_quality_path)
            receipts = _read_jsonl(receipts_path)
        else:
            decisions, automated_quality_records = audit_translation_decisions(
                [_unit_for_model(unit) for unit in inputs.units], decisions
            )
            try:
                decisions, audit_receipts, model_records = audit_translations_with_sol(
                    provider,
                    title_id=config.title_id.upper(),
                    decisions=decisions,
                    batch_size=max(1, min(config.translation_batch_size, 20)),
                    resume=config.resume,
                    max_batch_workers=config.model_batch_workers,
                )
                receipts.extend(audit_receipts)
                deterministic_by_id = {
                    str(row["unit_id"]): dict(row) for row in automated_quality_records
                }
                for model_record in model_records:
                    base = deterministic_by_id[str(model_record["unit_id"])]
                    base.update(model_record)
                automated_quality_records = list(deterministic_by_id.values())
            except (CodexUsageLimitError, CodexTimeoutError) as exc:
                # A service quota is not a semantic finding. Preserve the completed
                # genre-first candidate and record that its independent semantic
                # audit could not run; do not relabel every line as a literal
                # fallback.
                blocked_reason = (
                    "usage-limit" if isinstance(exc, CodexUsageLimitError) else "timeout"
                )
                for decision in decisions:
                    decision["automated_quality_call_id"] = None
                    decision["semantic_audit_status"] = "blocked"
                    decision["semantic_audit_issue_codes"] = []
                    decision["automated_quality_reasons"] = list(
                        dict.fromkeys(
                            [
                                *decision.get("automated_quality_reasons", []),
                                f"sol_independent_audit_{blocked_reason}",
                            ]
                        )
                    )
                for record in automated_quality_records:
                    record["model_status"] = "blocked"
                    record["model_block_reason"] = blocked_reason
                    record["retry_after"] = getattr(exc, "retry_after", None)
                    record["semantic_audit_status"] = "blocked"
                    record["semantic_audit_issue_codes"] = []
            # Visual review writes its pre-audit decisions first; the audited
            # decisions replace them atomically before packaging.
            _write_jsonl_atomic(decisions_path, decisions)
            _write_jsonl_atomic(automated_quality_path, automated_quality_records)
            _write_jsonl_atomic(receipts_path, receipts)
    else:
        _write_jsonl_atomic(decisions_path, decisions)
        _write_jsonl_atomic(receipts_path, receipts)
        _write_jsonl_atomic(automated_quality_path, ())
    _write_checkpoint(stage, quality_checkpoint_path, quality_stage_artifacts)
    visual_status_counts: dict[str, int] = {}
    for row in visual_context_rows:
        status = str(row.get("external_transfer_receipt", {}).get("status") or "unknown")
        visual_status_counts[status] = visual_status_counts.get(status, 0) + 1

    review_decisions = load_review_decisions(stage / "review_decisions.jsonl")
    automated = config.quality_policy == "automated"
    unit_translations = _package_translations(
        inputs.units, decisions, bundle, review_decisions, automated_quality=automated
    )
    output_dir = stage / "outputs"
    packaging_checkpoint_path = stage / "packaging-stage.json"
    packaging_stage_artifacts = (
        "translation_decisions.jsonl",
        "automated_quality.jsonl",
        "outputs/viewer_complete_ko.srt",
        "outputs/source_faithful_ko.srt",
        "outputs/viewer_natural_ko.srt",
        "outputs/translation_qa.json",
    )
    if not _checkpoint_is_valid(
        stage, packaging_checkpoint_path, packaging_stage_artifacts
    ):
        packaged = package_dual_outputs(
            inputs.units,
            unit_translations,
            output_dir,
            review_decisions=review_decisions,
            bundle=bundle,
            automated_quality=automated,
            replace_existing=output_dir.exists(),
        )
    else:
        qa = _read_json(output_dir / "translation_qa.json")
        packaged = _ExistingOutputs(output_dir, qa)

    if automated:
        automated_quality_records = _attach_automated_render_hashes(
            inputs.units,
            decisions,
            output_dir,
            automated_quality_records,
            resume=config.resume,
        )
        _write_checkpoint(stage, quality_checkpoint_path, quality_stage_artifacts)
    _write_checkpoint(
        stage, packaging_checkpoint_path, packaging_stage_artifacts
    )

    review_queue = [] if automated else _review_queue(inputs.units, decisions, bundle, review_decisions)
    _write_jsonl_atomic(stage / "review_queue.jsonl", review_queue)
    qa_report = _integrated_qa(
        inputs,
        decisions,
        receipts,
        bundle,
        output_dir,
        source_repeat=source_repeat,
        review_queue=review_queue,
        quality_policy=config.quality_policy,
        automated_quality_records=automated_quality_records,
        visual_context_records=visual_context_rows,
    )
    _write_json_atomic(stage / "qa_report.json", qa_report)
    verification_passed = bool(qa_report["verification_passed"])
    if not verification_passed:
        raise BundleBlockedError("integrated QA failed: " + ", ".join(qa_report["errors"]))

    all_automated_passed = bool(automated_quality_records) and all(
        row.get("status") == "passed" for row in automated_quality_records
    )
    all_semantic_audits_resolved = bool(automated_quality_records) and all(
        row.get("semantic_audit_status") in {"pass", "not-selected"}
        for row in automated_quality_records
    )
    unresolved_bundle_gate = any(
        gate.status != "passed"
        for gate in (bundle.recognition, bundle.alignment, bundle.presentation)
    ) or not bundle.accepted
    machine_uncertain = automated and (
        not all_automated_passed
        or not all_semantic_audits_resolved
        or unresolved_bundle_gate
    )
    run_stage = (
        "machine-uncertain" if automated and machine_uncertain
        else "machine-final" if automated
        else "machine-draft-not-demonstrated"
    )
    run_status = (
        "machine-uncertain" if automated and machine_uncertain
        else "machine-verified" if automated
        else "machine-draft-not-demonstrated"
    )
    run_manifest = {
        "schema_name": "translation-forensics/integrated-run-manifest",
        "schema_version": PROCESS_SCHEMA_VERSION,
        "title_id": config.title_id.upper(),
        "run_id": run_id,
        "status": run_status,
        "stage": "packaged",
        "cache_identity": cache_identity,
        "translation_architecture": "block_v1",
        "backend": bundle.backend,
        "bundle_valid": bundle.valid,
        "bundle_accepted": bundle.accepted,
        "complete_units": len(inputs.units),
        "candidate_units": packaged.candidate_units,
        "pending_review_count": len(review_queue),
        "quality_policy": config.quality_policy,
        "audit_attempt": config.audit_attempt,
        "automated_quality_units": len(automated_quality_records),
        "automated_quality_fallback_units": sum(
            row.get("status") == "fallback" for row in automated_quality_records
        ),
        "semantic_audit": {
            "scope": "targeted-semantic-only",
            "passed": sum(
                row.get("semantic_audit_status") == "pass"
                for row in automated_quality_records
            ),
            "issues": sum(
                row.get("semantic_audit_status") == "issue"
                for row in automated_quality_records
            ),
            "inconclusive": sum(
                row.get("semantic_audit_status") == "inconclusive"
                for row in automated_quality_records
            ),
            "not_selected": sum(
                row.get("semantic_audit_status") == "not-selected"
                for row in automated_quality_records
            ),
            "blocked": sum(
                row.get("semantic_audit_status") == "blocked"
                for row in automated_quality_records
            ),
        },
        "visual_policy": config.visual_policy,
        "visual_selected_units": len(visual_context_rows),
        "visual_capture_mode": (
            "legacy"
            if indexed_legacy_frames
            else "auto-generated"
            if any(row.get("frames") for row in visual_context_rows)
            else "none"
        ),
        "auto_generated_frame_count": sum(
            1
            for row in _read_jsonl(stage / "capture_index.jsonl")
            if row.get("source") == "auto-generated"
        ),
        "visual_status_counts": visual_status_counts,
        "external_image_transfer_count": sum(
            0
            if receipt.get("cache_hit")
            else int(receipt.get("pixel_external_transfer_count", 0) or 0)
            for receipt in receipts
        ),
        "human_reviewed": False,
        "human_final_allowed": False,
        "human_reference_equality": "unidentifiable",
        "100_percent_equal": False,
        "machine_final_allowed": automated and not machine_uncertain,
        "final_promotion_allowed": False,
        "verification_status": (
            "machine-uncertain" if automated and machine_uncertain
            else "machine-verified" if automated
            else "not-demonstrated"
        ),
        "created_at": _read_json(stage / "run-state.json").get("started_at"),
        "artifact_sha256": {
            name: stream_sha256(stage / name) for name in _HASHED_RUN_ARTIFACTS
        },
    }
    _write_json_atomic(stage / "run_manifest.json", run_manifest)
    promoted = promote_staged_run(
        output_root,
        run_id,
        verification_passed=True,
        stage=run_stage,
        pending_review_count=len(review_queue),
        required_artifacts=_REQUIRED_RUN_ARTIFACTS,
        manifest_sha256=stream_sha256(stage / "run_manifest.json"),
    )
    return {**run_manifest, "cache_hit": False, "run_dir": str(promoted)}


def _process_scene_v2(
    *,
    config: ProcessTitleConfig,
    provider: Any,
    stage: Path,
    output_root: Path,
    run_id: str,
    cache_identity: Mapping[str, Any],
    inputs: TranslationInputArtifacts,
    bundle: JapaneseSubtitleBundle,
    source_repeat: tuple[int, str],
    indexed_legacy_frames: Sequence[VisualFrame],
) -> dict[str, Any]:
    """Run scene_v2 without changing the legacy block_v1 execution path."""

    from .scene_segmentation import SceneSegmentationConfig, build_dialogue_scenes
    from .scene_translation import run_scene_translation_v2

    scenes = build_dialogue_scenes(
        inputs.units,
        config=SceneSegmentationConfig(
            gap_threshold_seconds=config.scene_gap_threshold_seconds,
            max_units=config.scene_max_units,
            max_source_characters=config.scene_max_source_characters,
        ),
    )
    visual_context_by_unit, visual_context_rows = _scene_visual_context(
        config=config,
        units=inputs.units,
        indexed_frames=indexed_legacy_frames,
    )
    scene_result = run_scene_translation_v2(
        provider,
        title_id=config.title_id.upper(),
        units=inputs.units,
        scenes=scenes,
        prompts_dir=Path(config.project_root).expanduser().resolve() / "prompts",
        schemas_dir=Path(config.project_root).expanduser().resolve() / "schemas",
        resume=config.resume,
        repair_attempts=config.naturalness_repair_attempts,
        cache_context={
            "process_cache_identity_sha256": cache_identity["identity_sha256"],
            "semantic_audit_scope": config.semantic_audit_scope,
            "dialogue_memory_policy": config.dialogue_memory_policy,
            "style_memory_by_scene": {},
        },
        visual_policy=config.visual_policy,
        visual_context_by_unit=visual_context_by_unit,
    )
    decisions = [dict(row) for row in scene_result.decisions]
    expected_unit_ids = [unit.unit_id for unit in inputs.units]
    if [str(row.get("unit_id") or "") for row in decisions] != expected_unit_ids:
        raise BundleBlockedError("scene_v2 decision order or unit coverage mismatch")
    if len(set(expected_unit_ids)) != len(decisions):
        raise BundleBlockedError("scene_v2 decisions do not cover every unit exactly once")

    scene_artifact_names: list[str] = []
    for raw_name, rows in scene_result.artifacts.items():
        name = _safe_run_artifact_name(raw_name)
        if name in _HASHED_RUN_ARTIFACTS or name in scene_artifact_names:
            raise ValueError(f"scene_v2 artifact name collides with an integration artifact: {name}")
        _write_jsonl_once_or_match(stage / name, rows, resume=config.resume)
        scene_artifact_names.append(name)

    receipts = [dict(row) for row in scene_result.receipts]
    quality_records: list[dict[str, Any]] = []
    for decision in decisions:
        semantic_status = str(decision.get("semantic_drift_status") or "unknown")
        dialogue_status = str(decision.get("korean_dialogue_status") or "unknown")
        passed = semantic_status not in {"issue", "fail"} and dialogue_status not in {
            "issue",
            "fail",
        }
        evidence_id = f"scene-v2:{decision['scene_id']}:{decision['unit_id']}"
        decision["confidence"] = decision.get("confidence") or ("high" if passed else "low")
        decision["review_required_reasons"] = list(
            dict.fromkeys(
                [
                    *decision.get("review_required_reasons", []),
                    *([] if passed else ["scene_v2_unresolved_quality_issue"]),
                ]
            )
        )
        decision["automated_quality_status"] = "passed" if passed else "fallback"
        decision["automated_quality_evidence_id"] = evidence_id
        quality_records.append(
            {
                "unit_id": decision["unit_id"],
                "scene_id": decision["scene_id"],
                "status": "passed" if passed else "fallback",
                "semantic_audit_status": "pass" if semantic_status != "issue" else "issue",
                "dialogue_audit_status": dialogue_status,
                "automated_quality_evidence_id": evidence_id,
                "translation_architecture": "scene_v2",
            }
        )

    _bind_scene_visual_receipts(visual_context_rows, receipts)
    _make_generated_frame_paths_portable(stage, visual_context_rows, receipts)
    _write_jsonl_atomic(stage / "visual_context.jsonl", visual_context_rows)
    _ensure_capture_index(stage, visual_context_rows, resume=config.resume)
    _write_jsonl_atomic(stage / "translation_draft_decisions.jsonl", decisions)
    _write_jsonl_atomic(stage / "translation_draft_receipts.jsonl", receipts)
    _write_jsonl_atomic(stage / "translation_decisions.jsonl", decisions)
    _write_jsonl_atomic(stage / "model_call_receipts.jsonl", receipts)
    _write_jsonl_atomic(stage / "automated_quality.jsonl", quality_records)
    _write_checkpoint(
        stage,
        stage / "translation-stage.json",
        (
            "translation_draft_decisions.jsonl",
            "translation_draft_receipts.jsonl",
            "visual_context.jsonl",
        ),
    )

    review_decisions = load_review_decisions(stage / "review_decisions.jsonl")
    unit_translations = _package_translations(
        inputs.units,
        decisions,
        bundle,
        review_decisions,
        automated_quality=True,
    )
    output_dir = stage / "outputs"
    packaged = package_dual_outputs(
        inputs.units,
        unit_translations,
        output_dir,
        review_decisions=review_decisions,
        bundle=bundle,
        automated_quality=True,
        replace_existing=output_dir.exists(),
    )
    quality_records = _attach_automated_render_hashes(
        inputs.units,
        decisions,
        output_dir,
        quality_records,
        resume=config.resume,
    )
    _write_checkpoint(
        stage,
        stage / "quality-stage.json",
        (
            "translation_decisions.jsonl",
            "automated_quality.jsonl",
            "model_call_receipts.jsonl",
        ),
    )
    _write_checkpoint(
        stage,
        stage / "packaging-stage.json",
        (
            "translation_decisions.jsonl",
            "automated_quality.jsonl",
            "outputs/viewer_complete_ko.srt",
            "outputs/source_faithful_ko.srt",
            "outputs/viewer_natural_ko.srt",
            "outputs/translation_qa.json",
        ),
    )
    _write_jsonl_atomic(stage / "review_queue.jsonl", ())
    qa_report = _integrated_qa(
        inputs,
        decisions,
        receipts,
        bundle,
        output_dir,
        source_repeat=source_repeat,
        review_queue=[],
        quality_policy="automated",
        translation_architecture="scene_v2",
        automated_quality_records=quality_records,
        visual_context_records=visual_context_rows,
    )
    qa_report["scene_v2"] = dict(scene_result.qa)
    qa_report["architecture_status"] = scene_result.architecture_status
    qa_report["benchmark_status"] = scene_result.benchmark_status
    _write_json_atomic(stage / "qa_report.json", qa_report)
    if not qa_report["verification_passed"]:
        raise BundleBlockedError("scene_v2 integrated QA failed: " + ", ".join(qa_report["errors"]))

    all_quality_passed = bool(quality_records) and all(
        row.get("status") == "passed" and row.get("semantic_audit_status") == "pass"
        for row in quality_records
    )
    unresolved_bundle_gate = any(
        gate.status != "passed"
        for gate in (bundle.recognition, bundle.alignment, bundle.presentation)
    ) or not bundle.accepted
    machine_uncertain = not all_quality_passed or unresolved_bundle_gate
    run_status = "machine-uncertain" if machine_uncertain else "machine-verified"
    run_stage = "machine-uncertain" if machine_uncertain else "machine-final"
    visual_status_counts: dict[str, int] = {}
    for row in visual_context_rows:
        status = str(row.get("external_transfer_receipt", {}).get("status") or "unknown")
        visual_status_counts[status] = visual_status_counts.get(status, 0) + 1
    hashed_artifacts = (*_HASHED_RUN_ARTIFACTS, *scene_artifact_names)
    run_manifest = {
        "schema_name": "translation-forensics/integrated-run-manifest",
        "schema_version": PROCESS_SCHEMA_VERSION,
        "title_id": config.title_id.upper(),
        "run_id": run_id,
        "status": run_status,
        "stage": "packaged",
        "cache_identity": dict(cache_identity),
        "translation_architecture": "scene_v2",
        "architecture_status": scene_result.architecture_status,
        "benchmark_status": scene_result.benchmark_status,
        "scene_cache_identity": scene_result.cache_identity,
        "scene_artifacts": scene_artifact_names,
        "scene_qa": dict(scene_result.qa),
        "backend": bundle.backend,
        "bundle_valid": bundle.valid,
        "bundle_accepted": bundle.accepted,
        "complete_units": len(inputs.units),
        "candidate_units": packaged.candidate_units,
        "pending_review_count": 0,
        "quality_policy": "automated",
        "audit_attempt": config.audit_attempt,
        "semantic_audit": {
            "scope": config.semantic_audit_scope,
            "passed": sum(row.get("semantic_audit_status") == "pass" for row in quality_records),
            "issues": sum(row.get("semantic_audit_status") == "issue" for row in quality_records),
            "inconclusive": 0,
            "not_selected": 0,
            "blocked": 0,
        },
        "visual_policy": config.visual_policy,
        "visual_selected_units": len(visual_context_rows),
        "visual_capture_mode": "legacy" if indexed_legacy_frames else "none",
        "auto_generated_frame_count": 0,
        "visual_status_counts": visual_status_counts,
        "external_image_transfer_count": sum(
            0
            if receipt.get("cache_hit")
            else int(receipt.get("pixel_external_transfer_count", 0) or 0)
            for receipt in receipts
        ),
        "final_package_structure": {
            "status": "locked-to-translation-units",
            "unit_count": len(inputs.units),
            "numbering_timing_order_preserved": True,
        },
        "human_reviewed": False,
        "human_final_allowed": False,
        "human_reference_equality": "unidentifiable",
        "100_percent_equal": False,
        "machine_final_allowed": not machine_uncertain,
        "final_promotion_allowed": False,
        "verification_status": "machine-uncertain" if machine_uncertain else "machine-verified",
        "created_at": _read_json(stage / "run-state.json").get("started_at"),
        "artifact_sha256": {
            name: stream_sha256(stage / name) for name in hashed_artifacts
        },
    }
    _write_json_atomic(stage / "run_manifest.json", run_manifest)
    promoted = promote_staged_run(
        output_root,
        run_id,
        verification_passed=True,
        stage=run_stage,
        pending_review_count=0,
        required_artifacts=(*hashed_artifacts, "run_manifest.json"),
        manifest_sha256=stream_sha256(stage / "run_manifest.json"),
    )
    return {**run_manifest, "cache_hit": False, "run_dir": str(promoted)}


def _scene_visual_context(
    *,
    config: ProcessTitleConfig,
    units: Sequence[TranslationUnit],
    indexed_frames: Sequence[VisualFrame],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if (
        config.visual_policy == "off"
        or not indexed_frames
        or not config.max_visual_units
        or not config.max_frames_per_unit
    ):
        return {}, []
    context: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for unit in units[: config.max_visual_units]:
        record = build_visual_context_record(
            title_id=config.title_id.upper(),
            unit_id=unit.unit_id,
            start_seconds=unit.start,
            end_seconds=unit.end,
            frames=indexed_frames,
            transfer_mode=(
                "targeted" if config.visual_policy == "targeted" else "metadata-only"
            ),
            max_frames=config.max_frames_per_unit,
        )
        image_paths = [str(frame["path"]) for frame in record.get("frames", [])]
        context[unit.unit_id] = {
            "visual_eligible": bool(image_paths),
            "image_paths": image_paths,
            "available_frame_sha256": [
                str(frame.get("sha256") or "") for frame in record.get("frames", [])
            ],
            "transfer_mode": record.get("transfer_mode"),
        }
        record["external_transfer_receipt"] = {
            "status": "not-sent-scene-semantic-trigger-pending",
            "external_transfer": False,
            "pixel_transfer_count": 0,
            "transferred_frame_sha256": [],
            "provider": None,
            "request_id": None,
        }
        rows.append(record)
    return context, rows


def _bind_scene_visual_receipts(
    visual_context_rows: Sequence[dict[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> None:
    for row in visual_context_rows:
        unit_id = str(row.get("unit_id") or "")
        receipt = next(
            (
                value
                for value in receipts
                if str(value.get("call_id") or "").endswith(
                    f".{unit_id}.visual-semantic-observation"
                )
            ),
            None,
        )
        if receipt is None:
            metadata_only = row.get("transfer_mode") == "metadata-only"
            row["external_transfer_receipt"] = {
                "status": (
                    "metadata-only-no-pixel-transfer"
                    if metadata_only
                    else "not-selected-by-scene-visual-trigger"
                ),
                "external_transfer": False,
                "pixel_transfer_count": 0,
                "transferred_frame_sha256": [],
                "provider": None,
                "request_id": None,
            }
            continue
        cache_replay = bool(receipt.get("cache_hit"))
        attachment_hashes = [
            str(item.get("sha256") or "")
            for item in receipt.get("image_attachments", [])
            if isinstance(item, Mapping)
        ]
        row["external_transfer_receipt"] = {
            "status": "cache-replay" if cache_replay else receipt.get("status", "succeeded"),
            "external_transfer": False if cache_replay else bool(attachment_hashes),
            "pixel_transfer_count": 0 if cache_replay else len(attachment_hashes),
            "transferred_frame_sha256": [] if cache_replay else attachment_hashes,
            "cache_origin_frame_sha256": attachment_hashes if cache_replay else [],
            "provider": receipt.get("provider", "codex-cli"),
            "request_id": receipt.get("request_sha256") or receipt.get("call_id"),
        }


def _safe_run_artifact_name(value: object) -> str:
    name = str(value).replace("\\", "/")
    path = Path(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or name.startswith("/")
    ):
        raise ValueError(f"unsafe scene_v2 artifact path: {value!r}")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class _ExistingOutputs:
    output_dir: Path
    qa: Mapping[str, Any]

    @property
    def candidate_units(self) -> int:
        return int(self.qa.get("candidate_units", 0))


def load_japanese_bundle(
    bundle_dir: Path,
    *,
    media_sha256: str,
    expected_media: Path | None = None,
) -> JapaneseSubtitleBundle:
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    required = (
        "source_faithful_ja.srt",
        "viewer_ja.srt",
        "transcript_ja.jsonl",
        "subtitle_audit.jsonl",
        "qc_report.json",
        "comparison_report.html",
        "review_manifest.json",
        "review_report.html",
        "bundle_verification.json",
        "source_media.json",
    )
    missing = [name for name in required if not (bundle_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Japanese bundle is incomplete: {missing}")
    stored_verification = _read_json(bundle_dir / "bundle_verification.json")
    review_manifest = _read_json(bundle_dir / "review_manifest.json")
    windows = review_manifest.get("windows", [])
    expected_windows = len(windows) if isinstance(windows, list) else 0
    from subtitle_pipeline.verification import verify_artifact_bundle

    verification = verify_artifact_bundle(
        bundle_dir,
        expected_video_path=Path(expected_media).resolve() if expected_media else None,
        expected_review_windows=expected_windows,
        expected_media_sha256=media_sha256,
        require_media_binding=True,
    )
    if not verification.get("valid"):
        raise BundleBlockedError(
            "Japanese bundle failed independent verification: "
            + ", ".join(str(value) for value in verification.get("errors", []))
        )
    for field in ("valid", "accepted", "quality_gate_status"):
        if stored_verification.get(field) != verification.get(field):
            raise BundleBlockedError(
                f"Japanese bundle stored verification is stale for field {field}"
            )
    qc = _read_json(bundle_dir / "qc_report.json")
    reasons = _reason_codes(verification.get("quality_gate_reasons", []))
    errors = [str(value) for value in verification.get("errors", [])]
    structural = GateResult(
        "failed" if any("structur" in value.lower() for value in errors) else "passed",
        tuple(value for value in errors if "structur" in value.lower()),
    )
    recognition_reasons = tuple(
        code
        for code in reasons
        if any(
            token in code
            for token in (
                "repetition",
                "halluc",
                "asr",
                "confidence",
                "language",
                "compression",
            )
        )
    )
    alignment_reasons = tuple(code for code in reasons if any(token in code for token in ("alignment", "word_text", "engine_disagreement")))
    presentation_reasons = tuple(
        code for code in reasons if code not in set(recognition_reasons) | set(alignment_reasons)
    )
    gate_status = str(verification.get("quality_gate_status") or qc.get("content_quality_gate", {}).get("status") or "failed")
    recognition = _split_gate(gate_status, recognition_reasons)
    alignment = _split_gate(gate_status, alignment_reasons)
    presentation = _split_gate(gate_status, presentation_reasons)
    artifacts = {name: bundle_dir / name for name in required}
    return JapaneseSubtitleBundle(
        run_version=PROCESS_SCHEMA_VERSION,
        media_sha256=media_sha256,
        backend=str(qc.get("backend") or "unknown"),
        artifacts=artifacts,
        valid=bool(verification.get("valid")),
        accepted=bool(verification.get("accepted")),
        structural=structural,
        recognition=recognition,
        alignment=alignment,
        presentation=presentation,
    )


def _split_gate(overall: str, reasons: tuple[str, ...]) -> GateResult:
    if not reasons:
        return GateResult()
    return GateResult("failed" if overall == "failed" else "review_required", reasons)


def _reason_codes(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    result = []
    for value in values:
        if isinstance(value, Mapping):
            result.append(str(value.get("code") or "unknown"))
        else:
            result.append(str(value))
    return tuple(result)


def _bundle_manifest(bundle: JapaneseSubtitleBundle, *, run_root: Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    artifact_paths: dict[str, str] = {}
    for name, path_value in bundle.artifacts.items():
        path = Path(path_value).resolve()
        try:
            artifact_paths[name] = str(path.relative_to(run_root))
        except ValueError:
            artifact_paths[name] = str(path)
    return {
        "schema_name": "translation-forensics/japanese-subtitle-bundle",
        "schema_version": PROCESS_SCHEMA_VERSION,
        "run_version": bundle.run_version,
        "media_sha256": bundle.media_sha256,
        "backend": bundle.backend,
        "valid": bundle.valid,
        "accepted": bundle.accepted,
        "artifacts": artifact_paths,
        "artifact_sha256": {name: stream_sha256(Path(path)) for name, path in bundle.artifacts.items()},
        "gates": {
            name: asdict(getattr(bundle, name))
            for name in ("structural", "recognition", "alignment", "presentation")
        },
        "disposition": asdict(evaluate_bundle(bundle)),
    }


def _bundle_from_manifest(
    manifest: Mapping[str, Any], *, base_dir: Path, expected_media_sha256: str
) -> JapaneseSubtitleBundle:
    gates = manifest["gates"]
    artifacts = {
        name: (Path(path) if Path(path).is_absolute() else Path(base_dir) / Path(path)).resolve()
        for name, path in manifest["artifacts"].items()
    }
    bundle = JapaneseSubtitleBundle(
        run_version=str(manifest["run_version"]),
        media_sha256=str(manifest["media_sha256"]),
        backend=str(manifest["backend"]),
        artifacts=artifacts,
        valid=bool(manifest["valid"]),
        accepted=bool(manifest["accepted"]),
        structural=GateResult(**gates["structural"]),
        recognition=GateResult(**gates["recognition"]),
        alignment=GateResult(**gates["alignment"]),
        presentation=GateResult(**gates["presentation"]),
    )
    if bundle.media_sha256 != expected_media_sha256:
        raise ValueError("Japanese bundle media SHA-256 changed during resume")
    for name, path in bundle.artifacts.items():
        if not Path(path).is_file() or stream_sha256(Path(path)) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Japanese bundle artifact changed during resume: {name}")
    source_media = _read_json(Path(bundle.artifacts["source_media.json"]))
    if source_media.get("sha256") != expected_media_sha256:
        raise ValueError("Japanese bundle source_media binding changed during resume")
    return bundle


def _load_translation_inputs(root: Path) -> TranslationInputArtifacts:
    units = load_translation_units(root / "translation_units_ja.jsonl")
    blocks, _, _ = parse_srt(root / "translation_ja.srt")
    if len(blocks) != len(units):
        raise ValueError("resumed translation input has mismatched SRT/unit coverage")
    exact = normalize_whitespace("".join(unit.text_raw for unit in units)) == normalize_whitespace(
        "".join(block.text for block in blocks)
    )
    if not exact:
        raise ValueError("resumed translation_ja no longer matches transcript text_raw")
    for index, (unit, block) in enumerate(zip(units, blocks), 1):
        if block.number != index or abs(block.start_seconds - unit.start) > 0.0015 or abs(block.end_seconds - unit.end) > 0.0015:
            raise ValueError(f"resumed translation input timing mismatch at unit {unit.unit_id}")
    return TranslationInputArtifacts(
        units=units,
        translation_ja_srt=root / "translation_ja.srt",
        translation_units_ja_jsonl=root / "translation_units_ja.jsonl",
        exact_after_whitespace_normalization=exact,
    )


def _unit_for_model(unit: TranslationUnit) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "start": unit.start,
        "end": unit.end,
        "source_japanese": unit.text_raw,
        "source_segment_ids": list(unit.source_segment_ids),
        "words": [dict(value) for value in unit.words],
        "asr_warnings": list(unit.asr_warnings),
        "quality_status": unit.quality_status,
        "evidence_ids": list(unit.evidence_ids),
        "speaker": unit.speaker,
        "source_evidence": dict(unit.source_evidence),
    }


def _visual_slots(source: str, uncertain: Iterable[str]) -> set[str]:
    slots = {str(value) for value in uncertain if str(value) in ALLOWED_VISUAL_SLOTS}
    if _DEICTIC_RE.search(source):
        slots.add("deictic_location")
    if _ADDRESSEE_RE.search(source):
        slots.update(("speaker", "addressee"))
    if _SCREEN_TEXT_RE.search(source):
        slots.add("on_screen_text")
    if _CONTINUITY_RE.search(source):
        slots.add("scene_continuity")
    return slots


def _visual_candidate_ids(
    decisions: list[dict[str, Any]], units: Iterable[TranslationUnit], limit: int
) -> list[str]:
    unit_map = {unit.unit_id: unit for unit in units}
    candidates: list[tuple[tuple[int, int, int], str]] = []
    for index, decision in enumerate(decisions):
        unit_id = str(decision["unit_id"])
        unit = unit_map[unit_id]
        slots = _visual_slots(unit.text_raw, decision.get("uncertain_slots", []))
        ambiguous = (
            decision.get("confidence") != "high"
            or bool(decision.get("uncertain_slots"))
            or bool(decision.get("review_required_reasons"))
            or bool(decision.get("critic_required"))
            or unit.quality_status != "trusted"
        )
        if not ambiguous:
            continue
        priority = (
            0 if decision.get("confidence") == "low" else 1,
            0 if unit.quality_status != "trusted" else 1,
            index,
        )
        candidates.append((priority, unit_id))
    return [unit_id for _, unit_id in sorted(candidates)[:limit]]


def _visual_selection_reasons(
    decision: Mapping[str, Any], unit: TranslationUnit, slots: Iterable[str]
) -> list[str]:
    reasons: list[str] = []
    if decision.get("confidence") != "high":
        reasons.append(f"translation_confidence_{decision.get('confidence') or 'unknown'}")
    if decision.get("uncertain_slots"):
        reasons.append("translation_uncertain_slots")
    if decision.get("review_required_reasons"):
        reasons.append("translation_review_required")
    if decision.get("critic_required"):
        reasons.append("translation_semantic_critic_required")
    if unit.quality_status != "trusted":
        reasons.append(f"source_{unit.quality_status}")
    reasons.extend(f"visual_slot_{slot}" for slot in sorted(set(slots)))
    return reasons


def _extract_unit_frames(
    media: Path, output_dir: Path, unit: TranslationUnit, max_frames: int
) -> list[VisualFrame]:
    duration = unit.end - unit.start
    midpoint = unit.start + duration / 2
    end_epsilon = min(0.05, duration / 2)
    safe_end = max(unit.start, unit.end - end_epsilon)
    anchors = [midpoint]
    if max_frames >= 2:
        anchors = [unit.start, safe_end]
        if max_frames >= 3:
            anchors = [unit.start, midpoint, safe_end]
    anchors = list(dict.fromkeys(round(value, 6) for value in anchors))
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[VisualFrame] = []
    for index, seconds in enumerate(anchors, 1):
        path = output_dir / f"frame_{index:02d}_{seconds:.3f}s.jpg"
        if not _is_complete_jpeg(path):
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial.jpg")
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        f"{seconds:.3f}",
                        "-i",
                        str(media),
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        str(temporary),
                    ],
                    check=True,
                    timeout=60,
                )
                if not _is_complete_jpeg(temporary):
                    raise ValueError(f"ffmpeg produced an invalid JPEG frame: {temporary}")
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        frames.append(VisualFrame(path.resolve(), seconds, sha256_file(path)))
    return frames


def _is_complete_jpeg(path: Path) -> bool:
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 4:
        return False
    with path.open("rb") as handle:
        start = handle.read(2)
        handle.seek(-2, os.SEEK_END)
        end = handle.read(2)
    return start == b"\xff\xd8" and end == b"\xff\xd9"


def _ensure_capture_index(
    stage: Path, visual_context_rows: Iterable[Mapping[str, Any]], *, resume: bool
) -> None:
    """Persist generated visual frames in the same auditable index as legacy captures."""

    path = stage / "capture_index.jsonl"
    current = _read_jsonl(path) if path.is_file() else []
    merged: dict[str, dict[str, Any]] = {}
    for row in current:
        frame_path = str(row.get("path", ""))
        digest = str(row.get("sha256", ""))
        if not frame_path or not digest:
            raise ValueError(f"invalid capture index row: {path}")
        merged[digest] = dict(row)
    for context in visual_context_rows:
        for frame in context.get("frames", []):
            if not isinstance(frame, Mapping):
                raise ValueError("visual context frame must be an object")
            frame_path = str(frame.get("path", ""))
            digest = str(frame.get("sha256", ""))
            if not frame_path or not digest:
                raise ValueError("generated visual frame is missing path or SHA-256")
            source = "auto-generated" if "generated-frames" in frame_path.replace("\\", "/") else "legacy"
            merged.setdefault(
                digest,
                {
                    "path": frame_path,
                    "timestamp_seconds": float(frame.get("timestamp_seconds", 0.0)),
                    "sha256": digest,
                    "source": source,
                },
            )
    rows = sorted(
        merged.values(),
        key=lambda row: (float(row.get("timestamp_seconds", 0.0)), str(row.get("path", "")).casefold()),
    )
    if current == rows and path.is_file():
        return
    if current and resume:
        raise ValueError(f"capture index changed while resuming: {path}")
    _write_jsonl_atomic(path, rows)


def _make_generated_frame_paths_portable(
    stage: Path,
    visual_records: Iterable[dict[str, Any]],
    receipts: Iterable[dict[str, Any]],
) -> None:
    """Replace paths inside the staged run with run-relative portable paths."""

    stage = Path(stage).resolve()

    def portable(value: object) -> str:
        path = Path(str(value)).expanduser()
        try:
            return path.resolve().relative_to(stage).as_posix()
        except ValueError:
            return str(path.resolve())

    for record in visual_records:
        for frame in record.get("frames", []):
            if isinstance(frame, dict) and frame.get("path"):
                frame["path"] = portable(frame["path"])
    for receipt in receipts:
        for attachment in receipt.get("image_attachments", []):
            if isinstance(attachment, dict) and attachment.get("path"):
                attachment["path"] = portable(attachment["path"])


def _attach_automated_render_hashes(
    units: Sequence[TranslationUnit],
    decisions: Iterable[Mapping[str, Any]],
    output_dir: Path,
    records: Iterable[Mapping[str, Any]],
    *,
    resume: bool,
) -> list[dict[str, Any]]:
    """Bind the selected rendered SRT text back to its audit record."""

    faithful_blocks, _, _ = parse_srt(Path(output_dir) / "source_faithful_ko.srt")
    natural_blocks, _, _ = parse_srt(Path(output_dir) / "viewer_natural_ko.srt")
    complete_blocks, _, _ = parse_srt(Path(output_dir) / "viewer_complete_ko.srt")
    if not (len(faithful_blocks) == len(natural_blocks) == len(complete_blocks) == len(units)):
        raise BundleBlockedError("automated quality cannot bind rendered SRT hashes")
    record_by_id = {str(row["unit_id"]): dict(row) for row in records}
    if set(record_by_id) != {unit.unit_id for unit in units}:
        raise BundleBlockedError("automated quality record coverage mismatch before render binding")
    rendered: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        record = record_by_id[unit.unit_id]
        source_text = faithful_blocks[index].text
        natural_text = natural_blocks[index].text
        complete_text = complete_blocks[index].text
        # The semantic audit is non-destructive. Even when it flags a meaning
        # concern, its record must bind the displayed viewer-natural subtitle,
        # not silently substitute the source-faithful baseline.
        selected_variant = "viewer-natural"
        selected_text = natural_text
        record.update(
            {
                "source_faithful_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
                "viewer_natural_sha256": hashlib.sha256(natural_text.encode("utf-8")).hexdigest(),
                "viewer_complete_sha256": hashlib.sha256(complete_text.encode("utf-8")).hexdigest(),
                "selected_variant": selected_variant,
                "selected_sha256": hashlib.sha256(selected_text.encode("utf-8")).hexdigest(),
                "render_binding": "unit_order_and_unit_id",
            }
        )
        rendered.append(record)
    path = Path(output_dir).parent / "automated_quality.jsonl"
    existing = _read_jsonl(path) if path.is_file() else []
    if existing and existing != rendered and any("selected_sha256" in row for row in existing):
        if resume:
            raise BundleBlockedError("automated quality render hash binding changed during resume")
    _write_jsonl_atomic(path, rendered)
    return rendered


def _package_translations(
    units: Iterable[TranslationUnit],
    decisions: list[dict[str, Any]],
    bundle: JapaneseSubtitleBundle,
    review_decisions: Mapping[str, Any],
    *,
    automated_quality: bool = False,
) -> list[UnitTranslation]:
    units_by_id = {unit.unit_id: unit for unit in units}
    result: list[UnitTranslation] = []
    for decision in decisions:
        unit_id = str(decision["unit_id"])
        unit = units_by_id[unit_id]
        human_approved = getattr(review_decisions.get(unit_id), "status", None) == "approved"
        clean_machine_candidate = (
            evaluate_bundle(bundle).candidate_generation_allowed
            and unit.quality_status == "trusted"
            and decision.get("confidence") != "low"
            and not decision.get("review_required_reasons")
            and not decision.get("visual_critical")
        )
        automated_evidence = (
            (str(decision.get("automated_quality_evidence_id")),)
            if automated_quality and decision.get("automated_quality_evidence_id")
            else ()
        )
        evidence = (
            tuple(unit.evidence_ids) + automated_evidence
            if automated_quality and decision.get("automated_quality_status") in {"passed", "fallback"}
            else unit.evidence_ids if clean_machine_candidate or human_approved else ()
        )
        if automated_quality and decision.get("automated_quality_status") == "fallback":
            # A fallback is review metadata, not a reason to erase a complete
            # Japanese-to-Korean draft. Preserve each renderable model field and
            # reserve the marker for an actually empty or Japanese-residual field.
            source_candidate = str(decision.get("source_faithful_korean") or "").strip()
            viewer_candidate = str(decision.get("viewer_natural_korean") or "").strip()
            source_faithful = (
                source_candidate
                if source_candidate and not has_japanese(source_candidate)
                else "[원문 불명확]"
            )
            viewer_natural = (
                viewer_candidate
                if viewer_candidate and not has_japanese(viewer_candidate)
                else source_faithful
            )
            if source_faithful == "[원문 불명확]":
                evidence = ()
        else:
            source_faithful = str(decision["source_faithful_korean"])
            viewer_natural = str(decision["viewer_natural_korean"])
        result.append(
            UnitTranslation(
                unit_id=unit_id,
                viewer_complete_ko=viewer_natural,
                source_faithful_ko=source_faithful,
                viewer_natural_ko=viewer_natural,
                evidence_ids=tuple(evidence),
            )
        )
    return result


def _review_queue(
    units: Iterable[TranslationUnit],
    decisions: list[dict[str, Any]],
    bundle: JapaneseSubtitleBundle,
    review_decisions: Mapping[str, Any],
) -> list[dict[str, Any]]:
    unit_map = {unit.unit_id: unit for unit in units}
    result: list[dict[str, Any]] = []
    for decision in decisions:
        unit_id = str(decision["unit_id"])
        unit = unit_map[unit_id]
        reviewed = getattr(review_decisions.get(unit_id), "status", "pending")
        reasons = list(decision.get("review_required_reasons", []))
        if not bundle.accepted:
            reasons.append("japanese_bundle_not_accepted")
        if bundle.recognition.status == "review_required":
            reasons.append("japanese_recognition_review_required")
        if bundle.alignment.status == "review_required":
            reasons.append("japanese_alignment_review_required")
        if unit.quality_status != "trusted":
            reasons.append(f"source_{unit.quality_status}")
        if decision.get("confidence") == "low":
            reasons.append("translation_low_confidence")
        if decision.get("visual_critical"):
            reasons.append("critical_visual_context")
        reasons = list(dict.fromkeys(reasons))
        if reasons and reviewed != "approved":
            result.append(
                {
                    "unit_id": unit_id,
                    "start": unit.start,
                    "end": unit.end,
                    "source_japanese": unit.text_raw,
                    "source_quality_status": unit.quality_status,
                    "translation_confidence": decision.get("confidence"),
                    "reasons": reasons,
                    "review_status": reviewed,
                }
            )
    return result


def _integrated_qa(
    inputs: TranslationInputArtifacts,
    decisions: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    bundle: JapaneseSubtitleBundle,
    output_dir: Path,
    *,
    source_repeat: tuple[int, str],
    review_queue: list[dict[str, Any]],
    quality_policy: str = "legacy",
    translation_architecture: str = "block_v1",
    automated_quality_records: Iterable[Mapping[str, Any]] = (),
    visual_context_records: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    errors: list[str] = []
    unit_ids = [unit.unit_id for unit in inputs.units]
    decision_ids = [str(row.get("unit_id") or "") for row in decisions]
    if decision_ids != unit_ids:
        errors.append("translation_decision_order_or_coverage_mismatch")
    outputs = {}
    for name in ("viewer_complete_ko.srt", "source_faithful_ko.srt", "viewer_natural_ko.srt"):
        path = output_dir / name
        blocks, _, _ = parse_srt(path)
        outputs[name] = {
            "blocks": len(blocks),
            "empty": sum(not block.text.strip() for block in blocks),
            "japanese_residual": sum(has_japanese(block.text) for block in blocks),
            "ellipsis_only": sum(bool(re.fullmatch(r"[.…\s]+", block.text)) for block in blocks),
            "hold_markers": sum("검수 보류" in block.text for block in blocks),
            "maximum_consecutive_identical": _maximum_consecutive_text(block.text for block in blocks)[0],
        }
        if len(blocks) != len(inputs.units) or outputs[name]["empty"]:
            errors.append(f"invalid_output_coverage:{name}")
        for index, (block, unit) in enumerate(zip(blocks, inputs.units), 1):
            if block.number != index:
                errors.append(f"output_number_mismatch:{name}:{unit.unit_id}")
            if (
                abs(block.start_seconds - unit.start) > 0.001
                or abs(block.end_seconds - unit.end) > 0.001
            ):
                errors.append(f"output_timing_mismatch:{name}:{unit.unit_id}")
    if outputs["viewer_complete_ko.srt"]["ellipsis_only"]:
        errors.append("viewer_complete_contains_ellipsis_only_units")
    if outputs["viewer_complete_ko.srt"]["japanese_residual"]:
        errors.append("viewer_complete_contains_japanese_residual")
    automated_records = [dict(row) for row in automated_quality_records]
    automated_statuses = {str(row.get("status") or "") for row in automated_records}
    semantic_audit_statuses = {
        str(row.get("semantic_audit_status") or "not-selected")
        for row in automated_records
    }
    semantic_audits_complete = bool(automated_records) and all(
        row.get("semantic_audit_status") in {"pass", "not-selected"}
        for row in automated_records
    )
    if quality_policy == "automated":
        automated_ids = [str(row.get("unit_id") or "") for row in automated_records]
        expected_ids = [unit.unit_id for unit in inputs.units]
        if automated_ids != expected_ids or len(set(automated_ids)) != len(automated_ids):
            errors.append("automated_quality_coverage_mismatch")
        if not automated_statuses.issubset({"passed", "fallback"}) or not automated_statuses:
            errors.append("automated_quality_status_invalid")
        for name, metrics in outputs.items():
            if metrics["hold_markers"]:
                errors.append(f"automated_output_contains_human_hold_marker:{name}")
            if metrics["japanese_residual"]:
                errors.append(f"automated_output_contains_japanese_residual:{name}")
        faithful_blocks, _, _ = parse_srt(output_dir / "source_faithful_ko.srt")
        natural_blocks, _, _ = parse_srt(output_dir / "viewer_natural_ko.srt")
        complete_blocks, _, _ = parse_srt(output_dir / "viewer_complete_ko.srt")
        for index, unit in enumerate(inputs.units):
            record = next((row for row in automated_records if str(row.get("unit_id")) == unit.unit_id), None)
            if record is None:
                continue
            selected_text = natural_blocks[index].text
            expected_hashes = {
                "source_faithful_sha256": hashlib.sha256(faithful_blocks[index].text.encode("utf-8")).hexdigest(),
                "viewer_natural_sha256": hashlib.sha256(natural_blocks[index].text.encode("utf-8")).hexdigest(),
                "viewer_complete_sha256": hashlib.sha256(complete_blocks[index].text.encode("utf-8")).hexdigest(),
                "selected_sha256": hashlib.sha256(selected_text.encode("utf-8")).hexdigest(),
            }
            if any(record.get(key) != value for key, value in expected_hashes.items()):
                errors.append(f"automated_quality_render_hash_mismatch:{unit.unit_id}")
            if record.get("render_binding") != "unit_order_and_unit_id":
                errors.append(f"automated_quality_render_binding_missing:{unit.unit_id}")
        dual_qa = _read_json(output_dir / "translation_qa.json")
        if automated_statuses == {"passed"} and int(dual_qa.get("candidate_units", -1)) != len(inputs.units):
            errors.append("machine_verified_candidate_coverage_mismatch")
    for record in visual_context_records:
        frames = record.get("frames", [])
        indexed_hashes = [
            str(frame.get("sha256") or "")
            for frame in frames
            if isinstance(frame, Mapping)
        ]
        transfer = record.get("external_transfer_receipt", {})
        if not isinstance(transfer, Mapping):
            errors.append(f"invalid_visual_transfer_receipt:{record.get('unit_id')}")
            continue
        transferred = [str(value or "") for value in transfer.get("transferred_frame_sha256", [])]
        if transfer.get("external_transfer") is True and transferred != indexed_hashes:
            errors.append(f"visual_frame_receipt_hash_mismatch:{record.get('unit_id')}")
        if transfer.get("external_transfer") is not True and int(transfer.get("pixel_transfer_count", 0) or 0):
            errors.append(f"visual_nontransfer_has_pixel_count:{record.get('unit_id')}")
    receipt_errors = []
    receipts_by_call_id: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        if receipt.get("schema_name") != "translation-forensics/codex-model-call-receipt":
            receipt_errors.append("unsupported model-call receipt schema")
            continue
        receipt_errors.extend(validate_call_receipt(receipt))
        call_id = str(receipt.get("call_id") or "")
        if not call_id:
            receipt_errors.append("model-call receipt lacks call_id")
        elif call_id in receipts_by_call_id:
            receipt_errors.append(f"duplicate model-call receipt:{call_id}")
        else:
            receipts_by_call_id[call_id] = receipt
    expected_calls: dict[str, str] = {}
    if translation_architecture == "scene_v2":
        required_roles = {
            "meaning-frame-terra",
            "translation-terra",
            "translation-audit-sol",
            "dialogue-critic-sol",
        }
        observed_roles = {str(receipt.get("role") or "") for receipt in receipts}
        for role in sorted(required_roles - observed_roles):
            receipt_errors.append(f"missing scene_v2 model-call role:{role}")
        for receipt in receipts:
            if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("source_sha256") or "")):
                receipt_errors.append(
                    f"scene_v2 model-call receipt lacks source hash:{receipt.get('call_id')}"
                )
    else:
        for decision in decisions:
            terra_call_id = str(decision.get("terra_call_id") or "")
            if not terra_call_id:
                receipt_errors.append(f"missing terra_call_id:{decision.get('unit_id')}")
            else:
                expected_calls[terra_call_id] = "translation-terra"
            sol_call_id = str(decision.get("sol_call_id") or "")
            blocked_visual = any(
                str(reason).startswith("visual_review_blocked_")
                for reason in decision.get("review_required_reasons", [])
            )
            if sol_call_id and not blocked_visual:
                expected_calls[sol_call_id] = "critique-sol"
            audit_call_id = str(decision.get("automated_quality_call_id") or "")
            if audit_call_id:
                expected_calls[audit_call_id] = "translation-audit-sol"
            visual_repair_call_id = str(decision.get("visual_repair_terra_call_id") or "")
            if visual_repair_call_id:
                expected_calls[visual_repair_call_id] = "translation-terra"
    for call_id, role in expected_calls.items():
        receipt = receipts_by_call_id.get(call_id)
        if receipt is None:
            receipt_errors.append(f"missing model-call receipt:{call_id}")
        elif receipt.get("role") != role:
            receipt_errors.append(f"model-call receipt role mismatch:{call_id}:{role}")
    if receipt_errors:
        errors.append("invalid_model_call_receipt")
    source_questions = sum("?" in unit.text_raw or "？" in unit.text_raw for unit in inputs.units)
    ko_question_marks = sum(
        "?" in str(row.get("viewer_natural_korean") or "") or "？" in str(row.get("viewer_natural_korean") or "")
        for row in decisions
    )
    source_negations = sum(bool(_NEGATION_RE.search(unit.text_raw)) for unit in inputs.units)
    bundle_machine_ready = bundle.accepted and all(
        gate.status == "passed"
        for gate in (bundle.structural, bundle.recognition, bundle.alignment, bundle.presentation)
    )
    return {
        "schema_name": "translation-forensics/integrated-qa",
        "schema_version": PROCESS_SCHEMA_VERSION,
        "verification_passed": not errors,
        "errors": errors,
        "translation_input_exact": inputs.exact_after_whitespace_normalization,
        "total_units": len(inputs.units),
        "source_maximum_consecutive_identical": source_repeat[0],
        "source_maximum_consecutive_text": source_repeat[1],
        "source_repetition_under_100": source_repeat[0] < 100,
        "source_question_units": source_questions,
        "viewer_question_mark_units": ko_question_marks,
        "source_negation_units": source_negations,
        "outputs": outputs,
        "model_call_receipts": len(receipts),
        "receipt_validation_errors": receipt_errors,
        "external_image_transfer_count": sum(
            0
            if receipt.get("cache_hit")
            else int(receipt.get("pixel_external_transfer_count", 0) or 0)
            for receipt in receipts
        ),
        "review_queue_units": len(review_queue),
        "bundle_valid": bundle.valid,
        "bundle_accepted": bundle.accepted,
        "quality_policy": quality_policy,
        "translation_architecture": translation_architecture,
        "automated_quality": {
            "units": len(automated_records),
            "passed": sum(row.get("status") == "passed" for row in automated_records),
            "fallback": sum(row.get("status") == "fallback" for row in automated_records),
            "statuses": sorted(automated_statuses),
            "semantic_audit_scope": "targeted-semantic-only",
            "semantic_audit_statuses": sorted(semantic_audit_statuses),
            "semantic_audit": {
                "passed": sum(
                    row.get("semantic_audit_status") == "pass"
                    for row in automated_records
                ),
                "issues": sum(
                    row.get("semantic_audit_status") == "issue"
                    for row in automated_records
                ),
                "inconclusive": sum(
                    row.get("semantic_audit_status") == "inconclusive"
                    for row in automated_records
                ),
                "not_selected": sum(
                    row.get("semantic_audit_status") == "not-selected"
                    for row in automated_records
                ),
                "blocked": sum(
                    row.get("semantic_audit_status") == "blocked"
                    for row in automated_records
                ),
            },
        },
        "human_reviewed": False,
        "human_final_allowed": False,
        "human_reference_equality": "unidentifiable",
        "100_percent_equal": False,
        "machine_final_allowed": (
            quality_policy == "automated"
            and not errors
            and bool(automated_records)
            and all(row.get("status") == "passed" for row in automated_records)
            and semantic_audits_complete
            and bundle_machine_ready
        ),
        "verification_status": (
            "machine-uncertain"
            if quality_policy == "automated"
            and (
                any(row.get("status") == "fallback" for row in automated_records)
                or not semantic_audits_complete
                or not bundle_machine_ready
            )
            else "machine-verified" if quality_policy == "automated" and not errors
            else "not-demonstrated"
        ),
        "final_promotion_allowed": False,
    }


def _maximum_consecutive_text(values: Iterable[str]) -> tuple[int, str]:
    maximum = 0
    maximum_text = ""
    previous = None
    current = 0
    for value in values:
        normalized = re.sub(r"\s+", "", str(value))
        if normalized and normalized == previous:
            current += 1
        else:
            previous = normalized
            current = 1 if normalized else 0
        if current > maximum:
            maximum = current
            maximum_text = normalized
    return maximum, maximum_text


def _visual_ab_report(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    units = []
    for review in reviews:
        unit_id = str(review["unit_id"])
        left, right = before[unit_id], after[unit_id]
        units.append(
            {
                "unit_id": unit_id,
                "text_changed": (
                    left.get("source_faithful_korean"), left.get("viewer_natural_korean")
                )
                != (right.get("source_faithful_korean"), right.get("viewer_natural_korean")),
                "confidence_before": left.get("confidence"),
                "confidence_after": right.get("confidence"),
                "critical_visual_impact": review.get("critical_visual_impact"),
                "verdict": review.get("verdict"),
            }
        )
    return {
        "schema_name": "translation-forensics/visual-ab-report",
        "schema_version": PROCESS_SCHEMA_VERSION,
        "reviewed_units": len(units),
        "changed_units": sum(row["text_changed"] for row in units),
        "critical_units": sum(bool(row["critical_visual_impact"]) for row in units),
        "units": units,
    }


def _file_record(path: Path, *, known_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": known_sha256 or stream_sha256(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _path_contains_exact_title(title_id: str, path: Path) -> bool:
    parts = Path(path).parts
    matching_indexes = [
        index for index, part in enumerate(parts) if title_code_matches(title_id, part)
    ]
    if not matching_indexes:
        return False
    relevant_suffix = Path(*parts[matching_indexes[0] :])
    return title_code_matches(title_id, relevant_suffix)


def _verify_promoted_run(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("schema_name") != "translation-forensics/integrated-run-manifest"
        or manifest.get("schema_version") != PROCESS_SCHEMA_VERSION
    ):
        raise ValueError("existing integrated run has an unsupported manifest version")
    policy_expectations = {
        "human_reviewed": False,
        "human_final_allowed": False,
        "human_reference_equality": "unidentifiable",
        "100_percent_equal": False,
        "final_promotion_allowed": False,
    }
    for field, expected in policy_expectations.items():
        if manifest.get(field) != expected:
            raise ValueError(f"existing integrated run violates policy field: {field}")
    status = str(manifest.get("status") or "")
    machine_final_allowed = manifest.get("machine_final_allowed")
    if status not in {
        "machine-verified",
        "machine-uncertain",
        "machine-draft-not-demonstrated",
    }:
        raise ValueError("existing integrated run has an invalid status")
    if machine_final_allowed is not (status == "machine-verified"):
        raise ValueError("existing integrated run has inconsistent machine-final status")
    state = _read_json(Path(run_dir) / "run-state.json")
    if (
        state.get("schema_name") != "translation-forensics/integrated-run"
        or state.get("status") != "verified"
        or state.get("run_id") != manifest.get("run_id")
        or state.get("cache_identity") != manifest.get("cache_identity")
        or state.get("manifest_sha256") != stream_sha256(Path(run_dir) / "run_manifest.json")
    ):
        raise ValueError("existing integrated run manifest is not bound to run-state")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("existing integrated run lacks current artifact hashes")
    artifact_names = list(_HASHED_RUN_ARTIFACTS)
    architecture = str(manifest.get("translation_architecture") or "block_v1")
    if architecture == "scene_v2":
        if manifest.get("architecture_status") != "experimental-unbenchmarked":
            raise ValueError("existing scene_v2 run has an invalid architecture status")
        if manifest.get("benchmark_status") != "not-run":
            raise ValueError("existing scene_v2 run has an invalid benchmark status")
        scene_artifacts = manifest.get("scene_artifacts")
        if not isinstance(scene_artifacts, list) or not scene_artifacts:
            raise ValueError("existing scene_v2 run lacks its artifact manifest")
        artifact_names.extend(_safe_run_artifact_name(name) for name in scene_artifacts)
    elif architecture != "block_v1":
        raise ValueError("existing integrated run has an unknown translation architecture")
    if len(set(artifact_names)) != len(artifact_names):
        raise ValueError("existing integrated run has duplicate artifact paths")
    missing_hashes = [name for name in artifact_names if name not in hashes]
    if missing_hashes:
        raise ValueError(f"existing integrated run lacks artifact hashes: {missing_hashes}")
    for name in artifact_names:
        path = Path(run_dir) / name
        expected = str(hashes[name])
        if not path.is_file():
            raise ValueError(f"existing integrated run artifact is missing: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or stream_sha256(path) != expected:
            raise ValueError(f"existing integrated run artifact changed: {name}")
    qa = _read_json(Path(run_dir) / "qa_report.json")
    if qa.get("verification_passed") is not True:
        raise ValueError("existing integrated run no longer has passing QA")
    _verify_promoted_frame_artifacts(Path(run_dir))


def _verify_promoted_frame_artifacts(run_dir: Path) -> None:
    run_dir = Path(run_dir).resolve()
    records = [
        *_read_jsonl(run_dir / "capture_index.jsonl"),
        *(
            frame
            for record in _read_jsonl(run_dir / "visual_context.jsonl")
            for frame in record.get("frames", [])
            if isinstance(frame, Mapping)
        ),
    ]
    checked: set[tuple[str, str]] = set()
    for record in records:
        raw_path = str(record.get("path") or "")
        expected = str(record.get("sha256") or "")
        key = (raw_path, expected)
        if key in checked:
            continue
        checked.add(key)
        path = Path(raw_path)
        if not path.is_absolute():
            path = run_dir / path
        if not path.is_file():
            raise ValueError(f"promoted visual frame is missing: {raw_path}")
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or stream_sha256(path) != expected:
            raise ValueError(f"promoted visual frame changed: {raw_path}")


def _resolve_media_sha256(
    project_root: Path, title_id: str, media: Path
) -> tuple[str, dict[str, Any]]:
    """Bind every run to the current media bytes, never path/mtime metadata alone."""

    del project_root, title_id
    media = Path(media).resolve()
    return stream_sha256(media), {
        "method": "streamed-full-file",
        "manifest_path": None,
        "matched_path": None,
        "matched_size": None,
        "matched_mtime": None,
        "full_hash_recomputed": True,
    }


def _capture_inventory(captures: Path | None, frames: Iterable[VisualFrame]) -> dict[str, Any] | None:
    if captures is None:
        return None
    rows = sorted(
        (
            {
                "path": str(frame.path),
                "timestamp_seconds": frame.timestamp_seconds,
                "sha256": frame.sha256,
            }
            for frame in frames
        ),
        key=lambda row: (float(row["timestamp_seconds"]), str(row["path"]).casefold()),
    )
    aggregate = hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(Path(captures).resolve()),
        "unique_frame_count": len(rows),
        "aggregate_sha256": aggregate,
        "deduplication": "sha256",
    }


def _existing_bundle_identity(bundle_dir: Path) -> str:
    names = (
        "source_faithful_ja.srt",
        "viewer_ja.srt",
        "transcript_ja.jsonl",
        "subtitle_audit.jsonl",
        "qc_report.json",
        "comparison_report.html",
        "review_manifest.json",
        "review_report.html",
        "bundle_verification.json",
        "source_media.json",
    )
    digest = hashlib.sha256()
    for name in names:
        path = Path(bundle_dir) / name
        if not path.is_file():
            raise FileNotFoundError(f"Japanese bundle is incomplete: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(stream_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _code_version(translation_architecture: str = "block_v1") -> str:
    repository_root = Path(__file__).resolve().parents[2]
    roots = [
        Path(__file__),
        Path(__file__).with_name("automated_quality.py"),
        Path(__file__).with_name("codex_exec_provider.py"),
        Path(__file__).with_name("integrated_pipeline.py"),
        Path(__file__).with_name("integrated_translation.py"),
        Path(__file__).with_name("srt.py"),
        Path(__file__).with_name("visual_context.py"),
    ]
    subtitle_root = Path(__file__).parents[1] / "subtitle_pipeline"
    roots.extend(sorted(subtitle_root.glob("*.py")))
    roots.extend(
        repository_root / relative
        for relative in (
            "references/translation-prompt-v6.txt",
            "prompts/batch-adult-srt-translation-run.md",
            "prompts/terra-semantic-translation-v1.md",
            "prompts/integrated-noisy-asr-recovery-v1.md",
        )
    )
    if translation_architecture == "scene_v2":
        roots.extend(sorted((repository_root / "src" / "translation_forensics").glob("scene_*.py")))
        roots.extend(
            repository_root / "src" / "translation_forensics" / name
            for name in ("dialogue_quality.py", "style_memory.py")
        )
        roots.extend(sorted((repository_root / "prompts").glob("scene-*.md")))
        roots.extend(sorted((repository_root / "prompts").glob("scene-*.manifest.json")))
        roots.extend(sorted((repository_root / "schemas").glob("scene-*.schema.json")))
        roots.extend(
            repository_root / relative
            for relative in (
                "prompts/korean-dialogue-critic-v1.md",
                "prompts/korean-dialogue-critic-v1.manifest.json",
                "schemas/korean-dialogue-critic-v1.schema.json",
            )
        )
    digest = hashlib.sha256()
    for path in roots:
        if path.is_file():
            try:
                identity = path.resolve().relative_to(repository_root).as_posix()
            except ValueError:
                identity = str(path.resolve())
            digest.update(identity.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _prompt_contract_inventory(
    translation_architecture: str = "block_v1",
) -> list[dict[str, str]]:
    repository_root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "references/translation-prompt-v6.txt",
        "prompts/batch-adult-srt-translation-run.md",
        "prompts/terra-semantic-translation-v1.md",
        "prompts/integrated-noisy-asr-recovery-v1.md",
    )
    records: list[dict[str, str]] = []
    for relative in relative_paths:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing prompt contract: {path}")
        records.append({"path": relative, "sha256": stream_sha256(path)})
    if translation_architecture == "scene_v2":
        scene_contracts = [
            *sorted((repository_root / "prompts").glob("scene-*.md")),
            *sorted((repository_root / "prompts").glob("scene-*.manifest.json")),
            *sorted((repository_root / "schemas").glob("scene-*.schema.json")),
            repository_root / "prompts" / "korean-dialogue-critic-v1.md",
            repository_root / "prompts" / "korean-dialogue-critic-v1.manifest.json",
            repository_root / "schemas" / "korean-dialogue-critic-v1.schema.json",
        ]
        if not scene_contracts:
            raise FileNotFoundError("missing scene_v2 prompt/schema contracts")
        for path in scene_contracts:
            records.append(
                {
                    "path": path.relative_to(repository_root).as_posix(),
                    "sha256": stream_sha256(path),
                }
            )
    return records


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        result.append(value)
    return result


def _checkpoint_is_valid(
    stage: Path, checkpoint_path: Path, artifact_names: Iterable[str]
) -> bool:
    if not checkpoint_path.is_file():
        return False
    try:
        checkpoint = _read_json(checkpoint_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    names = list(artifact_names)
    if (
        checkpoint.get("schema_name") != "translation-forensics/stage-checkpoint"
        or checkpoint.get("schema_version") != PROCESS_SCHEMA_VERSION
        or checkpoint.get("status") != "complete"
        or checkpoint.get("artifacts") != names
    ):
        return False
    hashes = checkpoint.get("artifact_sha256")
    if not isinstance(hashes, Mapping):
        return False
    for name in names:
        path = Path(stage) / name
        expected = str(hashes.get(name) or "")
        if (
            not path.is_file()
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or stream_sha256(path) != expected
        ):
            return False
    return True


def _write_checkpoint(
    stage: Path, checkpoint_path: Path, artifact_names: Iterable[str]
) -> None:
    names = list(artifact_names)
    missing = [name for name in names if not (Path(stage) / name).is_file()]
    if missing:
        raise FileNotFoundError(f"stage checkpoint artifacts are missing: {missing}")
    _write_json_atomic(
        checkpoint_path,
        {
            "schema_name": "translation-forensics/stage-checkpoint",
            "schema_version": PROCESS_SCHEMA_VERSION,
            "status": "complete",
            "artifacts": names,
            "artifact_sha256": {
                name: stream_sha256(Path(stage) / name) for name in names
            },
        },
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl_atomic(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    _write_text_atomic(
        path,
        "".join(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n" for value in values),
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_once_or_match(path: Path, value: Mapping[str, Any], *, resume: bool) -> None:
    if path.exists():
        if not resume or _read_json(path) != dict(value):
            raise FileExistsError(f"refusing to overwrite integration artifact: {path}")
        return
    _write_json_atomic(path, value)


def _write_jsonl_once_or_match(
    path: Path, values: Iterable[Mapping[str, Any]], *, resume: bool
) -> None:
    rows = [dict(value) for value in values]
    if path.exists():
        if not resume or _read_jsonl(path) != rows:
            raise FileExistsError(f"refusing to overwrite integration artifact: {path}")
        return
    _write_jsonl_atomic(path, rows)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "PROCESS_SCHEMA_VERSION",
    "ProcessTitleConfig",
    "choose_subtitle_backend",
    "ensure_normalized_audio",
    "load_japanese_bundle",
    "process_title",
    "stream_sha256",
]
