from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

from .srt import SubtitleBlock, seconds_to_timecode, write_srt


INTEGRATION_SCHEMA_VERSION = "1"
CACHE_SCHEMA_VERSION = "1"
TRANSCRIPT_BASENAME = "transcript_ja.jsonl"
REVIEW_HOLD_TEXT = "[검수 보류]"
MACHINE_UNCERTAIN_TEXT = "[기계 불확실]"

_GATE_STATUSES = frozenset({"passed", "review_required", "failed"})
_UNIT_QUALITY_STATUSES = frozenset({"trusted", "suspect", "unusable"})
_REVIEW_STATUSES = frozenset({"pending", "approved", "rejected", "deferred"})
_TERMINAL_REVIEW_STATUSES = frozenset({"approved", "rejected"})
_EPHEMERAL_CACHE_KEYS = frozenset(
    {
        "temp_wav_path",
        "temporary_wav_path",
        "temporary_audio_path",
        "scratch_wav_path",
    }
)
_UNUSABLE_WARNING_CODES = frozenset(
    {
        "empty_text",
        "non_japanese_reference",
        "possible_runaway_repetition",
        "text_unusable",
        "unusable",
    }
)
_SUSPECT_WARNING_CODES = frozenset(
    {
        "low_asr_confidence",
        "possible_periodic_repetition",
        "possible_repetition",
        "possible_silence_hallucination",
        "qwen_recovery",
        "review_required",
        "timestamp_precision_limited",
        "word_alignment_review_required",
        "word_text_mismatch",
    }
)


class IntegrationPipelineError(ValueError):
    """Base error for deterministic integration-contract violations."""


class BundleBlockedError(IntegrationPipelineError):
    """Raised when Japanese evidence is not safe enough to translate."""


class CoverageError(IntegrationPipelineError):
    """Raised when the complete Korean draft does not cover every unit."""


class ResumeRejected(IntegrationPipelineError):
    """Raised when a partial run or cache cannot be resumed safely."""


@dataclass(slots=True)
class _HeldRunLock:
    handle: BinaryIO
    owner_pid: int
    owner_thread_id: int


_RUN_LOCKS: dict[str, _HeldRunLock] = {}
_RUN_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True, slots=True)
class ExtractRequest:
    """Input contract for the canonical Japanese subtitle extractor."""

    title_id: str
    media: Path
    project_root: Path
    reference_ja: Path | None = None
    reference_ja_approved: bool = False
    backend_policy: str = "auto"
    qwen_runtime: str | None = None
    cache_dir: Path | None = None
    review_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title_id.strip():
            raise IntegrationPipelineError("title_id must not be empty")
        if self.reference_ja_approved and self.reference_ja is None:
            raise IntegrationPipelineError(
                "reference_ja_approved requires an explicit reference_ja path"
            )
        if self.backend_policy not in {"auto", "reference", "ensemble"}:
            raise IntegrationPipelineError(
                "backend_policy must be auto, reference, or ensemble"
            )


@dataclass(frozen=True, slots=True)
class GateResult:
    status: str = "passed"
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in _GATE_STATUSES:
            raise IntegrationPipelineError(f"unsupported gate status: {self.status}")
        if self.status == "passed" and self.reasons:
            raise IntegrationPipelineError("a passed gate cannot contain failure reasons")


@dataclass(frozen=True, slots=True)
class JapaneseSubtitleBundle:
    """Verified Japanese bundle with independent quality gate states."""

    run_version: str
    media_sha256: str
    backend: str
    artifacts: Mapping[str, Path]
    valid: bool = True
    accepted: bool = False
    structural: GateResult = field(default_factory=GateResult)
    recognition: GateResult = field(default_factory=GateResult)
    alignment: GateResult = field(default_factory=GateResult)
    presentation: GateResult = field(default_factory=GateResult)

    def __post_init__(self) -> None:
        _validate_sha256(self.media_sha256, field_name="media_sha256")
        if not self.run_version.strip():
            raise IntegrationPipelineError("run_version must not be empty")
        if not self.backend.strip():
            raise IntegrationPipelineError("backend must not be empty")


@dataclass(frozen=True, slots=True)
class BundleDisposition:
    translation_allowed: bool
    complete_draft_allowed: bool
    candidate_generation_allowed: bool
    release_allowed: bool
    review_required: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TranslationUnit:
    unit_id: str
    start: float
    end: float
    text_raw: str
    source_segment_ids: tuple[int, ...]
    words: tuple[Mapping[str, Any], ...]
    asr_warnings: tuple[str, ...]
    quality_status: str
    evidence_ids: tuple[str, ...]
    speaker: str = "speaker_unknown"

    def __post_init__(self) -> None:
        if self.quality_status not in _UNIT_QUALITY_STATUSES:
            raise IntegrationPipelineError(
                f"unsupported unit quality status: {self.quality_status}"
            )
        if not self.unit_id.strip():
            raise IntegrationPipelineError("unit_id must not be empty")
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise IntegrationPipelineError(f"non-finite timing for {self.unit_id}")
        if self.start < 0 or self.end <= self.start:
            raise IntegrationPipelineError(f"invalid timing for {self.unit_id}")
        if not self.text_raw.strip():
            raise IntegrationPipelineError(f"empty text_raw for {self.unit_id}")

    def json(self) -> dict[str, Any]:
        return {
            "schema_version": INTEGRATION_SCHEMA_VERSION,
            "unit_id": self.unit_id,
            "start": self.start,
            "end": self.end,
            "text_raw": self.text_raw,
            "source_segment_ids": list(self.source_segment_ids),
            "words": [dict(word) for word in self.words],
            "asr_warnings": list(self.asr_warnings),
            "quality_status": self.quality_status,
            "evidence_ids": list(self.evidence_ids),
            "speaker": self.speaker,
        }


@dataclass(frozen=True, slots=True)
class TranslationInputArtifacts:
    units: tuple[TranslationUnit, ...]
    translation_ja_srt: Path
    translation_units_ja_jsonl: Path
    exact_after_whitespace_normalization: bool


@dataclass(frozen=True, slots=True)
class UnitTranslation:
    unit_id: str
    viewer_complete_ko: str
    source_faithful_ko: str | None = None
    viewer_natural_ko: str | None = None
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DualOutputArtifacts:
    viewer_complete_ko_srt: Path
    source_faithful_ko_srt: Path
    viewer_natural_ko_srt: Path
    qa_path: Path
    total_units: int
    candidate_units: int
    held_units: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    unit_id: str
    status: str
    reviewer: str
    reason: str
    previous_status: str
    decided_at: str

    def __post_init__(self) -> None:
        if self.status not in _REVIEW_STATUSES:
            raise IntegrationPipelineError(f"unsupported review status: {self.status}")
        if self.previous_status not in _REVIEW_STATUSES:
            raise IntegrationPipelineError(
                f"unsupported previous review status: {self.previous_status}"
            )

    def json(self) -> dict[str, str]:
        return asdict(self)


def evaluate_bundle(bundle: JapaneseSubtitleBundle) -> BundleDisposition:
    reasons: list[str] = []
    if not bundle.valid:
        reasons.append("bundle_invalid")
    if bundle.structural.status != "passed":
        reasons.append(f"structural_{bundle.structural.status}")
    if bundle.recognition.status == "failed":
        reasons.append("recognition_failed")
    if bundle.alignment.status == "failed":
        reasons.append("alignment_failed")

    translation_allowed = not reasons
    review_required = any(
        gate.status == "review_required"
        for gate in (bundle.recognition, bundle.alignment, bundle.presentation)
    )
    candidate_generation_allowed = (
        translation_allowed
        and bundle.accepted
        and bundle.recognition.status == "passed"
        and bundle.alignment.status == "passed"
    )
    release_allowed = (
        translation_allowed
        and bundle.accepted
        and all(
            gate.status == "passed"
            for gate in (
                bundle.structural,
                bundle.recognition,
                bundle.alignment,
                bundle.presentation,
            )
        )
    )
    if bundle.presentation.status != "passed":
        reasons.append(f"presentation_{bundle.presentation.status}")
    if not bundle.accepted:
        reasons.append("bundle_not_human_accepted")
    return BundleDisposition(
        translation_allowed=translation_allowed,
        complete_draft_allowed=translation_allowed,
        candidate_generation_allowed=candidate_generation_allowed,
        release_allowed=release_allowed,
        review_required=review_required or not bundle.accepted,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def select_extraction_backend(request: ExtractRequest) -> str:
    """Resolve reference versus ensemble without trusting an unapproved SRT."""

    if request.backend_policy == "ensemble":
        return "ensemble"
    if request.reference_ja_approved:
        reference = Path(request.reference_ja)  # guarded by ExtractRequest
        if not reference.is_file():
            raise FileNotFoundError(reference)
        return "reference"
    if request.backend_policy == "reference":
        raise BundleBlockedError(
            "reference backend requires an explicitly approved reference_ja"
        )
    return "ensemble"


def classify_unit_quality(
    warnings: Iterable[str], *, explicit_status: str | None = None
) -> str:
    warning_set = {str(value) for value in warnings}
    inferred = (
        "unusable" if warning_set & _UNUSABLE_WARNING_CODES
        else "suspect" if warning_set & _SUSPECT_WARNING_CODES
        else "trusted"
    )
    if explicit_status is not None:
        if explicit_status not in _UNIT_QUALITY_STATUSES:
            raise IntegrationPipelineError(
                f"unsupported unit quality status: {explicit_status}"
            )
        severity = {"trusted": 0, "suspect": 1, "unusable": 2}
        return max((explicit_status, inferred), key=severity.__getitem__)
    return inferred


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", text, flags=re.UNICODE)


def build_translation_inputs(
    transcript_ja: Path,
    output_dir: Path,
    *,
    bundle: JapaneseSubtitleBundle | None = None,
) -> TranslationInputArtifacts:
    """Create translation inputs exclusively from transcript_ja.jsonl.text_raw."""

    transcript_ja = Path(transcript_ja)
    if transcript_ja.name != TRANSCRIPT_BASENAME:
        raise IntegrationPipelineError(
            f"translation input must be {TRANSCRIPT_BASENAME}; viewer_ja is presentation-only"
        )
    if bundle is not None:
        disposition = evaluate_bundle(bundle)
        if not disposition.translation_allowed:
            raise BundleBlockedError(
                "Japanese bundle cannot be translated: " + ", ".join(disposition.reasons)
            )
        declared = bundle.artifacts.get(TRANSCRIPT_BASENAME)
        if declared is not None and Path(declared).resolve() != transcript_ja.resolve():
            raise IntegrationPipelineError(
                "transcript_ja does not match the bundle's declared canonical transcript"
            )
    if not transcript_ja.is_file():
        raise FileNotFoundError(transcript_ja)

    units: list[TranslationUnit] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        transcript_ja.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise IntegrationPipelineError(
                f"invalid transcript JSON at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise IntegrationPipelineError(
                f"transcript line {line_number} must be a JSON object"
            )
        if "text_raw" not in row:
            raise IntegrationPipelineError(
                f"transcript line {line_number} has no text_raw"
            )
        text_raw = row["text_raw"]
        if not isinstance(text_raw, str):
            raise IntegrationPipelineError(
                f"transcript line {line_number} text_raw must be a string"
            )
        unit_id = str(row.get("utterance_id") or f"unit_{line_number:06d}")
        if unit_id in seen_ids:
            raise IntegrationPipelineError(f"duplicate unit_id: {unit_id}")
        seen_ids.add(unit_id)
        warnings = tuple(str(value) for value in row.get("warnings", []))
        evidence_values = row.get("evidence_ids") or row.get("evidence_refs")
        if evidence_values is None:
            evidence_values = [f"{TRANSCRIPT_BASENAME}#L{line_number}"]
        if not isinstance(evidence_values, list):
            raise IntegrationPipelineError(
                f"evidence IDs at line {line_number} must be a list"
            )
        source_ids = row.get("source_segment_ids", [])
        words = row.get("words", [])
        if not isinstance(source_ids, list) or not isinstance(words, list):
            raise IntegrationPipelineError(
                f"source_segment_ids and words at line {line_number} must be lists"
            )
        if any(not isinstance(word, dict) for word in words):
            raise IntegrationPipelineError(f"word entries at line {line_number} must be objects")
        unit = TranslationUnit(
            unit_id=unit_id,
            start=float(row["start"]),
            end=float(row["end"]),
            text_raw=text_raw,
            source_segment_ids=tuple(int(value) for value in source_ids),
            words=tuple(dict(word) for word in words),
            asr_warnings=warnings,
            quality_status=classify_unit_quality(
                warnings, explicit_status=row.get("quality_status")
            ),
            evidence_ids=tuple(str(value) for value in evidence_values),
            speaker=str(row.get("speaker") or "speaker_unknown"),
        )
        units.append(unit)
    if not units:
        raise IntegrationPipelineError("transcript_ja contains no translation units")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    srt_path = output_dir / "translation_ja.srt"
    units_path = output_dir / "translation_units_ja.jsonl"
    _ensure_new_paths((srt_path, units_path))

    blocks = [_block_for_unit(index, unit, unit.text_raw) for index, unit in enumerate(units, 1)]
    source_signature = normalize_whitespace("".join(unit.text_raw for unit in units))
    rendered_signature = normalize_whitespace("".join(block.text for block in blocks))
    exact = source_signature == rendered_signature
    if not exact:
        raise IntegrationPipelineError(
            "translation_ja content differs from transcript_ja.text_raw"
        )

    write_translation_units(units_path, units)
    _atomic_write_srt(srt_path, blocks)
    return TranslationInputArtifacts(
        units=tuple(units),
        translation_ja_srt=srt_path,
        translation_units_ja_jsonl=units_path,
        exact_after_whitespace_normalization=exact,
    )


def write_translation_units(
    path: Path,
    units: Sequence[TranslationUnit],
    *,
    overwrite: bool = False,
) -> Path:
    """Write the canonical unit contract as UTF-8 JSONL."""

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite translation units: {path}")
    unit_ids = [unit.unit_id for unit in units]
    if not units:
        raise IntegrationPipelineError("translation units must not be empty")
    if len(set(unit_ids)) != len(unit_ids):
        raise IntegrationPipelineError("translation units contain duplicate unit IDs")
    _atomic_write_text(
        path,
        "".join(
            json.dumps(unit.json(), ensure_ascii=False, sort_keys=True) + "\n"
            for unit in units
        ),
    )
    return path


def load_translation_units(path: Path) -> tuple[TranslationUnit, ...]:
    """Load and validate a current-version translation_units_ja.jsonl file."""

    path = Path(path)
    units: list[TranslationUnit] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("row is not an object")
            if row.get("schema_version") != INTEGRATION_SCHEMA_VERSION:
                raise ResumeRejected(
                    f"translation unit schema is not current at line {line_number}"
                )
            unit = TranslationUnit(
                unit_id=str(row["unit_id"]),
                start=float(row["start"]),
                end=float(row["end"]),
                text_raw=str(row["text_raw"]),
                source_segment_ids=tuple(int(value) for value in row["source_segment_ids"]),
                words=tuple(dict(word) for word in row["words"]),
                asr_warnings=tuple(str(value) for value in row["asr_warnings"]),
                quality_status=str(row["quality_status"]),
                evidence_ids=tuple(str(value) for value in row["evidence_ids"]),
                speaker=str(row.get("speaker") or "speaker_unknown"),
            )
        except ResumeRejected:
            raise
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise IntegrationPipelineError(
                f"invalid translation unit at line {line_number}"
            ) from exc
        if unit.unit_id in seen_ids:
            raise IntegrationPipelineError(f"duplicate unit_id: {unit.unit_id}")
        seen_ids.add(unit.unit_id)
        units.append(unit)
    if not units:
        raise IntegrationPipelineError("translation units must not be empty")
    return tuple(units)


def transition_review_decision(
    current: ReviewDecision | str | None,
    *,
    unit_id: str,
    status: str,
    reviewer: str,
    reason: str,
    decided_at: str | None = None,
) -> ReviewDecision:
    status = {"held": "deferred", "hold": "deferred"}.get(status, status)
    if status not in _REVIEW_STATUSES:
        raise IntegrationPipelineError(f"unsupported review status: {status}")
    previous = current.status if isinstance(current, ReviewDecision) else (current or "pending")
    previous = {"held": "deferred", "hold": "deferred"}.get(previous, previous)
    if previous not in _REVIEW_STATUSES:
        raise IntegrationPipelineError(f"unsupported previous review status: {previous}")
    if previous in _TERMINAL_REVIEW_STATUSES and status != previous:
        raise IntegrationPipelineError(
            f"review decision is terminal: {previous} -> {status}"
        )
    if previous == "pending" and status == "pending" and current is not None:
        raise IntegrationPipelineError("pending -> pending is not a state transition")
    if previous == "deferred" and status not in {"deferred", "approved", "rejected"}:
        raise IntegrationPipelineError(f"invalid review transition: {previous} -> {status}")
    if not unit_id.strip() or not reviewer.strip() or not reason.strip():
        raise IntegrationPipelineError("unit_id, reviewer, and reason are required")
    return ReviewDecision(
        unit_id=unit_id,
        status=status,
        reviewer=reviewer,
        reason=reason,
        previous_status=previous,
        decided_at=decided_at or _utc_now(),
    )


def load_review_decisions(path: Path) -> dict[str, ReviewDecision]:
    latest: dict[str, ReviewDecision] = {}
    path = Path(path)
    if not path.exists():
        return latest
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            decision = ReviewDecision(**row)
        except (json.JSONDecodeError, TypeError, IntegrationPipelineError) as exc:
            raise IntegrationPipelineError(
                f"invalid review decision at line {line_number}"
            ) from exc
        current = latest.get(decision.unit_id)
        expected_previous = current.status if current else "pending"
        if decision.previous_status != expected_previous:
            raise IntegrationPipelineError(
                f"broken review history for {decision.unit_id} at line {line_number}"
            )
        if current and current.status in _TERMINAL_REVIEW_STATUSES and decision.status != current.status:
            raise IntegrationPipelineError(
                f"terminal review decision changed for {decision.unit_id}"
            )
        latest[decision.unit_id] = decision
    return latest


def append_review_decision(path: Path, decision: ReviewDecision) -> None:
    path = Path(path)
    latest = load_review_decisions(path)
    expected_previous = latest.get(decision.unit_id)
    expected_status = expected_previous.status if expected_previous else "pending"
    if decision.previous_status != expected_status:
        raise IntegrationPipelineError(
            f"stale review transition for {decision.unit_id}: expected {expected_status}"
        )
    existing = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        path,
        existing + json.dumps(decision.json(), ensure_ascii=False, sort_keys=True) + "\n",
    )


def package_dual_outputs(
    units: Sequence[TranslationUnit],
    translations: Iterable[UnitTranslation | Mapping[str, Any]],
    output_dir: Path,
    *,
    review_decisions: Mapping[str, ReviewDecision | str] | None = None,
    bundle: JapaneseSubtitleBundle | None = None,
    automated_quality: bool = False,
    replace_existing: bool = False,
) -> DualOutputArtifacts:
    if not units:
        raise CoverageError("cannot package an empty translation")
    if bundle is not None and not evaluate_bundle(bundle).translation_allowed:
        raise BundleBlockedError("Japanese bundle is blocked from translation packaging")
    unit_ids = [unit.unit_id for unit in units]
    if len(set(unit_ids)) != len(unit_ids):
        raise CoverageError("translation units contain duplicate unit IDs")

    by_id: dict[str, UnitTranslation] = {}
    for value in translations:
        translation = _coerce_translation(value)
        if translation.unit_id in by_id:
            raise CoverageError(f"duplicate translation for {translation.unit_id}")
        by_id[translation.unit_id] = translation
    missing = [unit_id for unit_id in unit_ids if unit_id not in by_id]
    extras = sorted(set(by_id) - set(unit_ids))
    empty_complete = [
        unit_id
        for unit_id in unit_ids
        if unit_id in by_id and not by_id[unit_id].viewer_complete_ko.strip()
    ]
    if missing or extras or empty_complete:
        raise CoverageError(
            "viewer_complete_ko must cover every unit exactly once; "
            f"missing={missing}, extras={extras}, empty={empty_complete}"
        )

    review_decisions = review_decisions or {}
    complete_blocks: list[SubtitleBlock] = []
    faithful_blocks: list[SubtitleBlock] = []
    natural_blocks: list[SubtitleBlock] = []
    held: list[str] = []
    candidate_units = 0
    unit_receipts: list[dict[str, Any]] = []
    for index, unit in enumerate(units, 1):
        translation = by_id[unit.unit_id]
        review_status = _review_status(review_decisions.get(unit.unit_id))
        disposition = evaluate_bundle(bundle) if bundle is not None else None
        bundle_or_unit_accepted = (
            bundle is None
            or (
                automated_quality
                and bundle.valid
                and disposition is not None
                and disposition.translation_allowed
            )
            or disposition is not None and disposition.candidate_generation_allowed
            or review_status == "approved"
        )
        if automated_quality:
            candidate_eligible = (
                bool(translation.evidence_ids)
                and bool((translation.source_faithful_ko or "").strip())
                and bool((translation.viewer_natural_ko or "").strip())
                and review_status != "rejected"
                and bundle_or_unit_accepted
            )
        else:
            candidate_eligible = (
                bool(translation.evidence_ids)
                and bool((translation.source_faithful_ko or "").strip())
                and bool((translation.viewer_natural_ko or "").strip())
                and (unit.quality_status == "trusted" or review_status == "approved")
                and review_status != "rejected"
                and bundle_or_unit_accepted
            )
        complete_blocks.append(
            _block_for_unit(index, unit, translation.viewer_complete_ko)
        )
        if candidate_eligible:
            candidate_units += 1
            faithful_text = translation.source_faithful_ko or REVIEW_HOLD_TEXT
            natural_text = translation.viewer_natural_ko or REVIEW_HOLD_TEXT
        else:
            held.append(unit.unit_id)
            if automated_quality:
                faithful_text = translation.source_faithful_ko or MACHINE_UNCERTAIN_TEXT
                natural_text = translation.viewer_natural_ko or MACHINE_UNCERTAIN_TEXT
            else:
                faithful_text = REVIEW_HOLD_TEXT
                natural_text = REVIEW_HOLD_TEXT
        faithful_blocks.append(_block_for_unit(index, unit, faithful_text))
        natural_blocks.append(_block_for_unit(index, unit, natural_text))
        unit_receipts.append(
            {
                "unit_id": unit.unit_id,
                "quality_status": unit.quality_status,
                "review_status": review_status,
                "candidate_eligible": candidate_eligible,
                "evidence_ids": list(translation.evidence_ids),
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "viewer_complete_ko.srt"
    faithful_path = output_dir / "source_faithful_ko.srt"
    natural_path = output_dir / "viewer_natural_ko.srt"
    qa_path = output_dir / "translation_qa.json"
    if not replace_existing:
        _ensure_new_paths((complete_path, faithful_path, natural_path, qa_path))
    _atomic_write_srt(complete_path, complete_blocks)
    _atomic_write_srt(faithful_path, faithful_blocks)
    _atomic_write_srt(natural_path, natural_blocks)
    qa = {
        "schema_name": "translation-forensics/dual-output-qa",
        "schema_version": INTEGRATION_SCHEMA_VERSION,
        "total_units": len(units),
        "viewer_complete_units": len(complete_blocks),
        "viewer_complete_coverage": 1.0,
        "candidate_units": candidate_units,
        "candidate_coverage": candidate_units / len(units),
        "held_units": held,
        "human_reviewed": False,
        "human_final_allowed": False,
        "automated_quality": automated_quality,
        "machine_final_allowed": automated_quality and not held,
        "final_promotion_allowed": False,
        "verification_status": "not-demonstrated",
        "units": unit_receipts,
    }
    _atomic_write_json(qa_path, qa)
    return DualOutputArtifacts(
        viewer_complete_ko_srt=complete_path,
        source_faithful_ko_srt=faithful_path,
        viewer_natural_ko_srt=natural_path,
        qa_path=qa_path,
        total_units=len(units),
        candidate_units=candidate_units,
        held_units=tuple(held),
    )


def build_cache_identity(
    *,
    code_version: str,
    schema_version: str,
    options: Mapping[str, Any],
    media_sha256: str,
) -> dict[str, Any]:
    """Build a stable cache key without temporary WAV-path identity leaks."""

    if not code_version.strip() or not schema_version.strip():
        raise IntegrationPipelineError("code_version and schema_version are required")
    _validate_sha256(media_sha256, field_name="media_sha256")
    payload = {
        "code_version": code_version,
        "schema_version": schema_version,
        "media_sha256": media_sha256.lower(),
        "options": _sanitize_cache_value(options),
    }
    encoded = _canonical_json(payload).encode("utf-8")
    return {**payload, "identity_sha256": hashlib.sha256(encoded).hexdigest()}


def cache_is_reusable(
    manifest: Mapping[str, Any] | Path,
    expected_identity: Mapping[str, Any],
) -> bool:
    try:
        value = _load_mapping(manifest)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return (
        value.get("schema_name") == "translation-forensics/integrated-cache"
        and value.get("schema_version") == CACHE_SCHEMA_VERSION
        and value.get("status") == "complete"
        and value.get("cache_identity") == dict(expected_identity)
    )


def require_reusable_cache(
    manifest: Mapping[str, Any] | Path,
    expected_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not cache_is_reusable(manifest, expected_identity):
        raise ResumeRejected("cache is partial, stale, malformed, or has a different identity")
    return _load_mapping(manifest)


def stage_run(
    output_root: Path,
    run_id: str,
    *,
    cache_identity: Mapping[str, Any],
    resume: bool = False,
) -> Path:
    _validate_run_id(run_id)
    output_root = Path(output_root)
    final_dir = output_root / run_id
    partial_dir = output_root / ".partial" / run_id
    state_path = partial_dir / "run-state.json"
    lock_acquired = _acquire_run_lock(output_root, run_id)
    try:
        if final_dir.exists():
            if not resume:
                raise FileExistsError(f"promoted run already exists: {final_dir}")
            if partial_dir.exists():
                raise ResumeRejected(
                    "both partial and promoted run directories exist for the same run_id"
                )
            state = _require_run_state(
                final_dir / "run-state.json",
                run_id=run_id,
                statuses={"verified"},
                cache_identity=cache_identity,
            )
            _write_latest_from_state(output_root, state)
            _release_run_lock(output_root, run_id)
            return final_dir
        if partial_dir.exists():
            if not resume:
                raise FileExistsError(f"partial run already exists: {partial_dir}")
            _require_run_state(
                state_path,
                run_id=run_id,
                statuses={"partial", "verified"},
                cache_identity=cache_identity,
            )
            return partial_dir

        partial_dir.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(
            state_path,
            {
                "schema_name": "translation-forensics/integrated-run",
                "schema_version": INTEGRATION_SCHEMA_VERSION,
                "run_id": run_id,
                "status": "partial",
                "cache_identity": dict(cache_identity),
                "started_at": _utc_now(),
            },
        )
        return partial_dir
    except Exception:
        if lock_acquired:
            _release_run_lock(output_root, run_id)
        raise


def promote_staged_run(
    output_root: Path,
    run_id: str,
    *,
    verification_passed: bool,
    stage: str = "packaged",
    pending_review_count: int = 0,
    required_artifacts: Iterable[str] = (),
    manifest_sha256: str | None = None,
) -> Path:
    _validate_run_id(run_id)
    if not verification_passed:
        raise BundleBlockedError("a staged run cannot be promoted before verification passes")
    if pending_review_count < 0:
        raise IntegrationPipelineError("pending_review_count cannot be negative")
    if manifest_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", manifest_sha256
    ):
        raise IntegrationPipelineError("manifest_sha256 must be lowercase SHA-256")
    output_root = Path(output_root)
    partial_dir = output_root / ".partial" / run_id
    final_dir = output_root / run_id
    _acquire_run_lock(output_root, run_id)
    try:
        if final_dir.exists():
            if partial_dir.exists():
                raise ResumeRejected(
                    "both partial and promoted run directories exist for the same run_id"
                )
            state = _require_run_state(
                final_dir / "run-state.json",
                run_id=run_id,
                statuses={"verified"},
            )
            _require_matching_promotion(
                state, stage, pending_review_count, manifest_sha256
            )
            _require_artifacts(final_dir, required_artifacts)
            _write_latest_from_state(output_root, state)
            return final_dir

        state_path = partial_dir / "run-state.json"
        state = _require_run_state(
            state_path,
            run_id=run_id,
            statuses={"partial", "verified"},
        )
        _require_artifacts(partial_dir, required_artifacts)
        if state["status"] == "verified":
            _require_matching_promotion(
                state, stage, pending_review_count, manifest_sha256
            )
        else:
            state.update(
                {
                    "status": "verified",
                    "stage": stage,
                    "pending_review_count": pending_review_count,
                    "promoted_at": _utc_now(),
                    "manifest_sha256": manifest_sha256,
                }
            )
            _atomic_write_json(state_path, state)
        os.replace(partial_dir, final_dir)
        _write_latest_from_state(output_root, state)
        return final_dir
    finally:
        _release_run_lock(output_root, run_id)


def _coerce_translation(value: UnitTranslation | Mapping[str, Any]) -> UnitTranslation:
    if isinstance(value, UnitTranslation):
        return value
    if not isinstance(value, Mapping):
        raise CoverageError("translation values must be UnitTranslation or mappings")
    try:
        evidence = value.get("evidence_ids", value.get("evidence_refs", ()))
        viewer_natural = value.get(
            "viewer_natural_ko", value.get("viewer_natural_korean")
        )
        viewer_complete = value.get("viewer_complete_ko", viewer_natural)
        source_faithful = value.get(
            "source_faithful_ko", value.get("source_faithful_korean")
        )
        return UnitTranslation(
            unit_id=str(value["unit_id"]),
            viewer_complete_ko=str(viewer_complete or ""),
            source_faithful_ko=_optional_string(source_faithful),
            viewer_natural_ko=_optional_string(viewer_natural),
            evidence_ids=tuple(str(item) for item in evidence),
        )
    except (KeyError, TypeError) as exc:
        raise CoverageError("invalid translation mapping") from exc


def _review_status(value: ReviewDecision | str | None) -> str:
    if value is None:
        return "pending"
    status = value.status if isinstance(value, ReviewDecision) else value
    status = {"held": "deferred", "hold": "deferred"}.get(status, status)
    if status not in _REVIEW_STATUSES:
        raise IntegrationPipelineError(f"unsupported review status: {status}")
    return status


def _block_for_unit(number: int, unit: TranslationUnit, text: str) -> SubtitleBlock:
    return SubtitleBlock(
        number=number,
        start=seconds_to_timecode(unit.start),
        end=seconds_to_timecode(unit.end),
        text=text,
        start_seconds=unit.start,
        end_seconds=unit.end,
    )


def _atomic_write_srt(path: Path, blocks: Sequence[SubtitleBlock]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_srt(temporary, blocks)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_run_state(
    path: Path,
    *,
    run_id: str,
    statuses: set[str],
    cache_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = _read_json_object(path)
    if (
        state.get("schema_name") != "translation-forensics/integrated-run"
        or state.get("schema_version") != INTEGRATION_SCHEMA_VERSION
        or state.get("status") not in statuses
        or state.get("run_id") != run_id
        or (
            cache_identity is not None
            and state.get("cache_identity") != dict(cache_identity)
        )
    ):
        raise ResumeRejected("run state is stale, malformed, or has a different identity")
    return state


def _require_matching_promotion(
    state: Mapping[str, Any],
    stage: str,
    pending_review_count: int,
    manifest_sha256: str | None = None,
) -> None:
    if (
        state.get("stage") != stage
        or state.get("pending_review_count") != pending_review_count
        or (
            manifest_sha256 is not None
            and state.get("manifest_sha256") != manifest_sha256
        )
    ):
        raise ResumeRejected("verified run has different promotion metadata")


def _require_artifacts(root: Path, required_artifacts: Iterable[str]) -> None:
    missing = [name for name in required_artifacts if not (root / name).is_file()]
    if missing:
        raise BundleBlockedError(f"required staged artifacts are missing: {missing}")


def _write_latest_from_state(output_root: Path, state: Mapping[str, Any]) -> None:
    promoted_at = state.get("promoted_at")
    if not isinstance(promoted_at, str) or not promoted_at:
        raise ResumeRejected("verified run is missing promoted_at")
    cache_identity = state.get("cache_identity")
    if not isinstance(cache_identity, Mapping):
        raise ResumeRejected("verified run is missing cache identity")
    run_id = state.get("run_id")
    stage = state.get("stage")
    pending_review_count = state.get("pending_review_count")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(stage, str)
        or not stage
        or not isinstance(pending_review_count, int)
        or pending_review_count < 0
    ):
        raise ResumeRejected("verified run has incomplete promotion metadata")
    latest_path = output_root / "latest.json"
    payload = {
        "schema_name": "translation-forensics/latest-integrated-run",
        "schema_version": INTEGRATION_SCHEMA_VERSION,
        "run_id": run_id,
        "run_path": run_id,
        "stage": stage,
        "status": "verified",
        "cache_identity_sha256": cache_identity.get("identity_sha256"),
        "pending_review_count": pending_review_count,
        "updated_at": promoted_at,
    }
    if latest_path.is_file():
        try:
            current = _read_json_object(latest_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            current = {}
        if current == payload:
            return
        current_updated_at = current.get("updated_at")
        if (
            current.get("schema_name")
            == "translation-forensics/latest-integrated-run"
            and current.get("schema_version") == INTEGRATION_SCHEMA_VERSION
            and current.get("status") == "verified"
            and current.get("run_id") != run_id
            and isinstance(current_updated_at, str)
            and _timestamp_is_at_or_after(current_updated_at, promoted_at)
        ):
            return
    _atomic_write_json(latest_path, payload)


def _timestamp_is_at_or_after(left: str, right: str) -> bool:
    try:
        return datetime.fromisoformat(left) >= datetime.fromisoformat(right)
    except ValueError:
        return False


def _acquire_run_lock(output_root: Path, run_id: str) -> bool:
    lock_path = (Path(output_root) / ".locks" / f"{run_id}.lock").resolve()
    key = str(lock_path).casefold() if os.name == "nt" else str(lock_path)
    owner_pid = os.getpid()
    owner_thread_id = threading.get_ident()
    with _RUN_LOCKS_GUARD:
        held = _RUN_LOCKS.get(key)
        if held is not None:
            if (
                held.owner_pid == owner_pid
                and held.owner_thread_id == owner_thread_id
            ):
                return False
            raise ResumeRejected(f"run is locked by another worker: {run_id}")

        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"1")
                handle.flush()
            handle.seek(0)
            _lock_file(handle)
        except Exception:
            handle.close()
            raise ResumeRejected(f"run is locked by another worker: {run_id}")
        _RUN_LOCKS[key] = _HeldRunLock(
            handle=handle,
            owner_pid=owner_pid,
            owner_thread_id=owner_thread_id,
        )
        return True


def _release_run_lock(output_root: Path, run_id: str) -> None:
    lock_path = (Path(output_root) / ".locks" / f"{run_id}.lock").resolve()
    key = str(lock_path).casefold() if os.name == "nt" else str(lock_path)
    with _RUN_LOCKS_GUARD:
        held = _RUN_LOCKS.get(key)
        if held is None:
            return
        if (
            held.owner_pid != os.getpid()
            or held.owner_thread_id != threading.get_ident()
        ):
            return
        try:
            _unlock_file(held.handle)
        finally:
            held.handle.close()
            del _RUN_LOCKS[key]


def _lock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _ensure_new_paths(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite integration artifacts: " + ", ".join(existing))


def _sanitize_cache_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_cache_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _EPHEMERAL_CACHE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_cache_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_sanitize_cache_value(item) for item in value), key=str)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise IntegrationPipelineError(f"cache option is not JSON serializable: {type(value).__name__}")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_mapping(value: Mapping[str, Any] | Path) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return _read_json_object(Path(value))


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise IntegrationPipelineError(f"expected a JSON object: {path}")
    return value


def _validate_sha256(value: str, *, field_name: str) -> None:
    if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise IntegrationPipelineError(f"{field_name} must be a 64-character SHA-256 hex digest")


def _validate_run_id(run_id: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
        raise IntegrationPipelineError(f"unsafe run_id: {run_id!r}")


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "INTEGRATION_SCHEMA_VERSION",
    "BundleBlockedError",
    "BundleDisposition",
    "CoverageError",
    "DualOutputArtifacts",
    "ExtractRequest",
    "GateResult",
    "IntegrationPipelineError",
    "JapaneseSubtitleBundle",
    "REVIEW_HOLD_TEXT",
    "ResumeRejected",
    "ReviewDecision",
    "TranslationInputArtifacts",
    "TranslationUnit",
    "UnitTranslation",
    "append_review_decision",
    "build_cache_identity",
    "build_translation_inputs",
    "cache_is_reusable",
    "classify_unit_quality",
    "evaluate_bundle",
    "load_review_decisions",
    "load_translation_units",
    "normalize_whitespace",
    "package_dual_outputs",
    "promote_staged_run",
    "require_reusable_cache",
    "select_extraction_backend",
    "stage_run",
    "transition_review_decision",
    "write_translation_units",
]
