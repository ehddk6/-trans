"""Independent verification for a completed subtitle artifact bundle."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .media import build_media_binding, validate_media_binding
from .reference import reference_language_diagnostics
from .srt import read_srt_rows
from .text import normalize_japanese


REQUIRED_ARTIFACTS = (
    "source_faithful_ja.srt",
    "viewer_ja.srt",
    "transcript_ja.jsonl",
    "subtitle_audit.jsonl",
    "qc_report.json",
    "comparison_report.html",
    "review_manifest.json",
    "review_report.html",
)
MEDIA_BINDING_ARTIFACT = "source_media.json"


def _read_json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid_{label}:{exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"invalid_{label}:expected_object")
        return {}
    return value


def _read_jsonl(path: Path, errors: list[str], label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"invalid_{label}:{exc}")
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"invalid_{label}:line_{line_number}")
            continue
        if not isinstance(value, dict):
            errors.append(f"invalid_{label}:line_{line_number}:expected_object")
            continue
        rows.append(value)
    return rows


def _srt_timestamp_value(value: float) -> float:
    """Apply the same millisecond rounding used when the pipeline writes SRT."""
    return max(0, round(value * 1000)) / 1000


def _verify_viewer_audit(
    viewer_rows: list[dict[str, object]], audit_rows: list[dict[str, Any]], errors: list[str],
) -> None:
    if len(viewer_rows) != len(audit_rows):
        errors.append("viewer_audit_cue_count_mismatch")
    for index, (viewer, audit) in enumerate(zip(viewer_rows, audit_rows), 1):
        if not {"cue_id", "start", "end", "text_raw"} <= audit.keys():
            errors.append(f"invalid_subtitle_audit:row_{index}")
            continue
        if str(viewer["id"]) != str(audit["cue_id"]):
            errors.append(f"viewer_audit_id_mismatch:{index}")
        if not isinstance(audit["text_raw"], str):
            errors.append(f"invalid_subtitle_audit:row_{index}:text_raw")
        else:
            # cue_audit intentionally flattens display line wrapping. Remove only
            # those line breaks; meaningful spaces and all other text stay exact.
            viewer_text = str(viewer["text"]).replace("\r", "").replace("\n", "")
            if viewer_text != audit["text_raw"]:
                errors.append(f"viewer_audit_text_mismatch:{index}")
        try:
            expected_start = _srt_timestamp_value(float(audit["start"]))
            expected_end = _srt_timestamp_value(float(audit["end"]))
        except (TypeError, ValueError):
            errors.append(f"invalid_subtitle_audit:row_{index}:timing")
            continue
        if (
            abs(float(viewer["start"]) - expected_start) > 1e-9
            or abs(float(viewer["end"]) - expected_end) > 1e-9
        ):
            errors.append(f"viewer_audit_timing_mismatch:{index}")


def _verify_source_transcript(
    source_rows: list[dict[str, object]], transcript_rows: list[dict[str, Any]], errors: list[str],
) -> None:
    """Require the machine-readable transcript to prove the source SRT."""
    if len(source_rows) != len(transcript_rows):
        errors.append("source_transcript_cue_count_mismatch")
    for index, (source, transcript) in enumerate(zip(source_rows, transcript_rows), 1):
        if not {"start", "end", "text_raw"} <= transcript.keys():
            errors.append(f"invalid_transcript:row_{index}")
            continue
        expected_text = normalize_japanese(str(transcript["text_raw"]), "strict")
        if str(source["text"]) != expected_text:
            errors.append(f"source_transcript_text_mismatch:{index}")
        try:
            expected_start = _srt_timestamp_value(float(transcript["start"]))
            expected_end = _srt_timestamp_value(float(transcript["end"]))
        except (TypeError, ValueError):
            errors.append(f"invalid_transcript:row_{index}:timing")
            continue
        if (
            abs(float(source["start"]) - expected_start) > 1e-9
            or abs(float(source["end"]) - expected_end) > 1e-9
        ):
            errors.append(f"source_transcript_timing_mismatch:{index}")


def _timeline_structure(
    rows: list[dict[str, Any]] | list[dict[str, object]], *, text_key: str,
) -> dict[str, int]:
    result = {
        "empty": 0,
        "non_positive": 0,
        "overlap": 0,
        "non_finite": 0,
    }
    previous_end = -math.inf
    for row in rows:
        if not str(row.get(text_key, "")).strip():
            result["empty"] += 1
        try:
            start, end = float(row["start"]), float(row["end"])
        except (KeyError, TypeError, ValueError):
            result["non_finite"] += 1
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            result["non_finite"] += 1
            continue
        if end <= start:
            result["non_positive"] += 1
        if start < previous_end - 1e-9:
            result["overlap"] += 1
        previous_end = max(previous_end, end)
    return result


def _verify_transcript_identity(
    transcript_rows: list[dict[str, Any]], errors: list[str],
) -> None:
    identifiers = [str(row.get("utterance_id") or "") for row in transcript_rows]
    if any(not identifier for identifier in identifiers):
        errors.append("transcript_missing_utterance_id")
    if len(set(identifiers)) != len(identifiers):
        errors.append("transcript_duplicate_utterance_id")


def _verify_media_binding(
    output_dir: Path,
    errors: list[str],
    *,
    expected_video_path: Path | None,
    expected_media_sha256: str | None,
    expected_media_duration: float | None,
    require_media_binding: bool,
) -> dict[str, Any] | None:
    path = output_dir / MEDIA_BINDING_ARTIFACT
    required = require_media_binding or expected_media_sha256 is not None or expected_media_duration is not None
    if not path.is_file():
        if required:
            errors.append(f"missing_artifact:{MEDIA_BINDING_ARTIFACT}")
        return None
    binding = _read_json(path, errors, "source_media")
    errors.extend(validate_media_binding(binding))
    if errors and not binding:
        return binding

    expected_path = Path(expected_video_path).resolve() if expected_video_path is not None else None
    if expected_path is None and require_media_binding and binding.get("path"):
        expected_path = Path(str(binding["path"])).expanduser().resolve()
    current: dict[str, Any] | None = None
    if expected_path is not None:
        if not expected_path.is_file():
            errors.append("source_media_file_missing")
        else:
            try:
                # Re-hash at final verification even when the caller supplies the
                # hash computed during prepare.  This closes the prepare/finalize
                # time-of-check/time-of-use gap required by the bundle contract.
                current = build_media_binding(expected_path)
            except (OSError, RuntimeError, ValueError) as exc:
                errors.append(f"source_media_probe_failed:{exc}")
    expected_digest = expected_media_sha256 or (str(current["sha256"]) if current else None)
    expected_duration = (
        expected_media_duration
        if expected_media_duration is not None
        else float(current["duration_seconds"]) if current else None
    )
    if expected_digest is not None and str(binding.get("sha256")) != str(expected_digest).lower():
        errors.append("source_media_sha256_mismatch")
    if (
        expected_media_sha256 is not None
        and current is not None
        and str(current.get("sha256")) != str(expected_media_sha256).lower()
    ):
        errors.append("source_media_changed_during_run")
    if expected_duration is not None:
        try:
            bound_duration = float(binding.get("duration_seconds"))
            expected_duration_value = float(expected_duration)
        except (TypeError, ValueError):
            errors.append("invalid_source_media_metrics")
        else:
            if abs(bound_duration - expected_duration_value) > 0.10:
                errors.append("source_media_duration_mismatch")
    return binding


def _verify_qwen_manifest(output_dir: Path, report: dict[str, Any], errors: list[str]) -> None:
    if report.get("backend") != "ensemble":
        return
    path = output_dir / "qwen_verification_manifest.json"
    if not path.is_file():
        errors.append("missing_artifact:qwen_verification_manifest.json")
        return
    try:
        intervals = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid_qwen_verification_manifest:{exc}")
        return
    if not isinstance(intervals, list):
        errors.append("invalid_qwen_verification_manifest:expected_list")
        return
    verification = report.get("qwen_verification")
    if not isinstance(verification, dict):
        errors.append("invalid_qwen_verification_report")
        return
    valid_intervals: list[tuple[float, float]] = []
    for index, interval in enumerate(intervals, 1):
        try:
            start, end = float(interval["start"]), float(interval["end"])
            reasons = interval["reasons"]
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid_qwen_verification_manifest:interval_{index}")
            continue
        if start < 0 or end <= start or not isinstance(reasons, list) or not reasons:
            errors.append(f"invalid_qwen_verification_manifest:interval_{index}")
            continue
        valid_intervals.append((start, end))
    if verification.get("interval_count") != len(intervals):
        errors.append("qwen_manifest_interval_count_mismatch")
    expected_seconds = round(sum(end - start for start, end in valid_intervals), 3)
    try:
        reported_seconds = round(float(verification.get("audio_seconds")), 3)
        video_seconds = float(verification.get("video_seconds"))
    except (TypeError, ValueError):
        errors.append("invalid_qwen_verification_report")
        return
    if abs(reported_seconds - expected_seconds) > 1e-3:
        errors.append("qwen_manifest_audio_seconds_mismatch")
    if any(end > video_seconds + 0.01 for _, end in valid_intervals):
        errors.append("qwen_manifest_interval_outside_video")
    if "recovery_candidates" in verification:
        recovery_path = output_dir / "qwen_recovery_audit.jsonl"
        if not recovery_path.is_file():
            errors.append("missing_artifact:qwen_recovery_audit.jsonl")
            return
        recovery_rows = _read_jsonl(recovery_path, errors, "qwen_recovery_audit")
        accepted = sum(row.get("decision") == "accepted" for row in recovery_rows)
        rejected = sum(row.get("decision") == "rejected" for row in recovery_rows)
        if verification.get("recovery_candidates") != len(recovery_rows):
            errors.append("qwen_recovery_audit_count_mismatch")
        if verification.get("recovery_accepted") != accepted:
            errors.append("qwen_recovery_audit_accepted_mismatch")
        if verification.get("recovery_rejected") != rejected:
            errors.append("qwen_recovery_audit_rejected_mismatch")


def _exact_cue_text_match(
    candidate: list[dict[str, object]], reference: list[dict[str, object]],
) -> bool:
    """Compare decoded cue text in order, preserving cue boundaries and spaces."""
    return [str(row.get("text", "")) for row in candidate] == [
        str(row.get("text", "")) for row in reference
    ]


def _resolved_path(value: str | Path | None) -> str | None:
    if not value:
        return None
    return str(Path(value).resolve())


def verify_artifact_bundle(
    output_dir: Path,
    *,
    reference_srt: Path | None = None,
    expected_video_path: Path | None = None,
    expected_review_windows: int = 30,
    require_source_match: bool = False,
    require_passed_gate: bool = False,
    expected_media_sha256: str | None = None,
    expected_media_duration: float | None = None,
    require_media_binding: bool = False,
) -> dict[str, Any]:
    """Return an evidence-rich verification result without mutating the bundle.

    ``valid`` covers structural integrity and requested strict assertions.  A
    review-required quality gate is reported separately unless
    ``require_passed_gate`` is set, so callers can distinguish a usable review
    bundle from an automatically accepted subtitle result.
    """

    errors: list[str] = []
    warnings: list[str] = []
    output_dir = output_dir.resolve()
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).is_file()]
    if missing:
        errors.extend(f"missing_artifact:{name}" for name in missing)
        return {
            "output_dir": str(output_dir),
            "valid": False,
            "accepted": False,
            "missing_artifacts": missing,
            "errors": errors,
            "warnings": warnings,
        }

    report = _read_json(output_dir / "qc_report.json", errors, "qc_report")
    manifest = _read_json(output_dir / "review_manifest.json", errors, "review_manifest")
    transcript_rows = _read_jsonl(output_dir / "transcript_ja.jsonl", errors, "transcript")
    audit_rows = _read_jsonl(output_dir / "subtitle_audit.jsonl", errors, "subtitle_audit")
    try:
        viewer_rows = read_srt_rows(output_dir / "viewer_ja.srt")
        source_rows = read_srt_rows(output_dir / "source_faithful_ja.srt")
    except (OSError, ValueError) as exc:
        errors.append(f"invalid_srt:{exc}")
        viewer_rows, source_rows = [], []

    source_language = reference_language_diagnostics(source_rows)
    if not source_language["compatible"]:
        errors.append("source_language_hangul_dominant")
    _verify_source_transcript(source_rows, transcript_rows, errors)
    _verify_transcript_identity(transcript_rows, errors)
    _verify_qwen_manifest(output_dir, report, errors)

    viewer_timeline = _timeline_structure(viewer_rows, text_key="text")
    source_structural = _timeline_structure(source_rows, text_key="text")
    transcript_structural = _timeline_structure(transcript_rows, text_key="text_raw")
    structural = {name: viewer_timeline[name] for name in ("empty", "non_positive", "overlap")}
    if any(viewer_timeline.values()):
        errors.append("viewer_structural_error")
    if any(source_structural.values()):
        errors.append("source_structural_error")
    if any(transcript_structural.values()):
        errors.append("transcript_structural_error")
    _verify_viewer_audit(viewer_rows, audit_rows, errors)
    reported_total = report.get("total_cues")
    if not isinstance(reported_total, int) or isinstance(reported_total, bool):
        errors.append("invalid_qc_total_cues")
    elif reported_total != len(viewer_rows):
        errors.append("qc_total_cues_does_not_match_viewer")
    report_counts = {
        "empty": report.get("empty_cue_count"),
        "non_positive": report.get("timecode_reversal_count"),
        "overlap": report.get("timecode_overlap_count"),
    }
    if report.get("validation_errors"):
        errors.append("qc_validation_errors_present")
    if report_counts != structural:
        errors.append("qc_structural_counts_do_not_match_srt")

    windows = manifest.get("windows", [])
    if not isinstance(windows, list) or len(windows) != expected_review_windows:
        errors.append(f"unexpected_review_window_count:{len(windows) if isinstance(windows, list) else 'invalid'}")
    elif any(
        not isinstance(window, dict)
        or float(window.get("end", 0)) <= float(window.get("start", 0))
        or float(window.get("duration", 0)) <= 0
        for window in windows
    ):
        errors.append("invalid_review_window")

    report_html = (output_dir / "review_report.html").read_text(encoding="utf-8")
    manifest_video = _resolved_path(manifest.get("video_path"))
    expected_video = _resolved_path(expected_video_path)
    if expected_video is not None:
        if not expected_video_path or not expected_video_path.is_file():
            errors.append("expected_video_missing")
        if manifest_video != expected_video:
            errors.append("review_video_path_mismatch")
    if manifest_video:
        expected_controls = len(windows) if isinstance(windows, list) else 0
        if report_html.count("이 검수 구간 영상 재생") != expected_controls:
            errors.append("review_video_controls_missing")
    elif expected_video is not None:
        errors.append("review_video_controls_missing")

    source_media = _verify_media_binding(
        output_dir,
        errors,
        expected_video_path=expected_video_path,
        expected_media_sha256=expected_media_sha256,
        expected_media_duration=expected_media_duration,
        require_media_binding=require_media_binding,
    )
    if source_media is not None:
        try:
            bound_duration = float(source_media["duration_seconds"])
        except (KeyError, TypeError, ValueError):
            pass
        else:
            if any(float(row["end"]) > bound_duration + 0.10 for row in source_rows):
                errors.append("source_timeline_outside_media")
            if any(float(row["end"]) > bound_duration + 0.10 for row in viewer_rows):
                errors.append("viewer_timeline_outside_media")

    source_matches_reference: bool | None = None
    if reference_srt is not None:
        try:
            reference_rows = read_srt_rows(reference_srt)
            source_matches_reference = _exact_cue_text_match(source_rows, reference_rows)
        except (OSError, ValueError) as exc:
            errors.append(f"invalid_reference_srt:{exc}")
        if source_matches_reference is False:
            message = "source_text_differs_from_reference"
            (errors if require_source_match else warnings).append(message)

    gate = report.get("content_quality_gate", {})
    gate_status = gate.get("status") if isinstance(gate, dict) else None
    if gate_status not in {"passed", "review_required", "failed"}:
        errors.append("invalid_content_quality_gate")
    if gate_status == "failed":
        errors.append("content_quality_gate_failed")
    if require_passed_gate and gate_status != "passed":
        errors.append("content_quality_gate_not_passed")

    valid = not errors
    return {
        "output_dir": str(output_dir),
        "valid": valid,
        "accepted": valid and gate_status == "passed",
        "quality_gate_status": gate_status,
        "quality_gate_reasons": gate.get("reasons", []) if isinstance(gate, dict) else [],
        "missing_artifacts": missing,
        "viewer_cue_count": len(viewer_rows),
        "source_cue_count": len(source_rows),
        "source_language": source_language,
        "structural": structural,
        "source_structural": source_structural,
        "transcript_structural": transcript_structural,
        "source_media": source_media,
        "review_window_count": len(windows) if isinstance(windows, list) else None,
        "review_video_path": manifest_video,
        "source_matches_reference": source_matches_reference,
        "errors": errors,
        "warnings": warnings,
    }
