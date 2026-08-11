from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from .artifacts import cue_audit, qc_report, reference_comparison, write_comparison_report, write_jsonl
from .checks import annotate_asr_warnings
from .ensemble import EnsembleResult, transcribe_ensemble
from .models import Cue, SourceSegment, Utterance, Word
from .optimizer import segments_to_utterances, source_faithful_cues, viewer_cues
from .profiles import get_profile
from .qwen import QwenRuntime, transcribe_qwen
from .recognition import transcribe
from .reference import require_japanese_reference, segments_from_reference_rows
from .scenes import detect_scene_changes
from .srt import read_srt_rows, write_srt


def _json_default(value: object) -> object:
    """Convert NumPy scalar metrics returned by ASR libraries without losing values."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _srt_time(value: float) -> float:
    """Mirror SRT millisecond rounding for cross-artifact identity checks."""
    return max(0, round(value * 1000)) / 1000


def _write_bundle_verification(
    output_dir: Path,
    *,
    reference_srt: Path | None,
    video_path: Path | None,
    review_samples: int,
    require_source_match: bool = False,
) -> dict:
    """Persist an independent post-write validation beside every bundle."""

    from .verification import verify_artifact_bundle

    verification = verify_artifact_bundle(
        output_dir,
        reference_srt=reference_srt,
        expected_video_path=video_path,
        expected_review_windows=review_samples,
        require_source_match=require_source_match,
    )
    (output_dir / "bundle_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    if not verification["valid"]:
        raise RuntimeError(f"Invalid subtitle artifact bundle: {', '.join(verification['errors'])}")
    return verification


def _clip_input(input_path: Path, max_duration: float | None) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if not max_duration:
        return input_path, None
    temporary = tempfile.TemporaryDirectory(prefix="subtitle_pipeline_")
    clip = Path(temporary.name) / "sample.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(input_path), "-t", str(max_duration), "-vn", "-ac", "1", "-ar", "16000", str(clip)], check=True)
    return clip, temporary


def run_pipeline(
    input_path: Path | None, output_dir: Path, model_name: str = "large-v3-turbo", language: str = "ja",
    profile_name: str = "viewer_ja", normalization: str = "conservative", word_timestamps: bool = True,
    vad: bool = False, beam_size: int = 5, temperature: float = 0.0,
    condition_on_previous_text: bool = False, max_duration: float | None = None, scene_detection: str = "off",
    backend: str = "faster-whisper", qwen_runtime: QwenRuntime | None = None,
    reference_srt: Path | None = None, review_samples: int = 30, reuse_qwen_cache: bool = True,
    qwen_audit_samples: int = 4,
    normalized_audio_path: Path | None = None,
) -> dict:
    if language != "ja":
        raise ValueError("This pipeline is intentionally Japanese-only; use --language ja.")
    if backend not in {"faster-whisper", "qwen", "ensemble", "reference"}:
        raise ValueError(f"Unsupported backend: {backend}")
    if backend == "reference" and not reference_srt:
        raise ValueError("The reference backend requires --reference-srt.")
    if backend != "reference" and input_path is None:
        raise ValueError("An input video is required unless --backend reference is used.")
    if normalized_audio_path is not None and not normalized_audio_path.is_file():
        raise ValueError(f"Normalized audio does not exist: {normalized_audio_path}")
    if backend in {"qwen", "ensemble"} and qwen_runtime is None:
        qwen_runtime = QwenRuntime.discover()
    if backend == "ensemble" and output_dir.name != "ensemble_qwen_whisper":
        output_dir = output_dir / "ensemble_qwen_whisper"
    if backend == "reference" and output_dir.name != "reference_primary":
        output_dir = output_dir / "reference_primary"
    output_dir.mkdir(parents=True, exist_ok=True)
    qwen_alignment_cache = output_dir / "qwen_targeted_alignment_ja.jsonl"
    review_video_path = input_path if input_path and input_path.is_file() else None
    external_reference = read_srt_rows(reference_srt) if reference_srt else None
    reference_language = require_japanese_reference(external_reference) if external_reference is not None else None
    ensemble_result: EnsembleResult | None = None
    scene_changes: list[float] = []
    if backend == "reference":
        if not external_reference:
            raise ValueError("The reference backend requires at least one valid, non-empty SRT cue.")
        segments = segments_from_reference_rows(external_reference or [])
        # A validated supplied subtitle is source evidence, not ASR output.  Applying
        # ASR hallucination/repetition heuristics to it would turn legitimate text
        # into a false model-quality warning.
        duration = max((segment.end for segment in segments), default=0.0)
    else:
        assert input_path is not None
        original_input = input_path
        review_video_path = original_input
        scene_changes = detect_scene_changes(original_input, max_duration) if scene_detection == "auto" else []
        recognition_input = normalized_audio_path or input_path
        input_path, temporary = _clip_input(recognition_input, max_duration)
        try:
            if backend == "faster-whisper":
                segments, duration = transcribe(
                    input_path, model_name, language, word_timestamps, vad, beam_size, temperature,
                    condition_on_previous_text,
                )
                annotate_asr_warnings(segments)
            elif backend == "qwen":
                segments, duration = transcribe_qwen(
                    input_path, qwen_runtime, language=language, alignment_cache_path=qwen_alignment_cache,
                    reuse_alignment_cache=reuse_qwen_cache,
                    cache_identity_path=original_input,
                )
            else:
                ensemble_result = transcribe_ensemble(
                    input_path, qwen_runtime, model_name, language=language,
                    reference_rows=external_reference,
                    alignment_cache_path=qwen_alignment_cache, reuse_alignment_cache=reuse_qwen_cache,
                    audit_samples=qwen_audit_samples,
                    cache_identity_path=original_input,
                )
                segments, duration = ensemble_result.segments, ensemble_result.duration
        finally:
            if temporary:
                temporary.cleanup()
    source_normalization = "strict"
    utterances = segments_to_utterances(segments, source_normalization)
    source = source_faithful_cues(utterances)
    viewer_normalization = "viewer" if backend == "reference" else normalization
    viewer = viewer_cues(
        utterances, get_profile(profile_name),
        viewer_normalization, scene_changes,
    )
    write_srt(output_dir / "source_faithful_ja.srt", source)
    write_srt(output_dir / "viewer_ja.srt", viewer)
    write_jsonl(output_dir / "transcript_ja.jsonl", [utterance.json() for utterance in utterances])
    write_jsonl(output_dir / "subtitle_audit.jsonl", cue_audit(viewer))
    report = qc_report(
        viewer, segments, duration if duration == duration else (max_duration or 0), source_cues=source,
    )
    report["backend"] = backend
    report["qwen_source_segments"] = sum(segment.metadata.get("backend") == "qwen" for segment in segments)
    report["qwen_recoveries"] = sum(segment.metadata.get("decision") == "qwen_recovery" for segment in segments)
    report["viewer_runaway_compactions"] = sum(
        any(code in cue.warnings for code in ("viewer_runaway_repetition_compacted", "viewer_high_confidence_runaway_compacted"))
        for cue in viewer
    )
    report["viewer_runaway_collapses"] = sum(
        any(code in cue.warnings for code in ("viewer_runaway_repetition_collapsed", "viewer_high_confidence_runaway_collapsed"))
        for cue in viewer
    )
    if backend == "reference":
        report["reference_primary"] = {
            "input_cues": len(external_reference or []),
            "timing_adjustments": sum("reference_timing_adjusted" in segment.warnings for segment in segments),
            "source_text_preserved": True,
            "viewer_runaway_compactions": report["viewer_runaway_compactions"],
            "language_diagnostics": reference_language,
        }
    if ensemble_result is not None:
        verification_seconds = sum(float(item["end"]) - float(item["start"]) for item in ensemble_result.verification_intervals)
        verification_segments = ensemble_result.qwen_segments
        recovery_candidates = [
            segment for segment in verification_segments if segment.metadata.get("recovery_candidate")
        ]
        rejected_recoveries = [
            segment for segment in recovery_candidates if segment.metadata.get("recovery_decision") == "rejected"
        ]
        rejection_reasons = Counter(
            reason
            for segment in rejected_recoveries
            for reason in segment.metadata.get("recovery_rejection_reasons", [])
        )
        report["qwen_verification"] = {
            "mode": "whisper_primary_targeted_qwen",
            "interval_count": len(ensemble_result.verification_intervals),
            "audio_seconds": round(verification_seconds, 3),
            "video_seconds": round(duration, 3),
            "full_video_pass": verification_seconds >= max(0.0, duration - 0.01),
            "cache": (
                "reused" if verification_segments and all(
                    segment.metadata.get("qwen_cache") == "reused" for segment in verification_segments
                ) else "created" if verification_segments else "empty"
            ),
            "recovery_candidates": len(recovery_candidates),
            "recovery_accepted": sum(
                segment.metadata.get("recovery_decision") == "accepted" for segment in recovery_candidates
            ),
            "recovery_rejected": len(rejected_recoveries),
            "recovery_rejection_reasons": dict(sorted(rejection_reasons.items())),
        }
        recovery_audit = []
        for segment in recovery_candidates:
            positive_words = sum(word.end - word.start > 1e-3 for word in segment.words)
            recovery_audit.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "decision": segment.metadata.get("recovery_decision"),
                "rejection_reasons": segment.metadata.get("recovery_rejection_reasons", []),
                "word_count": len(segment.words),
                "positive_duration_word_count": positive_words,
                "source_interval": segment.words[0].source_interval if segment.words else None,
            })
        write_jsonl(output_dir / "qwen_recovery_audit.jsonl", recovery_audit)
        (output_dir / "qwen_verification_manifest.json").write_text(
            json.dumps(ensemble_result.verification_intervals, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
    if external_reference is not None:
        report["reference_comparison"] = reference_comparison(viewer, external_reference, source)
    (output_dir / "qc_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    write_comparison_report(output_dir / "comparison_report.html", segments, source, viewer)
    from .review import write_review_artifacts
    source_evidence = ensemble_result.qwen_segments if ensemble_result is not None else segments
    whisper_evidence = ensemble_result.whisper_segments if ensemble_result is not None else []
    disagreements = [segment for segment in segments if "engine_disagreement" in segment.warnings]
    source_evidence_label = (
        "참조 기준 원문" if backend == "reference" else
        "Qwen 원문 근거" if backend == "ensemble" else
        "Qwen 원문" if backend == "qwen" else "ASR 원문"
    )
    write_review_artifacts(
        output_dir / "review_manifest.json", output_dir / "review_report.html", viewer, source_evidence,
        whisper_evidence, disagreements, external_reference,
        video_duration=duration, video_path=review_video_path, source_evidence_label=source_evidence_label,
        quality_gate=report["content_quality_gate"],
        target_windows=review_samples,
    )
    _write_bundle_verification(
        output_dir,
        reference_srt=reference_srt,
        video_path=review_video_path,
        review_samples=review_samples,
        require_source_match=backend == "reference",
    )
    return report


def rebuild_presentation_from_transcript(
    transcript_path: Path, output_dir: Path, profile_name: str = "viewer_ja", normalization: str = "conservative",
    reference_srt: Path | None = None, review_samples: int = 30, video_path: Path | None = None,
) -> dict:
    """Rebuild subtitle presentation artifacts without rerunning ASR.

    This is useful when only cue-boundary policy changes; raw recognition text and word
    timestamps remain untouched.
    """
    utterances: list[Utterance] = []
    segments: list[SourceSegment] = []
    external_reference = read_srt_rows(reference_srt) if reference_srt else None
    if reference_srt is not None and not external_reference:
        raise ValueError("The supplied reference SRT requires at least one valid, non-empty cue.")
    if external_reference is not None:
        require_japanese_reference(external_reference)
    if video_path is not None and not video_path.is_file():
        raise ValueError(f"Review video does not exist: {video_path}")
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        words = [Word(**word) for word in row.get("words", [])]
        metrics = row.get("asr_metrics", {})
        utterance = Utterance(
            id=row["utterance_id"], source_segment_ids=row["source_segment_ids"], start=row["start"], end=row["end"],
            text_raw=row["text_raw"], text_normalized=row["text_normalized"], words=words,
            speaker=row.get("speaker", "speaker_unknown"), metrics=metrics, warnings=row.get("warnings", []),
        )
        utterances.append(utterance)
        segments.append(SourceSegment(
            id=utterance.source_segment_ids[0], start=utterance.start, end=utterance.end, text=utterance.text_raw,
            words=words, no_speech_prob=metrics.get("no_speech_prob"), avg_logprob=metrics.get("avg_logprob"),
            compression_ratio=metrics.get("compression_ratio"), speaker=utterance.speaker, warnings=utterance.warnings,
        ))
    asr_warning_codes = {
        "empty_asr_text", "high_compression_ratio", "low_asr_confidence",
        "possible_silence_hallucination", "possible_repetition", "possible_periodic_repetition",
        "possible_runaway_repetition", "review_required",
    }
    for segment in segments:
        segment.warnings = [warning for warning in segment.warnings if warning not in asr_warning_codes]
    annotate_asr_warnings(segments)
    for utterance, segment in zip(utterances, segments):
        utterance.warnings = list(segment.warnings)
    # The JSONL text_raw field is the authoritative transcript.  Rebuilds may
    # alter only viewer presentation; source-faithful output remains untouched.
    source_utterances = segments_to_utterances(segments, "strict")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_origin = transcript_path.parent / "source_faithful_ja.srt"
    source_output = output_dir / "source_faithful_ja.srt"
    if source_origin.is_file():
        # A rebuild changes only viewer presentation.  If its source bundle is
        # available, keep its exact source SRT bytes (including intentional
        # whitespace) rather than reconstructing it from JSONL.
        source_rows = read_srt_rows(source_origin)
        source = [
            Cue(index, float(row["start"]), float(row["end"]), str(row["text"]), str(row["text"]), [])
            for index, row in enumerate(source_rows, 1)
        ]
        reconstructed_source = source_faithful_cues(source_utterances)
        if (
            [cue.text for cue in source] != [cue.text for cue in reconstructed_source]
            or [(_srt_time(cue.start), _srt_time(cue.end)) for cue in source]
            != [(_srt_time(cue.start), _srt_time(cue.end)) for cue in reconstructed_source]
        ):
            raise ValueError("Existing source-faithful SRT does not match the transcript JSONL.")
        if source_origin.resolve() != source_output.resolve():
            shutil.copyfile(source_origin, source_output)
    else:
        source = source_faithful_cues(source_utterances)
        write_srt(source_output, source)
    viewer = viewer_cues(source_utterances, get_profile(profile_name), normalization)
    write_srt(output_dir / "viewer_ja.srt", viewer)
    rebuilt_transcript = output_dir / "transcript_ja.jsonl"
    if transcript_path.resolve() != rebuilt_transcript.resolve():
        shutil.copyfile(transcript_path, rebuilt_transcript)
    write_jsonl(output_dir / "subtitle_audit.jsonl", cue_audit(viewer))
    report = qc_report(
        viewer, segments, max((segment.end for segment in segments), default=0.0), source_cues=source,
    )
    report["backend"] = "rebuild"
    report["rebuild"] = {
        "source_transcript": str(transcript_path),
        "reference_supplied": external_reference is not None,
    }
    report["viewer_runaway_compactions"] = sum(
        any(code in cue.warnings for code in ("viewer_runaway_repetition_compacted", "viewer_high_confidence_runaway_compacted"))
        for cue in viewer
    )
    report["viewer_runaway_collapses"] = sum(
        any(code in cue.warnings for code in ("viewer_runaway_repetition_collapsed", "viewer_high_confidence_runaway_collapsed"))
        for cue in viewer
    )
    if external_reference is not None:
        report["reference_comparison"] = reference_comparison(viewer, external_reference, source)
    (output_dir / "qc_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    write_comparison_report(output_dir / "comparison_report.html", segments, source, viewer)
    from .review import write_review_artifacts
    write_review_artifacts(
        output_dir / "review_manifest.json", output_dir / "review_report.html", viewer, segments,
        external_reference=external_reference,
        video_duration=max((segment.end for segment in segments), default=0.0), video_path=video_path,
        source_evidence_label="ASR 원문 (재분할)",
        quality_gate=report["content_quality_gate"],
        target_windows=review_samples,
    )
    _write_bundle_verification(
        output_dir,
        reference_srt=reference_srt,
        video_path=video_path,
        review_samples=review_samples,
    )
    return report
