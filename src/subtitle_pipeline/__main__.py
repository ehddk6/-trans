from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import rebuild_presentation_from_transcript, run_pipeline
from .profiles import PROFILES
from .qwen import QwenRuntime
from .review_candidate import build_source_review_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Create faithful and viewer-oriented Japanese SRT subtitles.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--rebuild-from-transcript", type=Path,
                        help="Regenerate presentation artifacts without rerunning ASR.")
    parser.add_argument("--build-source-review-candidate", action="store_true",
                        help="Build a separate Qwen anomaly-review candidate without changing source-faithful SRT.")
    parser.add_argument("--max-review-targets", type=int, default=8,
                        help="Maximum possible_periodic_repetition cues to send to targeted Qwen review.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--backend", choices=["faster-whisper", "qwen", "ensemble", "reference"], default="faster-whisper")
    parser.add_argument("--qwen-python", type=Path)
    parser.add_argument("--qwen-model", type=Path)
    parser.add_argument("--qwen-aligner", type=Path)
    parser.add_argument("--reference-srt", type=Path,
                        help="Reference SRT; required for --backend reference and optional for ensemble review.")
    parser.add_argument("--normalized-audio", type=Path,
                        help="Persistent 16 kHz mono WAV shared by Whisper and Qwen; review still uses --input.")
    parser.add_argument("--review-samples", type=int, default=30)
    parser.add_argument("--qwen-audit-samples", type=int, default=4,
                        help="Maximum distributed short Qwen audit windows after the full Whisper pass.")
    parser.add_argument("--reuse-qwen-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Reuse a completed qwen_alignment_ja.jsonl in the output folder.")
    parser.add_argument("--language", default="ja", choices=["ja"])
    parser.add_argument("--profile", default="viewer_ja", choices=PROFILES)
    parser.add_argument(
        "--normalization", default="conservative", choices=["layout", "strict", "conservative", "viewer"],
        help="layout/strict preserve transcript content; viewer enables audited display compaction.",
    )
    parser.add_argument("--word-timestamps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vad", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable VAD filtering. Disabled by default to preserve low-volume dialogue.")
    parser.add_argument("--scene-detection", default="off", choices=["off", "auto"], help="FFmpeg soft boundary signal; never a hard split boundary.")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--condition-on-previous-text", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-duration", type=float, help="Transcribe only the opening N seconds (useful for a validation sample).")
    args = parser.parse_args()
    if args.build_source_review_candidate:
        if not args.rebuild_from_transcript or not args.input:
            parser.error("--build-source-review-candidate requires --rebuild-from-transcript and --input.")
        runtime = QwenRuntime.discover()
        report = build_source_review_candidate(
            args.rebuild_from_transcript, args.input, args.output_dir,
            runtime, max_targets=args.max_review_targets,
        )
        print(f"Wrote review candidate with {report['proposal_count']} Qwen proposals to {report['candidate_path']}")
        return
    if args.rebuild_from_transcript:
        report = rebuild_presentation_from_transcript(
            args.rebuild_from_transcript, args.output_dir, args.profile, args.normalization,
            args.reference_srt, args.review_samples, args.input,
        )
    elif args.input or args.backend == "reference":
        runtime = None
        if args.backend in {"qwen", "ensemble"}:
            discovered = QwenRuntime.discover()
            runtime = QwenRuntime(args.qwen_python or discovered.python, args.qwen_model or discovered.model, args.qwen_aligner or discovered.aligner)
        report = run_pipeline(
            args.input, args.output_dir, args.model, args.language, args.profile, args.normalization,
            args.word_timestamps, args.vad, args.beam_size, args.temperature,
            args.condition_on_previous_text, args.max_duration, args.scene_detection,
            args.backend, runtime, args.reference_srt, args.review_samples, args.reuse_qwen_cache,
            args.qwen_audit_samples, args.normalized_audio,
        )
    else:
        parser.error("Specify --input, --rebuild-from-transcript, or --backend reference with --reference-srt.")
    print(f"Wrote {report['total_cues']} viewer cues to {args.output_dir}")


if __name__ == "__main__":
    main()
