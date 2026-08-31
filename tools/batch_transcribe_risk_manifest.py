from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transcribe_risk_windows import (  # noqa: E402
    append_jsonl,
    audio_duration,
    build_block_evidence,
    configure_nvidia_dlls,
    extract_wav_window,
    plan_windows,
    read_changed_numbers,
    sha256_file,
    suspected_hallucination,
    write_json,
)
from translation_forensics.srt import parse_srt  # noqa: E402


def load_manifest(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    items = value.get("items") if isinstance(value, dict) else value
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must contain a non-empty items list")
    return [{str(key): str(item[key]) for key in item} for item in items]


def transcribe_item(
    item: dict[str, str],
    *,
    ledger: Path,
    model: Any,
    model_name: str,
    device: str,
    compute_type: str,
    ffmpeg: Path | None,
    context: float,
    merge_gap: float,
    maximum_window: float,
    split_overlap: float,
) -> dict[str, Any]:
    title = item["title"]
    srt = Path(item["srt"]).expanduser().resolve()
    audio = Path(item["audio"]).expanduser().resolve()
    output_dir = Path(item["output_dir"]).expanduser().resolve()
    file_name = item.get("file_name") or srt.name
    blocks, _, _ = parse_srt(srt)
    changed_numbers = read_changed_numbers(ledger, file_name)
    duration = audio_duration(audio, ffmpeg=ffmpeg)
    windows = plan_windows(
        blocks,
        changed_numbers,
        duration=duration,
        context=context,
        merge_gap=merge_gap,
        maximum_window=maximum_window,
        split_overlap=split_overlap,
    )
    identity = {
        "schema_name": "translation-forensics/risk-window-asr",
        "schema_version": "1",
        "srt": str(srt),
        "srt_sha256": sha256_file(srt),
        "ledger": str(ledger),
        "ledger_sha256": sha256_file(ledger),
        "audio": str(audio),
        "audio_sha256": sha256_file(audio),
        "audio_duration_seconds": duration,
        "ffmpeg": str(ffmpeg) if ffmpeg else None,
        "file_name": file_name,
        "changed_block_count": len(changed_numbers),
        "window_count": len(windows),
        "total_window_seconds": round(sum(window.duration for window in windows), 3),
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "context_seconds": context,
        "merge_gap_seconds": merge_gap,
        "maximum_window_seconds": maximum_window,
        "split_overlap_seconds": split_overlap,
    }
    plan_path = output_dir / "asr-plan.json"
    rows_path = output_dir / "asr-windows.jsonl"
    evidence_path = output_dir / "block-acoustic-evidence.jsonl"
    report_path = output_dir / "asr-report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if all(report.get(key) == value for key, value in identity.items()):
            print(f"[{title}] already complete", flush=True)
            return report
        raise RuntimeError(f"Existing report does not match current inputs: {report_path}")
    if rows_path.exists():
        raise FileExistsError(f"Partial output exists and cannot be overwritten: {rows_path}")
    write_json(plan_path, identity)

    started = time.perf_counter()
    clips_dir = output_dir / "clips"
    window_rows: list[dict[str, Any]] = []
    for index, window in enumerate(windows, 1):
        clip_path = clips_dir / f"{window.window_id}.wav"
        extract_wav_window(audio, clip_path, window, ffmpeg=ffmpeg)
        raw_segments, info = model.transcribe(
            str(clip_path),
            language="ja",
            task="transcribe",
            word_timestamps=True,
            vad_filter=False,
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        segments: list[dict[str, Any]] = []
        for segment in raw_segments:
            text = segment.text.strip()
            if not text:
                continue
            segments.append(
                {
                    "start_seconds": round(window.start + float(segment.start), 3),
                    "end_seconds": round(window.start + float(segment.end), 3),
                    "text": text,
                    "avg_logprob": getattr(segment, "avg_logprob", None),
                    "no_speech_prob": getattr(segment, "no_speech_prob", None),
                    "compression_ratio": getattr(segment, "compression_ratio", None),
                    "suspected_hallucination": suspected_hallucination(text),
                    "words": [
                        {
                            "start_seconds": round(window.start + float(word.start), 3),
                            "end_seconds": round(window.start + float(word.end), 3),
                            "word": word.word,
                            "probability": getattr(word, "probability", None),
                        }
                        for word in (segment.words or [])
                    ],
                }
            )
        row = {
            "window_id": window.window_id,
            "start_seconds": window.start,
            "end_seconds": window.end,
            "duration_seconds": window.duration,
            "block_numbers": list(window.block_numbers),
            "clip": str(clip_path),
            "clip_sha256": sha256_file(clip_path),
            "decoded_audio_duration_seconds": float(info.duration),
            "segments": segments,
        }
        append_jsonl(rows_path, row)
        window_rows.append(row)
        print(f"[{title} {index}/{len(windows)}] {window.start:.3f}-{window.end:.3f}s", flush=True)

    block_rows = build_block_evidence(blocks, changed_numbers, window_rows)
    evidence_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in block_rows),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        **identity,
        "status": "complete",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "completed_windows": len(window_rows),
        "asr_segment_count": sum(len(row["segments"]) for row in window_rows),
        "suspected_hallucination_segment_count": sum(
            bool(segment["suspected_hallucination"])
            for row in window_rows
            for segment in row["segments"]
        ),
        "blocks_with_eligible_asr": sum(bool(row["eligible_asr_text"]) for row in block_rows),
        "blocks_without_eligible_asr": sum(not bool(row["eligible_asr_text"]) for row in block_rows),
        "external_transfer": False,
    }
    write_json(report_path, report)
    print(f"[{title}] complete in {report['elapsed_seconds']}s", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe risk windows for multiple titles with one model load.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        help="Required for non-WAV manifest audio; extracts each selected window directly to WAV.",
    )
    parser.add_argument("--context", type=float, default=2.0)
    parser.add_argument("--merge-gap", type=float, default=1.0)
    parser.add_argument("--maximum-window", type=float, default=28.0)
    parser.add_argument("--split-overlap", type=float, default=2.0)
    parser.add_argument(
        "--output-dir-name",
        help="Replace the final directory name from each manifest item (for a second-model pass).",
    )
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    ledger = args.ledger.expanduser().resolve()
    args.ffmpeg = args.ffmpeg.expanduser().resolve() if args.ffmpeg else None
    items = load_manifest(manifest)
    if args.output_dir_name:
        for item in items:
            item["output_dir"] = str(Path(item["output_dir"]).with_name(args.output_dir_name))
    dll_handles = configure_nvidia_dlls()
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    reports = [
        transcribe_item(
            item,
            ledger=ledger,
            model=model,
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            ffmpeg=args.ffmpeg,
            context=args.context,
            merge_gap=args.merge_gap,
            maximum_window=args.maximum_window,
            split_overlap=args.split_overlap,
        )
        for item in items
    ]
    del model
    del dll_handles
    print(
        json.dumps(
            {
                "status": "complete",
                "model": args.model,
                "title_count": len(reports),
                "window_count": sum(int(report["window_count"]) for report in reports),
                "external_transfer": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
