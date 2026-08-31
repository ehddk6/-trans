from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _configure_nvidia_dlls() -> list[Any]:
    site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    handles: list[Any] = []
    paths = [path for path in site_packages.glob("*/bin") if path.is_dir()]
    if paths:
        os.environ["PATH"] = os.pathsep.join(str(path) for path in paths) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        for path in paths:
            handles.append(os.add_dll_directory(str(path)))
    return handles


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_row(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _extract_clip(
    *, ffmpeg: Path, media: Path, output: Path, start: float, duration: float
) -> None:
    if output.is_file() and output.stat().st_size > 44:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            str(ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(media),
            "-t",
            f"{duration:.3f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(output),
        ],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size <= 44:
        raise RuntimeError(f"Audio extraction failed: {completed.stderr[-500:]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpointed full-media Japanese ASR.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--window-seconds", type=float, default=600.0)
    parser.add_argument("--overlap-seconds", type=float, default=3.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--max-windows", type=int, default=0)
    args = parser.parse_args()

    media = args.media.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not media.is_file() or not ffmpeg.is_file():
        raise FileNotFoundError("Media or ffmpeg is missing")
    if not 0 <= args.overlap_seconds < args.window_seconds:
        raise ValueError("Overlap must be smaller than the window")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "asr-windows.jsonl"
    plan_path = output_dir / "asr-plan.json"
    report_path = output_dir / "asr-report.json"
    stat = media.stat()
    identity = {
        "schema_name": "translation-forensics/full-media-window-asr",
        "schema_version": 1,
        "title": args.title.upper(),
        "media": str(media),
        "media_bytes": stat.st_size,
        "media_mtime_ns": stat.st_mtime_ns,
        "duration_seconds": args.duration,
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "window_seconds": args.window_seconds,
        "overlap_seconds": args.overlap_seconds,
        "start_seconds": args.start_seconds,
    }
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("Existing ASR plan does not match current inputs")
    else:
        _write_json(plan_path, identity)

    starts: list[float] = []
    cursor = args.start_seconds
    while cursor < args.duration - 0.001:
        starts.append(cursor)
        cursor += args.window_seconds - args.overlap_seconds
    if args.max_windows > 0:
        starts = starts[: args.max_windows]
    completed_rows = _read_rows(rows_path)
    completed_ids = {str(row["window_id"]) for row in completed_rows}

    _dll_handles = _configure_nvidia_dlls()
    from faster_whisper import WhisperModel

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        local_files_only=True,
    )
    started = time.perf_counter()
    for index, start in enumerate(starts, 1):
        window_id = f"window-{index:04d}"
        if window_id in completed_ids:
            print(f"[{args.title} {index}/{len(starts)}] cached", flush=True)
            continue
        end = min(args.duration, start + args.window_seconds)
        clip = output_dir / "clips" / f"{window_id}.wav"
        _extract_clip(
            ffmpeg=ffmpeg,
            media=media,
            output=clip,
            start=start,
            duration=end - start,
        )
        raw_segments, info = model.transcribe(
            str(clip),
            language="ja",
            task="transcribe",
            vad_filter=True,
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        segments = []
        for segment in raw_segments:
            text = str(segment.text).strip()
            if not text:
                continue
            segments.append(
                {
                    "start_seconds": round(start + float(segment.start), 3),
                    "end_seconds": round(start + float(segment.end), 3),
                    "text": text,
                    "avg_logprob": float(getattr(segment, "avg_logprob", 0.0) or 0.0),
                    "no_speech_prob": float(
                        getattr(segment, "no_speech_prob", 0.0) or 0.0
                    ),
                }
            )
        row = {
            "window_id": window_id,
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "decoded_audio_duration_seconds": float(info.duration),
            "segments": segments,
        }
        _append_row(rows_path, row)
        completed_rows.append(row)
        print(
            f"[{args.title} {index}/{len(starts)}] {start:.1f}-{end:.1f}s "
            f"segments={len(segments)}",
            flush=True,
        )

    report = {
        **identity,
        "status": "complete" if len(completed_rows) == len(starts) else "partial",
        "window_count": len(starts),
        "completed_window_count": len(completed_rows),
        "segment_count": sum(len(row["segments"]) for row in completed_rows),
        "elapsed_seconds_last_run": round(time.perf_counter() - started, 3),
        "external_transfer": False,
    }
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
