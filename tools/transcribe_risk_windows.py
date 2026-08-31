from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from translation_forensics.srt import SubtitleBlock, parse_srt


BANNED_HALLUCINATIONS = (
    "ご視聴ありがとうございました",
    "次の動画でお会いしましょう",
    "おやすみなさい",
)


@dataclass(frozen=True)
class Window:
    window_id: str
    start: float
    end: float
    block_numbers: tuple[int, ...]

    @property
    def duration(self) -> float:
        return self.end - self.start


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_changed_numbers(ledger_path: Path, file_name: str) -> set[int]:
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    numbers = {
        int(row["block"])
        for row in rows
        if str(row.get("file")) == file_name and "block" in row
    }
    if not numbers:
        raise ValueError(f"No ledger rows found for {file_name}")
    return numbers


def audio_duration(path: Path, *, ffmpeg: Path | None = None) -> float:
    try:
        with wave.open(str(path), "rb") as source:
            rate = source.getframerate()
            frames = source.getnframes()
    except (wave.Error, EOFError):
        if ffmpeg is None:
            raise ValueError(f"A WAV file or --ffmpeg is required for: {path}") from None
        result = subprocess.run(
            [str(ffmpeg), "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", result.stderr)
        if match is None:
            raise ValueError(f"Could not determine audio duration: {path}")
        hours, minutes, seconds = match.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if rate <= 0 or frames <= 0:
        raise ValueError(f"Invalid WAV header: {path}")
    return frames / rate


def _merge_intervals(
    intervals: Iterable[tuple[float, float]], *, merge_gap: float
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + merge_gap:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def plan_windows(
    blocks: list[SubtitleBlock],
    changed_numbers: set[int],
    *,
    duration: float,
    context: float = 2.0,
    merge_gap: float = 1.0,
    maximum_window: float = 28.0,
    split_overlap: float = 2.0,
) -> list[Window]:
    if not 0 <= split_overlap < maximum_window:
        raise ValueError("split_overlap must be smaller than maximum_window")
    selected = [block for block in blocks if block.number in changed_numbers]
    missing = sorted(changed_numbers - {block.number for block in selected})
    if missing:
        raise ValueError(f"Ledger block numbers are absent from SRT: {missing[:10]}")
    intervals = [
        (max(0.0, block.start_seconds - context), min(duration, block.end_seconds + context))
        for block in selected
    ]
    windows: list[Window] = []
    for merged_start, merged_end in _merge_intervals(intervals, merge_gap=merge_gap):
        cursor = merged_start
        while cursor < merged_end - 1e-6:
            end = min(merged_end, cursor + maximum_window)
            covered = tuple(
                block.number
                for block in selected
                if block.start_seconds < end and block.end_seconds > cursor
            )
            windows.append(
                Window(
                    window_id=f"risk-{len(windows) + 1:04d}",
                    start=round(cursor, 3),
                    end=round(end, 3),
                    block_numbers=covered,
                )
            )
            if end >= merged_end - 1e-6:
                break
            cursor = end - split_overlap
    return windows


def extract_wav_window(
    source_path: Path,
    destination: Path,
    window: Window,
    *,
    ffmpeg: Path | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.casefold() != ".wav":
        if ffmpeg is None:
            raise ValueError(f"A WAV file or --ffmpeg is required for: {source_path}")
        subprocess.run(
            [
                str(ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{window.start:.3f}",
                "-t",
                f"{window.duration:.3f}",
                "-i",
                str(source_path),
                "-map",
                "0:a:0",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(destination),
            ],
            check=True,
        )
        return
    with wave.open(str(source_path), "rb") as source:
        params = source.getparams()
        rate = source.getframerate()
        start_frame = max(0, round(window.start * rate))
        end_frame = min(source.getnframes(), round(window.end * rate))
        source.setpos(start_frame)
        payload = source.readframes(end_frame - start_frame)
    with wave.open(str(destination), "wb") as output:
        output.setparams(params)
        output.writeframes(payload)


def configure_nvidia_dlls() -> list[Any]:
    site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    handles: list[Any] = []
    if not site_packages.is_dir():
        return handles
    paths = [path for path in site_packages.glob("*/bin") if path.is_dir()]
    if paths:
        os.environ["PATH"] = os.pathsep.join(str(path) for path in paths) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        for path in paths:
            handles.append(os.add_dll_directory(str(path)))
    return handles


def normalized_text(text: str) -> str:
    return "".join(text.split()).strip("。.!！?？")


def suspected_hallucination(text: str) -> bool:
    normalized = normalized_text(text)
    return any(normalized == normalized_text(pattern) for pattern in BANNED_HALLUCINATIONS)


def overlap_seconds(start: float, end: float, block: SubtitleBlock) -> float:
    return max(0.0, min(end, block.end_seconds) - max(start, block.start_seconds))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_block_evidence(
    blocks: list[SubtitleBlock],
    changed_numbers: set[int],
    window_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in blocks:
        if block.number not in changed_numbers:
            continue
        matches: list[dict[str, Any]] = []
        seen: set[tuple[float, float, str]] = set()
        for window in window_rows:
            for segment in window.get("segments", []):
                start = float(segment["start_seconds"])
                end = float(segment["end_seconds"])
                if overlap_seconds(start, end, block) <= 0:
                    continue
                key = (start, end, str(segment["text"]))
                if key in seen:
                    continue
                seen.add(key)
                matches.append(segment)
        rows.append(
            {
                "block": block.number,
                "start": block.start,
                "end": block.end,
                "current_text": block.text,
                "asr_segments": sorted(matches, key=lambda row: (row["start_seconds"], row["end_seconds"])),
                "eligible_asr_text": " ".join(
                    str(row["text"])
                    for row in sorted(matches, key=lambda item: (item["start_seconds"], item["end_seconds"]))
                    if not row["suspected_hallucination"]
                ).strip(),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe only subtitle intervals listed in a change ledger.")
    parser.add_argument("--srt", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path, help="16 kHz mono PCM WAV")
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        help="Required for non-WAV input; extracts each selected window directly to 16 kHz mono WAV.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--file-name")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--context", type=float, default=2.0)
    parser.add_argument("--merge-gap", type=float, default=1.0)
    parser.add_argument("--maximum-window", type=float, default=28.0)
    parser.add_argument("--split-overlap", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    args.srt = args.srt.expanduser().resolve()
    args.ledger = args.ledger.expanduser().resolve()
    args.audio = args.audio.expanduser().resolve()
    args.ffmpeg = args.ffmpeg.expanduser().resolve() if args.ffmpeg else None
    args.output_dir = args.output_dir.expanduser().resolve()
    file_name = args.file_name or args.srt.name
    blocks, _, _ = parse_srt(args.srt)
    changed_numbers = read_changed_numbers(args.ledger, file_name)
    duration = audio_duration(args.audio, ffmpeg=args.ffmpeg)
    windows = plan_windows(
        blocks,
        changed_numbers,
        duration=duration,
        context=args.context,
        merge_gap=args.merge_gap,
        maximum_window=args.maximum_window,
        split_overlap=args.split_overlap,
    )
    identity = {
        "schema_name": "translation-forensics/risk-window-asr",
        "schema_version": "1",
        "srt": str(args.srt),
        "srt_sha256": sha256_file(args.srt),
        "ledger": str(args.ledger),
        "ledger_sha256": sha256_file(args.ledger),
        "audio": str(args.audio),
        "audio_sha256": sha256_file(args.audio),
        "audio_duration_seconds": duration,
        "ffmpeg": str(args.ffmpeg) if args.ffmpeg else None,
        "file_name": file_name,
        "changed_block_count": len(changed_numbers),
        "window_count": len(windows),
        "total_window_seconds": round(sum(window.duration for window in windows), 3),
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "context_seconds": args.context,
        "merge_gap_seconds": args.merge_gap,
        "maximum_window_seconds": args.maximum_window,
        "split_overlap_seconds": args.split_overlap,
    }
    plan_path = args.output_dir / "asr-plan.json"
    rows_path = args.output_dir / "asr-windows.jsonl"
    evidence_path = args.output_dir / "block-acoustic-evidence.jsonl"
    report_path = args.output_dir / "asr-report.json"
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError(f"Existing ASR plan does not match current inputs: {plan_path}")
    else:
        write_json(plan_path, identity)
    if args.plan_only:
        print(json.dumps(identity, ensure_ascii=False, indent=2))
        return

    existing_rows = load_jsonl(rows_path) if args.resume else []
    if rows_path.exists() and not args.resume:
        raise FileExistsError(f"Output exists; use --resume: {rows_path}")
    completed = {str(row["window_id"]) for row in existing_rows}
    dll_handles = configure_nvidia_dlls()
    from faster_whisper import WhisperModel

    started = time.perf_counter()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    clips_dir = args.output_dir / "clips"
    for index, window in enumerate(windows, 1):
        if window.window_id in completed:
            continue
        clip_path = clips_dir / f"{window.window_id}.wav"
        if not clip_path.is_file():
            extract_wav_window(args.audio, clip_path, window, ffmpeg=args.ffmpeg)
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
            start = round(window.start + float(segment.start), 3)
            end = round(window.start + float(segment.end), 3)
            words = [
                {
                    "start_seconds": round(window.start + float(word.start), 3),
                    "end_seconds": round(window.start + float(word.end), 3),
                    "word": word.word,
                    "probability": getattr(word, "probability", None),
                }
                for word in (segment.words or [])
            ]
            segments.append(
                {
                    "start_seconds": start,
                    "end_seconds": end,
                    "text": text,
                    "avg_logprob": getattr(segment, "avg_logprob", None),
                    "no_speech_prob": getattr(segment, "no_speech_prob", None),
                    "compression_ratio": getattr(segment, "compression_ratio", None),
                    "suspected_hallucination": suspected_hallucination(text),
                    "words": words,
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
        existing_rows.append(row)
        if index == 1 or index % 10 == 0 or index == len(windows):
            print(
                f"[{index}/{len(windows)}] {window.window_id} "
                f"{window.start:.3f}-{window.end:.3f}s elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    del model
    del dll_handles

    window_rows = sorted(load_jsonl(rows_path), key=lambda row: row["window_id"])
    block_rows = build_block_evidence(blocks, changed_numbers, window_rows)
    evidence_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in block_rows),
        encoding="utf-8",
        newline="\n",
    )
    hallucinations = sum(
        bool(segment["suspected_hallucination"])
        for row in window_rows
        for segment in row.get("segments", [])
    )
    report = {
        **identity,
        "status": "complete",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "completed_windows": len(window_rows),
        "asr_segment_count": sum(len(row.get("segments", [])) for row in window_rows),
        "suspected_hallucination_segment_count": hallucinations,
        "blocks_with_eligible_asr": sum(bool(row["eligible_asr_text"]) for row in block_rows),
        "blocks_without_eligible_asr": sum(not bool(row["eligible_asr_text"]) for row in block_rows),
        "external_transfer": False,
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
