"""Emit Qwen3-ASR forced-alignment items as JSONL.

This module is deliberately standalone: invoke it with the Python interpreter
from SubtitleTool's Qwen virtual environment so that its ``qwen_asr`` package
and local model directories are available.  It does not import the rest of
the subtitle pipeline and never writes an SRT file.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SAMPLE_RATE = 16_000
DEFAULT_CHUNK_SECONDS = 180
MIN_CHUNK_SECONDS = 1
MAX_CHUNK_SECONDS = 180
MIN_MAX_NEW_TOKENS = 32
MAX_MAX_NEW_TOKENS = 1_024
AUDIO_DURATION_TOLERANCE_SECONDS = 0.1
QWEN_LANGUAGE = {"ja": "Japanese"}


@dataclass(frozen=True)
class AudioChunk:
    """A contiguous zero-based source chunk and its global offset in seconds."""

    index: int
    samples: Any
    offset_seconds: float
    source_interval: int | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local Qwen3-ASR plus Qwen3-ForcedAligner and write alignment JSONL."
    )
    parser.add_argument("--model", required=True, help="Local Qwen3-ASR model directory")
    parser.add_argument("--aligner", required=True, help="Local Qwen3-ForcedAligner directory")
    parser.add_argument("--audio", required=True, help="16 kHz mono WAV input")
    parser.add_argument("--output", required=True, help="JSONL output path")
    parser.add_argument("--intervals-json", help="Optional JSON list of source-time ranges to verify")
    parser.add_argument("--language", default="ja", choices=tuple(QWEN_LANGUAGE), help="ASR language")
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help=f"Sequential recognition chunk length (1-{MAX_CHUNK_SECONDS}, default: {DEFAULT_CHUNK_SECONDS})",
    )
    parser.add_argument(
        "--max-new-tokens",
        "--max-tokens",
        dest="max_new_tokens",
        type=int,
        default=256,
        help=f"Qwen decoding limit ({MIN_MAX_NEW_TOKENS}-{MAX_MAX_NEW_TOKENS})",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    for option, value in (("--model", args.model), ("--aligner", args.aligner), ("--audio", args.audio)):
        if not Path(value).exists():
            raise RuntimeError(f"{option} path does not exist: {value}")
    if not MIN_CHUNK_SECONDS <= args.chunk_seconds <= MAX_CHUNK_SECONDS:
        raise RuntimeError(
            f"--chunk-seconds must be between {MIN_CHUNK_SECONDS} and {MAX_CHUNK_SECONDS}: {args.chunk_seconds}"
        )
    if not MIN_MAX_NEW_TOKENS <= args.max_new_tokens <= MAX_MAX_NEW_TOKENS:
        raise RuntimeError(
            "--max-new-tokens must be between "
            f"{MIN_MAX_NEW_TOKENS} and {MAX_MAX_NEW_TOKENS}: {args.max_new_tokens}"
        )


def load_intervals(path: str | None, duration_seconds: float) -> list[dict[str, float]]:
    """Load source-to-audio windows without changing their source offsets.

    Plain ``start``/``end`` entries describe a full-source WAV.  Compact Qwen
    inputs use ``source_*`` and ``audio_*`` pairs so only requested audio is
    decoded while aligned output keeps original video timestamps.
    """
    if path is None:
        return []
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"--intervals-json is not readable JSON: {path}") from exc
    if not isinstance(raw, list):
        raise RuntimeError("--intervals-json must contain a JSON list")
    intervals: list[dict[str, float]] = []
    previous_source_end = 0.0
    previous_audio_end = 0.0
    previous_allow_overlap = False
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RuntimeError(f"--intervals-json item {index} must be an object")
        try:
            if "source_start" in item or "audio_start" in item:
                source_start, source_end = float(item["source_start"]), float(item["source_end"])
                audio_start, audio_end = float(item["audio_start"]), float(item["audio_end"])
            else:
                source_start, source_end = float(item["start"]), float(item["end"])
                audio_start, audio_end = source_start, source_end
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"--intervals-json item {index} needs numeric source and audio ranges") from exc
        if not 0.0 <= source_start < source_end:
            raise RuntimeError(f"--intervals-json item {index} has an invalid source range")
        allow_overlap = bool(item.get("allow_overlap", False))
        if (
            not 0.0 <= audio_start < audio_end
            or audio_start >= duration_seconds
            or audio_end > duration_seconds + AUDIO_DURATION_TOLERANCE_SECONDS
        ):
            raise RuntimeError(f"--intervals-json item {index} is outside the audio duration")
        if (
            (source_start < previous_source_end and not (allow_overlap or previous_allow_overlap))
            or audio_start < previous_audio_end
        ):
            raise RuntimeError("--intervals-json ranges must be sorted and non-overlapping")
        if abs((source_end - source_start) - (audio_end - audio_start)) > 0.01:
            raise RuntimeError(f"--intervals-json item {index} changes the interval duration")
        interval: dict[str, float | bool] = {
            "source_start": source_start,
            "source_end": source_end,
            "audio_start": audio_start,
            "audio_end": audio_end,
        }
        if allow_overlap:
            interval["allow_overlap"] = True
        intervals.append(interval)
        previous_source_end = max(previous_source_end, source_end)
        previous_audio_end = audio_end
        previous_allow_overlap = allow_overlap
    return intervals


def read_16khz_mono_wav(audio_path: Path) -> Any:
    """Read and validate the required interchange audio without downmixing it."""
    if audio_path.suffix.lower() != ".wav":
        raise RuntimeError(f"Qwen input must be a 16 kHz mono WAV file: {audio_path}")

    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - exercised in SubtitleTool's venv
        raise RuntimeError("Qwen worker requires numpy and soundfile in the selected Python environment") from exc

    info = sf.info(str(audio_path))
    if info.samplerate != SAMPLE_RATE:
        raise RuntimeError(f"Qwen input WAV must be {SAMPLE_RATE} Hz, got {info.samplerate} Hz")
    if info.channels != 1:
        raise RuntimeError(f"Qwen input WAV must be mono, got {info.channels} channels")

    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE or audio.ndim != 2 or audio.shape[1] != 1:
        raise RuntimeError("Qwen input WAV failed 16 kHz mono validation")
    mono = audio[:, 0]
    if not np.isfinite(mono).all():
        raise RuntimeError("Qwen input WAV contains non-finite samples")
    return mono


def iter_chunks(audio: Any, chunk_seconds: int, sample_rate: int = SAMPLE_RATE) -> Iterable[AudioChunk]:
    """Yield sequential chunks; no overlap is introduced so offsets remain auditable."""
    chunk_samples = chunk_seconds * sample_rate
    for index, start in enumerate(range(0, len(audio), chunk_samples)):
        yield AudioChunk(index=index, samples=audio[start : start + chunk_samples], offset_seconds=start / sample_rate)


def iter_selected_chunks(
    audio: Any, intervals: Sequence[dict[str, float]], chunk_seconds: int, sample_rate: int = SAMPLE_RATE,
) -> Iterable[AudioChunk]:
    """Yield only requested source ranges while retaining original video time."""
    chunk_samples = chunk_seconds * sample_rate
    index = 0
    for interval_index, interval in enumerate(intervals):
        audio_start = float(interval["audio_start"]) if "audio_start" in interval else float(interval["start"])
        audio_end = float(interval["audio_end"]) if "audio_end" in interval else float(interval["end"])
        source_start = float(interval["source_start"]) if "source_start" in interval else float(interval["start"])
        start_sample = round(audio_start * sample_rate)
        end_sample = round(audio_end * sample_rate)
        for local_start in range(start_sample, end_sample, chunk_samples):
            local_end = min(end_sample, local_start + chunk_samples)
            yield AudioChunk(
                index=index,
                samples=audio[local_start:local_end],
                offset_seconds=source_start + (local_start - start_sample) / sample_rate,
                source_interval=interval_index,
            )
            index += 1


def make_alignment_rows(
    items: Sequence[Any], *, offset_seconds: float, source_chunk: int, language: str, source_interval: int | None = None,
) -> list[dict[str, object]]:
    """Convert Qwen alignment items without normalizing or changing their text."""
    rows: list[dict[str, object]] = []
    for item in items:
        row: dict[str, object] = {
            "text": item.text,
            "start": float(item.start_time) + offset_seconds,
            "end": float(item.end_time) + offset_seconds,
            "source_chunk": source_chunk,
            "language": language,
            "backend": "qwen",
        }
        if source_interval is not None:
            row["source_interval"] = source_interval
        rows.append(row)
    return rows


def write_jsonl_atomically(output_path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write completed alignment output only after all inference and alignment succeeds."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
                handle.write("\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run(args: argparse.Namespace) -> int:
    """Run ASR first, release it, then load the forced aligner and emit JSONL."""
    validate_arguments(args)
    audio_path = Path(args.audio)
    audio = read_16khz_mono_wav(audio_path)
    duration_seconds = len(audio) / SAMPLE_RATE
    intervals = load_intervals(args.intervals_json, duration_seconds)
    chunks = list(iter_selected_chunks(audio, intervals, args.chunk_seconds) if intervals else iter_chunks(audio, args.chunk_seconds))
    if not chunks:
        raise RuntimeError("Qwen input WAV has no audio samples")

    try:
        import torch
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
    except ImportError as exc:  # pragma: no cover - requires SubtitleTool's Qwen venv
        raise RuntimeError(
            "qwen_asr is unavailable. Invoke qwen_worker.py with SubtitleTool's Qwen Python interpreter."
        ) from exc

    qwen_language = QWEN_LANGUAGE[args.language]
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
    )
    transcripts: list[tuple[AudioChunk, str, str]] = []
    try:
        for chunk in chunks:
            print(f"Qwen ASR chunk {chunk.index + 1}/{len(chunks)} ({chunk.offset_seconds:.1f}s)", flush=True)
            result = model.transcribe(audio=(chunk.samples, SAMPLE_RATE), language=qwen_language)[0]
            text = result.text
            detected_language = getattr(result, "language", None) or qwen_language
            transcripts.append((chunk, text, detected_language))
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aligner = Qwen3ForcedAligner.from_pretrained(
        args.aligner,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    rows: list[dict[str, object]] = []
    try:
        for chunk, text, detected_language in transcripts:
            # Empty is the one value that cannot produce an alignment.  Do not strip
            # non-empty text: Qwen's original transcript must remain untouched.
            if text == "":
                continue
            print(f"Qwen alignment chunk {chunk.index + 1}/{len(chunks)} ({chunk.offset_seconds:.1f}s)", flush=True)
            aligned = aligner.align(
                audio=(chunk.samples, SAMPLE_RATE), text=text, language=detected_language
            )[0]
            rows.extend(
                make_alignment_rows(
                    aligned.items,
                    offset_seconds=chunk.offset_seconds,
                    source_chunk=chunk.index,
                    language=args.language,
                    source_interval=chunk.source_interval,
                )
            )
    finally:
        del aligner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for row in rows:
        for field in ("start", "end"):
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"Qwen aligner emitted a non-finite {field} timestamp")
    write_jsonl_atomically(Path(args.output), rows)
    print(f"Qwen alignment cache written: {args.output}", flush=True)
    return len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        count = run(args)
    except RuntimeError as exc:
        parser.error(str(exc))
    print(f"Qwen forced-alignment JSONL complete: {args.output} ({count} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
