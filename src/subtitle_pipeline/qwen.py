from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .models import SourceSegment, Word
from .text import display_width


DEFAULT_QWEN_ROOT = Path("Qwen3ASR")
QWEN_ROOT_ENV = "SUBTITLE_PIPELINE_QWEN_ROOT"


def _configured_qwen_root() -> Path:
    configured = os.environ.get(QWEN_ROOT_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_QWEN_ROOT


@dataclass(frozen=True, slots=True)
class QwenRuntime:
    python: Path
    model: Path
    aligner: Path

    @classmethod
    def discover(cls, root: Path | None = None) -> "QwenRuntime":
        runtime_root = root if root is not None else _configured_qwen_root()
        return cls(
            runtime_root / "venv" / "Scripts" / "python.exe",
            runtime_root / "Model",
            runtime_root / "ForcedAligner",
        )

    def validate(self) -> None:
        missing = [str(path) for path in (self.python, self.model, self.aligner) if not path.exists()]
        if missing:
            raise RuntimeError("Qwen runtime files are missing: " + ", ".join(missing))


def parse_qwen_jsonl(path: Path) -> list[Word]:
    """Parse forced-alignment items emitted by the project-local Qwen helper."""
    words: list[Word] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            text, start, end = str(row["text"]).strip(), float(row["start"]), float(row["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid Qwen alignment item at line {line_number}") from exc
        if text and end >= start:
            words.append(Word(
                text, start, end, backend="qwen", source_chunk=row.get("source_chunk"),
                source_interval=row.get("source_interval"),
            ))
    return sorted(words, key=lambda word: (word.start, word.end))


def _extract_wav(input_path: Path, destination: Path, max_duration: float | None) -> None:
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(input_path)]
    if max_duration:
        command += ["-t", str(max_duration)]
    command += ["-vn", "-ac", "1", "-ar", "16000", str(destination)]
    subprocess.run(command, check=True)


def _compact_targeted_intervals(
    intervals: list[dict[str, object]], source_duration: float,
) -> list[dict[str, float | bool]]:
    """Map sparse source ranges into one compact WAV while retaining source time."""
    compact: list[dict[str, float]] = []
    previous_end = 0.0
    previous_allow_overlap = False
    audio_cursor = 0.0
    for index, item in enumerate(intervals):
        try:
            source_start, source_end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Targeted Qwen interval {index} needs numeric start and end") from exc
        if not 0.0 <= source_start < source_end <= source_duration:
            raise RuntimeError(f"Targeted Qwen interval {index} is outside the source duration")
        allow_overlap = bool(item.get("allow_overlap", False))
        if source_start < previous_end and not (allow_overlap or previous_allow_overlap):
            raise RuntimeError("Targeted Qwen intervals must be sorted and non-overlapping")
        length = source_end - source_start
        compact_item: dict[str, float | bool] = {
            "source_start": source_start,
            "source_end": source_end,
            "audio_start": audio_cursor,
            "audio_end": audio_cursor + length,
        }
        if allow_overlap:
            compact_item["allow_overlap"] = True
        compact.append(compact_item)
        previous_end = max(previous_end, source_end)
        previous_allow_overlap = allow_overlap
        audio_cursor += length
    return compact


def _extract_targeted_wav(input_path: Path, destination: Path, intervals: list[dict[str, float]]) -> None:
    """Extract only selected source ranges into a compact 16 kHz mono WAV."""
    if not intervals:
        raise RuntimeError("Targeted Qwen extraction requires at least one interval")
    filters = [
        f"[0:a]atrim=start={item['source_start']:.6f}:end={item['source_end']:.6f},asetpts=PTS-STARTPTS[a{index}]"
        for index, item in enumerate(intervals)
    ]
    if len(intervals) == 1:
        filter_graph = filters[0] + ";[a0]aresample=16000,aformat=channel_layouts=mono[outa]"
    else:
        inputs = "".join(f"[a{index}]" for index in range(len(intervals)))
        filter_graph = ";".join(filters) + f";{inputs}concat=n={len(intervals)}:v=0:a=1,aresample=16000,aformat=channel_layouts=mono[outa]"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(input_path),
        "-filter_complex", filter_graph, "-map", "[outa]", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(destination),
    ], check=True)


def _media_duration(path: Path) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(completed.stdout.strip())


def _cache_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _canonical_intervals(intervals: list[dict[str, object]] | None) -> list[dict[str, float]]:
    return [
        {"start": round(float(item["start"]), 6), "end": round(float(item["end"]), 6)}
        for item in intervals or []
    ]


def _qwen_cache_metadata(
    input_path: Path,
    runtime: QwenRuntime,
    helper: Path,
    duration: float,
    language: str,
    chunk_seconds: int,
    max_new_tokens: int,
    targeted_intervals: list[dict[str, object]] | None,
    cache_identity_path: Path | None = None,
) -> dict[str, object]:
    """Describe exactly the audio, runtime and audit scope represented by a cache."""
    identity_path = cache_identity_path or input_path
    stat = identity_path.stat()
    return {
        "schema_version": 2,
        "mode": "targeted" if targeted_intervals is not None else "full",
        "input": {
            # The inference path may be a newly-created normalized temporary
            # WAV.  Cache identity must follow the stable source container or
            # every rerun misses even when the media and options are unchanged.
            "path": str(identity_path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "duration_seconds": round(duration, 6),
        },
        "runtime": {
            "model": str(runtime.model.resolve()),
            "aligner": str(runtime.aligner.resolve()),
            "helper_sha256": sha256(helper.read_bytes()).hexdigest(),
        },
        "language": language,
        "chunk_seconds": chunk_seconds,
        "max_new_tokens": max_new_tokens,
        "intervals": _canonical_intervals(targeted_intervals),
    }


def _cache_matches(cache_path: Path, expected: dict[str, object]) -> bool:
    metadata_path = _cache_metadata_path(cache_path)
    if not cache_path.is_file() or not metadata_path.is_file():
        return False
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8")) == expected
    except (OSError, json.JSONDecodeError):
        return False


def _write_cache_metadata(cache_path: Path, metadata: dict[str, object]) -> None:
    metadata_path = _cache_metadata_path(cache_path)
    temporary_path = metadata_path.with_name(f".{metadata_path.name}.tmp")
    temporary_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(metadata_path)


def _tag_cache_status(segments: list[SourceSegment], status: str) -> list[SourceSegment]:
    for segment in segments:
        segment.metadata["qwen_cache"] = status
    return segments


def _is_sentence_end(text: str) -> bool:
    return text.rstrip().endswith(("。", "！", "？", ".", "!", "?"))


def qwen_words_to_segments(
    words: list[Word], max_chars: int = 42, max_duration: float = 4.5, decision: str = "qwen_primary",
) -> list[SourceSegment]:
    """Group aligned Qwen items without modifying their text or timing."""
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current and word.source_interval != current[-1].source_interval:
            groups.append(current)
            current = []
        current.append(word)
        text = "".join(item.text for item in current)
        duration = current[-1].end - current[0].start
        if display_width(text) >= max_chars or duration >= max_duration or (_is_sentence_end(text) and duration >= 1.2):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return [
        SourceSegment(
            id=index,
            start=group[0].start,
            end=max(group[-1].end, group[0].start + 0.01),
            text="".join(word.text for word in group).strip(),
            words=group,
            metadata={"backend": "qwen", "decision": decision, "qwen_item_count": len(group)},
        )
        for index, group in enumerate(groups)
        if "".join(word.text for word in group).strip()
    ]


def transcribe_qwen(
    input_path: Path,
    runtime: QwenRuntime,
    language: str = "ja",
    chunk_seconds: int = 180,
    max_new_tokens: int = 256,
    max_duration: float | None = None,
    alignment_cache_path: Path | None = None,
    reuse_alignment_cache: bool = True,
    targeted_intervals: list[dict[str, object]] | None = None,
    cache_identity_path: Path | None = None,
) -> tuple[list[SourceSegment], float]:
    helper = Path(__file__).with_name("qwen_worker.py")
    if not helper.is_file():
        raise RuntimeError("Missing project Qwen helper: subtitle_pipeline/qwen_worker.py")
    duration = _media_duration(input_path)
    if max_duration is not None:
        duration = min(duration, max_duration)
    targeted = targeted_intervals is not None
    decision = "qwen_verification" if targeted else "qwen_primary"
    cache_metadata = _qwen_cache_metadata(
        input_path, runtime, helper, duration, language, chunk_seconds, max_new_tokens, targeted_intervals,
        cache_identity_path,
    )
    if (
        max_duration is None
        and alignment_cache_path is not None
        and reuse_alignment_cache
        and _cache_matches(alignment_cache_path, cache_metadata)
    ):
        segments = qwen_words_to_segments(parse_qwen_jsonl(alignment_cache_path), decision=decision)
        return _tag_cache_status(segments, "reused"), duration

    runtime.validate()
    with tempfile.TemporaryDirectory(prefix="subtitle_pipeline_qwen_") as temporary:
        temporary_path = Path(temporary)
        wav_path = temporary_path / "input.wav"
        intervals_path = temporary_path / "intervals.json"
        jsonl_path = alignment_cache_path or temporary_path / "aligned_items.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        compact_intervals: list[dict[str, float]] | None = None
        if targeted:
            compact_intervals = _compact_targeted_intervals(targeted_intervals or [], duration)
            _extract_targeted_wav(input_path, wav_path, compact_intervals)
        else:
            _extract_wav(input_path, wav_path, max_duration)
        command = [
            str(runtime.python), str(helper), "--model", str(runtime.model), "--aligner", str(runtime.aligner),
            "--audio", str(wav_path), "--output", str(jsonl_path), "--language", language,
            "--chunk-seconds", str(chunk_seconds), "--max-new-tokens", str(max_new_tokens),
        ]
        if targeted:
            intervals_path.write_text(json.dumps(compact_intervals, ensure_ascii=False), encoding="utf-8")
            command += ["--intervals-json", str(intervals_path)]
        subprocess.run(command, check=True)
        words = parse_qwen_jsonl(jsonl_path)
    if alignment_cache_path is not None:
        _write_cache_metadata(alignment_cache_path, cache_metadata)
    segments = qwen_words_to_segments(words, decision=decision)
    return _tag_cache_status(segments, "created" if alignment_cache_path is not None else "ephemeral"), duration
