from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .asr_evidence import normalize_japanese, text_similarity
from .asr_fusion import add_asr_fusion
from .srt import SubtitleBlock, parse_srt


WHISPER_FAMILY = "faster-whisper-large-v3"
REAZON_FAMILY = "reazonspeech-k2-v2"
SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")


class LocalASRError(RuntimeError):
    pass


class ASRBackend(Protocol):
    source_family: str
    model_name: str

    def transcribe(self, audio_path: Path) -> str: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class AudioWindow:
    window_id: str
    core_start: float
    core_end: float
    clip_start: float
    clip_end: float

    @property
    def duration(self) -> float:
        return self.clip_end - self.clip_start

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "core_start": round(self.core_start, 3),
            "core_end": round(self.core_end, 3),
            "clip_start": round(self.clip_start, 3),
            "clip_end": round(self.clip_end, 3),
            "duration": round(self.duration, 3),
        }


def plan_vad_windows(
    duration_seconds: float,
    silence_midpoints: list[float],
    *,
    minimum_window_seconds: float = 24.0,
    target_window_seconds: float = 26.0,
    maximum_window_seconds: float = 28.0,
    context_overlap_seconds: float = 2.0,
) -> list[AudioWindow]:
    if duration_seconds <= 0:
        raise ValueError("Audio duration must be positive")
    if not 0 < minimum_window_seconds <= target_window_seconds <= maximum_window_seconds:
        raise ValueError("Window duration contract is invalid")
    if context_overlap_seconds < 0 or context_overlap_seconds >= minimum_window_seconds:
        raise ValueError("Context overlap contract is invalid")
    midpoints = sorted({round(float(value), 6) for value in silence_midpoints if 0 < value < duration_seconds})
    windows: list[AudioWindow] = []
    clip_start = 0.0
    previous_core_end = 0.0
    while clip_start < duration_seconds - 1e-6:
        remaining = duration_seconds - clip_start
        if remaining <= maximum_window_seconds:
            clip_end = duration_seconds
        else:
            candidates = [
                point
                for point in midpoints
                if clip_start + minimum_window_seconds <= point <= clip_start + maximum_window_seconds
            ]
            clip_end = (
                min(candidates, key=lambda point: abs(point - (clip_start + target_window_seconds)))
                if candidates
                else min(duration_seconds, clip_start + target_window_seconds)
            )
        core_start = previous_core_end
        core_end = clip_end
        windows.append(
            AudioWindow(
                window_id=f"window-{len(windows) + 1:05d}",
                core_start=core_start,
                core_end=core_end,
                clip_start=clip_start,
                clip_end=clip_end,
            )
        )
        if clip_end <= clip_start:
            raise LocalASRError("VAD window planner made no progress")
        if clip_end >= duration_seconds - 1e-6:
            break
        previous_core_end = clip_end
        clip_start = max(0.0, clip_end - context_overlap_seconds)
    return windows


def probe_audio_duration(audio_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise LocalASRError("ffprobe is required for full local ASR")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise LocalASRError(f"ffprobe failed: {completed.stderr.strip()[-400:]}")
    try:
        return float(completed.stdout.strip())
    except ValueError as exc:
        raise LocalASRError("ffprobe returned an invalid duration") from exc


def detect_silence_midpoints(audio_path: Path) -> list[float]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise LocalASRError("ffmpeg is required for VAD windowing")
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(audio_path),
            "-af",
            "silencedetect=noise=-35dB:d=0.35",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise LocalASRError(f"ffmpeg silencedetect failed: {completed.stderr.strip()[-400:]}")
    starts: list[float] = []
    midpoints: list[float] = []
    for line in completed.stderr.splitlines():
        start_match = SILENCE_START_RE.search(line)
        if start_match:
            starts.append(float(start_match.group(1)))
        end_match = SILENCE_END_RE.search(line)
        if end_match and starts:
            start = starts.pop(0)
            end = float(end_match.group(1))
            if end >= start:
                midpoints.append((start + end) / 2.0)
    return midpoints


def extract_audio_windows(audio_path: Path, windows: list[AudioWindow], clips_dir: Path) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise LocalASRError("ffmpeg is required for audio extraction")
    clips_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for window in windows:
        destination = clips_dir / f"{window.window_id}.wav"
        if destination.is_file():
            paths.append(destination)
            continue
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{window.clip_start:.3f}",
                "-t",
                f"{window.duration:.3f}",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(destination),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0 or not destination.is_file():
            raise LocalASRError(f"ffmpeg window extraction failed for {window.window_id}: {completed.stderr[-400:]}")
        paths.append(destination)
    return paths


class FasterWhisperBackend:
    source_family = WHISPER_FAMILY
    model_name = "large-v3"

    def __init__(self, *, force_cpu: bool = False, local_files_only: bool = False) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise LocalASRError("faster-whisper is not installed") from exc
        cuda_available = not force_cpu and shutil.which("nvidia-smi") is not None
        device = "cuda" if cuda_available else "cpu"
        compute_type = "float16" if cuda_available else "int8"
        self._model = WhisperModel(
            self.model_name,
            device=device,
            compute_type=compute_type,
            local_files_only=local_files_only,
        )

    def transcribe(self, audio_path: Path) -> str:
        return "".join(segment["text"] for segment in self.transcribe_segments(audio_path)).strip()

    def transcribe_segments(self, audio_path: Path) -> list[dict[str, Any]]:
        segments, _ = self._model.transcribe(
            str(audio_path),
            language="ja",
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=False,
        )
        return [
            {
                "start_seconds": float(segment.start),
                "end_seconds": float(segment.end),
                "text": str(segment.text).strip(),
                "avg_logprob": float(getattr(segment, "avg_logprob", 0.0) or 0.0),
                "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
            }
            for segment in segments
            if str(segment.text).strip()
        ]


class ReazonSpeechBackend:
    source_family = REAZON_FAMILY
    model_name = "reazonspeech-k2-v2"

    def __init__(self) -> None:
        try:
            from reazonspeech.k2.asr import load_model
        except ImportError as exc:
            raise LocalASRError("reazonspeech.k2.asr is not installed") from exc
        self._model = load_model(device="cpu", precision="fp32", language="ja")

    def transcribe(self, audio_path: Path) -> str:
        return "".join(segment["text"] for segment in self.transcribe_segments(audio_path)).strip()

    def transcribe_segments(self, audio_path: Path) -> list[dict[str, Any]]:
        from reazonspeech.k2.asr import audio_from_path, transcribe

        audio = audio_from_path(str(audio_path))
        result = transcribe(self._model, audio)
        duration = float(audio.waveform.shape[0]) / float(audio.samplerate)
        return _segments_from_reazon_subwords(
            [
                {"seconds": float(subword.seconds), "token": str(subword.token)}
                for subword in result.subwords
            ],
            duration_seconds=duration,
        )


def _segments_from_reazon_subwords(
    subwords: list[dict[str, Any]],
    *,
    duration_seconds: float,
    silence_gap_seconds: float = 0.9,
    maximum_segment_seconds: float = 5.5,
) -> list[dict[str, Any]]:
    """Turn ReazonSpeech's native point timestamps into conservative spans."""
    usable = [
        (float(row["seconds"]), str(row["token"]).strip())
        for row in subwords
        if str(row.get("token") or "").strip()
        and 0 <= float(row.get("seconds", -1)) <= duration_seconds
    ]
    if not usable:
        return []
    groups: list[list[tuple[float, str]]] = []
    for timestamp, token in sorted(usable):
        if (
            not groups
            or timestamp - groups[-1][-1][0] >= silence_gap_seconds
            or timestamp - groups[-1][0][0] >= maximum_segment_seconds
        ):
            groups.append([])
        groups[-1].append((timestamp, token))
    segments: list[dict[str, Any]] = []
    for group in groups:
        start = max(0.0, group[0][0] - 0.2)
        end = min(duration_seconds, max(start + 0.1, group[-1][0] + 0.4))
        text = "".join(token for _, token in group).replace("▁", "").strip()
        if text:
            segments.append(
                {
                    "start_seconds": start,
                    "end_seconds": end,
                    "text": text,
                    "subword_count": len(group),
                }
            )
    return segments


def map_transcripts_to_blocks(
    blocks: list[SubtitleBlock],
    windows: list[AudioWindow],
    transcript_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_window: dict[str, list[dict[str, Any]]] = {}
    for record in transcript_records:
        if record.get("status") == "completed" and str(record.get("text") or "").strip():
            by_window.setdefault(str(record.get("window_id")), []).append(record)
    result: list[dict[str, Any]] = []
    for block in blocks:
        overlapping = [
            window
            for window in windows
            if window.clip_start < block.end_seconds and window.clip_end > block.start_seconds
        ]
        transcripts: list[dict[str, str]] = []
        evidence_refs: list[str] = []
        for window in overlapping:
            for record in by_window.get(window.window_id, []):
                transcripts.append(
                    {
                        "window_id": window.window_id,
                        "source_family": str(record["source_family"]),
                        "model": str(record["model"]),
                        "text": str(record["text"]),
                        "audio_sha256": str(record["audio_sha256"]),
                        "clip_start_seconds": window.clip_start,
                        "clip_end_seconds": window.clip_end,
                        "alignment_scope": "overlapping-window",
                        "block_aligned": False,
                    }
                )
                evidence_refs.append(f"asr:{window.window_id}:{record['source_family']}")
        result.append(
            {
                "schema_name": "translation-forensics/block-acoustic-evidence",
                "schema_version": "1",
                "block_number": block.number,
                "start": block.start,
                "end": block.end,
                "transcripts": transcripts,
                "evidence_refs": sorted(set(evidence_refs)),
                "independent_source_families": sorted({row["source_family"] for row in transcripts}),
                "block_alignment_status": "window-context-only" if transcripts else "no-acoustic-evidence",
            }
        )
    return result


def run_full_local_asr(
    *,
    title_id: str,
    audio_path: Path,
    structure_path: Path,
    output_dir: Path,
    force_cpu: bool = False,
    allow_model_download: bool = True,
    resume: bool = True,
    max_windows: int = 0,
    backends: list[ASRBackend] | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "local-asr-manifest.json"
    transcripts_path = output_dir / "local-asr-transcripts.jsonl"
    block_evidence_path = output_dir / "block-acoustic-evidence.jsonl"
    if resume and manifest_path.is_file() and transcripts_path.is_file() and block_evidence_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("audio_sha256") == _sha256(audio_path)
            and manifest.get("structure_sha256") == _sha256(structure_path)
            and manifest.get("window_contract", {}).get("schema_version") == "2"
        ):
            return {**manifest, "cache_hit": True, "output": str(output_dir)}
    if any(path.exists() for path in (manifest_path, transcripts_path, block_evidence_path)):
        raise FileExistsError(f"Local ASR outputs already exist in {output_dir}")

    duration = probe_audio_duration(audio_path)
    silences = detect_silence_midpoints(audio_path)
    windows = plan_vad_windows(duration, silences)
    if max_windows > 0:
        windows = windows[:max_windows]
    clip_paths = extract_audio_windows(audio_path, windows, output_dir / "clips")

    backend_errors: list[dict[str, str]] = []
    if backends is None:
        backends = []
        try:
            backends.append(
                FasterWhisperBackend(force_cpu=force_cpu, local_files_only=not allow_model_download)
            )
        except LocalASRError as exc:
            backend_errors.append({"source_family": WHISPER_FAMILY, "error": str(exc)})
        try:
            backends.append(ReazonSpeechBackend())
        except LocalASRError as exc:
            backend_errors.append({"source_family": REAZON_FAMILY, "error": str(exc)})

    records: list[dict[str, Any]] = []
    for window, clip_path in zip(windows, clip_paths):
        audio_hash = _sha256(clip_path)
        for backend in backends:
            try:
                transcript = backend.transcribe(clip_path)
                status = "completed"
                error = ""
            except Exception as exc:  # local model libraries expose heterogeneous errors
                transcript = ""
                status = "failed"
                error = str(exc)
            records.append(
                {
                    "schema_name": "translation-forensics/local-asr-transcript",
                    "schema_version": "1",
                    "title_id": title_id,
                    "window_id": window.window_id,
                    "source_family": backend.source_family,
                    "model": backend.model_name,
                    "audio_sha256": audio_hash,
                    "clip_start_seconds": window.clip_start,
                    "clip_end_seconds": window.clip_end,
                    "text": transcript,
                    "status": status,
                    "error": error,
                    "external_transfer": False,
                }
            )
    transcripts_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    blocks, _, _ = parse_srt(structure_path)
    block_records = [
        add_asr_fusion(record)
        for record in map_transcripts_to_blocks(blocks, windows, records)
    ]
    block_evidence_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in block_records),
        encoding="utf-8",
        newline="\n",
    )
    completed_families = sorted(
        {record["source_family"] for record in records if record["status"] == "completed" and record["text"]}
    )
    manifest = {
        "schema_name": "translation-forensics/local-asr-manifest",
        "schema_version": "1",
        "title_id": title_id,
        "status": "completed" if len(completed_families) == 2 else "partial",
        "audio_sha256": _sha256(audio_path),
        "structure_sha256": _sha256(structure_path),
        "duration_seconds": round(duration, 3),
        "window_contract": {
            "schema_version": "2",
            "vad": "ffmpeg-silencedetect",
            "minimum_window_seconds": 24.0,
            "target_window_seconds": 26.0,
            "maximum_window_seconds": 28.0,
            "context_overlap_seconds": 2.0,
        },
        "windows": [window.as_dict() for window in windows],
        "completed_source_families": completed_families,
        "backend_errors": backend_errors,
        "transcripts": {"path": transcripts_path.name, "sha256": _sha256(transcripts_path), "records": len(records)},
        "block_acoustic_evidence": {
            "path": block_evidence_path.name,
            "sha256": _sha256(block_evidence_path),
            "records": len(block_records),
        },
        "external_asr_transfer": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return {**manifest, "cache_hit": False, "output": str(output_dir)}


def run_conflict_asr_rerun(
    *,
    title_id: str,
    audio_path: Path,
    blocks: list[SubtitleBlock],
    output_dir: Path,
    force_cpu: bool = False,
    allow_model_download: bool = True,
    resume: bool = True,
    backends: list[ASRBackend] | None = None,
) -> dict[int, dict[str, Any]]:
    """Run both ASR families once per clustered boundary-expanded conflict clip."""
    output_dir = output_dir.expanduser().resolve()
    evidence_path = output_dir / "conflict-block-acoustic-evidence.jsonl"
    requested = sorted(block.number for block in blocks)
    if resume and evidence_path.is_file():
        rows = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if sorted(int(row["block_number"]) for row in rows) == requested:
            return {int(row["block_number"]): row for row in rows}
    if evidence_path.exists():
        raise FileExistsError(f"Conflict ASR rerun evidence already exists: {evidence_path}")
    duration = probe_audio_duration(audio_path)
    planned = plan_conflict_rerun_windows(blocks, duration_seconds=duration)
    windows = [window for window, _ in planned]
    clip_paths = extract_audio_windows(audio_path, windows, output_dir / "clips")
    if backends is None:
        backends = [
            FasterWhisperBackend(force_cpu=force_cpu, local_files_only=not allow_model_download),
            ReazonSpeechBackend(),
        ]
    records: list[dict[str, Any]] = []
    for (window, covered_blocks), clip_path in zip(planned, clip_paths):
        transcripts: list[dict[str, Any]] = []
        for backend in backends:
            text = backend.transcribe(clip_path)
            if text:
                transcripts.append(
                    {
                        "window_id": window.window_id,
                        "source_family": backend.source_family,
                        "model": backend.model_name,
                        "text": text,
                        "audio_sha256": _sha256(clip_path),
                        "clip_start_seconds": window.clip_start,
                        "clip_end_seconds": window.clip_end,
                        "alignment_scope": "boundary-expanded-window-rerun",
                        "block_aligned": False,
                    }
                )
        for block in covered_blocks:
            records.append(
                {
                    "schema_name": "translation-forensics/block-acoustic-evidence",
                    "schema_version": "1",
                    "block_number": block.number,
                    "start": block.start,
                    "end": block.end,
                    "transcripts": transcripts,
                    "evidence_refs": sorted(
                        f"asr:{window.window_id}:{row['source_family']}" for row in transcripts
                    ),
                    "independent_source_families": sorted({row["source_family"] for row in transcripts}),
                    "block_alignment_status": "boundary-expanded-cluster-rerun",
                    "asr_rerun_count": 1,
                    "rerun_cluster_block_numbers": [item.number for item in covered_blocks],
                    "external_transfer": False,
                }
            )
    records = [add_asr_fusion(record) for record in records]
    _write = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(_write, encoding="utf-8", newline="\n")
    return {int(row["block_number"]): row for row in records}


def plan_conflict_rerun_windows(
    blocks: list[SubtitleBlock],
    *,
    duration_seconds: float,
    maximum_core_span_seconds: float = 20.0,
    clip_seconds: float = 28.0,
) -> list[tuple[AudioWindow, list[SubtitleBlock]]]:
    """Cluster adjacent conflicts so a scene rerun does not re-ASR the same audio repeatedly."""
    if duration_seconds <= 0 or maximum_core_span_seconds <= 0 or clip_seconds <= 0:
        raise ValueError("Conflict rerun window parameters must be positive")
    ordered = sorted(blocks, key=lambda block: (block.start_seconds, block.end_seconds, block.number))
    groups: list[list[SubtitleBlock]] = []
    for block in ordered:
        if not groups or block.end_seconds - groups[-1][0].start_seconds > maximum_core_span_seconds:
            groups.append([])
        groups[-1].append(block)
    planned: list[tuple[AudioWindow, list[SubtitleBlock]]] = []
    for index, group in enumerate(groups, 1):
        core_start = group[0].start_seconds
        core_end = group[-1].end_seconds
        midpoint = (core_start + core_end) / 2.0
        clip_start = max(0.0, midpoint - clip_seconds / 2.0)
        clip_end = min(duration_seconds, clip_start + clip_seconds)
        clip_start = max(0.0, clip_end - clip_seconds)
        planned.append(
            (
                AudioWindow(
                    window_id=f"conflict-group-{index:05d}",
                    core_start=core_start,
                    core_end=core_end,
                    clip_start=clip_start,
                    clip_end=clip_end,
                ),
                group,
            )
        )
    return planned


def _split_reazon_by_whisper_segments(reazon_text: str, segments: list[dict[str, Any]]) -> list[str]:
    compact = normalize_japanese(reazon_text)
    if not segments:
        return []
    normalized_segments = [normalize_japanese(str(segment.get("text") or "")) for segment in segments]
    whisper = "".join(normalized_segments)
    if not whisper or not compact:
        return [""] * len(segments)
    matcher = SequenceMatcher(None, whisper, compact, autojunk=False)
    anchors: list[tuple[int, int]] = [(0, 0), (len(whisper), len(compact))]
    for match in matcher.get_matching_blocks():
        if match.size:
            anchors.append((match.a, match.b))
            anchors.append((match.a + match.size, match.b + match.size))
    anchors = sorted(set(anchors))

    def project(position: int) -> int:
        exact = [right for left, right in anchors if left == position]
        if exact:
            return min(exact)
        lower = max((anchor for anchor in anchors if anchor[0] < position), default=(0, 0))
        upper = min((anchor for anchor in anchors if anchor[0] > position), default=(len(whisper), len(compact)))
        if upper[0] == lower[0]:
            return lower[1]
        ratio = (position - lower[0]) / (upper[0] - lower[0])
        return round(lower[1] + ratio * (upper[1] - lower[1]))

    whisper_boundaries = [0]
    for text in normalized_segments:
        whisper_boundaries.append(whisper_boundaries[-1] + len(text))
    projected = [project(position) for position in whisper_boundaries]
    for index in range(1, len(projected)):
        projected[index] = max(projected[index], projected[index - 1])
    return [compact[projected[index] : projected[index + 1]] for index in range(len(segments))]


def _utterance_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(
        0.0,
        min(float(left["end_seconds"]), float(right["end_seconds"]))
        - max(float(left["start_seconds"]), float(right["start_seconds"])),
    )


def _deduplicate_overlapping_utterances(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: (value["start_seconds"], value["end_seconds"], value["window_id"])):
        row_family = str(row["transcripts"][0]["source_family"])
        row_text = "".join(
            str(item.get("text") or "")
            for item in row["transcripts"]
            if item.get("source_family") == row_family
        )
        duplicate = False
        for existing in reversed(kept[-6:]):
            if float(row["start_seconds"]) - float(existing["end_seconds"]) > 3.0:
                break
            existing_families = {str(item.get("source_family") or "") for item in existing["transcripts"]}
            if row_family not in existing_families:
                continue
            existing_text = "".join(
                str(item.get("text") or "")
                for item in existing["transcripts"]
                if item.get("source_family") == row_family
            )
            minimum_duration = max(
                0.1,
                min(
                    float(row["end_seconds"]) - float(row["start_seconds"]),
                    float(existing["end_seconds"]) - float(existing["start_seconds"]),
                ),
            )
            if (
                _utterance_overlap(row, existing) / minimum_duration >= 0.5
                and text_similarity(row_text, existing_text) >= 0.8
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
    for index, row in enumerate(kept, 1):
        row["utterance_id"] = f"utterance-{index:05d}"
    return kept


def build_timestamped_utterance_evidence(
    *,
    title_id: str,
    structure_path: Path,
    local_asr_dir: Path,
    output_dir: Path,
    force_cpu: bool = False,
    allow_model_download: bool = True,
    resume: bool = True,
    whisper_backend: FasterWhisperBackend | None = None,
    reazon_backend: ReazonSpeechBackend | None = None,
) -> dict[str, Any]:
    """Recover block-local evidence from both families' native timestamps."""
    local_asr_dir = local_asr_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    report_path = output_dir / "utterance-evidence-report.json"
    ledger_path = output_dir / "utterance-ledger.jsonl"
    block_path = output_dir / "block-acoustic-evidence.jsonl"
    manifest_path = local_asr_dir / "local-asr-manifest.json"
    transcripts_path = local_asr_dir / "local-asr-transcripts.jsonl"
    input_hashes = {
        "structure_sha256": _sha256(structure_path),
        "local_asr_manifest_sha256": _sha256(manifest_path),
        "local_asr_transcripts_sha256": _sha256(transcripts_path),
    }
    if resume and report_path.is_file() and ledger_path.is_file() and block_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if all(report.get(key) == value for key, value in input_hashes.items()):
            return {**report, "cache_hit": True, "output": str(output_dir)}
    if any(path.exists() for path in (report_path, ledger_path, block_path)):
        raise FileExistsError(f"Timestamped utterance evidence already exists in {output_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    windows = {str(row["window_id"]): row for row in manifest["windows"]}
    transcript_rows = [
        json.loads(line)
        for line in transcripts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_window: dict[str, dict[str, dict[str, Any]]] = {}
    for row in transcript_rows:
        by_window.setdefault(str(row["window_id"]), {})[str(row["source_family"])] = row
    backend = whisper_backend or FasterWhisperBackend(
        force_cpu=force_cpu,
        local_files_only=not allow_model_download,
    )
    reazon = reazon_backend or ReazonSpeechBackend()
    utterances: list[dict[str, Any]] = []
    for window_id, window in windows.items():
        clip_path = local_asr_dir / "clips" / f"{window_id}.wav"
        segments = backend.transcribe_segments(clip_path)
        reazon_segments = (
            reazon.transcribe_segments(clip_path)
            if by_window.get(window_id, {}).get(REAZON_FAMILY, {}).get("status") == "completed"
            else []
        )
        for segment in segments:
            utterances.append(
                {
                    "schema_name": "translation-forensics/utterance-evidence",
                    "schema_version": "2",
                    "title_id": title_id,
                    "utterance_id": "pending",
                    "window_id": window_id,
                    "start_seconds": round(float(window["clip_start"]) + float(segment["start_seconds"]), 3),
                    "end_seconds": round(float(window["clip_start"]) + float(segment["end_seconds"]), 3),
                    "audio_sha256": _sha256(clip_path),
                    "whisper_avg_logprob": segment["avg_logprob"],
                    "whisper_no_speech_prob": segment["no_speech_prob"],
                    "transcripts": [
                        {
                            "source_family": WHISPER_FAMILY,
                            "model": "large-v3",
                            "text": str(segment["text"]),
                            "alignment_method": "native-segment-timestamp",
                            "alignment_confidence": 1.0,
                        }
                    ],
                    "independent_source_families": [WHISPER_FAMILY],
                    "external_transfer": False,
                }
            )
        for segment in reazon_segments:
            utterances.append(
                {
                    "schema_name": "translation-forensics/utterance-evidence",
                    "schema_version": "2",
                    "title_id": title_id,
                    "utterance_id": "pending",
                    "window_id": window_id,
                    "start_seconds": round(float(window["clip_start"]) + float(segment["start_seconds"]), 3),
                    "end_seconds": round(float(window["clip_start"]) + float(segment["end_seconds"]), 3),
                    "audio_sha256": _sha256(clip_path),
                    "transcripts": [
                        {
                            "source_family": REAZON_FAMILY,
                            "model": "reazonspeech-k2-v2",
                            "text": str(segment["text"]),
                            "alignment_method": "native-subword-timestamp-span",
                            "alignment_confidence": 1.0,
                            "subword_count": int(segment["subword_count"]),
                        }
                    ],
                    "independent_source_families": [REAZON_FAMILY],
                    "external_transfer": False,
                }
            )
    utterances = _deduplicate_overlapping_utterances(utterances)
    blocks, _, _ = parse_srt(structure_path)
    block_records: list[dict[str, Any]] = []
    for block in blocks:
        matched = [
            row
            for row in utterances
            if float(row["start_seconds"]) < block.end_seconds
            and float(row["end_seconds"]) > block.start_seconds
        ]
        transcripts: list[dict[str, Any]] = []
        for utterance in matched:
            for item in utterance["transcripts"]:
                transcripts.append(
                    {
                        **item,
                        "utterance_id": utterance["utterance_id"],
                        "window_id": utterance["window_id"],
                        "audio_sha256": utterance["audio_sha256"],
                        "start_seconds": utterance["start_seconds"],
                        "end_seconds": utterance["end_seconds"],
                        "alignment_scope": "utterance-timestamp",
                        "block_aligned": True,
                    }
                )
        families = sorted({item["source_family"] for item in transcripts})
        block_records.append(
            {
                "schema_name": "translation-forensics/block-acoustic-evidence",
                "schema_version": "1",
                "block_number": block.number,
                "start": block.start,
                "end": block.end,
                "transcripts": transcripts,
                "evidence_refs": sorted(
                    f"utterance:{item['utterance_id']}:{item['source_family']}" for item in transcripts
                ),
                "independent_source_families": families,
                "block_alignment_status": "utterance-timestamp-aligned" if transcripts else "no-acoustic-evidence",
            }
        )
    block_records = [add_asr_fusion(record) for record in block_records]
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in utterances),
        encoding="utf-8",
        newline="\n",
    )
    block_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in block_records),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_name": "translation-forensics/utterance-evidence-report",
        "schema_version": "1",
        "title_id": title_id,
        "status": "completed",
        **input_hashes,
        "utterance_count": len(utterances),
        "block_count": len(block_records),
        "dual_family_utterance_count": sum(len(row["independent_source_families"]) == 2 for row in utterances),
        "blocks_with_utterance_evidence": sum(bool(row["transcripts"]) for row in block_records),
        "blocks_with_dual_family_evidence": sum(len(row["independent_source_families"]) == 2 for row in block_records),
        "utterance_ledger": {"path": ledger_path.name, "sha256": _sha256(ledger_path)},
        "block_acoustic_evidence": {"path": block_path.name, "sha256": _sha256(block_path)},
        "external_transfer": False,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return {**report, "cache_hit": False, "output": str(output_dir)}
