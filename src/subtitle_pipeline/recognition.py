from __future__ import annotations

import math
import gc
from pathlib import Path
import subprocess
import tempfile
import wave

from .models import SourceSegment, Word


_LONG_AUDIO_CHUNK_SECONDS = 600
_LONG_AUDIO_OVERLAP_SECONDS = 2


def _prepare_whisper_audio(input_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Decode container audio with FFmpeg before passing it to PyAV/Whisper.

    Some OneDrive videos contain AAC streams that FFmpeg can decode completely,
    while PyAV reports only the first short packet range. Feeding a normalized
    WAV keeps Whisper's duration and coverage aligned with the actual video.
    """
    if input_path.suffix.lower() == ".wav":
        return input_path, None

    temporary = tempfile.TemporaryDirectory(prefix="subtitle_pipeline_audio_")
    audio_path = Path(temporary.name) / "audio.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", str(input_path), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(audio_path),
            ],
            check=True,
            timeout=7_200,
        )
    except Exception:
        temporary.cleanup()
        raise
    return audio_path, temporary


def transcribe(
    input_path: Path,
    model_name: str,
    language: str = "ja",
    word_timestamps: bool = True,
    vad: bool = False,
    beam_size: int = 5,
    temperature: float = 0.0,
    condition_on_previous_text: bool = False,
    clip_timestamps: list[float] | None = None,
) -> tuple[list[SourceSegment], float]:
    """Run faster-whisper while preserving its timing and quality metrics verbatim."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install ASR support: pip install -e .[asr]") from exc

    audio_path, temporary_audio = _prepare_whisper_audio(input_path)
    model = WhisperModel(model_name, device="auto", compute_type="auto")

    def append_segments(
        raw_segments,
        offset: float,
        destination: list[SourceSegment],
        *,
        owner_start: float | None = None,
        owner_end: float | None = None,
        include_owner_end: bool = True,
        chunk_index: int | None = None,
    ) -> None:
        for segment in raw_segments:
            global_start = segment.start + offset
            global_end = segment.end + offset
            midpoint = global_start + ((global_end - global_start) / 2)
            if owner_start is not None and midpoint < owner_start - 1e-9:
                continue
            if owner_end is not None and (
                midpoint > owner_end + 1e-9
                or (not include_owner_end and midpoint >= owner_end - 1e-9)
            ):
                continue
            words = [
                Word(w.word, w.start + offset, w.end + offset, getattr(w, "probability", None), backend="whisper")
                for w in (segment.words or [])
            ]
            destination.append(SourceSegment(
                id=len(destination), start=global_start, end=global_end,
                text=segment.text.strip(), words=words,
                no_speech_prob=getattr(segment, "no_speech_prob", None),
                avg_logprob=getattr(segment, "avg_logprob", None),
                compression_ratio=getattr(segment, "compression_ratio", None),
                metadata=(
                    {
                        "whisper_chunk_index": chunk_index,
                        "whisper_chunk_overlap_seconds": _LONG_AUDIO_OVERLAP_SECONDS,
                    }
                    if chunk_index is not None
                    else {}
                ),
            ))

    try:
        segments: list[SourceSegment] = []
        duration = math.nan
        wav_info = None
        if clip_timestamps is None and audio_path.suffix.lower() == ".wav":
            try:
                with wave.open(str(audio_path), "rb") as wav:
                    wav_info = (wav.getframerate(), wav.getnframes())
            except (EOFError, OSError, wave.Error):
                wav_info = None

        if wav_info is not None:
            sample_rate, total_frames = wav_info
            duration = total_frames / sample_rate if sample_rate else math.nan
            if duration > _LONG_AUDIO_CHUNK_SECONDS:
                with tempfile.TemporaryDirectory(prefix="subtitle_pipeline_whisper_chunks_") as chunk_dir:
                    with wave.open(str(audio_path), "rb") as source_wav:
                        params = source_wav.getparams()
                        frames_per_chunk = sample_rate * _LONG_AUDIO_CHUNK_SECONDS
                        overlap_frames = sample_rate * _LONG_AUDIO_OVERLAP_SECONDS
                        chunk_start = 0
                        chunk_index = 0
                        while chunk_start < total_frames:
                            frame_count = min(frames_per_chunk, total_frames - chunk_start)
                            core_end = chunk_start + frame_count
                            decode_start = max(0, chunk_start - overlap_frames)
                            decode_end = min(total_frames, core_end + overlap_frames)
                            source_wav.setpos(decode_start)
                            payload = source_wav.readframes(decode_end - decode_start)
                            chunk_path = Path(chunk_dir) / f"chunk_{chunk_index:05d}.wav"
                            with wave.open(str(chunk_path), "wb") as chunk_wav:
                                chunk_wav.setparams(params)
                                chunk_wav.writeframes(payload)
                            raw_segments, _ = model.transcribe(
                                str(chunk_path), language=language, task="transcribe", word_timestamps=word_timestamps,
                                vad_filter=vad, vad_parameters={"min_silence_duration_ms": 500}, beam_size=beam_size,
                                temperature=temperature, condition_on_previous_text=condition_on_previous_text,
                                clip_timestamps="0",
                            )
                            append_segments(
                                raw_segments,
                                decode_start / sample_rate,
                                segments,
                                owner_start=chunk_start / sample_rate,
                                owner_end=core_end / sample_rate,
                                include_owner_end=core_end >= total_frames,
                                chunk_index=chunk_index,
                            )
                            chunk_start += frame_count
                            chunk_index += 1
            else:
                raw_segments, info = model.transcribe(
                    str(audio_path), language=language, task="transcribe", word_timestamps=word_timestamps,
                    vad_filter=vad, vad_parameters={"min_silence_duration_ms": 500}, beam_size=beam_size,
                    temperature=temperature, condition_on_previous_text=condition_on_previous_text,
                    clip_timestamps=clip_timestamps or "0",
                )
                append_segments(raw_segments, 0.0, segments)
                duration = float(getattr(info, "duration", duration))
        else:
            raw_segments, info = model.transcribe(
                str(audio_path), language=language, task="transcribe", word_timestamps=word_timestamps,
                vad_filter=vad, vad_parameters={"min_silence_duration_ms": 500}, beam_size=beam_size,
                temperature=temperature, condition_on_previous_text=condition_on_previous_text,
                clip_timestamps=clip_timestamps or "0",
            )
            append_segments(raw_segments, 0.0, segments)
            duration = float(getattr(info, "duration", math.nan))
    finally:
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        if temporary_audio is not None:
            temporary_audio.cleanup()
    return segments, duration
