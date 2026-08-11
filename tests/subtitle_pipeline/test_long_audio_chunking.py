from __future__ import annotations

import sys
import types
import wave
from pathlib import Path

from subtitle_pipeline.recognition import transcribe


def test_audio_over_600_seconds_is_transcribed_in_bounded_sequential_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    audio = tmp_path / "long-601-seconds.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(1)
        handle.writeframes(b"\x00\x00" * 601)

    calls: list[tuple[int, str]] = []

    class FakeModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def transcribe(self, path, **kwargs):
            with wave.open(str(path), "rb") as handle:
                calls.append((handle.getnframes(), kwargs["clip_timestamps"]))
            segment = types.SimpleNamespace(
                start=0.0,
                end=1.0,
                text="テスト",
                words=[],
                no_speech_prob=0.0,
                avg_logprob=-0.1,
                compression_ratio=1.0,
            )
            return [segment], types.SimpleNamespace(duration=1.0)

    monkeypatch.setitem(
        sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeModel)
    )
    segments, duration = transcribe(audio, "fake-model")

    assert calls == [(600, "0"), (1, "0")]
    assert duration == 601.0
    assert [(segment.start, segment.end) for segment in segments] == [
        (0.0, 1.0),
        (600.0, 601.0),
    ]
