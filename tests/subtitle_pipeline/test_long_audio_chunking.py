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
            call_index = len(calls) - 1
            # The second decode starts two seconds before its owned 600-second
            # boundary.  Emit the final cue at local 2s so its global timing is
            # 600s after offset restoration.
            start = 0.0 if call_index == 0 else 2.0
            segment = types.SimpleNamespace(
                start=start,
                end=start + 1.0,
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

    assert calls == [(601, "0"), (3, "0")]
    assert duration == 601.0
    assert [(segment.start, segment.end) for segment in segments] == [
        (0.0, 1.0),
        (600.0, 601.0),
    ]
    assert segments[1].metadata["whisper_chunk_index"] == 1
    assert segments[1].metadata["whisper_chunk_overlap_seconds"] == 2


def test_long_audio_overlap_assigns_a_boundary_utterance_to_exactly_one_chunk(
    tmp_path: Path, monkeypatch
) -> None:
    audio = tmp_path / "boundary-601-seconds.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(1)
        handle.writeframes(b"\x00\x00" * 601)

    calls = 0

    class FakeModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def transcribe(self, path, **kwargs):
            nonlocal calls
            # Both overlapping decodes recognize the same global 599-601s cue.
            start = 599.0 if calls == 0 else 1.0
            calls += 1
            segment = types.SimpleNamespace(
                start=start,
                end=start + 2.0,
                text="境界の発話",
                words=[],
                no_speech_prob=0.0,
                avg_logprob=-0.1,
                compression_ratio=1.0,
            )
            return [segment], types.SimpleNamespace(duration=601.0)

    monkeypatch.setitem(
        sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeModel)
    )

    segments, _ = transcribe(audio, "fake-model")

    assert calls == 2
    assert [(segment.start, segment.end, segment.text) for segment in segments] == [
        (599.0, 601.0, "境界の発話")
    ]
    assert segments[0].metadata["whisper_chunk_index"] == 1
