from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from subtitle_pipeline.qwen import _compact_targeted_intervals
from subtitle_pipeline.qwen_worker import iter_chunks, iter_selected_chunks, load_intervals, make_alignment_rows


def test_qwen_runtime_discovery_uses_environment_root_with_portable_fallback(tmp_path, monkeypatch) -> None:
    from subtitle_pipeline import qwen

    monkeypatch.setenv(qwen.QWEN_ROOT_ENV, str(tmp_path))
    configured = qwen.QwenRuntime.discover()
    assert configured.python == tmp_path / "venv" / "Scripts" / "python.exe"
    assert configured.model == tmp_path / "Model"
    assert configured.aligner == tmp_path / "ForcedAligner"

    monkeypatch.delenv(qwen.QWEN_ROOT_ENV)
    fallback = qwen.QwenRuntime.discover()
    assert fallback.model == Path("Qwen3ASR") / "Model"


def test_completed_matching_qwen_alignment_cache_skips_the_qwen_runtime(tmp_path, monkeypatch) -> None:
    from subtitle_pipeline import qwen

    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"cached audio")
    cache = tmp_path / "qwen_alignment_ja.jsonl"
    cache.write_text('{"text":"cached","start":1.0,"end":2.0,"source_chunk":0}\n', encoding="utf-8")
    monkeypatch.setattr(qwen, "_media_duration", lambda _path: 12.0)
    runtime = qwen.QwenRuntime(tmp_path / "missing-python", tmp_path / "missing-model", tmp_path / "missing-aligner")
    helper = Path(qwen.__file__).with_name("qwen_worker.py")
    qwen._write_cache_metadata(cache, qwen._qwen_cache_metadata(
        input_path, runtime, helper, 12.0, "ja", 180, 256, None,
    ))

    segments, duration = qwen.transcribe_qwen(input_path, runtime, alignment_cache_path=cache)

    assert duration == 12.0
    assert [segment.text for segment in segments] == ["cached"]
    assert all(segment.metadata["qwen_cache"] == "reused" for segment in segments)


def test_qwen_cache_without_matching_metadata_is_not_reused(tmp_path, monkeypatch) -> None:
    from subtitle_pipeline import qwen

    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"new audio")
    cache = tmp_path / "qwen_alignment_ja.jsonl"
    cache.write_text('{"text":"stale","start":1.0,"end":2.0,"source_chunk":0}\n', encoding="utf-8")
    monkeypatch.setattr(qwen, "_media_duration", lambda _path: 12.0)
    runtime = qwen.QwenRuntime(tmp_path / "missing-python", tmp_path / "missing-model", tmp_path / "missing-aligner")

    try:
        qwen.transcribe_qwen(input_path, runtime, alignment_cache_path=cache)
    except RuntimeError as exc:
        assert "runtime files are missing" in str(exc)
    else:
        raise AssertionError("a cache without matching metadata must not be reused")


def test_qwen_cache_uses_stable_source_identity_instead_of_temporary_wav(tmp_path, monkeypatch) -> None:
    from subtitle_pipeline import qwen

    source = tmp_path / "movie.mp4"
    source.write_bytes(b"stable source container")
    first_wav = tmp_path / "first-temp.wav"
    second_wav = tmp_path / "second-temp.wav"
    first_wav.write_bytes(b"normalized audio one")
    second_wav.write_bytes(b"normalized audio two")
    cache = tmp_path / "qwen_targeted_alignment_ja.jsonl"
    cache.write_text('{"text":"cached","start":1.0,"end":2.0,"source_chunk":0}\n', encoding="utf-8")
    monkeypatch.setattr(qwen, "_media_duration", lambda _path: 12.0)
    runtime = qwen.QwenRuntime(tmp_path / "missing-python", tmp_path / "missing-model", tmp_path / "missing-aligner")
    helper = Path(qwen.__file__).with_name("qwen_worker.py")
    qwen._write_cache_metadata(cache, qwen._qwen_cache_metadata(
        first_wav, runtime, helper, 12.0, "ja", 180, 256, None, source,
    ))

    segments, duration = qwen.transcribe_qwen(
        second_wav, runtime, alignment_cache_path=cache, cache_identity_path=source,
    )

    assert duration == 12.0
    assert [segment.text for segment in segments] == ["cached"]
    assert segments[0].metadata["qwen_cache"] == "reused"


def test_qwen_cache_invalidates_when_stable_source_changes(tmp_path, monkeypatch) -> None:
    from subtitle_pipeline import qwen

    source = tmp_path / "movie.mp4"
    source.write_bytes(b"original source")
    wav = tmp_path / "temp.wav"
    wav.write_bytes(b"normalized audio")
    cache = tmp_path / "qwen_targeted_alignment_ja.jsonl"
    cache.write_text('{"text":"stale","start":1.0,"end":2.0,"source_chunk":0}\n', encoding="utf-8")
    monkeypatch.setattr(qwen, "_media_duration", lambda _path: 12.0)
    runtime = qwen.QwenRuntime(tmp_path / "missing-python", tmp_path / "missing-model", tmp_path / "missing-aligner")
    helper = Path(qwen.__file__).with_name("qwen_worker.py")
    qwen._write_cache_metadata(cache, qwen._qwen_cache_metadata(
        wav, runtime, helper, 12.0, "ja", 180, 256, None, source,
    ))
    source.write_bytes(b"changed source with a different size")

    try:
        qwen.transcribe_qwen(wav, runtime, alignment_cache_path=cache, cache_identity_path=source)
    except RuntimeError as exc:
        assert "runtime files are missing" in str(exc)
    else:
        raise AssertionError("changing the stable source must invalidate the Qwen cache")


def test_qwen_targeted_cache_for_a_different_interval_is_not_reused(tmp_path, monkeypatch) -> None:
    from subtitle_pipeline import qwen

    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    cache = tmp_path / "qwen_targeted_alignment_ja.jsonl"
    cache.write_text('{"text":"cached","start":1.0,"end":2.0,"source_chunk":0}\n', encoding="utf-8")
    monkeypatch.setattr(qwen, "_media_duration", lambda _path: 12.0)
    runtime = qwen.QwenRuntime(tmp_path / "missing-python", tmp_path / "missing-model", tmp_path / "missing-aligner")
    helper = Path(qwen.__file__).with_name("qwen_worker.py")
    qwen._write_cache_metadata(cache, qwen._qwen_cache_metadata(
        input_path, runtime, helper, 12.0, "ja", 180, 256, [{"start": 0.0, "end": 2.0}],
    ))

    try:
        qwen.transcribe_qwen(
            input_path, runtime, alignment_cache_path=cache, targeted_intervals=[{"start": 4.0, "end": 6.0}],
        )
    except RuntimeError as exc:
        assert "runtime files are missing" in str(exc)
    else:
        raise AssertionError("a cache for a different verification interval must not be reused")


def test_alignment_rows_preserve_text_and_apply_global_chunk_offset() -> None:
    rows = make_alignment_rows(
        [
            SimpleNamespace(text=" そのまま ", start_time=0.25, end_time=0.75),
            SimpleNamespace(text="話", start_time=1.0, end_time=1.5),
        ],
        offset_seconds=180.0,
        source_chunk=1,
        language="ja",
    )

    assert rows == [
        {
            "text": " そのまま ",
            "start": 180.25,
            "end": 180.75,
            "source_chunk": 1,
            "language": "ja",
            "backend": "qwen",
        },
        {
            "text": "話",
            "start": 181.0,
            "end": 181.5,
            "source_chunk": 1,
            "language": "ja",
            "backend": "qwen",
        },
    ]


def test_chunks_are_sequential_zero_based_with_global_offsets() -> None:
    chunks = list(iter_chunks(list(range(13)), chunk_seconds=2, sample_rate=3))

    assert [(chunk.index, chunk.samples, chunk.offset_seconds) for chunk in chunks] == [
        (0, [0, 1, 2, 3, 4, 5], 0.0),
        (1, [6, 7, 8, 9, 10, 11], 2.0),
        (2, [12], 4.0),
    ]


def test_targeted_chunks_keep_original_video_offsets_and_audit_interval_ids() -> None:
    chunks = list(iter_selected_chunks(
        list(range(30)), [{"start": 2.0, "end": 4.0}, {"start": 7.0, "end": 9.0}], chunk_seconds=1, sample_rate=3,
    ))

    assert [(chunk.offset_seconds, chunk.source_interval) for chunk in chunks] == [
        (2.0, 0), (3.0, 0), (7.0, 1), (8.0, 1),
    ]


def test_compact_targeted_audio_keeps_original_source_offsets() -> None:
    intervals = _compact_targeted_intervals(
        [{"start": 20.0, "end": 22.0}, {"start": 40.0, "end": 43.0}], 60.0,
    )

    chunks = list(iter_selected_chunks(list(range(15)), intervals, chunk_seconds=1, sample_rate=3))

    assert intervals == [
        {"source_start": 20.0, "source_end": 22.0, "audio_start": 0.0, "audio_end": 2.0},
        {"source_start": 40.0, "source_end": 43.0, "audio_start": 2.0, "audio_end": 5.0},
    ]
    assert [(chunk.offset_seconds, chunk.source_interval) for chunk in chunks] == [
        (20.0, 0), (21.0, 0), (40.0, 1), (41.0, 1), (42.0, 1),
    ]


def test_compact_targeted_audio_allows_explicit_core_context_overlap() -> None:
    intervals = _compact_targeted_intervals(
        [
            {"start": 20.0, "end": 24.0, "allow_overlap": True},
            {"start": 18.0, "end": 26.0, "allow_overlap": True},
        ], 60.0,
    )

    assert intervals[0]["source_start"] == 20.0
    assert intervals[1]["source_start"] == 18.0
    assert intervals[0]["audio_end"] == intervals[1]["audio_start"]


def test_interval_loader_rejects_overlapping_windows(tmp_path) -> None:
    path = tmp_path / "intervals.json"
    path.write_text('[{"start": 1, "end": 3}, {"start": 2, "end": 4}]', encoding="utf-8")

    try:
        load_intervals(str(path), 10.0)
    except RuntimeError as exc:
        assert "non-overlapping" in str(exc)
    else:
        raise AssertionError("overlapping audit windows must be rejected")


def test_interval_loader_allows_small_resampling_end_drift(tmp_path) -> None:
    path = tmp_path / "intervals.json"
    path.write_text('[{"start": 9.95, "end": 10.05}]', encoding="utf-8")

    assert load_intervals(str(path), 10.0) == [{
        "source_start": 9.95,
        "source_end": 10.05,
        "audio_start": 9.95,
        "audio_end": 10.05,
    }]


def test_interval_loader_allows_explicit_core_context_overlap(tmp_path) -> None:
    path = tmp_path / "intervals.json"
    path.write_text(
        '[{"source_start": 1, "source_end": 3, "audio_start": 0, "audio_end": 2, "allow_overlap": true}, '
        '{"source_start": 2, "source_end": 4, "audio_start": 2, "audio_end": 4, "allow_overlap": true}]',
        encoding="utf-8",
    )

    intervals = load_intervals(str(path), 4.0)

    assert intervals[1]["source_start"] == 2.0
    assert intervals[1]["allow_overlap"] is True
