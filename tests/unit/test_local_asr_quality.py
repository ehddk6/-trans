from __future__ import annotations

from translation_forensics.local_asr import (
    AudioWindow,
    _deduplicate_overlapping_utterances,
    _deduplicate_repair_transcripts,
    _segments_from_reazon_subwords,
    _split_reazon_by_whisper_segments,
    map_transcripts_to_blocks,
    plan_conflict_rerun_windows,
    plan_vad_windows,
    run_pre_ceiling_evidence_repair,
)
from translation_forensics.srt import SubtitleBlock


def test_pre_ceiling_repair_uses_native_overlap_and_never_copies_cluster_text(tmp_path, monkeypatch):
    import translation_forensics.local_asr as local_asr

    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")
    clip = tmp_path / "clips" / "repair-group-00001.wav"
    clip.parent.mkdir()
    clip.write_bytes(b"clip")
    monkeypatch.setattr(local_asr, "probe_audio_duration", lambda path: 10.0)
    monkeypatch.setattr(local_asr, "extract_audio_windows", lambda audio_path, windows, clips_dir: [clip])

    block = SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "x", 1.0, 2.0)
    base = {
        1: {
            "block_number": 1,
            "start": block.start,
            "end": block.end,
            "transcripts": [],
            "evidence_refs": [],
            "independent_source_families": [],
            "block_alignment_status": "no-acoustic-evidence",
        }
    }

    class FakeBackend:
        def __init__(self, family):
            self.source_family = family
            self.model_name = family

        def transcribe_segments(self, path):
            return [{"start_seconds": 1.0, "end_seconds": 1.5, "text": "same"}]

    repaired = run_pre_ceiling_evidence_repair(
        title_id="TEST",
        audio_path=audio,
        blocks=[block],
        base_acoustic=base,
        output_dir=tmp_path / "repair",
        backends=[FakeBackend("family-a"), FakeBackend("family-b")],
    )
    assert sorted(repaired) == [1]
    assert repaired[1]["independent_source_families"] == ["family-a", "family-b"]
    assert repaired[1]["block_alignment_status"] == "pre-ceiling-repair-utterance-timestamp-aligned"
    assert repaired[1]["asr_fusion"]["state"] == "dual_agreement"
    assert all(item["alignment_scope"] == "utterance-timestamp" for item in repaired[1]["transcripts"])


def test_pre_ceiling_repair_does_not_mark_long_segment_block_local_everywhere(tmp_path, monkeypatch):
    import translation_forensics.local_asr as local_asr

    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")
    clip = tmp_path / "clips" / "repair-group-00001.wav"
    clip.parent.mkdir()
    clip.write_bytes(b"clip")
    monkeypatch.setattr(local_asr, "probe_audio_duration", lambda path: 20.0)
    monkeypatch.setattr(local_asr, "extract_audio_windows", lambda audio_path, windows, clips_dir: [clip])
    blocks = [
        SubtitleBlock(1, "00:00:10,000", "00:00:11,000", "a", 10.0, 11.0),
        SubtitleBlock(2, "00:00:11,000", "00:00:12,000", "b", 11.0, 12.0),
        SubtitleBlock(3, "00:00:12,000", "00:00:15,000", "c", 12.0, 15.0),
    ]
    base = {
        block.number: {
            "block_number": block.number,
            "start": block.start,
            "end": block.end,
            "transcripts": [],
            "evidence_refs": [],
            "independent_source_families": [],
        }
        for block in blocks
    }

    class FakeBackend:
        def __init__(self, family):
            self.source_family = family
            self.model_name = family

        def transcribe_segments(self, path):
            return [{"start_seconds": 10.0, "end_seconds": 15.0, "text": "long utterance"}]

    repaired = run_pre_ceiling_evidence_repair(
        title_id="TEST",
        audio_path=audio,
        blocks=blocks,
        base_acoustic=base,
        output_dir=tmp_path / "repair",
        backends=[FakeBackend("family-a"), FakeBackend("family-b")],
    )
    aligned_counts = {
        number: sum(bool(item.get("block_aligned")) for item in row["transcripts"])
        for number, row in repaired.items()
    }
    assert set(repaired) == {1, 2, 3}
    assert sum(count > 0 for count in aligned_counts.values()) <= 1
    assert all(row["asr_fusion"]["alignment_strength"] != "block-aligned" for row in repaired.values() if aligned_counts[row["block_number"]] == 0)


def test_vad_windows_choose_silence_with_24_to_28_second_core():
    windows = plan_vad_windows(80.0, [25.8, 49.7, 74.0])
    assert windows[0].core_end == 25.8
    assert windows[1].core_end == 49.7
    assert all(24.0 <= window.duration <= 28.0 for window in windows[:-1])
    assert windows[0].clip_end - windows[1].clip_start == 2.0


def test_block_evidence_keeps_independent_families():
    blocks = [SubtitleBlock(1, "00:00:01,000", "00:00:02,000", "x", 1.0, 2.0)]
    windows = [AudioWindow("window-00001", 0.0, 26.0, 0.0, 27.0)]
    transcripts = [
        {"window_id": "window-00001", "status": "completed", "text": "止めて", "source_family": "faster-whisper-large-v3", "model": "large-v3", "audio_sha256": "a" * 64},
        {"window_id": "window-00001", "status": "completed", "text": "やめて", "source_family": "reazonspeech-k2-v2", "model": "reazonspeech-k2-v2", "audio_sha256": "a" * 64},
    ]
    evidence = map_transcripts_to_blocks(blocks, windows, transcripts)
    assert evidence[0]["independent_source_families"] == ["faster-whisper-large-v3", "reazonspeech-k2-v2"]
    assert len(evidence[0]["transcripts"]) == 2


def test_reazon_native_subword_timestamps_split_on_silence():
    segments = _segments_from_reazon_subwords(
        [
            {"seconds": 1.0, "token": "a"},
            {"seconds": 1.2, "token": "b"},
            {"seconds": 3.0, "token": "c"},
            {"seconds": 3.2, "token": "d"},
        ],
        duration_seconds=4.0,
    )
    assert [row["text"] for row in segments] == ["ab", "cd"]
    assert segments[0]["start_seconds"] == 0.8
    assert segments[1]["end_seconds"] == 3.6


def test_overlap_deduplication_does_not_merge_independent_families():
    common = {
        "utterance_id": "pending",
        "window_id": "window-1",
        "start_seconds": 1.0,
        "end_seconds": 2.0,
    }
    rows = [
        {**common, "transcripts": [{"source_family": "faster-whisper-large-v3", "text": "same"}]},
        {**common, "transcripts": [{"source_family": "reazonspeech-k2-v2", "text": "same"}]},
    ]
    kept = _deduplicate_overlapping_utterances(rows)
    assert len(kept) == 2


def test_repair_dedup_uses_parent_audio_across_overlapping_clips():
    rows = [
        {
            "source_family": "family-a",
            "model": "model-a",
            "text": "same utterance",
            "start_seconds": 10.0,
            "end_seconds": 12.0,
            "audio_sha256": "clip-a",
            "parent_audio_sha256": "parent",
        },
        {
            "source_family": "family-a",
            "model": "model-a",
            "text": "same utterance",
            "start_seconds": 10.2,
            "end_seconds": 12.1,
            "audio_sha256": "clip-b",
            "parent_audio_sha256": "parent",
        },
    ]
    kept, duplicates = _deduplicate_repair_transcripts(rows)
    assert len(kept) == 1
    assert duplicates == 1


def test_conflict_rerun_clusters_adjacent_blocks_into_shared_windows():
    blocks = [
        SubtitleBlock(index, f"00:00:{index:02d},000", f"00:00:{index + 1:02d},000", "x", float(index), float(index + 1))
        for index in range(1, 41)
    ]
    planned = plan_conflict_rerun_windows(blocks, duration_seconds=60.0)
    assert len(planned) == 2
    assert sum(len(group) for _, group in planned) == len(blocks)
    assert all(window.duration <= 28.0 for window, _ in planned)


def test_reazon_text_uses_sequence_aligned_segment_boundaries():
    segments = [{"text": "やめて"}, {"text": "ください"}]
    chunks = _split_reazon_by_whisper_segments("やめてねください", segments)
    assert "".join(chunks) == "やめてねください"
    assert chunks[0].startswith("やめて")
    assert chunks[1].endswith("ください")
