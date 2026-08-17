from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from translation_forensics.cli import main
from translation_forensics.pilot_audio_review import (
    PilotAlignmentError,
    build_pilot_audio_review_packet,
    evaluate_pilot_alignment,
    write_blocked_audio_review_manifest,
    write_pilot_alignment,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


def _alignment_inputs(tmp_path: Path, *, human: bool = True, mismatched_offset: bool = False) -> tuple[Path, Path, Path]:
    media = tmp_path / "source.wav"
    media.write_bytes(b"test-original-audio")
    anchors = tmp_path / "anchors.csv"
    rows = [
        {"anchor_id": "A-early", "srt_time_seconds": 1.0, "media_time_seconds": 1.5, "source": "human-direct-listening"},
        {"anchor_id": "A-middle", "srt_time_seconds": 5.0, "media_time_seconds": 5.5, "source": "human-direct-listening"},
        {"anchor_id": "A-late", "srt_time_seconds": 9.0, "media_time_seconds": 10.5 if mismatched_offset else 9.5, "source": "human-direct-listening" if human else "machine-asr"},
    ]
    with anchors.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=["anchor_id", "srt_time_seconds", "media_time_seconds", "source"])
        writer.writeheader()
        writer.writerows(rows)
    offset_map = tmp_path / "offset-map.json"
    offset_map.write_text(json.dumps({
        "schema_name": "translation-forensics/pilot-offset-map",
        "schema_version": "1",
        "approval_status": "approved",
        "human_verified": True,
        "approved_by": "timeline-reviewer",
        "scope": "full-title",
        "evidence_refs": [row["anchor_id"] for row in rows],
        "offset_seconds": 0.5,
    }, ensure_ascii=False), encoding="utf-8")
    return media, anchors, offset_map


def _resolved_alignment(tmp_path: Path) -> tuple[Path, Path]:
    media, anchors, offset_map = _alignment_inputs(tmp_path)
    report = evaluate_pilot_alignment(
        "SAMPLE",
        FIXTURES / "sample.structure.srt",
        media,
        anchors,
        offset_map,
        expected_blocks=3,
        duration_seconds=12.0,
    )
    alignment = tmp_path / "alignment.json"
    write_pilot_alignment(alignment, report)
    return media, alignment


def _fake_clip(_media: Path, destination: Path, start_seconds: float, duration_seconds: float) -> None:
    destination.write_bytes(f"{start_seconds:.6f}:{duration_seconds:.6f}".encode("ascii"))


def test_human_anchors_and_approved_offset_map_resolve_every_block(tmp_path: Path) -> None:
    media, anchors, offset_map = _alignment_inputs(tmp_path)
    report = evaluate_pilot_alignment(
        "SAMPLE",
        FIXTURES / "sample.structure.srt",
        media,
        anchors,
        offset_map,
        expected_blocks=3,
        duration_seconds=12.0,
    )

    assert report["status"] == "resolved"
    assert report["clip_preparation_allowed"] is True
    assert report["mapped_block_count"] == 3
    assert report["block_mappings"][0]["media_start_seconds"] == 1.5
    assert report["errors"] == []
    schema = json.loads((ROOT / "schemas" / "pilot-timeline-alignment.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(report)


@pytest.mark.parametrize("human,mismatched_offset", [(False, False), (True, True)])
def test_alignment_failure_keeps_clip_gate_closed(tmp_path: Path, human: bool, mismatched_offset: bool) -> None:
    media, anchors, offset_map = _alignment_inputs(tmp_path, human=human, mismatched_offset=mismatched_offset)
    report = evaluate_pilot_alignment(
        "SAMPLE",
        FIXTURES / "sample.structure.srt",
        media,
        anchors,
        offset_map,
        expected_blocks=3,
        duration_seconds=12.0,
    )
    alignment = tmp_path / "alignment.json"
    write_pilot_alignment(alignment, report)

    assert report["status"] == "unresolved"
    assert report["clip_preparation_allowed"] is False
    with pytest.raises(PilotAlignmentError, match="생성을 중단"):
        build_pilot_audio_review_packet(
            "SAMPLE",
            FIXTURES / "sample.structure.srt",
            FIXTURES / "sample.ja.srt",
            media,
            alignment,
            tmp_path / "packet",
            expected_blocks=3,
            clip_extractor=_fake_clip,
        )
    assert not (tmp_path / "packet").exists()

    blocked = write_blocked_audio_review_manifest(
        "SAMPLE",
        alignment,
        tmp_path / "blocked-packet",
        expected_blocks=3,
        reason="alignment unresolved",
    )
    assert blocked["status"] == "blocked"
    assert blocked["block_count"] == 0
    assert not (tmp_path / "blocked-packet" / "clips").exists()
    schema = json.loads((ROOT / "schemas" / "pilot-audio-review-manifest.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(blocked)


def test_full_audio_review_packet_has_audio_japanese_and_context_for_every_block(tmp_path: Path) -> None:
    media, alignment = _resolved_alignment(tmp_path)
    output = tmp_path / "packet"
    manifest = build_pilot_audio_review_packet(
        "SAMPLE",
        FIXTURES / "sample.structure.srt",
        FIXTURES / "sample.ja.srt",
        media,
        alignment,
        output,
        expected_blocks=3,
        clip_extractor=_fake_clip,
    )

    assert manifest["status"] == "ready"
    assert manifest["block_count"] == 3
    assert manifest["all_blocks_have_audio"] is True
    records = [json.loads(line) for line in (output / "blocks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["block_number"] for record in records] == [1, 2, 3]
    assert all(record["japanese"] and record["scene_context"] for record in records)
    assert all((output / record["audio"]["path"]).is_file() for record in records)
    schema = json.loads((ROOT / "schemas" / "pilot-audio-review-manifest.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(manifest)


def test_audio_review_generation_is_reproducible_and_detects_changed_audio(tmp_path: Path) -> None:
    media, alignment = _resolved_alignment(tmp_path)
    first = build_pilot_audio_review_packet(
        "SAMPLE",
        FIXTURES / "sample.structure.srt",
        FIXTURES / "sample.ja.srt",
        media,
        alignment,
        tmp_path / "first",
        expected_blocks=3,
        clip_extractor=_fake_clip,
    )
    second = build_pilot_audio_review_packet(
        "SAMPLE",
        FIXTURES / "sample.structure.srt",
        FIXTURES / "sample.ja.srt",
        media,
        alignment,
        tmp_path / "second",
        expected_blocks=3,
        clip_extractor=_fake_clip,
    )
    assert first == second
    assert (tmp_path / "first" / "blocks.jsonl").read_bytes() == (tmp_path / "second" / "blocks.jsonl").read_bytes()

    media.write_bytes(b"changed-audio")
    with pytest.raises(PilotAlignmentError, match="원음이 변경"):
        build_pilot_audio_review_packet(
            "SAMPLE",
            FIXTURES / "sample.structure.srt",
            FIXTURES / "sample.ja.srt",
            media,
            alignment,
            tmp_path / "changed",
            expected_blocks=3,
            clip_extractor=_fake_clip,
        )
    assert not (tmp_path / "changed").exists()


def test_cli_persists_stop_manifest_when_alignment_is_unresolved(tmp_path: Path) -> None:
    media, anchors, offset_map = _alignment_inputs(tmp_path, human=False)
    alignment = tmp_path / "timeline" / "alignment.json"
    assert main([
        "build-pilot-alignment",
        "--title", "SAMPLE",
        "--structure", str(FIXTURES / "sample.structure.srt"),
        "--audio", str(media),
        "--anchors", str(anchors),
        "--offset-map", str(offset_map),
        "--expected-blocks", "3",
        "--output", str(alignment),
    ]) == 1

    packet = tmp_path / "audio-review"
    assert main([
        "build-pilot-audio-review",
        "--title", "SAMPLE",
        "--structure", str(FIXTURES / "sample.structure.srt"),
        "--ja", str(FIXTURES / "sample.ja.srt"),
        "--audio", str(media),
        "--alignment", str(alignment),
        "--expected-blocks", "3",
        "--output", str(packet),
    ]) == 2
    blocked = json.loads((packet / "manifest.json").read_text(encoding="utf-8"))
    assert blocked["status"] == "blocked"
    assert blocked["clip_preparation_allowed"] is False
    assert blocked["block_count"] == 0
    assert not (packet / "clips").exists()
