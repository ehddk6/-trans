from __future__ import annotations

import json
from pathlib import Path

import pytest

from translation_forensics.visual_context import (
    ALLOWED_VISUAL_SLOTS,
    VisualFrame,
    build_visual_context_record,
    index_legacy_captures,
    parse_capture_timestamp,
    prepare_model_attachments,
    select_nearest_frames,
    sha256_file,
    title_code_matches,
)


def _image(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def test_capture_timestamp_ignores_subtitle_like_index() -> None:
    assert parse_capture_timestamp("sub_0231_7181.1s.jpg") == 7181.1
    assert parse_capture_timestamp("sub_9999_42s.png") == 42.0
    with pytest.raises(ValueError, match="no seconds timestamp"):
        parse_capture_timestamp("sub_7181.jpg")


def test_capture_index_sha256_deduplicates_pixels(tmp_path: Path) -> None:
    first = _image(tmp_path / "sub_0001_1.0s.jpg", b"same pixels")
    _image(tmp_path / "sub_9999_1.1s.jpg", b"same pixels")
    third = _image(tmp_path / "sub_0002_5.0s.jpg", b"different pixels")
    _image(tmp_path / "cover.jpg", b"not timestamped")

    frames = index_legacy_captures(tmp_path)

    assert [(frame.path, frame.timestamp_seconds) for frame in frames] == [
        (first.resolve(), 1.0),
        (third.resolve(), 5.0),
    ]
    assert [frame.sha256 for frame in frames] == [sha256_file(first), sha256_file(third)]


def test_selects_nearest_unique_frames_for_start_middle_end(tmp_path: Path) -> None:
    paths = [
        _image(tmp_path / "sub_1000_0.1s.jpg", b"start"),
        _image(tmp_path / "sub_0001_5.1s.jpg", b"middle"),
        _image(tmp_path / "sub_9000_5.2s.jpg", b"middle"),
        _image(tmp_path / "sub_0002_9.9s.jpg", b"end"),
    ]
    frames = [
        VisualFrame(path=path.resolve(), timestamp_seconds=parse_capture_timestamp(path), sha256=sha256_file(path))
        for path in paths
    ]

    selected = select_nearest_frames(frames, start_seconds=0, end_seconds=10)

    assert [item.selection_reason for item in selected] == [
        "nearest_to_unit_start",
        "nearest_to_unit_middle",
        "nearest_to_unit_end",
    ]
    assert [item.frame.timestamp_seconds for item in selected] == [0.1, 5.1, 9.9]
    assert len({item.frame.sha256 for item in selected}) == 3


def test_reduced_frame_limit_uses_informative_anchors(tmp_path: Path) -> None:
    paths = [
        _image(tmp_path / "sub_0001_0.0s.jpg", b"start"),
        _image(tmp_path / "sub_0002_5.0s.jpg", b"middle"),
        _image(tmp_path / "sub_0003_10.0s.jpg", b"end"),
    ]
    frames = [
        VisualFrame(path=path.resolve(), timestamp_seconds=parse_capture_timestamp(path), sha256=sha256_file(path))
        for path in paths
    ]

    one = select_nearest_frames(frames, start_seconds=0, end_seconds=10, max_frames=1)
    two = select_nearest_frames(frames, start_seconds=0, end_seconds=10, max_frames=2)

    assert [(item.selection_reason, item.frame.timestamp_seconds) for item in one] == [
        ("nearest_to_unit_middle", 5.0)
    ]
    assert [(item.selection_reason, item.frame.timestamp_seconds) for item in two] == [
        ("nearest_to_unit_start", 0.0),
        ("nearest_to_unit_end", 10.0),
    ]


def test_title_code_matching_is_exact_and_rejects_ambiguous_paths() -> None:
    assert title_code_matches("ABP-169", r"C:\captures\ABP-169_kr\timestamp_frames")
    assert title_code_matches("adn-622", "ADN-622.ja.srt")
    assert not title_code_matches("ABP-169", r"C:\captures\ABF-169_kr\timestamp_frames")
    assert not title_code_matches("ABP-169", "ABP-169-and-ABF-169")


def test_metadata_only_record_has_allowed_slots_and_zero_pixel_transfers(tmp_path: Path) -> None:
    path = _image(tmp_path / "sub_0001_5.0s.jpg", b"frame")
    frame = VisualFrame(path=path.resolve(), timestamp_seconds=5.0, sha256=sha256_file(path))

    record = build_visual_context_record(
        title_id="ADN-622",
        unit_id="unit-0001",
        start_seconds=4.0,
        end_seconds=6.0,
        frames=[frame],
        visual_slots={"speaker": "left person", "on_screen_text": None},
    )

    assert tuple(record["visual_slots"]) == ALLOWED_VISUAL_SLOTS
    assert record["external_transfer_receipt"] == {
        "status": "not-sent",
        "external_transfer": False,
        "pixel_transfer_count": 0,
        "transferred_frame_sha256": [],
        "provider": None,
        "request_id": None,
    }
    assert prepare_model_attachments(record) == []
    json.dumps(record, ensure_ascii=False)


def test_targeted_mode_prepares_descriptors_without_claiming_transfer(tmp_path: Path) -> None:
    path = _image(tmp_path / "sub_0001_5.0s.jpg", b"frame")
    frame = VisualFrame(path=path.resolve(), timestamp_seconds=5.0, sha256=sha256_file(path))
    record = build_visual_context_record(
        title_id="ADN-622",
        unit_id="unit-0001",
        start_seconds=4.0,
        end_seconds=6.0,
        frames=[frame],
        transfer_mode="targeted",
    )

    attachments = prepare_model_attachments(record)

    assert attachments == [
        {
            "path": str(path.resolve()),
            "mime_type": "image/jpeg",
            "sha256": sha256_file(path),
            "selection_reason": "nearest_to_unit_start",
        }
    ]
    assert record["external_transfer_receipt"]["external_transfer"] is False
    assert record["external_transfer_receipt"]["pixel_transfer_count"] == 0


def test_record_rejects_visual_inference_outside_allowed_slots(tmp_path: Path) -> None:
    path = _image(tmp_path / "sub_0001_5.0s.jpg", b"frame")
    frame = VisualFrame(path=path.resolve(), timestamp_seconds=5.0, sha256=sha256_file(path))

    with pytest.raises(ValueError, match="unsupported visual slots: action"):
        build_visual_context_record(
            title_id="ADN-622",
            unit_id="unit-0001",
            start_seconds=4.0,
            end_seconds=6.0,
            frames=[frame],
            visual_slots={"action": "unsupported inference"},
        )
