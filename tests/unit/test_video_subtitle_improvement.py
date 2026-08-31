from pathlib import Path

import pytest

from translation_forensics.srt import compare_structure, parse_srt, parse_srt_text
from translation_forensics.video_subtitle_improvement import (
    ImprovementError,
    _improve_file,
    discover_common_targets,
    residual_file_stats,
    select_overlap_translation,
    sha256_file,
)


def _blocks(text: str):
    return parse_srt_text(text)


def test_overlap_translation_excludes_boilerplate_and_deduplicates() -> None:
    target = _blocks("1\n00:00:00,000 --> 00:00:04,000\n日本語\n")[0]
    source_ja = _blocks(
        "1\n00:00:00,000 --> 00:00:01,000\nご視聴ありがとうございました\n\n"
        "2\n00:00:01,000 --> 00:00:03,000\n分かった\n\n"
        "3\n00:00:02,000 --> 00:00:04,000\n分かった\n"
    )
    source_ko = _blocks(
        "1\n00:00:00,000 --> 00:00:01,000\n시청해 주셔서 감사합니다\n\n"
        "2\n00:00:01,000 --> 00:00:03,000\n알았어.\n\n"
        "3\n00:00:02,000 --> 00:00:04,000\n알았어.\n"
    )

    text, source_numbers = select_overlap_translation(target, source_ja, source_ko)

    assert text == "알았어."
    assert source_numbers == [2]


def test_overlap_translation_rejects_a_tiny_boundary_touch() -> None:
    target = _blocks("1\n00:00:00,000 --> 00:00:10,000\n日本語\n")[0]
    source_ja = _blocks("1\n00:00:09,900 --> 00:00:20,000\n言葉\n")
    source_ko = _blocks("1\n00:00:09,900 --> 00:00:20,000\n말\n")

    text, source_numbers = select_overlap_translation(target, source_ja, source_ko)

    assert text is None
    assert source_numbers == []


def test_manual_improvement_is_hash_guarded_and_preserves_structure(tmp_path: Path) -> None:
    input_path = tmp_path / "input.srt"
    output_path = tmp_path / "output.srt"
    input_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n아ー!\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n그대로 둔다.\n",
        encoding="utf-8",
    )
    config = {
        "sha256": sha256_file(input_path),
        "mode": "manual",
        "default_method": "punctuation-normalization",
        "overrides": {"1": "아아!"},
    }

    ledger, report = _improve_file(
        input_path=input_path,
        output_path=output_path,
        file_config=config,
        source_root=tmp_path,
    )

    before, _, _ = parse_srt(input_path)
    after, _, _ = parse_srt(output_path)
    assert compare_structure(before, after)["pass"] is True
    assert after[0].text == "아아!"
    assert after[1].text == "그대로 둔다."
    assert report["residual_after"]["characters"] == 0
    assert ledger[0]["method"] == "punctuation-normalization"


def test_manual_improvement_rejects_hash_mismatch(tmp_path: Path) -> None:
    input_path = tmp_path / "input.srt"
    input_path.write_text("1\n00:00:00,000 --> 00:00:01,000\n日本語\n", encoding="utf-8")

    with pytest.raises(ImprovementError, match="input hash mismatch"):
        _improve_file(
            input_path=input_path,
            output_path=tmp_path / "output.srt",
            file_config={
                "sha256": "0" * 64,
                "mode": "manual",
                "overrides": {"1": "한국어"},
            },
            source_root=tmp_path,
        )


def test_manual_improvement_requires_full_residual_coverage(tmp_path: Path) -> None:
    input_path = tmp_path / "input.srt"
    input_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n日本語\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n言葉\n",
        encoding="utf-8",
    )

    with pytest.raises(ImprovementError, match="do not cover every residual"):
        _improve_file(
            input_path=input_path,
            output_path=tmp_path / "output.srt",
            file_config={
                "sha256": sha256_file(input_path),
                "mode": "manual",
                "overrides": {"1": "한국어"},
            },
            source_root=tmp_path,
        )


def test_common_target_discovery_ignores_non_subtitle_directories(tmp_path: Path) -> None:
    source_root = tmp_path / "Videos"
    target_root = tmp_path / "video"
    (source_root / ".cache").mkdir(parents=True)
    (source_root / "TITLE-001").mkdir()
    target_root.mkdir()
    (source_root / "TITLE-001" / "TITLE-001_ja.srt").write_text("x", encoding="utf-8")
    (target_root / "TITLE-001.srt").write_text("x", encoding="utf-8")

    mapping = discover_common_targets(source_root, target_root, {})

    assert mapping == {"TITLE-001": target_root / "TITLE-001.srt"}


def test_unrelated_duplicate_numbers_do_not_block_a_residual_fix(tmp_path: Path) -> None:
    input_path = tmp_path / "input.srt"
    output_path = tmp_path / "output.srt"
    input_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n아ー!\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n그대로.\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\n이것도 그대로.\n",
        encoding="utf-8",
    )

    _improve_file(
        input_path=input_path,
        output_path=output_path,
        file_config={
            "sha256": sha256_file(input_path),
            "mode": "manual",
            "overrides": {"1": "아아!"},
        },
        source_root=tmp_path,
    )

    after, _, _ = parse_srt(output_path)
    assert [block.number for block in after] == [1, 2, 2]


def test_residual_file_audit_tolerates_an_unrelated_bad_timecode(tmp_path: Path) -> None:
    path = tmp_path / "bad-time.srt"
    path.write_text("1\n00:09:99,990 --> 00:10:00,000\n日本語\n", encoding="utf-8")

    assert residual_file_stats(path) == {"blocks": 1, "lines": 1, "characters": 3}
