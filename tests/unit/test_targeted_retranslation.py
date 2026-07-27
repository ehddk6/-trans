from __future__ import annotations

import json
from pathlib import Path

from translation_forensics.targeted_retranslation import apply_targeted_retranslations
from translation_forensics.srt import parse_srt


def _write(path: Path, value: str) -> Path:
    path.write_text(value, encoding="utf-8", newline="\n")
    return path


def _srt(lines: list[str]) -> str:
    return "\n\n".join(f"{index}\n00:00:0{index},000 --> 00:00:0{index + 1},000\n{text}" for index, text in enumerate(lines, 1)) + "\n"


def test_targeted_retranslation_replaces_supported_and_marks_hold(tmp_path: Path) -> None:
    structure = _write(tmp_path / "structure.srt", _srt(["一", "二"]))
    source = _write(tmp_path / "source.srt", _srt(["낡은 하나", "낡은 둘"]))
    viewer = _write(tmp_path / "viewer.srt", _srt(["낡은 하나", "낡은 둘"]))
    response = _write(
        tmp_path / "response.json",
        json.dumps({"results": [
            {"title": "SAMPLE", "block_number": 1, "decision": "replace", "source_faithful_korean": "새 하나", "viewer_natural_korean": "새로운 하나", "confidence": "high", "basis": ["japanese_srt"], "reason": "명확"},
            {"title": "SAMPLE", "block_number": 2, "decision": "hold", "source_faithful_korean": "", "viewer_natural_korean": "", "confidence": "low", "basis": ["japanese_srt"], "reason": "손상"},
        ]}, ensure_ascii=False),
    )
    result = apply_targeted_retranslations(title="SAMPLE", structure_path=structure, source_path=source, viewer_path=viewer, response_paths=[response], output_dir=tmp_path / "v2")
    blocks, _, _ = parse_srt(Path(result["source_output"]))
    assert [block.text for block in blocks] == ["새 하나", "…"]
    assert result["replacement_blocks"] == 1
    assert result["hold_marker_blocks"] == 1


def test_targeted_retranslation_review_can_revert_a_replace(tmp_path: Path) -> None:
    structure = _write(tmp_path / "structure.srt", _srt(["一"]))
    source = _write(tmp_path / "source.srt", _srt(["낡은 하나"]))
    viewer = _write(tmp_path / "viewer.srt", _srt(["낡은 하나"]))
    response = _write(tmp_path / "response.json", json.dumps({"results": [{"title": "SAMPLE", "block_number": 1, "decision": "replace", "source_faithful_korean": "새 하나", "viewer_natural_korean": "새 하나", "confidence": "high", "basis": ["japanese_srt"], "reason": "명확"}]}, ensure_ascii=False))
    review = _write(tmp_path / "review.json", json.dumps({"reviews": [{"title": "SAMPLE", "block_number": 1, "verdict": "revert", "reason": "근거 부족"}]}, ensure_ascii=False))
    result = apply_targeted_retranslations(title="SAMPLE", structure_path=structure, source_path=source, viewer_path=viewer, response_paths=[response], review_paths=[review], output_dir=tmp_path / "v3", version="v3")
    blocks, _, _ = parse_srt(Path(result["source_output"]))
    assert blocks[0].text == "…"
    assert result["replacement_blocks"] == 0
    assert result["review_reverted_blocks"] == 1
    assert Path(result["source_output"]).name.endswith("targeted-retranslation-v3.srt")
