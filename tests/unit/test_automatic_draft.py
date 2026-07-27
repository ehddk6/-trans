from __future__ import annotations

import json
from pathlib import Path

from translation_forensics.automatic_draft import build_automatic_draft
from translation_forensics.srt import parse_srt


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _srt(texts: list[str]) -> str:
    return "\n\n".join(
        f"{index}\n00:00:0{index},000 --> 00:00:0{index + 1},000\n{text}"
        for index, text in enumerate(texts, 1)
    ) + "\n"


def test_automatic_draft_uses_partial_decisions_then_aligned_fallback(tmp_path: Path) -> None:
    structure = _write(tmp_path / "structure.srt", _srt(["一", "二", "三"]))
    fallback = _write(tmp_path / "fallback.srt", _srt(["첫째", "둘째", "셋째"]))
    decisions = _write(
        tmp_path / "partial.jsonl",
        json.dumps({"block_number": 1, "source_faithful_korean": "직역 하나", "viewer_natural_korean": "자연 하나"}, ensure_ascii=False) + "\n",
    )
    result = build_automatic_draft(
        title="SAMPLE",
        structure_path=structure,
        output_dir=tmp_path / "automatic",
        fallback_path=fallback,
        decision_candidate_path=decisions,
    )
    source, _, _ = parse_srt(Path(result["source_output"]))
    viewer, _, _ = parse_srt(Path(result["viewer_output"]))
    assert result["status"] == "automatic-draft-complete"
    assert [block.text for block in source] == ["직역 하나", "둘째", "셋째"]
    assert [block.text for block in viewer] == ["자연 하나", "둘째", "셋째"]
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert report["unresolved_blocks"] == []
    assert report["final_promotion_allowed"] is False
