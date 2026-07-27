"""Mechanically reconstruct a decision ledger from two generated Korean SRTs.

This utility deliberately accepts no source subtitle: it records only exact text
present in the supplied generated artifacts and carries their unreviewed status.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def parse_srt(path: Path) -> list[tuple[int, str, str]]:
    raw = path.read_text(encoding="utf-8")
    if "\r" in raw:
        raise ValueError(f"{path}: expected LF-only input")
    rows: list[tuple[int, str, str]] = []
    for position, block in enumerate(re.split(r"\n{2,}", raw.rstrip("\n")), 1):
        lines = block.split("\n")
        if len(lines) < 3 or lines[0] != str(position) or " --> " not in lines[1]:
            raise ValueError(f"{path}: invalid SRT block {position}")
        text = "\n".join(lines[2:])
        if not text:
            raise ValueError(f"{path}: blank Korean text at block {position}")
        rows.append((position, lines[1], text))
    return rows


def main(source_faithful: str, viewer_natural: str, report: str, output: str) -> None:
    source = parse_srt(Path(source_faithful))
    viewer = parse_srt(Path(viewer_natural))
    if len(source) != 577 or len(viewer) != 577:
        raise ValueError("expected exactly 577 generated SRT blocks")
    if not Path(report).is_file():
        raise ValueError("generated GPT-5.6 report is required")

    ledger: list[dict[str, object]] = []
    for (number, timecode, source_text), (viewer_number, viewer_timecode, viewer_text) in zip(source, viewer, strict=True):
        if (number, timecode) != (viewer_number, viewer_timecode):
            raise ValueError(f"generated SRT structures differ at block {number}")
        ledger.append({
            "block_number": number,
            "source_japanese": "",
            "previous_korean": "",
            "source_faithful_korean": source_text,
            "viewer_natural_korean": viewer_text,
            "translation_method": "reconstructed_from_generated_gpt56_srt",
            "translation_model": "gpt-5.6-terra",
            "status": "translated",
            "confidence": "not_evaluated",
            "uncertain_slots": ["source_and_translation_not_independently_re_evaluated"],
            "evidence_refs": [
                "generated_gpt56_source_faithful_srt",
                "generated_gpt56_viewer_natural_srt",
                "generated_gpt56_report",
            ],
            "review_note": "Mechanically extracted exactly from generated GPT-5.6 SRTs; no human review, audio verification, evaluation, or final verification.",
            "provenance": "reconstructed_from_generated_gpt56_artifacts_only",
        })
    Path(output).write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in ledger),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: extract_generated_srt_ledger.py SOURCE_SRT VIEWER_SRT REPORT OUTPUT")
    main(*sys.argv[1:])
