"""Compile audited direct-translation checkpoints into an SRT without altering timing."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uff66-\uff9f]")


def split_srt(raw: str) -> list[list[str]]:
    blocks = [block.splitlines() for block in re.split(r"\n{2,}", raw.strip()) if block.strip()]
    for expected, lines in enumerate(blocks, 1):
        if len(lines) < 3 or lines[0] != str(expected) or " --> " not in lines[1]:
            raise ValueError(f"Invalid source SRT block {expected}")
    return blocks


def main(source_path: str, checkpoints: str, output_path: str, report_path: str) -> None:
    source = split_srt(Path(source_path).read_text(encoding="utf-8").replace("\r\n", "\n"))
    translations: dict[int, str] = {}
    for path in sorted(Path(checkpoints).glob("JUQ-778.gpt56-direct-v1.checkpoint-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            n, ko = row["block_number"], row["korean"]
            if n in translations:
                raise ValueError(f"Duplicate decision for block {n}")
            if row.get("status") != "translated" or not isinstance(ko, str) or not ko.strip():
                raise ValueError(f"Invalid translation decision for block {n}")
            translations[n] = ko.strip()
    missing = [str(i) for i in range(1, len(source) + 1) if i not in translations]
    residue = [str(n) for n, ko in translations.items() if JAPANESE.search(ko)]
    report = {
        "source_blocks": len(source),
        "translation_decisions": len(translations),
        "missing_blocks": missing,
        "japanese_residue_blocks": residue,
        "complete": not missing and not residue,
        "method": "GPT-5.6 direct translation only; no external/local MT or legacy Korean subtitles used",
    }
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    if not report["complete"]:
        raise SystemExit("Refusing to create final SRT until all checkpoints are valid.")
    rendered = []
    for lines in source:
        n = int(lines[0])
        rendered.append("\n".join([lines[0], lines[1], translations[n]]))
    Path(output_path).write_text("\n\n".join(rendered) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: compile_direct_translation.py SOURCE CHECKPOINT_DIR OUTPUT REPORT")
    main(*sys.argv[1:])
