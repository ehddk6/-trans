from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .alignment import align_by_overlap
from .srt import SubtitleBlock, parse_srt, write_srt


def build_korean_aligned_draft(ja_path: Path, previous_ko_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    japanese, _, _ = parse_srt(ja_path)
    previous, _, _ = parse_srt(previous_ko_path)
    alignments = align_by_overlap(japanese, previous)
    previous_by_number = {block.number: block for block in previous}
    used = Counter()
    draft_blocks: list[SubtitleBlock] = []
    unresolved: list[int] = []
    for source, alignment in zip(japanese, alignments):
        if alignment.candidate_number is None:
            text = "[번역 필요]"
            unresolved.append(source.number)
        else:
            text = previous_by_number[alignment.candidate_number].text or "[번역 필요]"
            used[alignment.candidate_number] += 1
            if not text.strip():
                unresolved.append(source.number)
        draft_blocks.append(SubtitleBlock(source.number, source.start, source.end, text, source.start_seconds, source.end_seconds))
    write_srt(output_path, draft_blocks)
    repeated = {str(number): count for number, count in used.items() if count > 1}
    report = {
        "status": "draft-unverified",
        "source_japanese": str(ja_path),
        "previous_korean": str(previous_ko_path),
        "output": str(output_path),
        "structure_locked_to": str(ja_path),
        "japanese_blocks": len(japanese),
        "previous_korean_blocks": len(previous),
        "draft_blocks": len(draft_blocks),
        "matched_blocks": len(japanese) - len(unresolved),
        "unresolved_blocks": unresolved,
        "reused_previous_blocks": repeated,
        "note": "기존 한국어 후보를 시간 겹침으로 정렬한 번역 초안이며, 일본어 의미 검수 전에는 최종본이 아닙니다.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report
