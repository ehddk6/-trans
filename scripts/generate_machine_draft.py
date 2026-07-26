"""Generate explicitly unverified Korean SRT drafts from a Japanese SRT.

This is a draft-production utility, not a semantic-review or final-packaging step.
It never reads a prior Korean subtitle and writes only explicitly versioned outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import requests

from translation_forensics.srt import SubtitleBlock, has_japanese, parse_srt, write_srt


ENGINE = "google-translate-web-endpoint"
ENDPOINT = "https://translate.googleapis.com/translate_a/single"
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def translate_text(text: str, session: requests.Session) -> str:
    for attempt in range(4):
        try:
            response = session.get(
                ENDPOINT,
                params={"client": "gtx", "sl": "ja", "tl": "ko", "dt": "t", "q": text},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            translated = "".join(part[0] for part in payload[0] if part and part[0]).strip()
            if translated:
                return translated
            raise ValueError("empty translation response")
        except (requests.RequestException, ValueError, TypeError, IndexError, json.JSONDecodeError):
            if attempt == 3:
                raise
            time.sleep(0.75 * (attempt + 1))
    raise AssertionError("unreachable")


def build_draft(title: str, source_path: Path, workspace: Path, version: str, workers: int) -> dict[str, Any]:
    blocks, encoding, newline = parse_srt(source_path)
    cache: dict[str, str] = {}
    lock = Lock()

    def task(block: SubtitleBlock) -> tuple[int, str]:
        with lock:
            cached = cache.get(block.text)
        if cached is not None:
            return block.number, cached
        with requests.Session() as session:
            translated = translate_text(block.text, session)
        with lock:
            cache.setdefault(block.text, translated)
            return block.number, cache[block.text]

    translations: dict[int, str] = {}
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(task, block): block for block in blocks}
        for future in as_completed(futures):
            block = futures[future]
            try:
                number, translated = future.result()
                if JAPANESE_RE.search(translated):
                    raise ValueError("machine output retains Japanese characters")
                translations[number] = translated
            except Exception as exc:  # records are deliberately blocked rather than replaced with placeholders
                failures.append({"block_number": block.number, "reason": str(exc)})

    intermediate = workspace / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    stem = f"{title}.machine-assisted-ja-ko-{version}"
    decisions_path = intermediate / f"{stem}.decisions.jsonl"
    blocked_path = intermediate / f"{stem}.blocked.json"
    report_path = intermediate / f"{stem}.report.json"
    source_output = intermediate / f"{stem}.source-faithful.srt"
    viewer_output = intermediate / f"{stem}.viewer-natural.srt"

    records: list[dict[str, Any]] = []
    for block in blocks:
        translated = translations.get(block.number, "")
        records.append(
            {
                "block_number": block.number,
                "timecode": f"{block.start} --> {block.end}",
                "source_japanese": block.text,
                "source_faithful_korean": translated,
                "viewer_natural_korean": translated,
                "translation_method": "machine_translation_from_japanese_unverified",
                "status": "machine_draft" if translated else "blocked",
                "confidence": "low",
                "uncertain_slots": ["all_semantic_slots"] if translated else ["translation_unavailable"],
                "evidence_refs": ["japanese_srt"],
                "review_note": "일본어 SRT만을 입력으로 한 기계 번역 초안. 기존 한국어·오디오·ASR·화면·사람 검수는 사용하지 않았다.",
            }
        )
    with decisions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report: dict[str, Any] = {
        "title": title,
        "status": "machine-assisted-draft-unverified" if not failures else "machine-assisted-draft-blocked",
        "source_japanese": str(source_path.resolve()),
        "source_encoding": encoding,
        "source_line_endings": newline,
        "structure_blocks": len(blocks),
        "translated_blocks": len(translations),
        "blocked_blocks": len(failures),
        "decisions": str(decisions_path.resolve()),
        "source_faithful_output": str(source_output.resolve()) if not failures else None,
        "viewer_natural_output": str(viewer_output.resolve()) if not failures else None,
        "machine_engine": ENGINE,
        "legacy_korean_read": False,
        "audio_or_asr_used": False,
        "human_review_claimed": False,
        "text_crosschecked_claimed": False,
        "final_promotion_allowed": False,
        "viewer_natural_note": "기계 초안 단계에서는 의미 보존 확인 전이므로 두 SRT의 문구를 동일하게 유지했다.",
    }
    if failures:
        blocked_path.write_text(json.dumps({"title": title, "blocked": failures}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        report["blocked"] = str(blocked_path.resolve())
    else:
        source_blocks = [SubtitleBlock(block.number, block.start, block.end, translations[block.number], block.start_seconds, block.end_seconds) for block in blocks]
        viewer_blocks = [SubtitleBlock(block.number, block.start, block.end, translations[block.number], block.start_seconds, block.end_seconds) for block in blocks]
        write_srt(source_output, source_blocks)
        write_srt(viewer_output, viewer_blocks)
        report["structure_preserved"] = True
        report["japanese_residue_blocks"] = [block.number for block in source_blocks if has_japanese(block.text)]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--workers", default=6, type=int)
    args = parser.parse_args()
    report = build_draft(args.title, args.source, args.workspace, args.version, args.workers)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["blocked_blocks"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
