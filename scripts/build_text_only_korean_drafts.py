"""Build structure-locked, text-only Korean subtitle drafts from Japanese SRT files.

This helper deliberately does not consult legacy Korean subtitles, audio, video, or
ASR.  It records one decision per source block and labels every output as an
unreviewed machine draft; it must never be promoted as a final subtitle.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable

from deep_translator import GoogleTranslator

from translation_forensics.srt import JAPANESE_RE, SubtitleBlock, compare_structure, parse_srt, write_srt


MARKER_RE = re.compile(r"__TFBLOCK(\d{6})__")


def chunks(blocks: list[SubtitleBlock], limit: int = 3_000) -> Iterable[list[SubtitleBlock]]:
    current: list[SubtitleBlock] = []
    size = 0
    for block in blocks:
        addition = len(block.text) + 28
        if current and size + addition > limit:
            yield current
            current, size = [], 0
        current.append(block)
        size += addition
    if current:
        yield current


def translate_request(text: str) -> str:
    for attempt in range(4):
        try:
            result = GoogleTranslator(source="ja", target="ko").translate(text)
            if result:
                return result
        except Exception:
            pass
        time.sleep(1 + attempt)
    raise RuntimeError("translation request returned no text after retries")


def parse_batch(result: str, batch: list[SubtitleBlock]) -> dict[int, str] | None:
    matches = list(MARKER_RE.finditer(result))
    expected = [block.number for block in batch]
    if [int(match.group(1)) for match in matches] != expected:
        return None
    translated: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(result)
        text = result[match.end():end].strip(" \t\r\n:-")
        translated[int(match.group(1))] = text
    return translated


def translate_blocks(blocks: list[SubtitleBlock]) -> tuple[dict[int, str], dict[int, str]]:
    translated: dict[int, str] = {}
    failures: dict[int, str] = {}
    for batch in chunks(blocks):
        payload = "\n".join(f"__TFBLOCK{block.number:06d}__\n{block.text}" for block in batch)
        try:
            parsed = parse_batch(translate_request(payload), batch)
        except RuntimeError:
            parsed = None
        if parsed is not None:
            translated.update(parsed)
            continue
        # A changed marker is not trusted. Retry only the affected blocks.
        for block in batch:
            try:
                translated[block.number] = translate_request(block.text).strip()
            except RuntimeError as exc:
                failures[block.number] = str(exc)
    return translated, failures


def build(title: str, project_root: Path, *, version: str) -> dict[str, object]:
    source = project_root / title / f"{title}.ja.srt"
    workspace = project_root / "translation-forensics" / "workspaces" / title
    intermediate = workspace / "intermediate"
    decisions_path = intermediate / f"{title}.translation-decisions-text-only-{version}.jsonl"
    source_output = intermediate / f"{title}.ko-source-faithful-text-only-{version}.srt"
    viewer_output = intermediate / f"{title}.ko-viewer-natural-text-only-{version}.srt"
    report_path = intermediate / f"{title}.text-only-draft-{version}.report.json"
    outputs = (decisions_path, source_output, viewer_output, report_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite versioned outputs: {existing}")

    blocks, encoding, newline = parse_srt(source)
    translations, failures = translate_blocks(blocks)
    decisions: list[dict[str, object]] = []
    residual_japanese: list[int] = []
    blocked: list[int] = []
    for block in blocks:
        text = translations.get(block.number, "").strip()
        problem = failures.get(block.number, "")
        if not text or JAPANESE_RE.search(text):
            residual_japanese.append(block.number) if JAPANESE_RE.search(text) else None
            blocked.append(block.number)
            problem = problem or "machine translation missing or retained Japanese characters"
            text = ""
        status = "machine_text_draft" if block.number not in blocked else "blocked_machine_output"
        note = (
            "Japanese SRT text was translated directly by a machine service; no audio, video, ASR, "
            "legacy Korean subtitle, or human semantic review was used."
            if status == "machine_text_draft"
            else f"Text-only machine draft could not be produced safely: {problem}"
        )
        decisions.append({
            "block_number": block.number,
            "source_japanese": block.text,
            "previous_korean": "",
            "source_faithful_korean": text,
            "viewer_natural_korean": text,
            "translation_method": "machine_text_translation_from_japanese_srt",
            "status": status,
            "confidence": "unverified",
            "uncertain_slots": ["no_human_semantic_review"],
            "evidence_refs": ["japanese_srt", "machine_text_translation"],
            "review_note": note,
        })

    intermediate.mkdir(parents=True, exist_ok=True)
    with decisions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
    if not blocked:
        output_blocks = [SubtitleBlock(block.number, block.start, block.end, translations[block.number], block.start_seconds, block.end_seconds) for block in blocks]
        write_srt(source_output, output_blocks)
        write_srt(viewer_output, output_blocks)
        source_check = compare_structure(blocks, parse_srt(source_output)[0])
        viewer_check = compare_structure(blocks, parse_srt(viewer_output)[0])
    else:
        source_check = {"pass": False, "issues": [{"code": "blocked_machine_output", "blocks": blocked}]}
        viewer_check = source_check
    report = {
        "status": "text-only-machine-draft",
        "title": title,
        "scope": "all source SRT blocks",
        "source": str(source),
        "source_encoding": encoding,
        "source_newline": newline,
        "translation_inputs": ["japanese_srt"],
        "excluded_inputs": ["legacy_korean", "audio", "video", "asr", "human_review"],
        "decisions": str(decisions_path),
        "source_faithful_draft": str(source_output) if not blocked else None,
        "viewer_natural_draft": str(viewer_output) if not blocked else None,
        "blocks": len(blocks),
        "machine_draft_blocks": len(blocks) - len(blocked),
        "blocked_machine_output_blocks": blocked,
        "residual_japanese_blocks": residual_japanese,
        "source_structure_check": source_check,
        "viewer_structure_check": viewer_check,
        "final_promotion_allowed": False,
        "limitations": "Machine text translation only. These are not audio-verified, human-reviewed, evaluated, or final subtitles.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--version", default="v2")
    parser.add_argument("titles", nargs="+")
    args = parser.parse_args()
    reports = [build(title, args.project_root, version=args.version) for title in args.titles]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
