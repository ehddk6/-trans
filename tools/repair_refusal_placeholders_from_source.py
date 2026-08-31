from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from translation_forensics.srt import (
    JAPANESE_RE,
    SubtitleBlock,
    parse_srt,
    render_srt,
    seconds_to_timecode,
)


PLACEHOLDER = "[안전상 번역 불가]"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.split()).strip()


def split_chunks(text: str, limit: int = 84) -> list[str]:
    value = normalized(text)
    if len(value) <= limit:
        return [value]
    words = value.split()
    if len(words) == 1:
        return [value[index : index + limit] for index in range(0, len(value), limit)]
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > limit:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def wrap_two_lines(text: str, limit: int = 42) -> str:
    value = normalized(text)
    if len(value) <= limit:
        return value
    fitting = [
        index
        for index in range(1, len(value))
        if len(value[:index].rstrip()) <= limit
        and len(value[index:].lstrip()) <= limit
    ]
    preferred = [index for index in fitting if value[index - 1] in " ,.?!…"]
    pool = preferred or fitting
    if not pool:
        raise ValueError(f"Text cannot fit a two-line layout: {value!r}")
    split_at = min(
        pool,
        key=lambda index: (
            max(len(value[:index].rstrip()), len(value[index:].lstrip())),
            abs(len(value[:index].rstrip()) - len(value[index:].lstrip())),
        ),
    )
    return f"{value[:split_at].rstrip()}\n{value[split_at:].lstrip()}"


def placeholder_runs(blocks: list[SubtitleBlock]) -> list[list[SubtitleBlock]]:
    runs: list[list[SubtitleBlock]] = []
    current: list[SubtitleBlock] = []
    for block in blocks:
        if normalized(block.text) == PLACEHOLDER:
            current.append(block)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def normalize_layout(
    blocks: list[SubtitleBlock],
) -> tuple[list[SubtitleBlock], int]:
    output: list[SubtitleBlock] = []
    repaired = 0
    for block in blocks:
        chunks = split_chunks(block.text)
        duration = block.end_seconds - block.start_seconds
        rendered = [wrap_two_lines(chunk) for chunk in chunks]
        if len(rendered) == 1 and rendered[0] == block.text:
            output.append(block)
            continue
        repaired += 1
        for index, text in enumerate(rendered):
            start = block.start_seconds + duration * index / len(rendered)
            end = block.start_seconds + duration * (index + 1) / len(rendered)
            if end - start < 0.08:
                raise ValueError(
                    f"Block {block.number} is too short for {len(rendered)} layout chunks"
                )
            output.append(
                replace(
                    block,
                    start=seconds_to_timecode(start),
                    end=seconds_to_timecode(end),
                    start_seconds=start,
                    end_seconds=end,
                    text=text,
                )
            )
    return output, repaired


def apply_title_cleanup(
    title: str, blocks: list[SubtitleBlock]
) -> tuple[list[SubtitleBlock], list[dict[str, Any]]]:
    ledger: list[dict[str, Any]] = []
    output: list[SubtitleBlock] = []
    for block in blocks:
        if title == "SONE-676" and block.start == "02:16:23,720":
            ledger.append(
                {
                    "action": "drop",
                    "start": block.start,
                    "text": block.text,
                    "reason": "partial preceding source fragment at replacement boundary",
                }
            )
            continue
        new_block = block
        if title == "SONE-676" and block.start == "02:16:24,420":
            new_block = replace(
                block,
                start="02:16:23,720",
                start_seconds=block.start_seconds - 0.7,
                text="엄청 기분 좋아요.",
            )
        elif title == "START-306" and block.start == "01:12:54,190":
            new_block = replace(block, text="그러지 마… 울지 마.")
        elif title == "SNOS-232" and block.start == "02:26:22,700":
            new_block = replace(block, text="괜찮아?")
        if new_block != block:
            ledger.append(
                {
                    "action": "replace",
                    "start": block.start,
                    "old_text": block.text,
                    "new_text": new_block.text,
                    "reason": "source-boundary fragment normalized for readable Korean",
                }
            )
        output.append(new_block)
    return output, ledger


def source_replacements(
    run: list[SubtitleBlock], source: list[SubtitleBlock]
) -> tuple[list[SubtitleBlock], dict[str, Any]]:
    run_start = min(block.start_seconds for block in run)
    run_end = max(block.end_seconds for block in run)
    selected = [
        block
        for block in source
        if block.end_seconds > run_start
        and block.start_seconds < run_end
        and normalized(block.text) != PLACEHOLDER
    ]
    replacements: list[SubtitleBlock] = []
    source_numbers: list[int] = []
    for block in selected:
        start = max(run_start, block.start_seconds)
        end = min(run_end, block.end_seconds)
        if end <= start:
            continue
        chunks = split_chunks(block.text)
        duration = end - start
        for index, chunk in enumerate(chunks):
            chunk_start = start + duration * index / len(chunks)
            chunk_end = start + duration * (index + 1) / len(chunks)
            if chunk_end - chunk_start < 0.08:
                continue
            replacements.append(
                replace(
                    block,
                    start=seconds_to_timecode(chunk_start),
                    end=seconds_to_timecode(chunk_end),
                    start_seconds=chunk_start,
                    end_seconds=chunk_end,
                    text=wrap_two_lines(chunk),
                )
            )
        source_numbers.append(block.number)
    unresolved = not replacements
    if unresolved:
        template = run[0]
        replacements = [replace(template, text="…")]
    return replacements, {
        "placeholder_blocks": [block.number for block in run],
        "start": seconds_to_timecode(run_start),
        "end": seconds_to_timecode(run_end),
        "source_blocks": source_numbers,
        "replacement_count": len(replacements),
        "unresolved_to_ellipsis": unresolved,
    }


def qa(blocks: list[SubtitleBlock]) -> dict[str, Any]:
    invalid = [block.number for block in blocks if block.end_seconds <= block.start_seconds]
    out_of_order = [
        right.number
        for left, right in zip(blocks, blocks[1:])
        if right.start_seconds < left.start_seconds
    ]
    japanese = [block.number for block in blocks if JAPANESE_RE.search(block.text)]
    placeholders = [block.number for block in blocks if PLACEHOLDER in block.text]
    empty = [block.number for block in blocks if not block.text.strip()]
    layout = [
        block.number
        for block in blocks
        if len(block.lines) > 2 or max(len(line) for line in block.lines) > 42
    ]
    return {
        "pass": not (invalid or out_of_order or japanese or placeholders or empty or layout),
        "block_count": len(blocks),
        "invalid_duration_blocks": invalid,
        "out_of_order_blocks": out_of_order,
        "japanese_blocks": japanese,
        "placeholder_blocks": placeholders,
        "empty_blocks": empty,
        "layout_violation_blocks": layout,
        "first_start": blocks[0].start if blocks else None,
        "last_end": blocks[-1].end if blocks else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace refusal placeholders with time-matched Korean source cues."
    )
    parser.add_argument("--title", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--source-ko", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    input_hash = sha256(args.input)
    source_hash = sha256(args.source_ko)
    if input_hash != args.expected_sha256.casefold():
        raise ValueError(
            f"Input SHA-256 drifted: expected {args.expected_sha256}, found {input_hash}"
        )
    if source_hash != args.expected_source_sha256.casefold():
        raise ValueError(
            "Source SHA-256 drifted: "
            f"expected {args.expected_source_sha256}, found {source_hash}"
        )

    blocks, _, _ = parse_srt(args.input)
    source, _, _ = parse_srt(args.source_ko)
    runs = placeholder_runs(blocks)
    if not runs:
        raise ValueError(f"No exact refusal placeholders found in {args.input}")

    run_by_first = {run[0].number: run for run in runs}
    placeholder_numbers = {block.number for run in runs for block in run}
    candidate: list[SubtitleBlock] = []
    ledger: list[dict[str, Any]] = []
    for block in blocks:
        run = run_by_first.get(block.number)
        if run is not None:
            replacements, row = source_replacements(run, source)
            candidate.extend(replacements)
            ledger.append(row)
        elif block.number not in placeholder_numbers:
            candidate.append(block)

    candidate, cleanup_ledger = apply_title_cleanup(args.title.upper(), candidate)
    candidate, layout_repair_count = normalize_layout(candidate)
    candidate.sort(key=lambda block: (block.start_seconds, block.end_seconds, block.number))
    candidate = [replace(block, number=index) for index, block in enumerate(candidate, 1)]
    result = qa(candidate)
    if not result["pass"]:
        raise ValueError(f"Candidate QA failed: {json.dumps(result, ensure_ascii=False)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_srt(candidate), encoding="utf-8-sig", newline="\n")
    report = {
        "schema_name": "translation-forensics/source-backed-placeholder-repair",
        "schema_version": 1,
        "title": args.title,
        "input": str(args.input),
        "input_sha256": input_hash,
        "source_ko": str(args.source_ko),
        "source_ko_sha256": source_hash,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "source_block_count": len(source),
        "placeholder_count_before": len(placeholder_numbers),
        "placeholder_run_count": len(runs),
        "source_cues_inserted": sum(row["replacement_count"] for row in ledger),
        "unresolved_run_count": sum(row["unresolved_to_ellipsis"] for row in ledger),
        "layout_repair_count": layout_repair_count,
        "title_cleanup_ledger": cleanup_ledger,
        "ledger": ledger,
        **result,
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
