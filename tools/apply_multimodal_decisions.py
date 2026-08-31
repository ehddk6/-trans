from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from translation_forensics.srt import SubtitleBlock, parse_srt, render_srt, timecode_to_seconds


JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_decisions(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_decisions(
    blocks: list[SubtitleBlock],
    decisions: list[dict[str, Any]],
) -> tuple[list[SubtitleBlock], list[dict[str, Any]], list[int]]:
    by_number = {int(row["block"]): row for row in decisions}
    if len(by_number) != len(decisions):
        raise ValueError("Duplicate decision block numbers")

    source_numbers = {block.number for block in blocks}
    missing = sorted(set(by_number) - source_numbers)
    if missing:
        raise ValueError(f"Decision blocks not found in source: {missing}")

    output: list[SubtitleBlock] = []
    output_source_numbers: list[int] = []
    ledger: list[dict[str, Any]] = []
    for block in blocks:
        decision = by_number.get(block.number)
        if decision is None:
            output.append(block)
            output_source_numbers.append(block.number)
            continue

        expected = str(decision.get("expected_before", ""))
        if block.text != expected:
            raise ValueError(
                f"Block {block.number} drifted: expected {expected!r}, found {block.text!r}"
            )
        action = str(decision["action"])
        if action == "drop":
            ledger.append(
                {
                    **decision,
                    "source_start": block.start,
                    "source_end": block.end,
                    "applied": True,
                }
            )
            continue
        if action != "replace":
            raise ValueError(f"Unsupported action for block {block.number}: {action}")

        after = str(decision["after"])
        if not after.strip():
            raise ValueError(f"Replacement for block {block.number} is empty")
        end = str(decision.get("end") or block.end)
        end_seconds = timecode_to_seconds(end)
        if end_seconds <= block.start_seconds:
            raise ValueError(f"Invalid end time for block {block.number}: {end}")
        output.append(replace(block, text=after, end=end, end_seconds=end_seconds))
        output_source_numbers.append(block.number)
        ledger.append(
            {
                **decision,
                "source_start": block.start,
                "source_end": block.end,
                "applied": True,
            }
        )

    renumbered = [replace(block, number=index) for index, block in enumerate(output, 1)]
    return renumbered, ledger, output_source_numbers


def qa_report(
    source: list[SubtitleBlock],
    candidate: list[SubtitleBlock],
    ledger: list[dict[str, Any]],
    output_source_numbers: list[int],
) -> dict[str, Any]:
    japanese_blocks = [block.number for block in candidate if JAPANESE_RE.search(block.text)]
    empty_blocks = [block.number for block in candidate if not block.text.strip()]
    source_invalid_durations = {
        block.number for block in source if block.end_seconds <= block.start_seconds
    }
    candidate_invalid_source_numbers = {
        source_number
        for block, source_number in zip(candidate, output_source_numbers)
        if block.end_seconds <= block.start_seconds
    }
    new_invalid_durations = sorted(candidate_invalid_source_numbers - source_invalid_durations)
    source_out_of_order_count = sum(
        right.start_seconds < left.start_seconds for left, right in zip(source, source[1:])
    )
    candidate_out_of_order = [
        right.number
        for left, right in zip(candidate, candidate[1:])
        if right.start_seconds < left.start_seconds
    ]
    new_out_of_order_count = max(0, len(candidate_out_of_order) - source_out_of_order_count)
    numbering_ok = [block.number for block in candidate] == list(range(1, len(candidate) + 1))
    action_counts = {
        action: sum(row["action"] == action for row in ledger)
        for action in sorted({str(row["action"]) for row in ledger})
    }
    passed = not (
        japanese_blocks
        or empty_blocks
        or new_invalid_durations
        or new_out_of_order_count
    ) and numbering_ok
    return {
        "pass": passed,
        "source_block_count": len(source),
        "output_block_count": len(candidate),
        "removed_block_count": len(source) - len(candidate),
        "decision_count": len(ledger),
        "action_counts": action_counts,
        "renumbered_sequentially": numbering_ok,
        "japanese_block_count": len(japanese_blocks),
        "japanese_blocks": japanese_blocks,
        "empty_block_count": len(empty_blocks),
        "empty_blocks": empty_blocks,
        "legacy_invalid_duration_source_blocks": sorted(
            candidate_invalid_source_numbers & source_invalid_durations
        ),
        "new_invalid_duration_source_blocks": new_invalid_durations,
        "legacy_out_of_order_count": source_out_of_order_count,
        "candidate_out_of_order_blocks": candidate_out_of_order,
        "new_out_of_order_count": new_out_of_order_count,
        "retimed_source_blocks": [int(row["block"]) for row in ledger if "end" in row],
        "removed_source_blocks": [int(row["block"]) for row in ledger if row["action"] == "drop"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply evidence-backed multimodal subtitle decisions.")
    parser.add_argument("--base-srt", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output-srt", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--qa-report", required=True, type=Path)
    args = parser.parse_args()

    package = load_decisions(args.decisions)
    expected_sha = str(package["base_sha256"])
    actual_sha = sha256(args.base_srt)
    if actual_sha != expected_sha:
        raise ValueError(f"Base SRT SHA-256 drifted: expected {expected_sha}, found {actual_sha}")

    source, source_encoding, source_newline = parse_srt(args.base_srt)
    candidate, ledger, output_source_numbers = apply_decisions(source, list(package["decisions"]))
    report = qa_report(source, candidate, ledger, output_source_numbers)
    if not report["pass"]:
        raise ValueError(f"QA failed: {json.dumps(report, ensure_ascii=False)}")

    args.output_srt.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.qa_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_srt.write_text(render_srt(candidate), encoding="utf-8-sig", newline="\n")
    args.ledger.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ledger),
        encoding="utf-8",
        newline="\n",
    )
    report.update(
        {
            "base_srt": str(args.base_srt),
            "base_sha256": actual_sha,
            "base_encoding": source_encoding,
            "base_newline": source_newline,
            "output_srt": str(args.output_srt),
            "output_sha256": sha256(args.output_srt),
            "decisions": str(args.decisions),
            "ledger": str(args.ledger),
        }
    )
    args.qa_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
