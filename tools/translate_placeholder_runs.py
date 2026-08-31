from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from subtitle_pipeline.text import wrap_two_lines
from translation_forensics.codex_exec_provider import CodexExecProvider
from translation_forensics.integrated_translation import translate_units_with_terra
from translation_forensics.srt import SubtitleBlock, parse_srt, render_srt

from recover_placeholder_runs import (
    overlap_seconds,
    placeholder_runs,
    qa_report,
    recover,
    sha256,
)


_NONLEXICAL_KOREAN_RE = re.compile(
    r"^[\s,.!?…·~〜\-]*(?:하아+|아+|으+|음+|우+|하+|어+|오+|응+|흐+|후+)"
    r"(?:[\s,.!?…·~〜\-]+(?:하아+|아+|으+|음+|우+|하+|어+|오+|응+|흐+|후+))*"
    r"[\s,.!?…·~〜\-]*$"
)


class _FirstRowDuplicateProvider:
    """Keep the first model row when a structured batch repeats one unit ID.

    The underlying receipt retains an explicit warning.  Missing unit IDs are
    still rejected by ``translate_units_with_terra``; this wrapper only handles
    the observed case where an extra continuation row reused the final ID.
    """

    def __init__(self, provider: CodexExecProvider) -> None:
        self._provider = provider

    def run_structured(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        response, receipt = self._provider.run_structured(**kwargs)
        rows = response.get("translations")
        if not isinstance(rows, list):
            return response, receipt
        seen: set[str] = set()
        kept: list[Any] = []
        duplicates: list[str] = []
        for row in rows:
            unit_id = str(row.get("unit_id") or "") if isinstance(row, dict) else ""
            if unit_id and unit_id in seen:
                duplicates.append(unit_id)
                continue
            if unit_id:
                seen.add(unit_id)
            kept.append(row)
        if not duplicates:
            return response, receipt
        warning = "duplicate_unit_ids_after_first_dropped:" + ",".join(
            sorted(set(duplicates))
        )
        return {**response, "translations": kept}, {
            **receipt,
            "response_contract_warnings": [
                *receipt.get("response_contract_warnings", []),
                warning,
            ],
        }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _viewer_text(value: object) -> str:
    flattened = " ".join(str(value or "").split())
    if not flattened:
        raise ValueError("Model returned an empty viewer translation")
    lines = wrap_two_lines(flattened, 22, 42)
    if len(lines) <= 2 and max(len(line) for line in lines) <= 42:
        return "\n".join(lines)
    if len(flattened) > 84:
        raise ValueError("Viewer translation cannot fit within two 42-character lines")
    preferred = [
        index
        for index, character in enumerate(flattened, 1)
        if character in " ,.?!…"
    ]
    fitting = [
        index
        for index in range(1, len(flattened))
        if len(flattened[:index].rstrip()) <= 42
        and len(flattened[index:].lstrip()) <= 42
    ]
    preferred_fitting = [index for index in preferred if index in fitting]
    pool = preferred_fitting or fitting
    if not pool:
        raise ValueError("Viewer translation has no valid two-line split")
    split_at = min(
        pool,
        key=lambda index: (
            max(
                len(flattened[:index].rstrip()),
                len(flattened[index:].lstrip()),
            ),
            abs(len(flattened[:index].rstrip()) - len(flattened[index:].lstrip())),
            index,
        ),
    )
    return flattened[:split_at].rstrip() + "\n" + flattened[split_at:].lstrip()


def _selected_source_blocks(
    base: list[SubtitleBlock], source: list[SubtitleBlock], marker: str
) -> tuple[list[SubtitleBlock], list[dict[str, Any]]]:
    selected: dict[int, SubtitleBlock] = {}
    run_rows: list[dict[str, Any]] = []
    for run_index, run in enumerate(placeholder_runs(base, marker), 1):
        start_seconds = run[0].start_seconds
        end_seconds = run[-1].end_seconds
        source_rows = [
            block
            for block in source
            if overlap_seconds(start_seconds, end_seconds, block) > 0
            and (
                start_seconds <= (block.start_seconds + block.end_seconds) / 2 <= end_seconds
                or overlap_seconds(start_seconds, end_seconds, block) / max(block.duration, 0.001)
                >= 0.5
            )
        ]
        for block in source_rows:
            selected[block.number] = block
        run_rows.append(
            {
                "run": run_index,
                "start": run[0].start,
                "end": run[-1].end,
                "placeholder_blocks": len(run),
                "selected_japanese_blocks": [block.number for block in source_rows],
            }
        )
    return sorted(selected.values(), key=lambda block: block.number), run_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate only approved-Japanese cues overlapping placeholder runs."
    )
    parser.add_argument("--title", required=True)
    parser.add_argument("--base-srt", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--source-ja", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--marker", default="[안전상 번역 불가]")
    parser.add_argument(
        "--overrides",
        type=Path,
        help="Optional JSON object mapping approved-Japanese block numbers to Korean text or null.",
    )
    parser.add_argument(
        "--base-overrides",
        type=Path,
        help="Optional JSON object mapping retained base-SRT block numbers to Korean text or null.",
    )
    parser.add_argument(
        "--run-overrides",
        type=Path,
        help="Optional JSON object mapping 1-based placeholder run numbers to fallback Korean text.",
    )
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--collapse-nonlexical-runs",
        action="store_true",
        help="Keep only the first cue in each contiguous run of Korean vocalisations.",
    )
    args = parser.parse_args()

    base_digest = sha256(args.base_srt)
    if base_digest != args.base_sha256.casefold():
        raise ValueError(
            f"Base SRT SHA-256 drifted: expected {args.base_sha256}, found {base_digest}"
        )
    base, base_encoding, base_newline = parse_srt(args.base_srt)
    source, source_encoding, source_newline = parse_srt(args.source_ja)
    selected, run_rows = _selected_source_blocks(base, source, args.marker)
    if not selected:
        raise ValueError("No approved Japanese cues overlap the placeholder runs")

    units = [
        {
            "unit_id": f"ja-{block.number:06d}",
            "start": block.start_seconds,
            "end": block.end_seconds,
            "source_japanese": block.text.replace("\n", " ").strip(),
            # The approved SRT is authoritative for timing and Japanese evidence,
            # but individual cues can still be clipped/noisy ASR fragments.  Mark
            # them suspect so the translator may explicitly record a bounded
            # context recovery instead of failing the whole targeted run.
            "quality_status": "suspect",
            "asr_warnings": [],
            "evidence_ids": [f"approved-ja-block-{block.number}"],
            "source_evidence": {
                "kind": "approved_japanese_srt",
                "block_number": block.number,
                "source_sha256": sha256(args.source_ja),
            },
        }
        for block in selected
    ]
    provider = CodexExecProvider(
        cache_dir=args.cache_dir,
        timeout_seconds=args.timeout,
    )
    preflight = provider.preflight()
    if preflight.get("status") != "pass":
        raise RuntimeError(
            "Codex provider preflight failed: " + ", ".join(preflight.get("errors", []))
        )
    decisions, receipts = translate_units_with_terra(
        _FirstRowDuplicateProvider(provider),
        title_id=args.title.upper(),
        units=units,
        resume=True,
        batch_size=args.batch_size,
    )
    decisions_by_id = {str(row["unit_id"]): row for row in decisions}
    translated_source = [
        SubtitleBlock(
            number=block.number,
            start=block.start,
            end=block.end,
            text=_viewer_text(
                decisions_by_id[f"ja-{block.number:06d}"]["viewer_natural_korean"]
            ),
            start_seconds=block.start_seconds,
            end_seconds=block.end_seconds,
        )
        for block in selected
    ]
    overrides: dict[int, str | None] = {}
    if args.overrides:
        raw_overrides = json.loads(args.overrides.read_text(encoding="utf-8"))
        overrides = {
            int(key): None if value is None else str(value).strip()
            for key, value in raw_overrides.items()
        }
        unknown = sorted(set(overrides) - {block.number for block in selected})
        if unknown:
            raise ValueError(f"Override block numbers are outside the selected source: {unknown}")
    override_rows: list[dict[str, Any]] = []
    refined_source: list[SubtitleBlock] = []
    for block in translated_source:
        if block.number not in overrides:
            refined_source.append(block)
            continue
        replacement = overrides[block.number]
        override_rows.append(
            {
                "block": block.number,
                "start": block.start,
                "end": block.end,
                "model_text": block.text,
                "replacement_text": replacement,
                "action": "drop" if replacement is None else "replace",
            }
        )
        if replacement is not None:
            if not replacement:
                raise ValueError(f"Override text is empty for block {block.number}")
            refined_source.append(replace(block, text=_viewer_text(replacement)))
    translated_source = refined_source
    if args.collapse_nonlexical_runs:
        collapsed_source: list[SubtitleBlock] = []
        previous_source_number: int | None = None
        in_nonlexical_run = False
        for block in translated_source:
            is_nonlexical = bool(
                _NONLEXICAL_KOREAN_RE.fullmatch(block.text.replace("\n", " "))
            )
            contiguous = (
                previous_source_number is not None
                and block.number == previous_source_number + 1
            )
            if is_nonlexical and contiguous and in_nonlexical_run:
                override_rows.append(
                    {
                        "block": block.number,
                        "start": block.start,
                        "end": block.end,
                        "model_text": block.text,
                        "replacement_text": None,
                        "action": "drop_repeated_nonlexical",
                    }
                )
            else:
                collapsed_source.append(block)
            in_nonlexical_run = is_nonlexical
            previous_source_number = block.number
        translated_source = collapsed_source
    run_overrides: dict[int, str] = {}
    if args.run_overrides:
        raw_run_overrides = json.loads(args.run_overrides.read_text(encoding="utf-8"))
        run_overrides = {
            int(key): str(value).strip() for key, value in raw_run_overrides.items()
        }
        if any(not value for value in run_overrides.values()):
            raise ValueError("Run override text must be non-empty")
    candidate, recovery_ledger = recover(
        base,
        translated_source,
        marker=args.marker,
        unresolved_overrides=run_overrides,
    )
    base_override_rows: list[dict[str, Any]] = []
    if args.base_overrides:
        raw_base_overrides = json.loads(args.base_overrides.read_text(encoding="utf-8"))
        base_overrides = {
            int(key): None if value is None else str(value).strip()
            for key, value in raw_base_overrides.items()
        }
        base_by_number = {block.number: block for block in base}
        unknown = sorted(set(base_overrides) - set(base_by_number))
        if unknown:
            raise ValueError(f"Base override block numbers are absent: {unknown}")
        candidate_by_time = {
            (block.start, block.end): block for block in candidate
        }
        remove_times: set[tuple[str, str]] = set()
        replacements_by_time: dict[tuple[str, str], str] = {}
        for number, replacement in base_overrides.items():
            original = base_by_number[number]
            key = (original.start, original.end)
            if key not in candidate_by_time:
                raise ValueError(f"Base override block {number} is absent from the candidate")
            base_override_rows.append(
                {
                    "block": number,
                    "start": original.start,
                    "end": original.end,
                    "model_text": original.text,
                    "replacement_text": replacement,
                    "action": "drop_base" if replacement is None else "replace_base",
                }
            )
            if replacement is None:
                remove_times.add(key)
            else:
                if not replacement:
                    raise ValueError(f"Base override text is empty for block {number}")
                replacements_by_time[key] = _viewer_text(replacement)
        candidate = [
            replace(block, text=replacements_by_time.get((block.start, block.end), block.text))
            for block in candidate
            if (block.start, block.end) not in remove_times
        ]
        candidate = [
            replace(block, number=index) for index, block in enumerate(candidate, 1)
        ]
    candidate = [
        replace(block, text=_viewer_text(block.text))
        if len(block.lines) > 2 or max(len(line) for line in block.lines) > 42
        else block
        for block in candidate
    ]
    report = qa_report(base, candidate, recovery_ledger, marker=args.marker)
    if not report["pass"]:
        raise ValueError(f"QA failed: {json.dumps(report, ensure_ascii=False)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_ko_path = args.output_dir / f"{args.title.upper()}.targeted-source-ko.srt"
    candidate_path = args.output_dir / f"{args.title.upper()}.srt"
    decisions_path = args.output_dir / "translation-decisions.jsonl"
    receipts_path = args.output_dir / "model-call-receipts.jsonl"
    runs_path = args.output_dir / "placeholder-runs.jsonl"
    recovery_path = args.output_dir / "recovery-ledger.jsonl"
    overrides_path = args.output_dir / "override-ledger.jsonl"
    qa_path = args.output_dir / "qa-report.json"

    source_ko_path.write_text(
        render_srt(translated_source), encoding="utf-8-sig", newline="\n"
    )
    candidate_path.write_text(render_srt(candidate), encoding="utf-8-sig", newline="\n")
    _write_jsonl(decisions_path, decisions)
    _write_jsonl(receipts_path, receipts)
    _write_jsonl(runs_path, run_rows)
    _write_jsonl(recovery_path, recovery_ledger)
    _write_jsonl(overrides_path, [*override_rows, *base_override_rows])
    report.update(
        {
            "title": args.title.upper(),
            "base_srt": str(args.base_srt),
            "base_sha256": base_digest,
            "base_encoding": base_encoding,
            "base_newline": base_newline,
            "source_ja": str(args.source_ja),
            "source_ja_sha256": sha256(args.source_ja),
            "source_ja_encoding": source_encoding,
            "source_ja_newline": source_newline,
            "selected_japanese_block_count": len(selected),
            "translation_decision_count": len(decisions),
            "model_call_count": len(receipts),
            "override_count": len(override_rows) + len(base_override_rows),
            "dropped_override_count": sum(row["action"] == "drop" for row in override_rows),
            "base_override_count": len(base_override_rows),
            "dropped_base_override_count": sum(
                row["action"] == "drop_base" for row in base_override_rows
            ),
            "collapsed_nonlexical_count": sum(
                row["action"] == "drop_repeated_nonlexical" for row in override_rows
            ),
            "run_override_count": len(run_overrides),
            "output_srt": str(candidate_path),
            "output_sha256": sha256(candidate_path),
        }
    )
    qa_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
