from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def reliable_segments(row: dict[str, Any], *, min_logprob: float, max_compression: float) -> list[dict[str, Any]]:
    return [
        {
            "start_seconds": segment["start_seconds"],
            "end_seconds": segment["end_seconds"],
            "text": segment["text"],
            "avg_logprob": segment["avg_logprob"],
            "compression_ratio": segment["compression_ratio"],
        }
        for segment in row.get("asr_segments", [])
        if not segment.get("suspected_hallucination")
        and segment.get("avg_logprob") is not None
        and float(segment["avg_logprob"]) >= min_logprob
        and segment.get("compression_ratio") is not None
        and float(segment["compression_ratio"]) <= max_compression
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a compact human-review report from source-MP3 ASR evidence."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-logprob", type=float, default=-1.2)
    parser.add_argument("--max-compression", type=float, default=2.4)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    title_rows = []
    for item in manifest["items"]:
        evidence_path = Path(item["output_dir"]) / "block-acoustic-evidence.jsonl"
        evidence = load_jsonl(evidence_path)
        candidates = []
        for row in evidence:
            segments = reliable_segments(
                row,
                min_logprob=args.min_logprob,
                max_compression=args.max_compression,
            )
            if not segments:
                continue
            candidates.append(
                {
                    "block": row["block"],
                    "start": row["start"],
                    "end": row["end"],
                    "current_text": row["current_text"],
                    "asr_segments": segments,
                    "asr_text": " ".join(segment["text"] for segment in segments),
                }
            )
        title_rows.append(
            {
                "title": item["title"],
                "source_audio": item["audio"],
                "selected_block_count": len(evidence),
                "reliable_candidate_count": len(candidates),
                "candidates": candidates,
            }
        )
    report = {
        "schema_name": "translation-forensics/source-mp3-audio-review-brief",
        "schema_version": "1",
        "manifest": str(manifest_path),
        "min_logprob": args.min_logprob,
        "max_compression": args.max_compression,
        "title_count": len(title_rows),
        "reliable_candidate_count": sum(row["reliable_candidate_count"] for row in title_rows),
        "titles": title_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "title_count": report["title_count"],
                "reliable_candidate_count": report["reliable_candidate_count"],
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
