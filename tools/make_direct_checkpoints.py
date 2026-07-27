"""Turn model-authored TSV decisions into 100-block JSONL checkpoint files."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uff66-\uff9f]")


def main(tsv_path: str, destination: str) -> None:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    seen: set[int] = set()
    for raw in Path(tsv_path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        number, korean = raw.split("\t", 1)
        n, ko = int(number), korean.strip()
        if n in seen or not ko or JAPANESE.search(ko):
            raise ValueError(f"Invalid direct decision: {n}")
        seen.add(n)
        start = ((n - 1) // 100) * 100 + 1
        grouped[start].append({
            "block_number": n,
            "korean": ko,
            "method": "gpt-5.6-direct",
            "status": "translated",
            "confidence": "source-only",
            "evidence": ["japanese_srt"],
        })
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    for start, rows in grouped.items():
        rows.sort(key=lambda row: int(row["block_number"]))
        expected = list(range(start, min(start + 100, 1956)))
        actual = [int(row["block_number"]) for row in rows]
        if actual != expected:
            raise ValueError(f"Checkpoint {start:04d} is incomplete")
        out = target / f"JUQ-778.gpt56-direct-v1.checkpoint-{start:04d}.jsonl"
        out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: make_direct_checkpoints.py ENTRIES_TSV DESTINATION")
    main(sys.argv[1], sys.argv[2])
