"""Fail closed when an SRT is clearly a Korean translation, not Japanese source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from subtitle_pipeline.reference import reference_language_diagnostics
from subtitle_pipeline.srt import read_srt_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-srt", required=True, type=Path)
    args = parser.parse_args()
    try:
        rows = read_srt_rows(args.reference_srt)
    except (OSError, ValueError) as exc:
        print(json.dumps({"compatible": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1)
    diagnostics = reference_language_diagnostics(rows)
    print(json.dumps(diagnostics, ensure_ascii=False))
    raise SystemExit(0 if rows and diagnostics["compatible"] else 1)


if __name__ == "__main__":
    main()
