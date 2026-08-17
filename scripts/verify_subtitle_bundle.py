"""Verify a generated Japanese subtitle artifact bundle without changing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from subtitle_pipeline.verification import verify_artifact_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-srt", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--review-windows", type=int, default=30)
    parser.add_argument("--require-source-match", action="store_true")
    parser.add_argument("--require-passed-gate", action="store_true")
    parser.add_argument(
        "--require-media-binding",
        action="store_true",
        help="Require source_media.json and verify it against --video.",
    )
    parser.add_argument("--expected-media-sha256", help="Known full media SHA-256; avoids re-hashing --video.")
    parser.add_argument("--expected-media-duration", type=float)
    parser.add_argument("--output-json", type=Path, help="Write UTF-8 JSON directly, avoiding shell encoding loss.")
    args = parser.parse_args()
    result = verify_artifact_bundle(
        args.output_dir,
        reference_srt=args.reference_srt,
        expected_video_path=args.video,
        expected_review_windows=args.review_windows,
        require_source_match=args.require_source_match,
        require_passed_gate=args.require_passed_gate,
        expected_media_sha256=args.expected_media_sha256,
        expected_media_duration=args.expected_media_duration,
        require_media_binding=args.require_media_binding,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
