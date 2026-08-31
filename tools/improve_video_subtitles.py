from __future__ import annotations

import argparse
import json
from pathlib import Path

from translation_forensics.video_subtitle_improvement import run_improvement


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a verified, non-destructive subtitle improvement set.")
    parser.add_argument("--target-root", type=Path, required=True, help="Current playback subtitle directory")
    parser.add_argument("--source-root", type=Path, required=True, help="Videos directory containing JA/KO pairs")
    parser.add_argument("--output-root", type=Path, required=True, help="New directory; must not already exist")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/video-subtitle-overrides.json"),
        help="Hash-guarded improvement configuration",
    )
    parser.add_argument(
        "--resume-incomplete",
        action="store_true",
        help="Restore and reuse an incomplete output directory that has no QA report",
    )
    parser.add_argument(
        "--refresh-generated",
        action="store_true",
        help="Refresh a completed output only after its prior QA report proves ownership",
    )
    args = parser.parse_args()
    report = run_improvement(
        target_root=args.target_root,
        source_root=args.source_root,
        output_root=args.output_root,
        config_path=args.config,
        resume_incomplete=args.resume_incomplete,
        refresh_generated=args.refresh_generated,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
