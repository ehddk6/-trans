#!/usr/bin/env bash
set -euo pipefail
python "$(dirname "$0")/subtitle_forensics.py" --root /mnt/data --output /mnt/data/subtitle_forensics_v1
