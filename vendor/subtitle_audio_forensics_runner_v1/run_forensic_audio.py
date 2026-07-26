#!/usr/bin/env python3
"""Generic local audio-evidence runner for Subtitle Forensics projects."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".m2ts"}


def run(cmd: list[str]) -> None:
    print("\n> " + " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def find_candidates(root: Path, kind: str) -> list[Path]:
    files = [p for p in root.rglob("*") if p.is_file()]
    if kind == "queue":
        return sorted(
            [p for p in files if p.suffix.lower() == ".csv" and "review-queue" in p.name.lower()],
            key=lambda p: ("audio-pending" not in p.name.lower(), len(p.parts), p.name.lower()),
        )
    if kind == "audio":
        return sorted(
            [p for p in files if p.suffix.lower() in AUDIO_EXTS and "clips" not in {x.lower() for x in p.parts}],
            key=lambda p: (-p.stat().st_size, len(p.parts), p.name.lower()),
        )
    raise ValueError(kind)


def choose(candidates: list[Path], explicit: Path | None, label: str) -> Path:
    if explicit:
        p = explicit.expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"{label} 파일이 없습니다: {p}")
        return p
    if not candidates:
        raise SystemExit(f"{label} 파일을 찾지 못했습니다. --{'queue' if label == 'review queue' else 'audio'}로 지정하세요.")
    if len(candidates) == 1:
        return candidates[0]
    print(f"{label} 후보가 여러 개입니다:")
    for i, p in enumerate(candidates[:20], 1):
        print(f"  {i}. {p}")
    raise SystemExit(f"후보가 여러 개라 자동 선택하지 않았습니다. --{'queue' if label == 'review queue' else 'audio'}로 정확한 경로를 지정하세요.")


def infer_title(queue: Path, audio: Path) -> str:
    name = queue.name
    lower = name.lower()
    idx = lower.find(".review-queue")
    if idx > 0:
        return name[:idx].rstrip(". _-")
    title = audio.stem.rstrip(". _-")
    return title or "subtitle-project"


def validate_queue(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"block_number", "timecode", "review_band"}
        missing = required - fields
        if missing:
            raise SystemExit(f"review queue 필수 열 누락: {', '.join(sorted(missing))}")
        if next(reader, None) is None:
            raise SystemExit("review queue가 비어 있습니다.")


def detect_device(force_cpu: bool) -> tuple[str, str]:
    if not force_cpu and shutil.which("nvidia-smi"):
        return "cuda", "float16"
    return "cpu", "int8"


def zip_tree(source: Path, destination: Path, include_clips: bool) -> None:
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(source.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(source)
            if not include_clips and rel.parts and rel.parts[0].lower() == "clips":
                continue
            zf.write(p, rel.as_posix())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=Path, default=Path.cwd())
    ap.add_argument("--queue", type=Path)
    ap.add_argument("--audio", type=Path)
    ap.add_argument("--bands", default="P1,P2")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--include-clips", action="store_true", help="결과 ZIP에 WAV 조각도 포함")
    ap.add_argument("--padding", type=float, default=2.5)
    ap.add_argument("--merge-gap", type=float, default=1.5)
    ap.add_argument("--max-scene", type=float, default=45.0)
    args = ap.parse_args()

    root = args.project_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"작품 폴더가 없습니다: {root}")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg를 찾지 못했습니다. PATH에 ffmpeg를 추가한 뒤 다시 실행하세요.")

    queue = choose(find_candidates(root, "queue"), args.queue, "review queue")
    audio = choose(find_candidates(root, "audio"), args.audio, "audio")
    validate_queue(queue)
    title = infer_title(queue, audio)

    script_dir = Path(__file__).resolve().parent
    work = root / f"{title}.work_audio"
    work.mkdir(parents=True, exist_ok=True)
    device, compute = detect_device(args.cpu)

    print(json.dumps({
        "project_dir": str(root),
        "title": title,
        "queue": str(queue),
        "audio": str(audio),
        "bands": args.bands,
        "model": args.model,
        "device": device,
        "compute_type": compute,
        "work_dir": str(work),
    }, ensure_ascii=False, indent=2), flush=True)

    started = time.time()
    scenes = work / "review-scenes.csv"
    if not scenes.exists():
        run([
            sys.executable, str(script_dir / "prepare_review_pack.py"),
            "--queue", str(queue), "--audio", str(audio), "--out", str(work),
            "--bands", args.bands, "--padding", str(args.padding),
            "--merge-gap", str(args.merge_gap), "--max-scene", str(args.max_scene),
        ])
    else:
        print("기존 review-scenes.csv가 있어 음원 추출을 건너뜁니다.")

    candidates = work / "asr-candidates.csv"
    run([
        sys.executable, str(script_dir / "run_asr_adaptive.py"),
        "--scenes", str(scenes), "--model", args.model,
        "--device", device, "--compute-type", compute,
        "--prompt", str(script_dir / "asr_prompt_ja.txt"), "--out", str(candidates),
    ])

    summary = {
        "schema_version": "subtitle-audio-forensics/1",
        "status": "audio-asr-crosschecked-input",
        "title": title,
        "project_dir": str(root),
        "queue": str(queue),
        "audio": str(audio),
        "bands": [x.strip() for x in args.bands.split(",") if x.strip()],
        "model": args.model,
        "device": device,
        "compute_type": compute,
        "elapsed_seconds": round(time.time() - started, 1),
        "direct_human_listening": False,
        "note": "같은 Whisper 모델의 여러 패스는 독립 증거가 아니라 교차검토 후보입니다.",
    }
    (work / "run-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    dest = root / f"{title}.work_audio.zip"
    zip_tree(work, dest, args.include_clips)
    print(f"\n완료: {dest}")
    if not args.include_clips:
        print("결과 ZIP은 분석 CSV 중심이며 WAV 조각은 제외했습니다. WAV가 필요하면 --include-clips를 사용하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
