#!/usr/bin/env python3
"""Adaptive faster-whisper runner for Subtitle Forensics review scenes.

Fast path: two unbiased passes for every scene.
Escalation: no-VAD and prompted passes only when the first two disagree,
are empty, or look low-confidence. Results are checkpointed after every pass.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

FIELDS = [
    "scene_id", "model", "profile", "audio", "text",
    "avg_logprob", "no_speech_prob", "language_probability", "error",
]


def load_prompt(path: Path | None) -> str:
    return path.read_text(encoding="utf-8").strip() if path else ""


def mean(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else ""


def norm(text: str) -> str:
    return re.sub(r"[\s。、！？!?・…ー]+", "", text or "")


def similarity(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def read_existing(path: Path):
    if not path.exists():
        return [], set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    done = {(r.get("scene_id", ""), r.get("model", ""), r.get("profile", "")) for r in rows}
    return rows, done


def write_checkpoint(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def transcribe_one(model, scene, root: Path, model_name: str, profile: dict, fixed_prompt: str):
    rel = scene.get(profile["audio"], "")
    if not rel:
        return None
    audio = root / rel
    prompt_parts = []
    if fixed_prompt:
        prompt_parts.append(fixed_prompt)
    if scene.get("context_japanese"):
        prompt_parts.append(scene["context_japanese"])

    kwargs = {
        "language": "ja",
        "task": "transcribe",
        "beam_size": profile["beam"],
        "vad_filter": profile["vad"],
        "word_timestamps": False,
        "condition_on_previous_text": False,
    }
    if profile["use_prompt"] and prompt_parts:
        kwargs["initial_prompt"] = "。".join(prompt_parts)

    started = time.perf_counter()
    try:
        segments, info = model.transcribe(str(audio), **kwargs)
        segs = list(segments)
        text = "".join(s.text.strip() for s in segs).strip()
        row = {
            "scene_id": scene["scene_id"],
            "model": model_name,
            "profile": profile["name"],
            "audio": rel,
            "text": text,
            "avg_logprob": mean([getattr(s, "avg_logprob", None) for s in segs]),
            "no_speech_prob": mean([getattr(s, "no_speech_prob", None) for s in segs]),
            "language_probability": getattr(info, "language_probability", ""),
            "error": "",
        }
    except Exception as exc:
        row = {
            "scene_id": scene["scene_id"], "model": model_name,
            "profile": profile["name"], "audio": rel, "text": "",
            "avg_logprob": "", "no_speech_prob": "",
            "language_probability": "", "error": repr(exc),
        }
    elapsed = time.perf_counter() - started
    return row, elapsed


def as_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def needs_escalation(base_rows: list[dict], threshold: float) -> tuple[bool, str]:
    by_profile = {r["profile"]: r for r in base_rows}
    a = by_profile.get("original_unbiased", {})
    b = by_profile.get("dialogue_unbiased", {})
    ta, tb = a.get("text", ""), b.get("text", "")
    reasons = []
    if not ta or not tb:
        reasons.append("empty")
    sim = similarity(ta, tb)
    if sim < threshold:
        reasons.append(f"disagree:{sim:.2f}")
    for r in (a, b):
        lp = as_float(r.get("avg_logprob"))
        nsp = as_float(r.get("no_speech_prob"))
        if lp is not None and lp < -1.0:
            reasons.append("low_logprob")
        if nsp is not None and nsp > 0.60:
            reasons.append("high_no_speech")
        if r.get("error"):
            reasons.append("error")
    return bool(reasons), ",".join(sorted(set(reasons))) or "stable"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, type=Path)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--prompt", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--disagreement-threshold", type=float, default=0.82)
    ap.add_argument("--max-scenes", type=int, default=0)
    args = ap.parse_args()

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        print("faster-whisper가 필요합니다: python -m pip install faster-whisper", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    with args.scenes.open("r", encoding="utf-8-sig", newline="") as f:
        scenes = list(csv.DictReader(f))
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]

    root = args.scenes.parent
    fixed_prompt = load_prompt(args.prompt)
    rows, done = read_existing(args.out)

    print(
        f"Loading {args.model} on {args.device} / {args.compute_type} ...",
        flush=True,
    )
    started_load = time.perf_counter()
    try:
        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
        )
    except Exception as exc:
        print(f"모델 로드 실패: {exc!r}", file=sys.stderr, flush=True)
        return 3
    print(f"Model ready in {time.perf_counter() - started_load:.1f}s", flush=True)

    base_profiles = [
        {"name": "original_unbiased", "audio": "original_audio", "vad": True, "use_prompt": False, "beam": 3},
        {"name": "dialogue_unbiased", "audio": "dialogue_audio", "vad": True, "use_prompt": False, "beam": 3},
    ]
    extra_profiles = [
        {"name": "original_no_vad", "audio": "original_audio", "vad": False, "use_prompt": False, "beam": 3},
        {"name": "original_prompted", "audio": "original_audio", "vad": True, "use_prompt": True, "beam": 3},
    ]

    total_started = time.perf_counter()
    for idx, scene in enumerate(scenes, 1):
        sid = scene["scene_id"]
        scene_rows = [r for r in rows if r.get("scene_id") == sid and r.get("model") == args.model]
        print(f"\n[{idx}/{len(scenes)}] {sid}", flush=True)

        for profile in base_profiles:
            key = (sid, args.model, profile["name"])
            if key in done:
                print(f"  - {profile['name']}: resume/skip", flush=True)
                continue
            result = transcribe_one(model, scene, root, args.model, profile, fixed_prompt)
            if result is None:
                continue
            row, elapsed = result
            rows.append(row); done.add(key); scene_rows.append(row)
            write_checkpoint(args.out, rows)
            print(f"  - {profile['name']}: {elapsed:.1f}s | {row['text'][:100] or '[empty]'}", flush=True)

        current_base = [r for r in rows if r.get("scene_id") == sid and r.get("model") == args.model and r.get("profile") in {"original_unbiased", "dialogue_unbiased"}]
        escalate, reason = needs_escalation(current_base, args.disagreement_threshold)
        print(f"  - decision: {'ESCALATE' if escalate else 'stable'} ({reason})", flush=True)

        if escalate:
            for profile in extra_profiles:
                key = (sid, args.model, profile["name"])
                if key in done:
                    print(f"  - {profile['name']}: resume/skip", flush=True)
                    continue
                result = transcribe_one(model, scene, root, args.model, profile, fixed_prompt)
                if result is None:
                    continue
                row, elapsed = result
                rows.append(row); done.add(key)
                write_checkpoint(args.out, rows)
                print(f"  - {profile['name']}: {elapsed:.1f}s | {row['text'][:100] or '[empty]'}", flush=True)

        elapsed_total = time.perf_counter() - total_started
        avg = elapsed_total / idx
        eta = avg * (len(scenes) - idx)
        print(f"  - elapsed {elapsed_total/60:.1f}m | ETA {eta/60:.1f}m", flush=True)

    print(f"\nDone: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
