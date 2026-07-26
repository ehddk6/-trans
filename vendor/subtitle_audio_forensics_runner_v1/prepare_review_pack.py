#!/usr/bin/env python3
"""Create merged review scenes and audio variants from a Subtitle Forensics queue."""
from __future__ import annotations
import argparse, csv, json, re, subprocess
from pathlib import Path

TC_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$")

def tc_to_sec(value: str) -> float:
    m = TC_RE.match(value.strip())
    if not m:
        raise ValueError(f"Invalid SRT time: {value}")
    h, mi, s, ms = map(int, m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000

def sec_to_tc(value: float) -> str:
    value = max(0.0, value)
    ms_total = int(round(value * 1000))
    h, rem = divmod(ms_total, 3600000)
    mi, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{mi:02d}:{s:02d},{ms:03d}"

def read_queue(path: Path, bands: set[str]) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if bands and r.get("review_band", "") not in bands:
            continue
        start_s, end_s = [x.strip() for x in r["timecode"].split("-->")]
        rr = dict(r)
        rr["block_number"] = int(r["block_number"])
        rr["start_sec"] = tc_to_sec(start_s)
        rr["end_sec"] = tc_to_sec(end_s)
        out.append(rr)
    return sorted(out, key=lambda x: (x["start_sec"], x["block_number"]))

def merge_scenes(rows: list[dict], padding: float, gap: float, max_scene: float) -> list[dict]:
    scenes: list[dict] = []
    for r in rows:
        s = max(0.0, r["start_sec"] - padding)
        e = r["end_sec"] + padding
        if not scenes or s - scenes[-1]["end_sec"] > gap or e - scenes[-1]["start_sec"] > max_scene:
            scenes.append({"start_sec": s, "end_sec": e, "blocks": [r]})
        else:
            scenes[-1]["end_sec"] = max(scenes[-1]["end_sec"], e)
            scenes[-1]["blocks"].append(r)
    for i, sc in enumerate(scenes, 1):
        sc["scene_id"] = f"S{i:04d}"
    return scenes

def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def extract_audio(ffmpeg: str, audio: Path, sc: dict, out_dir: Path) -> tuple[str, str]:
    sid = sc["scene_id"]
    dur = sc["end_sec"] - sc["start_sec"]
    orig = out_dir / f"{sid}.original.wav"
    enhanced = out_dir / f"{sid}.dialogue.wav"
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{sc['start_sec']:.3f}", "-t", f"{dur:.3f}", "-i", str(audio)]
    run(base + ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(orig)])
    # Conservative dialogue enhancement. This is a second listening/ASR view, not a replacement for the original.
    filt = "aformat=sample_rates=16000:channel_layouts=mono,highpass=f=80,lowpass=f=12000,afftdn=nf=-24,loudnorm=I=-18:TP=-2:LRA=7"
    run(base + ["-af", filt, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(enhanced)])
    return orig.name, enhanced.name

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, type=Path)
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bands", default="P1,P2", help="Comma-separated bands; use ALL for all rows")
    ap.add_argument("--padding", type=float, default=2.5)
    ap.add_argument("--merge-gap", type=float, default=1.5)
    ap.add_argument("--max-scene", type=float, default=45.0)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    bands = set() if args.bands.upper() == "ALL" else {x.strip() for x in args.bands.split(",") if x.strip()}
    rows = read_queue(args.queue, bands)
    if not rows:
        raise SystemExit("선택한 review band에 해당하는 블록이 없습니다.")
    scenes = merge_scenes(rows, args.padding, args.merge_gap, args.max_scene)
    args.out.mkdir(parents=True, exist_ok=True)
    clips = args.out / "clips"
    clips.mkdir(exist_ok=True)

    manifest_rows = []
    for sc in scenes:
        orig = enhanced = ""
        if not args.no_audio:
            orig, enhanced = extract_audio(args.ffmpeg, args.audio, sc, clips)
        blocks = []
        for r in sc["blocks"]:
            blocks.append({k: r.get(k, "") for k in [
                "block_number", "timecode", "source_japanese", "provisional_korean",
                "confidence", "uncertain_scope", "required_evidence", "forensics_risk",
                "review_band", "priority_action"
            ]})
        manifest_rows.append({
            "scene_id": sc["scene_id"],
            "start_time": sec_to_tc(sc["start_sec"]),
            "end_time": sec_to_tc(sc["end_sec"]),
            "duration_sec": f"{sc['end_sec'] - sc['start_sec']:.3f}",
            "block_numbers": ",".join(str(b["block_number"]) for b in blocks),
            "highest_band": min((b["review_band"] for b in blocks), key=lambda x: int(x[1:]) if x.startswith("P") else 99),
            "original_audio": f"clips/{orig}" if orig else "",
            "dialogue_audio": f"clips/{enhanced}" if enhanced else "",
            "context_japanese": " / ".join(str(b.get("source_japanese") or "") for b in blocks if str(b.get("source_japanese") or "").strip()),
            "blocks_json": json.dumps(blocks, ensure_ascii=False),
        })

    out_csv = args.out / "review-scenes.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [])
        if manifest_rows:
            w.writeheader(); w.writerows(manifest_rows)
    summary = {
        "selected_bands": sorted(bands) if bands else "ALL",
        "blocks": len(rows), "scenes": len(scenes),
        "review_audio_minutes": round(sum(float(r["duration_sec"]) for r in manifest_rows) / 60, 3),
        "padding_sec": args.padding, "merge_gap_sec": args.merge_gap, "max_scene_sec": args.max_scene,
    }
    (args.out / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
