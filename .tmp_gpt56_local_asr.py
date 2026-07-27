"""One-off, local-only ASR recovery for known garbled Japanese SRT blocks.

Uses exact SRT windows, ffmpeg, and a locally cached faster-whisper model.
No web/API call and no subtitle candidate is consulted.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from translation_forensics.srt import parse_srt
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path.home() / ".cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
TARGETS = {
    "ADN-622": [54, 57, 142, 185, 464, 474, 475, 479, 542, 643],
    "ADN-746": [217, 264, 313, 318, 336, 361, 363, 371],
    "IPX-998": [218, 285, 287, 330, 664, 690, 691, 692, 693, 790, 814],
    "JUQ-439": [373, 375, 436, 598, 618, 643, 644],
}


def transcribe(model: WhisperModel, wav: Path, vad: bool) -> tuple[str, float | None, float | None, float | None, str]:
    try:
        segments, info = model.transcribe(
            str(wav), language="ja", task="transcribe", beam_size=5,
            vad_filter=vad, condition_on_previous_text=False,
        )
        got = list(segments)
        text = "".join(s.text.strip() for s in got).strip()
        avg_logprob = sum(s.avg_logprob for s in got) / len(got) if got else None
        no_speech = sum(s.no_speech_prob for s in got) / len(got) if got else None
        return text, avg_logprob, no_speech, info.language_probability, ""
    except Exception as exc:  # retain recovery evidence instead of concealing it
        return "", None, None, None, repr(exc)


def main() -> None:
    if not MODEL.is_dir():
        raise SystemExit(f"Missing local model: {MODEL}")
    model = WhisperModel(str(MODEL), device="cuda", compute_type="float16")
    for title, numbers in TARGETS.items():
        source = ROOT / title / f"{title}.ja.srt"
        blocks, _, _ = parse_srt(source)
        selected = {b.number: b for b in blocks if b.number in numbers}
        outdir = ROOT / "translation-forensics/workspaces" / title / "intermediate"
        clip_dir = outdir / f"{title}.gpt56-direct-v1.asr-clips"
        clip_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for number in numbers:
            b = selected[number]
            wav = clip_dir / f"{number:04d}.exact.wav"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{b.start_seconds:.3f}", "-t", f"{b.duration:.3f}",
                "-i", str(ROOT / title / f"{title}.mp3"), "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(wav),
            ], check=True)
            for profile, vad in (("exact_unbiased_no_vad", False), ("exact_unbiased_vad", True)):
                text, avg_logprob, no_speech, language_probability, error = transcribe(model, wav, vad)
                rows.append({
                    "block_number": number,
                    "timecode": f"{b.start} --> {b.end}",
                    "source_japanese_srt": b.text,
                    "model": "faster-whisper-large-v3-turbo (local cache)",
                    "profile": profile,
                    "audio_window": "exact_srt_window_no_padding",
                    "candidate_japanese": text,
                    "avg_logprob": avg_logprob,
                    "no_speech_prob": no_speech,
                    "language_probability": language_probability,
                    "error": error,
                })
                print(title, number, profile, repr(text), flush=True)
        dest = outdir / f"{title}.gpt56-direct-v1.asr-candidates.jsonl"
        dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
        print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
