from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .alignment import align_by_overlap
from .srt import SubtitleBlock, parse_srt
from .manifest import write_json


def _band(score: float) -> str:
    if score >= 72:
        return "P1"
    # 구조 기반 fallback은 vendor 위험 모델보다 보수적인 신호만 가지므로
    # P2 후보를 놓치지 않도록 별도 임계값을 사용한다. 최종 의미 판정은 하지 않는다.
    if score >= 45:
        return "P2"
    if score >= 28:
        return "P3"
    return "P4"


def _fallback_score(block: SubtitleBlock, previous_text: str = "") -> tuple[float, list[str]]:
    text = block.text
    compact = re.sub(r"\s+", "", text)
    score = 0.0
    flags: list[str] = []
    if len(compact) > 42:
        score += 22; flags.append("over_42_chars")
    if len(block.lines) > 2:
        score += 18; flags.append("over_2_lines")
    if "?" in text or "？" in text or any(token in text for token in ("何", "誰", "どこ", "どう", "なに")):
        score += 22; flags.append("question_candidate")
    if re.search(r"ない|ません|ぬ|だめ|駄目|嫌|やめ|無理|いや", text):
        score += 28; flags.append("negation_or_refusal_candidate")
    if re.search(r"入れ|入る|いれ|はいる|出す|出る|だす|行く|いく|いって|濡れ|ぬれ|触|さわ|挿|舐|なめ|動|うご|抜|ぬい|奥|おく|中|なか|外|そと|前|まえ|後|あと|イク|して", text):
        score += 22; flags.append("action_or_location_candidate")
    if block.duration < 0.8 and len(compact) > 10:
        score += 12; flags.append("short_timing")
    if previous_text and previous_text == text:
        score += 10; flags.append("duplicate_candidate")
    return min(100.0, score), flags


def build_fallback_queue(ja_path: Path, previous_path: Path | None, output_path: Path, *, reason: str) -> dict[str, Any]:
    ja_blocks, _, _ = parse_srt(ja_path)
    previous_blocks = parse_srt(previous_path)[0] if previous_path and previous_path.exists() else []
    aligned_previous = align_by_overlap(ja_blocks, previous_blocks) if previous_blocks else []
    previous_by_number = {item.source_number: item.candidate_number for item in aligned_previous}
    previous_text_by_number = {block.number: block.text for block in previous_blocks}
    rows: list[dict[str, object]] = []
    for block in ja_blocks:
        previous_number = previous_by_number.get(block.number)
        previous_text = previous_text_by_number.get(previous_number or -1, "")
        score, flags = _fallback_score(block, previous_text)
        band = _band(score)
        rows.append({
            "block_number": block.number,
            "timecode": f"{block.start} --> {block.end}",
            "review_band": band,
            "source_japanese": block.text,
            "provisional_korean": previous_text,
            "confidence": "medium" if previous_text else "low",
            "uncertain_scope": "자동 큐 생성 단계에서는 의미 판정하지 않음",
            "required_evidence": "원음·앞뒤 문맥·일본어 문법",
            "forensics_risk": f"{score:.2f}",
            "priority_action": "vendor 결과 또는 원음 교차검토" if score >= 58 else "문자·문맥 검토",
            "risk_flags": ";".join(flags),
            "queue_method": "structure_fallback",
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["block_number", "timecode", "review_band"])
        writer.writeheader()
        writer.writerows(rows)
    return {"status": "created", "method": "structure_fallback", "reason": reason, "rows": len(rows), "output": str(output_path)}


def _vendor_requirements(vendor_root: Path) -> tuple[bool, list[str]]:
    required = [
        vendor_root / "_v6_srt_extract" / "ABF-303.final-deep-natural-ko.v6.srt",
        vendor_root / "ABF-303_photo_verified_retranslation" / "ABF-303.final-photo-verified-retranslated-ko.v1.changes.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    return not missing, missing


def run_vendor_forensics(vendor_root: Path, output_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    script = Path(__file__).resolve().parents[2] / "vendor" / "subtitle_forensics_v1" / "subtitle_forensics.py"
    if not script.exists():
        return {"status": "skipped", "reason": "vendor script missing", "script": str(script)}
    ready, missing = _vendor_requirements(vendor_root)
    command = [sys.executable, str(script), "--root", str(vendor_root), "--output", str(output_dir)]
    result: dict[str, Any] = {"status": "ready" if ready else "skipped", "command": command, "missing": missing, "method": "subtitle_forensics_v1"}
    if not ready or dry_run:
        if dry_run:
            result["dry_run"] = True
        return result
    completed = subprocess.run(command, cwd=str(script.parent), capture_output=True, text=True, check=False)
    result.update({"returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]})
    result["status"] = "completed" if completed.returncode == 0 else "failed"
    return result


def analyze_title(ja_path: Path, previous_path: Path | None, queue_path: Path, *, vendor_root: Path | None = None, vendor_output: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    if vendor_root:
        vendor_result = run_vendor_forensics(vendor_root, vendor_output or queue_path.parent / "vendor-forensics", dry_run=dry_run)
        if vendor_result.get("status") == "completed":
            return vendor_result
        reason = "; ".join(vendor_result.get("missing", [])) or str(vendor_result.get("reason", "vendor 실행 불가"))
    else:
        vendor_result = {"status": "not_requested"}
        reason = "공용 Subtitle Forensics 원자료의 원본 학습 입력이 현재 작업 폴더에 없어 구조 기반 큐를 사용"
    if dry_run:
        return {"status": "dry-run", "method": "structure_fallback", "reason": reason, "output": str(queue_path)}
    fallback = build_fallback_queue(ja_path, previous_path, queue_path, reason=reason)
    fallback["vendor"] = vendor_result
    write_json(queue_path.parent / "forensics-run.json", fallback)
    return fallback
