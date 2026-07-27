from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .forensic_model import read_jsonl


REVIEW_DECISIONS = {"approve", "approve-with-edit", "hold", "reject", "needs-human-listening", "needs-more-context", "nonverbal", "unresolved"}


def _safe_relative(path: str, *, from_dir: Path, audio_root: Path | None) -> str:
    if not path or audio_root is None:
        return ""
    candidate = (audio_root / path).resolve()
    if not candidate.exists():
        return ""
    try:
        return str(candidate.relative_to(from_dir.resolve())).replace("\\", "/")
    except ValueError:
        return ""


def build_review_pack(title: str, context_path: Path, output_dir: Path, *, audio_root: Path | None = None) -> dict[str, Any]:
    """Build a self-contained local review page without inventing any decision."""
    if output_dir.exists():
        raise FileExistsError(f"기존 review pack을 덮어쓰지 않습니다: {output_dir}")
    contexts = read_jsonl(context_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for scene in contexts:
        scene_id = str(scene.get("scene_id", ""))
        if not scene_id:
            raise ValueError("review context에 scene_id가 없습니다.")
        blocks = [block for block in scene.get("blocks", []) if isinstance(block, dict)]
        records.append({
            "scene_id": scene_id,
            "timecode": f"{scene.get('start_time', '')} --> {scene.get('end_time', '')}",
            "priority": scene.get("highest_band", ""),
            "audio": {
                "original": _safe_relative(str(scene.get("original_audio", "")), from_dir=output_dir, audio_root=audio_root),
                "dialogue": _safe_relative(str(scene.get("dialogue_audio", "")), from_dir=output_dir, audio_root=audio_root),
            },
            "blocks": blocks,
            "asr_candidates": [row for row in scene.get("asr_candidates", []) if isinstance(row, dict)],
            "decision_status": "unreviewed",
            "review_instruction": "ASR·화면·기존 한국어 후보는 근거 후보이며 의미 확정이 아닙니다.",
        })
    manifest = {
        "schema_name": "translation-forensics/review-pack",
        "schema_version": "1",
        "title_id": title,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "context_source": str(context_path.resolve()),
        "scenes": len(records),
        "status": "review-template-not-reviewed",
        "network": "disabled-by-design; generated HTML references no remote resources",
    }
    (output_dir / "review-context.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    template = []
    for record in records:
        for block in record["blocks"]:
            template.append({
                "review_id": f"R-{title}-{record['scene_id']}-{block.get('block_number', '')}",
                "scene_id": record["scene_id"],
                "block_number": block.get("block_number"),
                "decision": "unreviewed",
                "evidence_refs": [],
                "uncertain_slots": [],
                "review_note": "",
            })
    (output_dir / "review-decisions.template.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in template), encoding="utf-8", newline="\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    data_json = html.escape(json.dumps(records, ensure_ascii=False), quote=False)
    page = f"""<!doctype html><html lang=\"ko\"><meta charset=\"utf-8\"><title>{html.escape(title)} 검수 패킷</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem}}article{{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}}pre{{white-space:pre-wrap;background:#f6f6f6;padding:.75rem}}.warn{{color:#8a3b00}}button{{margin-right:.5rem}}</style>
<h1>{html.escape(title)} 로컬 검수 패킷</h1><p class=\"warn\">이 페이지는 판정을 만들지 않습니다. 원음과 독립 근거를 확인한 뒤 템플릿 JSONL에 결정을 기록하세요.</p><div id=\"app\"></div>
<script id=\"data\" type=\"application/json\">{data_json}</script><script>
const data=JSON.parse(document.getElementById('data').textContent), app=document.getElementById('app');
for(const s of data){{const a=document.createElement('article');a.innerHTML=`<h2>${{s.scene_id}} · ${{s.priority}} · ${{s.timecode}}</h2>`;
for(const [label,src] of Object.entries(s.audio)) if(src){{const p=document.createElement('p');p.textContent=label+' audio: ';const x=document.createElement('audio');x.controls=true;x.src=src;p.append(x);a.append(p)}}
const b=document.createElement('pre');b.textContent=JSON.stringify({{blocks:s.blocks,asr_candidates:s.asr_candidates,instruction:s.review_instruction}},null,2);a.append(b);app.append(a)}}
</script></html>"""
    (output_dir / "review.html").write_text(page, encoding="utf-8", newline="\n")
    return {"status": "review-pack-created", "output": str(output_dir), "scenes": len(records), "decision_template": str(output_dir / "review-decisions.template.jsonl"), "review_status": "not-reviewed"}


def validate_review_decisions(context_path: Path, decisions_path: Path) -> dict[str, Any]:
    expected: set[tuple[str, int]] = set()
    for scene in read_jsonl(context_path):
        scene_id = str(scene.get("scene_id", ""))
        for block in scene.get("blocks", []):
            if isinstance(block, dict):
                try:
                    expected.add((scene_id, int(block["block_number"])))
                except (KeyError, TypeError, ValueError):
                    pass
    actual: set[tuple[str, int]] = set(); errors: list[str] = []
    for index, decision in enumerate(read_jsonl(decisions_path), 1):
        try:
            key = (str(decision["scene_id"]), int(decision["block_number"]))
        except (KeyError, TypeError, ValueError):
            errors.append(f"결정 {index}: scene_id 또는 block_number가 잘못되었습니다."); continue
        if key in actual:
            errors.append(f"결정 {index}: 중복 결정 {key}")
        actual.add(key)
        status = decision.get("decision")
        if status not in REVIEW_DECISIONS:
            errors.append(f"결정 {index}: 허용되지 않는 decision {status!r}")
        if status in {"approve", "approve-with-edit", "reject"} and not decision.get("evidence_refs"):
            errors.append(f"결정 {index}: {status}에는 evidence_refs가 필요합니다.")
    missing = sorted(expected - actual); extra = sorted(actual - expected)
    errors.extend(f"결정 누락: {scene_id}/{number}" for scene_id, number in missing)
    errors.extend(f"패킷에 없는 결정: {scene_id}/{number}" for scene_id, number in extra)
    return {"status": "pass" if not errors else "fail", "expected_blocks": len(expected), "decisions": len(actual), "errors": errors, "final_promotion_allowed": False}
