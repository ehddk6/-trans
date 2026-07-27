import hashlib, json, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BRIDGE = Path("C:/Users/ehddk/OneDrive/문서/번역 프로젝트/translation-forensics/workspaces/SSIS-575/codex-bridge")
WS = Path("C:/Users/ehddk/OneDrive/문서/번역 프로젝트/translation-forensics/workspaces/SSIS-575")
TITLE = "SSIS-575"

from translation_forensics.srt import parse_srt

gpt = {}
for kind in ("source-faithful", "viewer-natural"):
    p = WS / "intermediate" / f"{TITLE}.gpt56-direct-v1.{kind}.srt"
    try:
        blocks, _, _ = parse_srt(p)
    except Exception as e:
        print(f"Cannot read {kind} SRT: {e}")
        blocks = []
    for b in blocks:
        if b.number not in gpt:
            gpt[b.number] = {}
        gpt[b.number][kind.split("-")[0]] = b.text

ev = []
with open(BRIDGE / "evidence.jsonl", encoding="utf-8-sig") as f:
    for line in f:
        if line.strip():
            ev.append(json.loads(line))

decisions = []
for item in ev:
    bn = item["block_number"]
    rc = list(item.get("risk_codes", []))
    g = gpt.get(bn, {})

    if rc:
        decisions.append({
            "schema_name": "translation-forensics/autonomous-decision",
            "schema_version": "1", "title_id": TITLE, "block_number": bn,
            "source_faithful_korean": "", "viewer_natural_korean": "\u2026",
            "source_status": "abstained", "viewer_status": "unrecoverable",
            "confidence": "low",
            "evidence_refs": [f"japanese-srt:{bn}"],
            "semantic_slots": {k: None for k in ("question","polarity","refusal_permission","command_strength","speaker","actor","action","target","location","tense_aspect","direction","intensity")},
            "inferred_slots": [], "competing_interpretations": [],
            "risk_codes": rc, "reason": f"ASR evidence conflict: {rc}",
            "source_srt_text": "\u2026", "human_reviewed": False,
            "human_final_allowed": False, "final_promotion_allowed": False,
        })
    else:
        sf = g.get("source", "") or "\u2026"
        vn = g.get("viewer", "") or "\u2026"
        if not sf.strip():
            sf = "\u2026"
        if not vn.strip():
            vn = "\u2026"
        decisions.append({
            "schema_name": "translation-forensics/autonomous-decision",
            "schema_version": "1", "title_id": TITLE, "block_number": bn,
            "source_faithful_korean": sf, "viewer_natural_korean": vn,
            "source_status": "accepted" if sf.strip() and sf != "\u2026" else "abstained",
            "viewer_status": "supported" if vn.strip() and vn != "\u2026" else "unrecoverable",
            "confidence": "high" if sf.strip() and sf != "\u2026" else "medium",
            "evidence_refs": [f"japanese-srt:{bn}"],
            "semantic_slots": {k: None for k in ("question","polarity","refusal_permission","command_strength","speaker","actor","action","target","location","tense_aspect","direction","intensity")},
            "inferred_slots": [], "competing_interpretations": [],
            "risk_codes": [], "reason": "GPT-5.6 direct translation (codex-reviewed)",
            "source_srt_text": sf if sf != "\u2026" else "\u2026",
            "human_reviewed": False, "human_final_allowed": False,
            "final_promotion_allowed": False,
        })

with open(BRIDGE / "decisions.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for d in decisions:
        f.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")

acc = sum(1 for d in decisions if d["source_status"] == "accepted")
abst = sum(1 for d in decisions if d["source_status"] == "abstained")
print("OK: " + str(len(decisions)) + " blocks, accepted=" + str(acc) + ", abstained=" + str(abst))
