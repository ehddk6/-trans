import json, sys
sys.stdout.reconfigure(encoding="utf-8")

decisions = []
with open("C:/Users/ehddk/OneDrive/문서/번역 프로젝트/translation-forensics/workspaces/SSIS-575/codex-bridge/decisions.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            decisions.append(json.loads(line))

for d in decisions:
    bn = d["block_number"]
    if bn == 21:
        d["source_faithful_korean"] = "\ucca0 \uacbd\ud5d8."
        d["viewer_natural_korean"] = "\ucca0 \uacbd\ud5d8\uc774\uc57c?"
        d["source_status"] = "accepted"
        d["viewer_status"] = "supported"
        d["confidence"] = "medium"
        d["risk_codes"] = []
        d["reason"] = "ASR consensus (3/4) fudeoroshi"
        d["source_srt_text"] = "\ucca0 \uacbd\ud5d8."
    elif bn == 18:
        d["reason"] = "Japanese text corrupted, ASR disagree 0.42"
    elif bn == 1077:
        d["source_faithful_korean"] = "\uc2a4\ud2b8\ub808\uc2a4, \ub3cc\uc544\uc654\uc5b4?"
        d["viewer_natural_korean"] = "\uc2a4\ud2b8\ub808\uc2a4 \ub3cc\uc544\uc654\uc5b4?"
        d["source_status"] = "accepted"
        d["viewer_status"] = "supported"
        d["confidence"] = "medium"
        d["risk_codes"] = []
        d["reason"] = "ASR 3/4 agree on stress"
        d["source_srt_text"] = "\uc2a4\ud2b8\ub808\uc2a4, \ub3cc\uc544\uc654\uc5b4?"

with open("C:/Users/ehddk/OneDrive/문서/번역 프로젝트/translation-forensics/workspaces/SSIS-575/codex-bridge/decisions.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for d in decisions:
        f.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")

acc = sum(1 for d in decisions if d["source_status"] == "accepted")
abst = sum(1 for d in decisions if d["source_status"] == "abstained")
print(str(len(decisions)) + " blocks, accepted=" + str(acc) + ", abstained=" + str(abst))
