import json
decisions = []
with open("workspaces/SSIS-575/codex-bridge/decisions.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            decisions.append(json.loads(line))

samples = [1, 5, 10, 100, 500, 1000, 1500]
for bn in samples:
    d = [x for x in decisions if x["block_number"] == bn][0]
    sf = d["source_faithful_korean"]
    vn = d["viewer_natural_korean"]
    ss = d["semantic_slots"]
    diff = "[DIFF]" if sf != vn else ""
    print("Block " + str(bn) + " " + diff)
    print("  SF: " + sf[:50])
    print("  VN: " + vn[:50])
    print("  Q:" + str(ss["question"]) + " P:" + str(ss["polarity"]) + " C:" + str(ss["command_strength"]) + " T:" + str(ss["tense_aspect"]))

vn_diffs = [d for d in decisions if d["source_faithful_korean"] != d["viewer_natural_korean"]]
print()
print("=== Gen 2 Stats ===")
print("Total: " + str(len(decisions)))
print("Viewer diff count: " + str(len(vn_diffs)))
for d in vn_diffs[:5]:
    print("  B" + str(d["block_number"]) + ": SF=" + d["source_faithful_korean"][:30])
