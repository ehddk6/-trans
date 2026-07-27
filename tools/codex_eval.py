import json
from pathlib import Path

pkg = Path("workspaces/SSIS-575/autonomous-release-codex-v1")
errors = []
warnings = []

required = ["structure.srt", "codex-decisions.jsonl", "codex-evidence.jsonl", "qa-report.json",
            "codex-release-report.json", "codex-proof.json", "codex-release-manifest.json",
            "evidence-graph.json", "machine-alignment.json", "title-memory.json"]
for name in required:
    f = pkg / name
    if not f.exists():
        errors.append("missing: " + name)

for suffix in ["source-faithful-ko.codex-release-v1.srt", "viewer-complete-ko.codex-release-v1.srt"]:
    candidates = list(pkg.glob("*" + suffix))
    if len(candidates) != 1:
        errors.append("SRT count: " + suffix + " = " + str(len(candidates)))

qa = json.loads((pkg / "qa-report.json").read_text(encoding="utf-8-sig"))
if qa.get("status") == "fail":
    errors.append("QA failed")
if not qa.get("structure_same"):
    errors.append("structure mismatch")

report = json.loads((pkg / "codex-release-report.json").read_text(encoding="utf-8-sig"))
if report.get("critical_conflicts_in_accepted", -1) > 0:
    errors.append("critical conflicts: " + str(report["critical_conflicts_in_accepted"]))

decisions = []
with open(pkg / "codex-decisions.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            decisions.append(json.loads(line))

source_statuses = [d["source_status"] for d in decisions]
accepted = sum(1 for s in source_statuses if s == "accepted")
abstained = sum(1 for s in source_statuses if s == "abstained")
empty_viewer = [d["block_number"] for d in decisions if not d.get("viewer_natural_korean","").strip()]
if empty_viewer:
    errors.append("empty viewer blocks: " + str(empty_viewer[:5]))

manifest = json.loads((pkg / "codex-release-manifest.json").read_text(encoding="utf-8-sig"))
for item in manifest.get("outputs", []):
    out_path = pkg / item["path"]
    if not out_path.exists():
        errors.append("manifest missing: " + item["path"])

print("=== Stage 1: Mechanical Verification ===")
print("Errors: " + str(len(errors)))
for e in errors:
    print("  ERR: " + e)
print()
print("Blocks: " + str(len(decisions)) + " total, " + str(accepted) + " accepted, " + str(abstained) + " abstained")
print("Empty viewer: " + str(len(empty_viewer)))
print("QA: " + qa.get("status", "unknown") + ", structure: " + str(qa.get("structure_same", False)))
print("Critical conflicts: " + str(report.get("critical_conflicts_in_accepted", "?")))
print("Release: " + report.get("status", "unknown"))
