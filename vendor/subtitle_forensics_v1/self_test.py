from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
errors = []
checks = []

def check(condition: bool, name: str, detail: str = ""):
    checks.append({"name": name, "pass": bool(condition), "detail": detail})
    if not condition:
        errors.append(f"{name}: {detail}")

summary = json.loads((ROOT / "implementation-summary.json").read_text(encoding="utf-8"))
portfolio = pd.read_csv(ROOT / "portfolio-priority-queue.csv", encoding="utf-8-sig")
asset_index = pd.read_csv(ROOT / "asset-index.csv", encoding="utf-8-sig")
model = joblib.load(ROOT / "model" / "risk_model.joblib")

check(len(asset_index) == summary["titles_analyzed"], "asset count", f"{len(asset_index)}")
check(len(portfolio) == summary["blocks_analyzed"], "portfolio block count", f"{len(portfolio)}")
check(portfolio["priority_score"].between(0, 100).all(), "priority score range")
check(portfolio["final_risk"].between(0, 100).all(), "absolute risk range")
check(set(portfolio["review_band"].unique()).issubset({"P1", "P2", "P3", "P4"}), "review bands")
check(set(portfolio["tier"].unique()).issubset({"T1", "T2", "T3", "T4"}), "absolute tiers")
check(portfolio["priority_score"].is_monotonic_decreasing, "portfolio sorted descending")
check(isinstance(model, dict) and {"classifier", "vectorizer", "scaler"}.issubset(model), "model load")

for _, asset in asset_index.iterrows():
    title = asset["title"]
    folder = ROOT / "titles" / title
    ledger_path = folder / f"{title}.evidence-ledger.csv"
    report_path = folder / f"{title}.qa-report.json"
    packet_path = folder / f"{title}.review-packets.jsonl"
    check(ledger_path.exists(), f"{title} ledger exists")
    check(report_path.exists(), f"{title} report exists")
    check(packet_path.exists(), f"{title} packets exists")
    if ledger_path.exists() and report_path.exists():
        ledger = pd.read_csv(ledger_path, encoding="utf-8-sig")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        check(len(ledger) == report["validation"]["block_count"], f"{title} row count")
        check(ledger["block"].tolist() == list(range(1, len(ledger) + 1)), f"{title} sequential blocks")
        check(ledger["priority_score"].between(0, 100).all(), f"{title} priority range")

result = {
    "pass": not errors,
    "check_count": len(checks),
    "failed_count": len(errors),
    "errors": errors,
    "checks": checks,
}
(ROOT / "self-test-report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: result[k] for k in ["pass", "check_count", "failed_count", "errors"]}, ensure_ascii=False, indent=2))
raise SystemExit(1 if errors else 0)
