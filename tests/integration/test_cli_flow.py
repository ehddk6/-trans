from __future__ import annotations

import json
from pathlib import Path

from translation_forensics.cli import main


ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


def test_cli_synthetic_flow(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    project.mkdir()
    assert main(["init-title", "--project-root", str(project), "--title", "SAMPLE"]) == 0
    inputs = project / "workspaces" / "SAMPLE" / "inputs"
    for name in ("sample.structure.srt", "sample.ja.srt", "sample.previous-ko.srt"):
        target = inputs / name.replace("sample", "SAMPLE")
        target.write_bytes((FIXTURES / name).read_bytes())
    assert main(["inspect", "--project-root", str(project), "--title", "SAMPLE"]) == 0
    assert main(["analyze", "--project-root", str(project), "--title", "SAMPLE"]) == 0
    queue_files = list((project / "workspaces" / "SAMPLE" / "intermediate").glob("*.review-queue*.csv"))
    assert len(queue_files) == 1
    assert main(["init-consistency-ledger", "--project-root", str(project), "--title", "SAMPLE"]) == 0
    ledger = project / "workspaces" / "SAMPLE" / "intermediate" / "SAMPLE.translation-consistency-v1.jsonl"
    assert main(["validate-consistency-ledger", "--input", str(ledger)]) == 0
    assert main(["build-translation-queue", "--project-root", str(project), "--title", "SAMPLE"]) == 0
    assert main(["init-translation-decisions", "--project-root", str(project), "--title", "SAMPLE"]) == 0
    assert main(["build-uncertainty-review-queue", "--project-root", str(project), "--title", "SAMPLE"]) == 0
    assert (project / "workspaces" / "SAMPLE" / "intermediate" / "SAMPLE.review-queue.uncertainty-v1.csv").exists()
    assert main(["validate-quality-regressions", "--input", str(ROOT / "tests" / "fixtures" / "translation-quality-regressions.json")]) == 0
    manifest = json.loads((project / "workspaces" / "SAMPLE" / "metadata" / "project-manifest.json").read_text(encoding="utf-8"))
    assert manifest["title"] == "SAMPLE"
