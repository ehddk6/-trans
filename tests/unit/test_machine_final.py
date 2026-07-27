from __future__ import annotations

import json
from pathlib import Path

from translation_forensics.machine_final import package_machine_final, repair_machine_final_asr


def _write(path: Path, value: str) -> Path:
    path.write_text(value, encoding="utf-8", newline="\n")
    return path


def _srt(text: str) -> str:
    return f"1\n00:00:01,000 --> 00:00:02,000\n{text}\n"


def test_machine_final_keeps_human_final_blocked(tmp_path: Path) -> None:
    structure = _write(tmp_path / "structure.srt", _srt("日本語"))
    source = _write(tmp_path / "source.srt", _srt("한국어"))
    viewer = _write(tmp_path / "viewer.srt", _srt("한국어"))
    report = _write(tmp_path / "automatic-report.json", json.dumps({"status": "automatic-draft-complete", "unresolved_blocks": []}))
    ledger = _write(tmp_path / "ledger.jsonl", json.dumps({"block_number": 1, "status": "automatic-draft", "source_faithful_korean": "한국어", "viewer_natural_korean": "한국어"}) + "\n")
    asr = _write(tmp_path / "asr.csv", "scene_id,model,profile,audio,text,avg_logprob,no_speech_prob,language_probability,error\nS1,m,original_unbiased,a.wav,日本語,-0.1,0.1,1,\n")
    timeline = _write(tmp_path / "timeline.json", json.dumps({"status": "unresolved"}))
    proof = _write(tmp_path / "proof.json", json.dumps({"human_reference_equality": "unidentifiable", "100_percent_equal": False}))
    result = package_machine_final(
        title="SAMPLE",
        structure_path=structure,
        source_path=source,
        viewer_path=viewer,
        automatic_report_path=report,
        automatic_ledger_path=ledger,
        asr_path=asr,
        timeline_path=timeline,
        proof_path=proof,
        output_dir=tmp_path / "machine-final",
    )
    assert result["machine_final_allowed"] is True
    assert result["human_final_allowed"] is False
    assert result["final_promotion_allowed"] is False
    package_asr = Path(result["output"]) / "asr-candidates.csv"
    package_asr.write_bytes(b"damaged")
    repaired = repair_machine_final_asr(source_asr_path=asr, package_dir=Path(result["output"]))
    assert repaired["status"] == "repaired"
    assert package_asr.read_bytes() == asr.read_bytes()
    assert (Path(result["output"]) / "asr-candidates.for-excel.utf8bom.csv").read_bytes().startswith(b"\xef\xbb\xbf")
