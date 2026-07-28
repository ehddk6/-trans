from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from translation_forensics.outputs import package_title_outputs
from translation_forensics.review_prioritization import build_uncertainty_review_queue
from translation_forensics.semantic_translation import apply_translation_decisions
from translation_forensics.validation import validate_srt_file


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")
    return path


def _srt(texts: list[str]) -> str:
    parts = []
    for number, text in enumerate(texts, 1):
        start = number * 2 - 1
        parts.append(f"{number}\n00:00:{start:02d},000 --> 00:00:{start + 1:02d},000\n{text}")
    return "\n\n".join(parts) + "\n"


def _decision(number: int, *, evidence_refs: list[str], status: str = "approved", confidence: str = "high") -> dict[str, object]:
    return {
        "block_number": number,
        "source_faithful_korean": f"원문 번역 {number}",
        "viewer_natural_korean": f"자연 번역 {number}",
        "translation_method": "semantic_review_from_japanese",
        "translation_model": "gpt-5.6-terra",
        "status": status,
        "confidence": confidence,
        "evidence_refs": evidence_refs,
    }


def test_strict_application_rejects_evidence_refs_not_declared_by_translation_queue(tmp_path: Path) -> None:
    structure = _write(tmp_path / "structure.srt", _srt(["最初です", "次です", "終わりです"]))
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "".join(json.dumps({"block_number": number, "evidence_refs": ["japanese_srt", "neighboring_context"]}, ensure_ascii=False) + "\n" for number in range(1, 4)),
        encoding="utf-8",
        newline="\n",
    )
    decisions = [_decision(number, evidence_refs=["japanese_srt"]) for number in range(1, 4)]
    decisions[1]["evidence_refs"] = ["invented_evidence"]
    decisions_path = tmp_path / "decisions.jsonl"
    decisions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions), encoding="utf-8", newline="\n")

    result = apply_translation_decisions(
        structure,
        decisions_path,
        tmp_path / "source.srt",
        tmp_path / "viewer.srt",
        tmp_path / "report.json",
        strict=True,
        translation_queue_path=queue,
    )

    assert result["status"] == "fail"
    assert any("translation queue에 없는 evidence_refs" in error for error in result["errors"])
    assert not (tmp_path / "source.srt").exists()


def test_validation_rejects_utf8_bom_for_final_srt_contract(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.srt"
    candidate.write_bytes(b"\xef\xbb\xbf" + _srt(["첫 문장"]).encode("utf-8"))

    report = validate_srt_file(candidate)

    assert report.status == "fail"
    assert any(issue.code == "encoding" and issue.detail == {"encoding": "utf-8-sig"} for issue in report.issues)


def test_packaging_cannot_bypass_translation_queue_evidence_link(tmp_path: Path) -> None:
    structure = _write(tmp_path / "structure.srt", _srt(["最初です"]))
    source = _write(tmp_path / "source.srt", _srt(["첫 문장"]))
    viewer = _write(tmp_path / "viewer.srt", _srt(["첫 문장이에요"]))
    decisions = _write(tmp_path / "decisions.jsonl", json.dumps(_decision(1, evidence_refs=["japanese_srt"]), ensure_ascii=False) + "\n")

    with pytest.raises(RuntimeError, match="translation queue"):
        package_title_outputs(
            "SAMPLE",
            structure,
            source,
            viewer,
            tmp_path / "final",
            japanese_path=structure,
            translation_decisions_path=decisions,
        )


def test_uncertainty_queue_keeps_each_review_reason_visible(tmp_path: Path) -> None:
    structure = _write(tmp_path / "structure.srt", _srt(["最初です", "質問ですか", "終わりです"]))
    queue = tmp_path / "translation-queue.jsonl"
    queue.write_text(
        "".join(json.dumps({"block_number": number, "evidence_refs": ["japanese_srt", "neighboring_context"]}, ensure_ascii=False) + "\n" for number in range(1, 4)),
        encoding="utf-8",
        newline="\n",
    )
    decisions = [
        _decision(1, evidence_refs=["japanese_srt"], confidence="high"),
        {
            **_decision(2, evidence_refs=[], status="unresolved", confidence="low"),
            "source_faithful_korean": "",
            "viewer_natural_korean": "",
            "uncertain_slots": ["polarity"],
            "competing_interpretations": ["질문", "평서"],
        },
        _decision(3, evidence_refs=["unmapped_ref"], confidence="medium"),
    ]
    decisions_path = tmp_path / "decisions.jsonl"
    decisions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions), encoding="utf-8", newline="\n")
    forensics = _write(
        tmp_path / "forensics.csv",
        "block_number,timecode,review_band,source_japanese,provisional_korean,confidence,uncertain_scope,required_evidence,forensics_risk,priority_action\n"
        "1,\"00:00:01,000 --> 00:00:02,000\",P2,最初です,처음이에요,high,,문맥,40,검토\n"
        "2,\"00:00:03,000 --> 00:00:04,000\",P1,質問ですか,질문인가요?,low,화행,원음,80,교차검토\n",
    )
    source = _write(tmp_path / "source.srt", _srt(["원문 번역 1", "원문 번역 2", "매우 길어서 일 초 안에 읽을 수 없는 한국어 자막 문장입니다"]))
    viewer = _write(tmp_path / "viewer.srt", _srt(["자연 번역 1", "자연 번역 2", "매우 길어서 일 초 안에 읽을 수 없는 한국어 자막 문장입니다"]))
    output = tmp_path / "priority.csv"

    result = build_uncertainty_review_queue(
        structure,
        output,
        decisions_path=decisions_path,
        translation_queue_path=queue,
        forensics_queue_path=forensics,
        source_path=source,
        viewer_path=viewer,
    )

    assert result["priority_counts"] == {"critical": 1, "high": 1, "medium": 1}
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["block_number"] for row in rows] == ["2", "3", "1"]
    assert "decision-status" in rows[0]["reason_codes"]
    assert "uncertain-slots" in rows[0]["reason_codes"]
    assert "unmapped-evidence-ref" in rows[1]["reason_codes"]
    assert "source_faithful:high_cps" in rows[1]["reason_codes"]
    assert "forensics-p2" in rows[2]["reason_codes"]
