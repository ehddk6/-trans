from __future__ import annotations

import json
from pathlib import Path

from translation_forensics.source_quality import audit_source_blocks, is_legitimate_short_repetition


def write_srt(path: Path, texts: list[str]) -> None:
    parts = []
    for index, text in enumerate(texts, 1):
        start = index * 2
        parts.append(f"{index}\n00:00:{start:02d},000 --> 00:00:{start + 1:02d},000\n{text}")
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def test_long_mid_program_outro_is_unusable(tmp_path):
    source = tmp_path / "source.srt"
    write_srt(source, ["普通の台詞"] + ["ご視聴ありがとうございました"] * 10 + ["別の台詞"])
    report, records = audit_source_blocks(title_id="SAMPLE", japanese_path=source)
    repeated = records[1:11]
    assert all(record["source_quality_status"] == "unusable" for record in repeated)
    assert all("mid-program-outro" in record["reason_codes"] for record in repeated)
    assert report["longest_consecutive_repeat"] == 10


def test_short_vocalization_repetition_is_not_corruption(tmp_path):
    source = tmp_path / "source.srt"
    write_srt(source, ["あ"] * 8)
    _, records = audit_source_blocks(title_id="SAMPLE", japanese_path=source)
    assert is_legitimate_short_repetition("あ…")
    assert all(record["source_quality_status"] == "trusted" for record in records)


def test_convergent_dual_asr_can_mark_source_conflict(tmp_path):
    source = tmp_path / "source.srt"
    asr = tmp_path / "asr.jsonl"
    write_srt(source, ["今日は晴れです"])
    asr.write_text(
        json.dumps(
            {
                "block_number": 1,
                "transcripts": [
                    {"source_family": "faster-whisper-large-v3", "text": "もうやめて"},
                    {"source_family": "reazonspeech-k2-v2", "text": "もうやめて"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _, records = audit_source_blocks(title_id="SAMPLE", japanese_path=source, asr_evidence_path=asr)
    assert records[0]["source_quality_status"] == "unusable"
    assert "dual-asr-source-conflict" in records[0]["reason_codes"]
