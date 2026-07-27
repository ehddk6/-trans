from __future__ import annotations

import csv
import json
from pathlib import Path

from translation_forensics.inferred_recovery import (
    apply_inferred_recovery,
    build_inference_audio_queue,
    build_inference_context,
)
from translation_forensics.srt import parse_srt


def _write(path: Path, value: str) -> Path:
    path.write_text(value, encoding="utf-8", newline="\n")
    return path


def _srt(lines: list[str]) -> str:
    return "\n\n".join(f"{index}\n00:00:0{index},000 --> 00:00:0{index + 1},000\n{text}" for index, text in enumerate(lines, 1)) + "\n"


def _hold_ledger(path: Path) -> Path:
    return _write(path, json.dumps({"title_id": "SAMPLE", "block_number": 2, "applied_method": "targeted-hold-marker"}, ensure_ascii=False) + "\n")


def test_build_inference_queue_and_context(tmp_path: Path) -> None:
    structure = _write(tmp_path / "structure.srt", _srt(["前", "壊れた", "後"]))
    source = _write(tmp_path / "source.srt", _srt(["앞", "…", "뒤"]))
    viewer = _write(tmp_path / "viewer.srt", _srt(["앞", "…", "뒤"]))
    ledger = _hold_ledger(tmp_path / "hold.jsonl")
    queue = tmp_path / "queue.csv"
    result = build_inference_audio_queue(title="SAMPLE", structure_path=structure, hold_ledger_path=ledger, output_path=queue)
    assert result["held_blocks"] == 1
    with queue.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["block_number"] == "2"
    scenes = _write(
        tmp_path / "scenes.csv",
        "scene_id,start_time,end_time,duration_sec,block_numbers,original_audio,dialogue_audio\nS0001,\"00:00:02,000\",\"00:00:03,000\",1.000,2,clips/x.wav,clips/y.wav\n",
    )
    asr = _write(
        tmp_path / "asr.csv",
        "scene_id,model,profile,audio,text,avg_logprob,no_speech_prob,language_probability,error\nS0001,large-v3,original_unbiased,clips/x.wav,テスト,-0.2,0.01,0.9,\n",
    )
    context = tmp_path / "context.json"
    built = build_inference_context(title="SAMPLE", structure_path=structure, source_path=source, viewer_path=viewer, hold_ledger_path=ledger, scenes_path=scenes, asr_path=asr, output_path=context)
    assert built["targets"] == 1
    payload = json.loads(context.read_text(encoding="utf-8"))
    assert payload["targets"][0]["block_number"] == 2
    assert payload["targets"][0]["asr_scene"]["profiles"][0]["text"] == "テスト"


def test_apply_inferred_recovery_is_explicitly_nonfinal(tmp_path: Path) -> None:
    structure = _write(tmp_path / "structure.srt", _srt(["前", "壊れた"]))
    source = _write(tmp_path / "source.srt", _srt(["앞", "…"]))
    viewer = _write(tmp_path / "viewer.srt", _srt(["앞", "…"]))
    ledger = _hold_ledger(tmp_path / "hold.jsonl")
    response = _write(
        tmp_path / "response.json",
        json.dumps({"results": [{
            "title": "SAMPLE", "block_number": 2, "decision": "infer-replace",
            "source_faithful_korean": "추론 복구", "viewer_natural_korean": "추론해 복구했어",
            "confidence": "medium", "basis": ["asr_same_family", "neighbor_context"], "reason": "정확한 블록 음성 전사와 문맥을 함께 사용함",
        }]}, ensure_ascii=False),
    )
    result = apply_inferred_recovery(title="SAMPLE", structure_path=structure, source_path=source, viewer_path=viewer, hold_ledger_path=ledger, response_paths=[response], output_dir=tmp_path / "v4")
    blocks, _, _ = parse_srt(Path(result["source_output"]))
    assert [block.text for block in blocks] == ["앞", "추론 복구"]
    assert result["inferred_replacement_blocks"] == 1
    assert result["final_promotion_allowed"] is False
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert report["machine_final_allowed"] is False
    assert report["human_final_allowed"] is False
