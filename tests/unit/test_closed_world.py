from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from translation_forensics.closed_world import (
    prove_quality_claim,
    run_closed_world,
    validate_closed_world_package,
)
from translation_forensics.outputs import package_title_outputs


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")
    return path


def _srt(rows: list[tuple[str, str]]) -> str:
    parts = []
    for number, (timecode, text) in enumerate(rows, 1):
        parts.append(f"{number}\n{timecode}\n{text}")
    return "\n\n".join(parts) + "\n"


def _inputs(tmp_path: Path) -> dict[str, Path]:
    times = [
        "00:00:01,000 --> 00:00:03,000",
        "00:00:04,000 --> 00:00:06,000",
        "00:00:07,000 --> 00:00:09,000",
    ]
    japanese = ["私は学校へ行きます。", "待ってるね。", "あなたは来ますか?"]
    source = ["나는 학교에 갑니다.", "기다릴게.", "당신은 와요?"]
    viewer = ["나는 학교에 갑니다.", "기다릴게.", "당신은 와요?"]
    previous = ["나는 학교에 갑니다.", "기다릴게.", "당신은 와요?"]
    structure = _write(tmp_path / "sample.structure.srt", _srt(list(zip(times, japanese))))
    ja = _write(tmp_path / "sample.ja.srt", _srt(list(zip(times, japanese))))
    source_path = _write(tmp_path / "model-a.source-faithful.srt", _srt(list(zip(times, source))))
    viewer_path = _write(tmp_path / "model-a.viewer-natural.srt", _srt(list(zip(times, viewer))))
    previous_path = _write(tmp_path / "sample.previous.srt", _srt(list(zip(times, previous))))

    for cand_path, family_name in [(source_path, "model-a/run-1"), (viewer_path, "model-b/run-2")]:
        sha = hashlib.sha256(cand_path.read_bytes()).hexdigest()
        _write(
            tmp_path / (cand_path.name + ".provenance.json"),
            json.dumps({
                "candidates": [{
                    "path": cand_path.name,
                    "sha256": sha,
                    "source_family": family_name,
                    "model": family_name.split("/")[0],
                    "run_id": family_name.split("/")[1],
                    "parent_sha256": "0" * 64,
                    "prompt_sha256": "0" * 64,
                }]
            }) + "\n",
        )

    context = {
        "scene_id": "S0001",
        "context_japanese": japanese[0],
        "blocks": [{"block_number": 1}],
        "asr_candidates": [
            {"profile": "original_unbiased", "text": japanese[0], "source_family": "whisper-family"},
            {"profile": "dialogue_unbiased", "text": japanese[0], "source_family": "whisper-family"},
        ],
    }
    review_context = _write(
        tmp_path / "review-context.jsonl",
        json.dumps(context, ensure_ascii=False) + "\n",
    )
    return {
        "structure": structure,
        "ja": ja,
        "source": source_path,
        "viewer": viewer_path,
        "previous": previous_path,
        "review_context": review_context,
    }


def _run(tmp_path: Path) -> tuple[dict[str, Path], Path, dict]:
    inputs = _inputs(tmp_path)
    output = tmp_path / "closed-world"
    result = run_closed_world(
        title="SAMPLE",
        project_root=tmp_path,
        structure_path=inputs["structure"],
        japanese_path=inputs["ja"],
        output_dir=output,
        candidate_paths=[inputs["source"], inputs["viewer"]],
        previous_path=inputs["previous"],
        review_context_path=inputs["review_context"],
    )
    return inputs, output, result


def test_closed_world_decides_every_block_and_abstains_on_ambiguity(tmp_path: Path) -> None:
    _, output, result = _run(tmp_path)
    decisions = [
        json.loads(line)
        for line in (output / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert result["decision_coverage"] == 1.0
    assert [row["status"] for row in decisions] == ["accepted", "abstained", "accepted"]
    assert "ambiguous-omitted-participant" in decisions[1]["abstention_reasons"]
    assert all(row["direct_human_listening"] is False for row in decisions)
    assert all(row["final_promotion_allowed"] is False for row in decisions)
    assert result["validation"]["status"] == "pass"


def test_closed_world_collapses_same_family_asr_and_never_claims_reference_equality(tmp_path: Path) -> None:
    inputs, output, _ = _run(tmp_path)
    first_observation = json.loads((output / "observations.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first_observation["asr"]["independent_family_count"] == 1
    assert first_observation["asr"]["same_family_duplicates_collapsed"] == 1

    proof = prove_quality_claim(inputs["structure"], output)
    assert proof["human_reference_equality"] == "unidentifiable"
    assert proof["100_percent_equal"] is False
    measured = prove_quality_claim(inputs["structure"], output, human_reference_path=inputs["viewer"])
    assert measured["human_reference_equality"] == "measured-not-proven"
    assert measured["100_percent_equal"] is False


def test_closed_world_rerun_is_byte_identical(tmp_path: Path) -> None:
    inputs, output, _ = _run(tmp_path)
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    run_closed_world(
        title="SAMPLE",
        project_root=tmp_path,
        structure_path=inputs["structure"],
        japanese_path=inputs["ja"],
        output_dir=output,
        candidate_paths=[inputs["source"], inputs["viewer"]],
        previous_path=inputs["previous"],
        review_context_path=inputs["review_context"],
    )
    after = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    assert before == after


def test_closed_world_records_explicit_model_download_permission(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "network-enabled"
    result = run_closed_world(
        title="SAMPLE",
        project_root=tmp_path,
        structure_path=inputs["structure"],
        japanese_path=inputs["ja"],
        output_dir=output,
        candidate_paths=[inputs["source"], inputs["viewer"]],
        previous_path=inputs["previous"],
        review_context_path=inputs["review_context"],
        network_model_download_permitted=True,
    )
    manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
    proof = json.loads((output / "proof.json").read_text(encoding="utf-8"))
    assert result["network_model_download_permitted"] is True
    assert manifest["network_model_download_permitted"] is True
    assert manifest["network_evidence_used"] is False
    assert proof["network_model_download_permitted"] is True
    assert proof["100_percent_equal"] is False


def test_closed_world_rejects_polarity_and_question_flip(tmp_path: Path) -> None:
    timecode = "00:00:01,000 --> 00:00:03,000"
    source = _write(tmp_path / "source.srt", _srt([(timecode, "あなたは行かないですか?")]))
    bad = _write(tmp_path / "model-b.source-faithful.srt", _srt([(timecode, "당신은 갑니다.")]))
    bad_viewer = _write(tmp_path / "model-b.viewer-natural.srt", _srt([(timecode, "당신은 가요.")]))
    previous = _write(tmp_path / "previous.srt", _srt([(timecode, "당신은 갑니다.")]))
    output = tmp_path / "closed-world"
    run_closed_world(
        title="FLIP",
        project_root=tmp_path,
        structure_path=source,
        japanese_path=source,
        output_dir=output,
        candidate_paths=[bad, bad_viewer],
        previous_path=previous,
    )
    decision = json.loads((output / "decisions.jsonl").read_text(encoding="utf-8").strip())
    assert decision["status"] == "abstained"
    observation = json.loads((output / "observations.jsonl").read_text(encoding="utf-8").strip())
    rejected = observation["candidate_translations"]
    assert all("polarity-flip" in row["validation_errors"] for row in rejected)
    assert all("question-flip" in row["validation_errors"] for row in rejected)


def test_closed_world_validation_detects_tampering(tmp_path: Path) -> None:
    inputs, output, _ = _run(tmp_path)
    decisions = (output / "decisions.jsonl").read_text(encoding="utf-8")
    (output / "decisions.jsonl").write_text(decisions.replace('"final_promotion_allowed": false', '"final_promotion_allowed": true', 1), encoding="utf-8")
    report = validate_closed_world_package(inputs["structure"], output)
    assert report["status"] == "fail"
    assert any("final 승격" in error or "해시 불일치" in error for error in report["errors"])


def test_generic_packager_cannot_masquerade_as_closed_world_stage(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    with pytest.raises(RuntimeError, match="run-closed-world"):
        package_title_outputs(
            "SAMPLE",
            inputs["structure"],
            inputs["source"],
            inputs["viewer"],
            tmp_path / "generic-output",
            stage="closed-world-validated",
            japanese_path=inputs["ja"],
        )
