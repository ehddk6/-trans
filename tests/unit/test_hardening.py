from __future__ import annotations

import json
from pathlib import Path

import pytest

from translation_forensics.offline_hybrid import build_offline_hybrid
from translation_forensics.srt import parse_srt
from translation_forensics.validation import validate_pair, validate_srt_file


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _srt(rows: list[tuple[str, str, str]]) -> str:
    return "\n\n".join(
        f"{index}\n{start} --> {end}\n{text}"
        for index, (start, end, text) in enumerate(rows, 1)
    ) + "\n"


def test_validate_pair_returns_fail_for_malformed_candidate(tmp_path: Path) -> None:
    reference = _write(tmp_path / "reference.srt", _srt([("00:00:01,000", "00:00:02,000", "원문")]))
    source = _write(tmp_path / "source.srt", "not an srt\n")
    viewer = _write(tmp_path / "viewer.srt", _srt([("00:00:01,000", "00:00:02,000", "번역")]))
    result = validate_pair(reference, source, viewer)
    assert result["status"] == "fail"
    assert result["structure_same"] is False
    assert result["source_faithful"]["issues"][0]["code"] == "parse"


def test_baseline_overlap_is_warning_but_new_overlap_is_error(tmp_path: Path) -> None:
    overlap = _srt([
        ("00:00:01,000", "00:00:03,000", "하나"),
        ("00:00:02,500", "00:00:04,000", "둘"),
    ])
    baseline = _write(tmp_path / "baseline.srt", overlap)
    report = validate_srt_file(baseline)
    assert report.status == "warning"
    assert "baseline_overlap" in {issue.code for issue in report.issues}

    reference = _write(tmp_path / "reference.srt", _srt([
        ("00:00:01,000", "00:00:02,000", "하나"),
        ("00:00:02,500", "00:00:04,000", "둘"),
    ]))
    candidate_blocks, _, _ = parse_srt(baseline)
    candidate_report = validate_srt_file(baseline, reference=parse_srt(reference)[0])
    assert candidate_blocks
    assert candidate_report.status == "fail"
    assert "time_order" in {issue.code for issue in candidate_report.issues}


def test_offline_hybrid_uses_machine_source_fallback_and_writes_auditable_package(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces"
    title = "TITLE"
    title_dir = workspace / title
    structure = _srt([
        ("00:00:01,000", "00:00:02,000", "一"),
        ("00:00:03,000", "00:00:04,000", "二"),
    ])
    _write(title_dir / "inputs" / f"{title}.structure.srt", structure)
    _write(title_dir / "closed-world" / "closed-world-validated-v1" / "source-faithful.preview.srt", _srt([
        ("00:00:01,000", "00:00:02,000", "폐쇄형 하나"),
        ("00:00:03,000", "00:00:04,000", "[미확정]"),
    ]))
    _write(title_dir / "closed-world" / "closed-world-validated-v1" / "viewer-natural.preview.srt", _srt([
        ("00:00:01,000", "00:00:02,000", "자연 하나"),
        ("00:00:03,000", "00:00:04,000", "[미확정]"),
    ]))
    machine = title_dir / "runs" / "run-1" / "machine-final-v1"
    _write(machine / f"{title}.source-faithful-ko.machine-final-v1.srt", _srt([
        ("00:00:01,000", "00:00:02,000", "기계 하나"),
        ("00:00:03,000", "00:00:04,000", "기계 둘"),
    ]))
    _write(machine / f"{title}.viewer-natural-ko.machine-final-v1.srt", _srt([
        ("00:00:01,000", "00:00:02,000", "기계 자연 하나"),
        ("00:00:03,000", "00:00:04,000", "기계 자연 둘"),
    ]))
    titles = _write(tmp_path / "titles.txt", f"{title}\n")
    output = tmp_path / "output"

    results = build_offline_hybrid(titles, workspace, output)
    assert results[0]["status"] == "offline-hybrid-preview-packaged"
    package = output / title / "offline-hybrid-preview-v1"
    source_path = next(package.glob("*.source-faithful-ko.*.srt"))
    viewer_path = next(package.glob("*.viewer-complete-ko.*.srt"))
    source_blocks, encoding, newline = parse_srt(source_path)
    viewer_blocks, _, _ = parse_srt(viewer_path)
    assert [block.text for block in source_blocks] == ["폐쇄형 하나", "기계 둘"]
    assert [block.text for block in viewer_blocks] == ["자연 하나", "기계 자연 둘"]
    assert encoding in {"utf-8", "utf-8-sig"}
    assert newline == "LF"
    assert not source_path.read_bytes().startswith(b"\xef\xbb\xbf")
    validation = json.loads((package / "validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "pass"
    manifest = json.loads((package / "offline-hybrid-manifest.json").read_text(encoding="utf-8"))
    assert manifest["final_promotion_allowed"] is False

    with pytest.raises(FileExistsError):
        build_offline_hybrid(titles, workspace, output)


def test_reporting_uses_decisions_and_block_mapped_asr(tmp_path: Path) -> None:
    from translation_forensics.reporting import build_change_log, build_evidence_ledger, build_uncertainty_map

    reference_path = _write(tmp_path / "reference-reporting.srt", _srt([
        ("00:00:01,000", "00:00:02,000", "一"),
        ("00:00:03,000", "00:00:04,000", "二"),
    ]))
    japanese = parse_srt(reference_path)[0]
    source = parse_srt(_write(tmp_path / "source-reporting.srt", _srt([
        ("00:00:01,000", "00:00:02,000", "하나"),
        ("00:00:03,000", "00:00:04,000", "둘"),
    ])))[0]
    viewer = source
    scenes = _write(
        tmp_path / "scenes.csv",
        "scene_id,start_time,end_time,duration_sec,block_numbers,highest_band,original_audio,dialogue_audio,context_japanese,blocks_json\n"
        'S1,"00:00:01,000","00:00:04,000",3,1,P1,,,,[]\n',
    )
    decisions = [{
        "block_number": 1,
        "status": "accepted",
        "confidence": "high",
        "evidence_refs": ["japanese_srt", "local_asr_candidates"],
        "source_japanese": "一",
        "viewer_natural_korean": "하나",
        "review_note": "근거 일치",
    }]
    asr_rows = [{"scene_id": "S1", "profile": "original_unbiased", "text": "いち"}]

    ledger = build_evidence_ledger(
        japanese,
        status="text-crosschecked",
        asr_rows=asr_rows,
        scenes_path=scenes,
        decisions=decisions,
        japanese=japanese,
    )
    assert ledger[0]["japanese_reference"] == "一"
    assert ledger[0]["original_unbiased_asr"] == "いち"
    assert ledger[0]["decision_evidence_refs"] == "japanese_srt; local_asr_candidates"
    assert ledger[1]["original_unbiased_asr"] == "not_mapped"

    changes = build_change_log(japanese, None, source, viewer, status="text-crosschecked", japanese=japanese, decisions=decisions)
    assert changes[0]["confidence"] == "high"
    assert changes[0]["evidence_summary"] == "japanese_srt; local_asr_candidates"

    uncertainty = build_uncertainty_map(japanese, status="text-crosschecked", decisions=decisions)
    assert uncertainty[0]["uncertain_block"] == "no"
    assert uncertainty[1]["uncertain_block"] == "yes"
