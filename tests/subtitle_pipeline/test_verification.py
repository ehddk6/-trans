import json
from pathlib import Path
import subprocess
import sys

from subtitle_pipeline.pipeline import run_pipeline
from subtitle_pipeline.verification import verify_artifact_bundle


def test_verifier_accepts_complete_reference_bundle_with_video(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"not-a-real-video")
    reference = tmp_path / "reference.srt"
    reference.write_text("1\n00:00:00,000 --> 00:00:02,000\nsource text\n", encoding="utf-8")
    run_pipeline(
        video, tmp_path / "output", backend="reference", reference_srt=reference, review_samples=4,
    )

    result = verify_artifact_bundle(
        tmp_path / "output" / "reference_primary",
        reference_srt=reference,
        expected_video_path=video,
        expected_review_windows=4,
        require_source_match=True,
        require_passed_gate=True,
    )

    assert result["valid"] is True
    assert result["accepted"] is True
    assert result["source_matches_reference"] is True
    assert result["structural"] == {"empty": 0, "non_positive": 0, "overlap": 0}
    saved = (tmp_path / "output" / "reference_primary" / "bundle_verification.json").read_text(encoding="utf-8")
    assert '"accepted": true' in saved


def test_verifier_can_report_review_required_without_failing_structure(tmp_path):
    reference = tmp_path / "reference.srt"
    reference.write_text("1\n00:00:00,000 --> 00:00:00,200\nshort\n", encoding="utf-8")
    run_pipeline(None, tmp_path / "output", backend="reference", reference_srt=reference, review_samples=3)

    result = verify_artifact_bundle(tmp_path / "output" / "reference_primary", expected_review_windows=3)

    assert result["valid"] is True
    assert result["accepted"] is False
    assert result["quality_gate_status"] == "review_required"


def test_verifier_rejects_viewer_text_tampering(tmp_path):
    reference = tmp_path / "reference.srt"
    reference.write_text("1\n00:00:00,000 --> 00:00:02,000\n正しい日本語\n", encoding="utf-8")
    run_pipeline(None, tmp_path / "output", backend="reference", reference_srt=reference, review_samples=1)
    bundle = tmp_path / "output" / "reference_primary"
    (bundle / "viewer_ja.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nWRONG SUBTITLE\n", encoding="utf-8",
    )

    result = verify_artifact_bundle(
        bundle,
        reference_srt=reference,
        expected_review_windows=1,
        require_source_match=True,
        require_passed_gate=True,
    )

    assert result["valid"] is False
    assert result["accepted"] is False
    assert "viewer_audit_text_mismatch:1" in result["errors"]


def test_verifier_rejects_hangul_dominant_source_even_without_reference_assertion(tmp_path):
    reference = tmp_path / "reference.srt"
    reference.write_text("1\n00:00:00,000 --> 00:00:02,000\n正しい日本語です。\n", encoding="utf-8")
    run_pipeline(None, tmp_path / "output", backend="reference", reference_srt=reference, review_samples=1)
    bundle = tmp_path / "output" / "reference_primary"
    (bundle / "source_faithful_ja.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n이것은 일본어 원문이 아니라 잘못 들어온 한국어 번역 자막입니다.\n",
        encoding="utf-8",
    )

    result = verify_artifact_bundle(bundle, expected_review_windows=1)

    assert result["valid"] is False
    assert result["source_language"]["status"] == "incompatible_hangul_dominant"
    assert "source_language_hangul_dominant" in result["errors"]


def test_verifier_rejects_source_transcript_tampering(tmp_path):
    reference = tmp_path / "reference.srt"
    reference.write_text("1\n00:00:00,000 --> 00:00:02,000\n正しい日本語です。\n", encoding="utf-8")
    run_pipeline(None, tmp_path / "output", backend="reference", reference_srt=reference, review_samples=1)
    bundle = tmp_path / "output" / "reference_primary"
    transcript_path = bundle / "transcript_ja.jsonl"
    row = json.loads(transcript_path.read_text(encoding="utf-8"))
    row["text_raw"] = "改ざんされた文字列"
    transcript_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    result = verify_artifact_bundle(bundle, expected_review_windows=1)

    assert result["valid"] is False
    assert "source_transcript_text_mismatch:1" in result["errors"]


def test_verifier_requires_qwen_manifest_for_ensemble_bundle(tmp_path):
    reference = tmp_path / "reference.srt"
    reference.write_text("1\n00:00:00,000 --> 00:00:02,000\n正しい日本語です。\n", encoding="utf-8")
    run_pipeline(None, tmp_path / "output", backend="reference", reference_srt=reference, review_samples=1)
    bundle = tmp_path / "output" / "reference_primary"
    report_path = bundle / "qc_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["backend"] = "ensemble"
    report["qwen_verification"] = {
        "interval_count": 1,
        "audio_seconds": 2.0,
        "video_seconds": 2.0,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    result = verify_artifact_bundle(bundle, expected_review_windows=1)

    assert result["valid"] is False
    assert "missing_artifact:qwen_verification_manifest.json" in result["errors"]


def test_verifier_requires_recovery_audit_for_new_ensemble_report(tmp_path):
    reference = tmp_path / "reference.srt"
    reference.write_text("1\n00:00:00,000 --> 00:00:02,000\n正しい日本語です。\n", encoding="utf-8")
    run_pipeline(None, tmp_path / "output", backend="reference", reference_srt=reference, review_samples=1)
    bundle = tmp_path / "output" / "reference_primary"
    report_path = bundle / "qc_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["backend"] = "ensemble"
    report["qwen_verification"] = {
        "interval_count": 1,
        "audio_seconds": 2.0,
        "video_seconds": 2.0,
        "recovery_candidates": 0,
        "recovery_accepted": 0,
        "recovery_rejected": 0,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    (bundle / "qwen_verification_manifest.json").write_text(
        '[{"start":0.0,"end":2.0,"reasons":["distributed_audit"]}]', encoding="utf-8",
    )

    result = verify_artifact_bundle(bundle, expected_review_windows=1)

    assert result["valid"] is False
    assert "missing_artifact:qwen_recovery_audit.jsonl" in result["errors"]


def test_verifier_rejects_audit_identity_timing_count_and_qc_total_tampering(tmp_path):
    reference = tmp_path / "reference.srt"
    reference.write_text("1\n00:00:00,000 --> 00:00:02,000\nsource text\n", encoding="utf-8")
    run_pipeline(None, tmp_path / "output", backend="reference", reference_srt=reference, review_samples=1)
    bundle = tmp_path / "output" / "reference_primary"
    audit_path = bundle / "subtitle_audit.jsonl"
    audit_row = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_row["cue_id"] = 99
    audit_row["start"] = 0.125
    audit_path.write_text(
        json.dumps(audit_row, ensure_ascii=False) + "\n" + json.dumps(audit_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path = bundle / "qc_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["total_cues"] = 3
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    result = verify_artifact_bundle(bundle, expected_review_windows=1)

    assert result["valid"] is False
    assert {
        "viewer_audit_cue_count_mismatch",
        "viewer_audit_id_mismatch:1",
        "viewer_audit_timing_mismatch:1",
        "qc_total_cues_does_not_match_viewer",
    } <= set(result["errors"])


def test_verifier_requires_exact_reference_text_including_internal_spaces(tmp_path):
    reference = tmp_path / "reference.srt"
    reference.write_text("1\n00:00:00,000 --> 00:00:02,000\n原文  の  空白\n", encoding="utf-8")
    run_pipeline(None, tmp_path / "output", backend="reference", reference_srt=reference, review_samples=1)
    bundle = tmp_path / "output" / "reference_primary"
    (bundle / "source_faithful_ja.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n原文の空白\n", encoding="utf-8",
    )

    result = verify_artifact_bundle(
        bundle, reference_srt=reference, expected_review_windows=1, require_source_match=True,
    )

    assert result["valid"] is False
    assert result["source_matches_reference"] is False
    assert "source_text_differs_from_reference" in result["errors"]


def test_verifier_script_runs_from_its_own_scripts_directory():
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "verify_subtitle_bundle.py"), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--output-dir" in result.stdout


def test_verifier_script_writes_non_ascii_paths_directly_as_utf8(tmp_path):
    reference = tmp_path / "reference.srt"
    reference.write_text("1\n00:00:00,000 --> 00:00:02,000\n日本語\n", encoding="utf-8")
    non_ascii_root = tmp_path / "日本語 자막"
    run_pipeline(None, non_ascii_root, backend="reference", reference_srt=reference, review_samples=1)
    bundle = non_ascii_root / "reference_primary"
    destination = bundle / "standalone_verification.json"
    project_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "verify_subtitle_bundle.py"),
            "--output-dir",
            str(bundle),
            "--review-windows",
            "1",
            "--output-json",
            str(destination),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert saved["valid"] is True
    assert "日本語 자막" in saved["output_dir"]
