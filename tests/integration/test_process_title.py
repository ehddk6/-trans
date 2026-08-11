from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from pathlib import Path

import pytest

from translation_forensics.codex_exec_provider import CodexUsageLimitError
from translation_forensics.integrated_pipeline import BundleBlockedError
from translation_forensics.process_title import (
    ProcessTitleConfig,
    choose_subtitle_backend,
    process_title,
)
from translation_forensics.srt import parse_srt
from translation_forensics.visual_context import sha256_file


process_title_module = importlib.import_module("translation_forensics.process_title")


class FakeProvider:
    def __init__(self) -> None:
        self.calls = []

    def run_structured(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["role"] == "translation-terra":
            translations = []
            for unit in kwargs["payload"]["units"]:
                ambiguous = unit["unit_id"] == "utt_000002"
                translations.append(
                    {
                        "unit_id": unit["unit_id"],
                        "source_faithful_korean": "당신은 거기 있나요?" if ambiguous else "안녕하세요.",
                        "viewer_natural_korean": "거기 있어요?" if ambiguous else "안녕하세요.",
                        "confidence": "low" if ambiguous else "high",
                        "uncertain_slots": ["addressee", "deictic_location"] if ambiguous else [],
                        "review_required_reasons": [],
                    }
                )
            return {"translations": translations}, self._receipt(kwargs)
        if kwargs["role"] == "translation-audit-sol":
            audits = []
            for unit in kwargs["payload"]["units"]:
                ambiguous = unit["unit_id"] == "utt_000002"
                audits.append(
                    {
                        "unit_id": unit["unit_id"],
                        "verdict": "unknown" if ambiguous else "pass",
                        "backtranslation_japanese": unit["source_japanese"],
                        "source_fidelity_score": 0.7 if ambiguous else 0.98,
                        "question_preserved": True,
                        "negation_preserved": True,
                        "numeric_tokens_preserved": True,
                        "reasons": ["ambiguous context"] if ambiguous else [],
                        "meaning_flip": False,
                        "uncertainty_codes": ["context_ambiguous"] if ambiguous else [],
                    }
                )
            return {"audits": audits}, self._receipt(kwargs)
        frames = kwargs["image_paths"]
        return {
            "unit_id": kwargs["payload"]["unit"]["unit_id"],
            "verdict": "keep",
            "source_faithful_korean": kwargs["payload"]["draft"]["source_faithful_korean"],
            "viewer_natural_korean": kwargs["payload"]["draft"]["viewer_natural_korean"],
            "confidence": "medium",
            "critical_visual_impact": False,
            "visual_slots": {
                "speaker": [],
                "addressee": ["화면 속 한 사람"],
                "deictic_location": ["현재 장소"],
                "on_screen_text": [],
                "scene_continuity": [],
            },
            "review_required_reasons": [],
        }, self._receipt(kwargs, frames)

    @staticmethod
    def _receipt(kwargs, frames=()):
        role = kwargs["role"]
        model, effort = {
            "translation-terra": ("gpt-5.6-terra", "high"),
            "translation-audit-sol": ("gpt-5.6-sol", "xhigh"),
            "critique-sol": ("gpt-5.6-sol", "xhigh"),
        }[role]
        attachments = [
            {
                "path": str(path),
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in frames
        ]
        return {
            "schema_name": "translation-forensics/codex-model-call-receipt",
            "schema_version": "1",
            "status": "succeeded",
            "provider": "fake-codex",
            "role": role,
            "call_id": kwargs["call_id"],
            "requested_model": model,
            "reasoning_effort": effort,
            "ephemeral": True,
            "isolated_temporary_directory": True,
            "requested_model_verified_by_cli_invocation": True,
            "model_call_verified": True,
            "api_key_used": False,
            "sandbox": "read-only",
            "exit_code": 0,
            "external_transfer": bool(attachments),
            "pixel_external_transfer_count": len(attachments),
            "image_attachments": attachments,
            "thread_id": "test-thread",
            "codex_cli_version": "test-codex",
            "request_sha256": "request",
            "prompt_sha256": "prompt",
            "evidence_sha256": "evidence",
            "schema_sha256": "schema",
            "response_sha256": "response",
        }


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _make_existing_bundle(
    tmp_path: Path, *, accepted: bool = False, media_suffix: str = ".mp4"
) -> tuple[Path, Path, Path]:
    media = tmp_path / f"ADN-622{media_suffix}"
    media.write_bytes(b"test-media")
    bundle = tmp_path / "ADN-622" / "ensemble_qwen_whisper"
    bundle.mkdir(parents=True)
    source = (
        "1\n00:00:00,000 --> 00:00:01,000\nこんにちは。\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nあなたはそこ？\n"
    )
    (bundle / "source_faithful_ja.srt").write_text(source, encoding="utf-8")
    (bundle / "viewer_ja.srt").write_text(source, encoding="utf-8")
    rows = [
        {
            "utterance_id": "utt_000001",
            "source_segment_ids": [1],
            "start": 0.0,
            "end": 1.0,
            "text_raw": "こんにちは。",
            "text_normalized": "こんにちは。",
            "words": [],
            "speaker": "speaker_unknown",
            "asr_metrics": {},
            "warnings": [],
        },
        {
            "utterance_id": "utt_000002",
            "source_segment_ids": [2],
            "start": 1.0,
            "end": 2.0,
            "text_raw": "あなたはそこ？",
            "text_normalized": "あなたはそこ？",
            "words": [],
            "speaker": "speaker_unknown",
            "asr_metrics": {},
            "warnings": ["review_required"],
        },
    ]
    (bundle / "transcript_ja.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    audit_rows = [
        {"cue_id": 1, "start": 0.0, "end": 1.0, "text_raw": "こんにちは。"},
        {"cue_id": 2, "start": 1.0, "end": 2.0, "text_raw": "あなたはそこ？"},
    ]
    (bundle / "subtitle_audit.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows),
        encoding="utf-8",
    )
    gate_reasons = [] if accepted else [{"code": "possible_repetition", "count": 1}]
    _write_json(
        bundle / "qc_report.json",
        {
            "backend": "reference",
            "total_cues": 2,
            "empty_cue_count": 0,
            "timecode_reversal_count": 0,
            "timecode_overlap_count": 0,
            "validation_errors": [],
            "content_quality_gate": {
                "status": "passed" if accepted else "review_required",
                "reasons": gate_reasons,
            },
        },
    )
    (bundle / "comparison_report.html").write_text("<html></html>", encoding="utf-8")
    _write_json(
        bundle / "review_manifest.json",
        {
            "video_path": str(media.resolve()),
            "windows": [
                {"start": 0.0, "end": 1.0, "duration": 1.0},
                {"start": 1.0, "end": 2.0, "duration": 1.0},
            ],
        },
    )
    (bundle / "review_report.html").write_text(
        "이 검수 구간 영상 재생\n이 검수 구간 영상 재생\n", encoding="utf-8"
    )
    _write_json(
        bundle / "bundle_verification.json",
        {
            "valid": True,
            "accepted": accepted,
            "quality_gate_status": "review_required" if not accepted else "passed",
            "quality_gate_reasons": gate_reasons,
            "errors": [],
        },
    )
    captures = tmp_path / "ADN-622" / "timestamp_frames"
    captures.mkdir()
    (captures / "sub_0001_0.500s.jpg").write_bytes(b"one-frame")
    return media, bundle, captures


def test_process_title_packages_complete_draft_but_holds_unaccepted_bundle_candidates(tmp_path):
    media, bundle, captures = _make_existing_bundle(tmp_path, accepted=False)
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in bundle.iterdir() if path.is_file()}
    provider = FakeProvider()
    result = process_title(
        ProcessTitleConfig(
            project_root=tmp_path / "project",
            title_id="ADN-622",
            media=media,
            japanese_bundle=bundle,
            legacy_captures=captures,
            visual_policy="targeted",
            max_visual_units=2,
            resume=True,
        ),
        provider=provider,
    )

    run_dir = Path(result["run_dir"])
    complete, _, _ = parse_srt(run_dir / "outputs" / "viewer_complete_ko.srt")
    faithful, _, _ = parse_srt(run_dir / "outputs" / "source_faithful_ko.srt")
    assert len(complete) == 2 and all(block.text for block in complete)
    assert [block.text for block in faithful] == ["[검수 보류]", "[검수 보류]"]
    assert result["candidate_units"] == 0
    assert result["external_image_transfer_count"] == 1
    assert (run_dir / "translation-input" / "translation_ja.srt").is_file()
    assert (run_dir.parent / "latest.json").is_file()
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in bundle.iterdir() if path.is_file()} == before
    visual = [json.loads(line) for line in (run_dir / "visual_context.jsonl").read_text(encoding="utf-8").splitlines()]
    assert visual[0]["external_transfer_receipt"]["pixel_transfer_count"] == 1
    assert "source_suspect" in visual[0]["unit_selection_reasons"]
    assert visual[0]["requested_visual_slots"] == ["addressee", "deictic_location", "speaker"]

    cached = process_title(
        ProcessTitleConfig(
            project_root=tmp_path / "project",
            title_id="ADN-622",
            media=media,
            japanese_bundle=bundle,
            legacy_captures=captures,
            visual_policy="targeted",
            max_visual_units=2,
            resume=True,
        ),
        provider=provider,
    )
    assert cached["cache_hit"] is True
    assert cached["run_dir"] == str(run_dir)


def test_automated_quality_policy_replaces_human_holds_with_safe_fallback(tmp_path):
    media, bundle, _ = _make_existing_bundle(tmp_path, accepted=False)
    result = process_title(
        ProcessTitleConfig(
            project_root=tmp_path / "project",
            title_id="ADN-622",
            media=media,
            japanese_bundle=bundle,
            visual_policy="off",
            quality_policy="automated",
            resume=True,
        ),
        provider=FakeProvider(),
    )

    run_dir = Path(result["run_dir"])
    complete, _, _ = parse_srt(run_dir / "outputs" / "viewer_complete_ko.srt")
    faithful, _, _ = parse_srt(run_dir / "outputs" / "source_faithful_ko.srt")
    assert result["quality_policy"] == "automated"
    assert result["pending_review_count"] == 0
    assert result["candidate_units"] == 2
    assert result["status"] == "machine-uncertain"
    assert all("검수 보류" not in block.text for block in complete + faithful)
    assert faithful[1].text == "당신은 거기 있나요?"
    audits = [
        json.loads(line)
        for line in (run_dir / "automated_quality.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(audits) == 2
    assert {row["status"] for row in audits} == {"passed", "fallback"}
    assert all(row["selected_sha256"] for row in audits)
    assert all(row["render_binding"] == "unit_order_and_unit_id" for row in audits)


def test_metadata_visual_policy_never_transfers_pixels(tmp_path):
    media, bundle, captures = _make_existing_bundle(tmp_path)
    provider = FakeProvider()
    result = process_title(
        ProcessTitleConfig(
            project_root=tmp_path / "project",
            title_id="ADN-622",
            media=media,
            japanese_bundle=bundle,
            legacy_captures=captures,
            visual_policy="metadata",
            resume=True,
        ),
        provider=provider,
    )
    records = [
        json.loads(line)
        for line in (Path(result["run_dir"]) / "visual_context.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert result["external_image_transfer_count"] == 0
    assert records[0]["external_transfer_receipt"]["pixel_transfer_count"] == 0
    assert all(call["role"] != "critique-sol" for call in provider.calls)


def test_photo_less_title_auto_captures_selected_visual_units_and_indexes_them(tmp_path, monkeypatch):
    media, bundle, _ = _make_existing_bundle(tmp_path)
    generated = tmp_path / "generated-frame.jpg"
    generated.write_bytes(b"generated-frame")

    def fake_extract(media_path, output_dir, unit, max_frames):
        assert media_path == media.resolve()
        assert max_frames == 3
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "frame_01_1.000s.jpg"
        target.write_bytes(generated.read_bytes())
        return [process_title_module.VisualFrame(target.resolve(), unit.start, sha256_file(target))]

    monkeypatch.setattr(process_title_module, "_extract_unit_frames", fake_extract)
    result = process_title(
        ProcessTitleConfig(
            project_root=tmp_path / "project",
            title_id="ADN-622",
            media=media,
            japanese_bundle=bundle,
            legacy_captures=None,
            visual_policy="targeted",
            max_visual_units=2,
            resume=True,
        ),
        provider=FakeProvider(),
    )

    run_dir = Path(result["run_dir"])
    capture_rows = [
        json.loads(line)
        for line in (run_dir / "capture_index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert capture_rows and capture_rows[0]["source"] == "auto-generated"
    assert capture_rows[0]["sha256"] == sha256_file(generated)
    assert result["visual_capture_mode"] == "auto-generated"
    manifest = json.loads((run_dir / "input_manifest.json").read_text(encoding="utf-8"))
    assert manifest["visual_capture"]["mode"] == "legacy_or_auto_generated"


def test_reference_requires_explicit_approval_and_missing_reference_uses_ensemble(tmp_path):
    reference = tmp_path / "ADN-622.ja.srt"
    reference.write_text("1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n", encoding="utf-8")
    with pytest.raises(ValueError, match="explicit approval"):
        choose_subtitle_backend(reference_ja=reference, reference_ja_approved=False, japanese_bundle=None)
    assert choose_subtitle_backend(reference_ja=reference, reference_ja_approved=True, japanese_bundle=None) == "reference"
    assert choose_subtitle_backend(reference_ja=None, reference_ja_approved=False, japanese_bundle=None) == "ensemble"


def test_visual_quota_block_still_packages_machine_draft(tmp_path):
    media, bundle, captures = _make_existing_bundle(tmp_path)

    class LimitedProvider(FakeProvider):
        def run_structured(self, **kwargs):
            if kwargs["role"] == "critique-sol":
                raise CodexUsageLimitError(
                    "limited",
                    role="critique-sol",
                    call_id=kwargs["call_id"],
                    retry_after="2026-08-16T14:05:00+09:00",
                )
            return super().run_structured(**kwargs)

    result = process_title(
        ProcessTitleConfig(
            project_root=tmp_path / "project",
            title_id="ADN-622",
            media=media,
            japanese_bundle=bundle,
            legacy_captures=captures,
            visual_policy="targeted",
            resume=True,
        ),
        provider=LimitedProvider(),
    )
    run_dir = Path(result["run_dir"])
    assert (run_dir / "outputs" / "viewer_complete_ko.srt").is_file()
    assert result["visual_status_counts"] == {"blocked": 1}
    visual = json.loads((run_dir / "visual_context.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert visual["external_transfer_receipt"]["external_transfer"] is False
    assert visual["external_transfer_receipt"]["pixel_transfer_count"] == 0


def test_review_decisions_are_snapshotted_and_change_the_run_identity(tmp_path):
    media, bundle, _ = _make_existing_bundle(tmp_path, accepted=False)
    review_path = tmp_path / "ADN-622.review-decisions.jsonl"
    rows = [
        {
            "unit_id": "utt_000001",
            "status": "approved",
            "reviewer": "human-a",
            "reason": "direct listening confirmed the unit",
            "previous_status": "pending",
            "decided_at": "2026-08-10T00:00:00+00:00",
        }
    ]
    review_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    config = ProcessTitleConfig(
        project_root=tmp_path / "project",
        title_id="ADN-622",
        media=media,
        japanese_bundle=bundle,
        visual_policy="off",
        review_decisions=review_path,
        resume=True,
    )
    first = process_title(config, provider=FakeProvider())
    first_dir = Path(first["run_dir"])
    assert first["candidate_units"] == 1
    assert [
        json.loads(line)
        for line in (first_dir / "review_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ] == rows

    rows.append(
        {
            "unit_id": "utt_000002",
            "status": "approved",
            "reviewer": "human-a",
            "reason": "direct listening confirmed the warning",
            "previous_status": "pending",
            "decided_at": "2026-08-10T00:01:00+00:00",
        }
    )
    review_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    second = process_title(config, provider=FakeProvider())
    assert second["candidate_units"] == 2
    assert second["run_dir"] != first["run_dir"]


def test_resume_rejects_a_changed_promoted_artifact(tmp_path):
    media, bundle, _ = _make_existing_bundle(tmp_path)
    config = ProcessTitleConfig(
        project_root=tmp_path / "project",
        title_id="ADN-622",
        media=media,
        japanese_bundle=bundle,
        visual_policy="off",
        resume=True,
    )
    result = process_title(config, provider=FakeProvider())
    output = Path(result["run_dir"]) / "outputs" / "viewer_complete_ko.srt"
    output.write_text(output.read_text(encoding="utf-8") + "tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact changed"):
        process_title(config, provider=FakeProvider())


def test_partial_resume_revalidates_translation_input_content(tmp_path):
    media, bundle, _ = _make_existing_bundle(tmp_path)
    config = ProcessTitleConfig(
        project_root=tmp_path / "project",
        title_id="ADN-622",
        media=media,
        japanese_bundle=bundle,
        visual_policy="off",
        resume=True,
    )

    class FailingProvider:
        def run_structured(self, **kwargs):
            raise RuntimeError("stop after translation input")

    with pytest.raises(RuntimeError, match="stop after translation input"):
        process_title(config, provider=FailingProvider())
    partials = list((tmp_path / "project" / "workspaces" / "ADN-622" / "integrated" / ".partial").iterdir())
    assert len(partials) == 1
    translation_srt = partials[0] / "translation-input" / "translation_ja.srt"
    translation_srt.write_text(
        translation_srt.read_text(encoding="utf-8").replace("こんにちは。", "改変しました。"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no longer matches"):
        process_title(config, provider=FakeProvider())


def test_process_title_rejects_cross_title_reference(tmp_path):
    media = tmp_path / "ABP-169.mp4"
    media.write_bytes(b"test-media")
    reference = tmp_path / "ABF-169.ja.srt"
    reference.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exact title code ABP-169"):
        process_title(
            ProcessTitleConfig(
                project_root=tmp_path / "project",
                title_id="ABP-169",
                media=media,
                reference_ja=reference,
                reference_ja_approved=True,
                visual_policy="off",
            ),
            provider=FakeProvider(),
        )

    mixed_root = tmp_path / "ABP-169"
    mixed_root.mkdir()
    mixed_reference = mixed_root / "ABF-169.ja.srt"
    mixed_reference.write_text(reference.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="exact title code ABP-169"):
        process_title(
            ProcessTitleConfig(
                project_root=tmp_path / "mixed-project",
                title_id="ABP-169",
                media=media,
                reference_ja=mixed_reference,
                reference_ja_approved=True,
                visual_policy="off",
            ),
            provider=FakeProvider(),
        )


def test_viewer_complete_japanese_residual_fails_integrated_qa(tmp_path):
    media, bundle, _ = _make_existing_bundle(tmp_path)

    class JapaneseResidualProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            if kwargs["role"] == "translation-terra":
                for row in response["translations"]:
                    row["source_faithful_korean"] = "こんにちは"
                    row["viewer_natural_korean"] = "こんにちは"
            return response, receipt

    with pytest.raises(BundleBlockedError, match="viewer_complete_contains_japanese_residual"):
        process_title(
            ProcessTitleConfig(
                project_root=tmp_path / "project",
                title_id="ADN-622",
                media=media,
                japanese_bundle=bundle,
                visual_policy="off",
                resume=True,
            ),
            provider=JapaneseResidualProvider(),
        )


def test_existing_bundle_is_independently_reverified_and_bound_to_media(tmp_path):
    media, bundle, _ = _make_existing_bundle(tmp_path)
    transcript = bundle / "transcript_ja.jsonl"
    transcript.write_text(
        transcript.read_text(encoding="utf-8").replace("こんにちは。", "改変しました。", 1),
        encoding="utf-8",
    )
    with pytest.raises(BundleBlockedError, match="source_transcript_text_mismatch"):
        process_title(
            ProcessTitleConfig(
                project_root=tmp_path / "project",
                title_id="ADN-622",
                media=media,
                japanese_bundle=bundle,
                visual_policy="off",
            ),
            provider=FakeProvider(),
        )

    other_media = tmp_path / "ADN-622-copy.mp4"
    other_media.write_bytes(media.read_bytes())
    transcript.write_text(
        transcript.read_text(encoding="utf-8").replace("改変しました。", "こんにちは。", 1),
        encoding="utf-8",
    )
    with pytest.raises(BundleBlockedError, match="review_video_path_mismatch"):
        process_title(
            ProcessTitleConfig(
                project_root=tmp_path / "other-project",
                title_id="ADN-622",
                media=other_media,
                japanese_bundle=bundle,
                visual_policy="off",
            ),
            provider=FakeProvider(),
        )


def test_existing_bundle_verification_accepts_mp3_media(tmp_path):
    media, bundle, _ = _make_existing_bundle(tmp_path, media_suffix=".mp3")
    result = process_title(
        ProcessTitleConfig(
            project_root=tmp_path / "project",
            title_id="ADN-622",
            media=media,
            japanese_bundle=bundle,
            visual_policy="off",
            resume=True,
        ),
        provider=FakeProvider(),
    )
    assert Path(result["run_dir"]).is_dir()


def test_model_receipts_are_fail_closed_and_linked_to_decisions(tmp_path):
    media, bundle, _ = _make_existing_bundle(tmp_path)

    class InvalidReceiptProvider(FakeProvider):
        def run_structured(self, **kwargs):
            response, receipt = super().run_structured(**kwargs)
            receipt.pop("schema_name", None)
            return response, receipt

    with pytest.raises(BundleBlockedError, match="invalid_model_call_receipt"):
        process_title(
            ProcessTitleConfig(
                project_root=tmp_path / "project",
                title_id="ADN-622",
                media=media,
                japanese_bundle=bundle,
                visual_policy="off",
                resume=True,
            ),
            provider=InvalidReceiptProvider(),
        )


def test_reference_process_path_invokes_canonical_subtitle_runner_settings(tmp_path):
    media, template_bundle, _ = _make_existing_bundle(tmp_path, accepted=True)
    reference = tmp_path / "ADN-622.ja.srt"
    reference.write_text(
        (template_bundle / "source_faithful_ja.srt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    captured = {}

    def recording_runner(input_path, output_dir, **kwargs):
        captured["input_path"] = input_path
        captured["output_dir"] = output_dir
        captured["kwargs"] = kwargs
        shutil.copytree(template_bundle, output_dir / "reference_primary")
        return {"status": "created-by-test-runner"}

    result = process_title(
        ProcessTitleConfig(
            project_root=tmp_path / "project",
            title_id="ADN-622",
            media=media,
            reference_ja=reference,
            reference_ja_approved=True,
            visual_policy="off",
            resume=True,
        ),
        provider=FakeProvider(),
        subtitle_runner=recording_runner,
    )

    assert Path(result["run_dir"]).is_dir()
    assert captured["input_path"] == media.resolve()
    assert captured["kwargs"]["backend"] == "reference"
    assert captured["kwargs"]["reference_srt"] == reference.resolve()
    assert captured["kwargs"]["model_name"] == "large-v3-turbo"
    assert captured["kwargs"]["language"] == "ja"
    assert captured["kwargs"]["vad"] is False
    assert captured["kwargs"]["condition_on_previous_text"] is False
    assert captured["kwargs"]["normalized_audio_path"] is None
