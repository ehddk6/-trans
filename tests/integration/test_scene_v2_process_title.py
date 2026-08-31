from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest

from subtitle_pipeline.media import build_media_binding
from translation_forensics.cli import build_parser
from translation_forensics.codex_exec_provider import ROLE_POLICY
from translation_forensics.process_title import ProcessTitleConfig, process_title
from translation_forensics.srt import parse_srt


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _make_existing_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    media = tmp_path / "TEST-001.wav"
    with wave.open(str(media), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8_000)
        handle.writeframes(b"\x00\x00" * 24_000)
    bundle = tmp_path / "TEST-001" / "ensemble_qwen_whisper"
    bundle.mkdir(parents=True)
    source = (
        "1\n00:00:00,000 --> 00:00:01,000\n行く？\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nうん。\n"
    )
    for name in ("source_faithful_ja.srt", "viewer_ja.srt"):
        (bundle / name).write_text(source, encoding="utf-8")
    transcript = [
        {
            "utterance_id": "utt_000001",
            "source_segment_ids": [1],
            "start": 0.0,
            "end": 1.0,
            "text_raw": "行く？",
            "text_normalized": "行く？",
            "words": [],
            "speaker": "speaker_a",
            "asr_metrics": {},
            "warnings": [],
        },
        {
            "utterance_id": "utt_000002",
            "source_segment_ids": [2],
            "start": 1.0,
            "end": 2.0,
            "text_raw": "うん。",
            "text_normalized": "うん。",
            "words": [],
            "speaker": "speaker_b",
            "asr_metrics": {},
            "warnings": [],
        },
    ]
    (bundle / "transcript_ja.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in transcript),
        encoding="utf-8",
    )
    (bundle / "subtitle_audit.jsonl").write_text(
        "".join(
            json.dumps(
                {"cue_id": index, "start": row["start"], "end": row["end"], "text_raw": row["text_raw"]},
                ensure_ascii=False,
            )
            + "\n"
            for index, row in enumerate(transcript, 1)
        ),
        encoding="utf-8",
    )
    _write_json(
        bundle / "qc_report.json",
        {
            "backend": "reference",
            "total_cues": 2,
            "empty_cue_count": 0,
            "timecode_reversal_count": 0,
            "timecode_overlap_count": 0,
            "validation_errors": [],
            "content_quality_gate": {"status": "passed", "reasons": []},
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
    _write_json(bundle / "source_media.json", build_media_binding(media))
    _write_json(
        bundle / "bundle_verification.json",
        {
            "valid": True,
            "accepted": True,
            "quality_gate_status": "passed",
            "quality_gate_reasons": [],
            "errors": [],
        },
    )
    captures = tmp_path / "TEST-001" / "timestamp_frames"
    captures.mkdir()
    # A structurally valid JPEG is enough because the fake provider never opens it.
    (captures / "sub_0001_0.500s.jpg").write_bytes(b"\xff\xd8fake\xff\xd9")
    return media, bundle, captures


def _semantic_frame(
    scene_id: str, unit: dict[str, object], *, visual_trigger: bool = False
) -> dict[str, object]:
    question = str(unit["unit_id"]) == "utt_000001"
    return {
        "schema_version": "1",
        "scene_id": scene_id,
        "unit_id": unit["unit_id"],
        "utterance_type": "dialogue",
        "semantic_summary": "질문" if question else "긍정 응답",
        "speech_act": "question" if question else "response",
        "question": "yes" if question else "no",
        "polarity": "positive",
        "refusal_permission": "neither",
        "stop_continue": "neither",
        "command_strength": "none",
        "speaker": unit.get("speaker"),
        "addressee": None,
        "actor": None,
        "action": "가다" if question else "동의하다",
        "target": None,
        "location": None,
        "direction": None,
        "tense_aspect": "present",
        "completion": "unknown",
        "intensity": None,
        "numeric_tokens": [],
        "register": "informal",
        "response_to_unit_id": "utt_000001" if not question else None,
        "continues_from_unit_id": None,
        "continues_to_unit_id": None,
        "must_preserve": ["polarity"],
        "uncertain_slots": ["speaker"] if visual_trigger and question else [],
        "competing_interpretations": ["화면 왼쪽", "화면 오른쪽"] if visual_trigger and question else [],
        "visual_resolvable_slots": ["speaker"] if visual_trigger and question else [],
        "expected_translation_delta": "major" if visual_trigger and question else "none",
        "audio_only_uncertainty": False,
        "confidence": "high",
        "evidence_refs": list(unit.get("evidence_ids", [])) or ["source"],
    }


class FakeSceneProvider:
    def __init__(
        self, *, visual_trigger: bool = False, dialogue_issue_then_repair: bool = False
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.visual_trigger = visual_trigger
        self.dialogue_issue_then_repair = dialogue_issue_then_repair

    def run_structured(self, **kwargs):
        self.calls.append(kwargs)
        role = kwargs["role"]
        call_id = kwargs["call_id"]
        scene = kwargs["payload"]["scene"]
        units = scene["units"]
        scene_id = scene["scene_id"]
        if role == "meaning-frame-terra":
            response = {
                "frames": [
                    _semantic_frame(scene_id, unit, visual_trigger=self.visual_trigger)
                    for unit in units
                ]
            }
        elif role == "meaning-frame-sol":
            unit_id = kwargs["payload"]["unit_id"]
            response = {
                "observations": [
                    {
                        "scene_id": scene_id,
                        "unit_id": unit_id,
                        "observed_slots": [
                            {"slot": "speaker", "observation": "화면 왼쪽 인물"}
                        ],
                        "unresolved_slots": [],
                        "confidence": "high",
                        "frame_evidence_refs": ["selected-frame"],
                        "unsupported_inference_warnings": [],
                    }
                ]
            }
        elif role == "translation-terra" and "dialogue-realization" in call_id:
            response = {
                "realizations": [
                    {
                        "scene_id": scene_id,
                        "turn_id": f"turn-{unit['unit_id']}",
                        "source_unit_ids": [unit["unit_id"]],
                        "speaker_id": unit.get("speaker"),
                        "korean": "갈래?" if unit["unit_id"] == "utt_000001" else "응.",
                        "relation_to_previous": "new" if unit["unit_id"] == "utt_000001" else "response",
                        "segmentation_hint": "single_cue",
                    }
                    for unit in units
                ]
            }
        elif role == "translation-terra" and "subtitle-segmentation" in call_id:
            response = {
                "projections": [
                    {
                        "scene_id": scene_id,
                        "unit_id": turn["source_unit_ids"][0],
                        "text": turn["korean"],
                        "source_turn_ids": [turn["turn_id"]],
                        "projection_notes": [],
                    }
                    for turn in kwargs["payload"]["realizations"]
                ]
            }
        elif role == "translation-terra" and "source-faithful" in call_id:
            response = {
                "translations": [
                    {
                        "scene_id": scene_id,
                        "unit_id": unit["unit_id"],
                        "source_faithful_korean": (
                            "갈 거야?" if unit["unit_id"] == "utt_000001" else "응."
                        ),
                    }
                    for unit in units
                ]
            }
        elif role == "translation-audit-sol":
            response = {"audits": []}
        elif role == "dialogue-critic-sol":
            initial = call_id.endswith(".initial")
            response = {
                "audits": (
                    [
                        {
                            "scene_id": scene_id,
                            "unit_ids": ["utt_000002"],
                            "target_span": "응.",
                            "category": "response_mismatch",
                            "severity": "major",
                            "reason": "장면 응답을 더 분명하게 연결해야 함",
                            "repair_scope": "dialogue_text",
                        }
                    ]
                    if self.dialogue_issue_then_repair and initial
                    else []
                )
            }
        elif role == "repair-terra":
            response = {
                "repair": [
                    {
                        "scene_id": scene_id,
                        "turn_id": "repair-utt_000002",
                        "affected_unit_ids": ["utt_000002"],
                        "korean": "그래.",
                        "repair_scope": "dialogue_text",
                        "issue_ids": ["dialogue-major-1"],
                    }
                ]
            }
        else:
            raise AssertionError((role, call_id))
        return response, self._receipt(kwargs, response)

    @staticmethod
    def _receipt(kwargs: dict[str, object], response: dict[str, object]) -> dict[str, object]:
        role = str(kwargs["role"])
        model, effort = ROLE_POLICY[role]
        image_paths = [Path(path) for path in kwargs.get("image_paths", [])]
        attachments = [
            {
                "path": str(path),
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in image_paths
        ]
        return {
            "schema_name": "translation-forensics/codex-model-call-receipt",
            "schema_version": "1",
            "status": "succeeded",
            "title_id": kwargs["title_id"],
            "role": role,
            "call_id": kwargs["call_id"],
            "requested_model": model,
            "reasoning_effort": effort,
            "thread_id": "fake-thread",
            "codex_cli_version": "fake-codex",
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
            "request_sha256": hashlib.sha256(str(kwargs["call_id"]).encode()).hexdigest(),
            "prompt_sha256": "0" * 64,
            "evidence_sha256": "1" * 64,
            "schema_sha256": "2" * 64,
            "response_sha256": hashlib.sha256(
                json.dumps(response, sort_keys=True).encode()
            ).hexdigest(),
            "cache_hit": False,
        }


@pytest.mark.parametrize("visual_policy", ["off", "metadata"])
def test_scene_v2_process_title_end_to_end_and_resume_without_pixels(
    tmp_path: Path, visual_policy: str
) -> None:
    media, bundle, captures = _make_existing_bundle(tmp_path)
    provider = FakeSceneProvider()
    config = ProcessTitleConfig(
        project_root=Path(__file__).resolve().parents[2],
        title_id="TEST-001",
        media=media,
        japanese_bundle=bundle,
        legacy_captures=captures,
        translation_architecture="scene_v2",
        visual_policy=visual_policy,
        output_root=tmp_path / f"runs-{visual_policy}",
        resume=True,
    )
    result = process_title(config, provider=provider)
    run_dir = Path(result["run_dir"])

    assert result["translation_architecture"] == "scene_v2"
    assert result["architecture_status"] == "experimental-unbenchmarked"
    assert result["benchmark_status"] == "not-run"
    assert result["cache_identity"]["options"]["translation_architecture"] == "scene_v2"
    assert result["external_image_transfer_count"] == 0
    assert all(call.get("image_paths", []) == [] for call in provider.calls)
    for name in result["scene_artifacts"]:
        assert (run_dir / name).is_file()
        assert name in result["artifact_sha256"]
    assert (run_dir / "model_call_receipts.jsonl").is_file()

    original, _, _ = parse_srt(run_dir / "translation-input" / "translation_ja.srt")
    natural, encoding, newline = parse_srt(run_dir / "outputs" / "viewer_natural_ko.srt")
    assert [(row.number, row.start, row.end) for row in natural] == [
        (row.number, row.start, row.end) for row in original
    ]
    assert [row.text for row in natural] == ["갈래?", "응."]
    assert encoding == "utf-8"
    assert newline == "LF"

    call_count = len(provider.calls)
    resumed = process_title(config, provider=provider)
    assert resumed["cache_hit"] is True
    assert resumed["run_dir"] == str(run_dir)
    assert len(provider.calls) == call_count


def test_scene_v2_cli_and_model_policy_contracts() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = build_parser()
    args = parser.parse_args(
        [
            "process-title",
            "--title",
            "TEST-001",
            "--media",
            "TEST-001.mp4",
            "--translation-architecture",
            "scene_v2",
            "--naturalness-repair-attempts",
            "2",
            "--semantic-audit-scope",
            "targeted",
            "--dialogue-memory-policy",
            "provisional-style-only",
        ]
    )
    assert args.translation_architecture == "scene_v2"
    assert args.naturalness_repair_attempts == 2
    assert args.semantic_audit_scope == "targeted"
    assert args.dialogue_memory_policy == "provisional-style-only"
    assert ROLE_POLICY["dialogue-critic-sol"] == ("gpt-5.6-sol", "xhigh")
    model_config = json.loads(
        (project_root / "config" / "translation-model.json").read_text(encoding="utf-8")
    )
    assert model_config["allowed_translation_models"] == ["gpt-5.6-terra"]
    assert model_config["default_translation_architecture"] == "block_v1"
    assert model_config["allowed_translation_architectures"] == ["block_v1", "scene_v2"]
    receipt_schema = json.loads(
        (project_root / "schemas" / "codex-model-call-receipt.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert "dialogue-critic-sol" in receipt_schema["properties"]["role"]["enum"]
    assert ProcessTitleConfig(
        project_root=Path("."), title_id="X", media=Path("X.mp4")
    ).translation_architecture == "block_v1"
    with pytest.raises(ValueError, match="scene_gap_threshold_seconds"):
        ProcessTitleConfig(
            project_root=Path("."),
            title_id="X",
            media=Path("X.mp4"),
            translation_architecture="scene_v2",
            scene_gap_threshold_seconds=0,
        ).validate()


@pytest.mark.parametrize(
    "argv",
    [
        ["init-scene-benchmark"],
        ["ingest-external-baseline", "--input", "in.json", "--output", "out.json"],
        [
            "build-scene-blind-review-pack",
            "--block-v1",
            "block.jsonl",
            "--scene-v2",
            "scene.jsonl",
            "--seed",
            "7",
            "--output",
            "review.zip",
        ],
        ["validate-scene-review", "--pack", "review.zip", "--review", "review.json"],
        [
            "summarize-scene-benchmark",
            "--pack",
            "review.zip",
            "--review",
            "review.json",
            "--internal-key",
            "key.json",
            "--output",
            "summary.json",
        ],
        [
            "build-visual-ablation-manifest",
            "--experiment-id",
            "visual-1",
            "--scene-id",
            "scene-1",
            "--output",
            "manifest.json",
        ],
        ["validate-visual-ablation-result", "--input", "result.json"],
        [
            "summarize-visual-ablation",
            "--manifest",
            "manifest.json",
            "--result",
            "result.json",
            "--output",
            "summary.json",
        ],
    ],
)
def test_scene_evaluation_cli_commands_are_registered(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    assert callable(args.func)


def test_scene_v2_targeted_visual_transfers_only_exact_trigger(tmp_path: Path) -> None:
    media, bundle, captures = _make_existing_bundle(tmp_path)
    provider = FakeSceneProvider(visual_trigger=True)
    result = process_title(
        ProcessTitleConfig(
            project_root=Path(__file__).resolve().parents[2],
            title_id="TEST-001",
            media=media,
            japanese_bundle=bundle,
            legacy_captures=captures,
            translation_architecture="scene_v2",
            visual_policy="targeted",
            output_root=tmp_path / "runs-targeted",
            resume=True,
        ),
        provider=provider,
    )
    run_dir = Path(result["run_dir"])
    visual_calls = [call for call in provider.calls if call["role"] == "meaning-frame-sol"]
    assert len(visual_calls) == 1
    assert visual_calls[0]["payload"]["unit_id"] == "utt_000001"
    assert len(visual_calls[0]["image_paths"]) == 1
    assert all(
        not call.get("image_paths")
        for call in provider.calls
        if call["role"] != "meaning-frame-sol"
    )
    assert result["external_image_transfer_count"] == 1
    assert "visual_semantic_observations.jsonl" in result["scene_artifacts"]
    visual_rows = [
        json.loads(line)
        for line in (run_dir / "visual_context.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    transferred = [
        row
        for row in visual_rows
        if row["external_transfer_receipt"]["external_transfer"] is True
    ]
    assert [row["unit_id"] for row in transferred] == ["utt_000001"]
    assert transferred[0]["external_transfer_receipt"]["pixel_transfer_count"] == 1


def test_scene_v2_process_title_runs_targeted_repair_and_reaudits(tmp_path: Path) -> None:
    media, bundle, _ = _make_existing_bundle(tmp_path)
    provider = FakeSceneProvider(dialogue_issue_then_repair=True)
    result = process_title(
        ProcessTitleConfig(
            project_root=Path(__file__).resolve().parents[2],
            title_id="TEST-001",
            media=media,
            japanese_bundle=bundle,
            translation_architecture="scene_v2",
            visual_policy="off",
            naturalness_repair_attempts=1,
            output_root=tmp_path / "runs-repair",
            resume=True,
        ),
        provider=provider,
    )
    run_dir = Path(result["run_dir"])
    natural, _, _ = parse_srt(run_dir / "outputs" / "viewer_natural_ko.srt")
    assert [row.text for row in natural] == ["갈래?", "그래."]
    assert result["scene_qa"]["repair_count"] == 1
    assert any(call["role"] == "repair-terra" for call in provider.calls)
    assert any(
        call["role"] == "dialogue-critic-sol" and "repair-1" in call["call_id"]
        for call in provider.calls
    )
    repair_history = [
        json.loads(line)
        for line in (run_dir / "repair_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert repair_history[0]["status"] == "applied"
