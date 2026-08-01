from __future__ import annotations

import json

from translation_forensics.codex_exec_provider import ROLE_POLICY
from translation_forensics.codex_quality import (
    _review_repair_eligible,
    build_scene_batches,
    compare_independent_frames,
    evaluate_evidence_ceiling,
    run_codex_quality_title,
    stratified_proxy_sample,
    validate_codex_quality,
)
from translation_forensics.asr_fusion import add_asr_fusion
from translation_forensics.srt import SubtitleBlock


def frame(number: int, *, question=False, action="stop"):
    return {
        "block_number": number,
        "speech_act": "command",
        "question": question,
        "polarity": "negative",
        "refusal_permission": None,
        "command_strength": "strong",
        "speaker": None,
        "actor": None,
        "action": action,
        "target": None,
        "location": None,
        "tense_aspect": "present",
        "direction": None,
        "intensity": "strong",
    }


def test_independent_frame_comparison_is_deterministic():
    agreements = compare_independent_frames(
        [frame(1, question=True), frame(2)],
        [frame(1, question=False), frame(2)],
        [1, 2],
    )
    assert agreements[0]["agreed"] is False
    assert agreements[0]["critical_slot_conflicts"] == ["question"]
    assert agreements[0]["consensus_frame"]["question"] is None
    assert agreements[1]["agreed"] is True


def test_missing_slot_is_coverage_gap_not_semantic_conflict():
    left = frame(1)
    right = frame(1)
    left["question"] = None
    right["question"] = False
    agreements = compare_independent_frames([left], [right], [1])
    assert agreements[0]["critical_slot_conflicts"] == []
    assert agreements[0]["slot_coverage_gaps"] == ["question"]
    assert agreements[0]["consensus_frame"]["question"] is None


def test_context_retry_is_recorded_for_nonblocking_conflict():
    left = frame(1)
    right = frame(1)
    left["location"] = "room-a"
    right["location"] = "room-b"
    agreement = compare_independent_frames(
        [left],
        [right],
        [1],
        context_retried=True,
    )[0]
    assert agreement["render_blocking_conflicts"] == []
    assert agreement["recovery_state"] == "recovered_context"


def test_repair_requires_claim_level_removable_finding():
    assert _review_repair_eligible(
        {
            "verdict": "repair",
            "unsupported_additions": ["location"],
            "claim_findings": [
                {"slot": "location", "disposition": "omit", "reason": "unsupported"}
            ],
        }
    )
    assert not _review_repair_eligible(
        {
            "verdict": "repair",
            "unsupported_additions": ["location"],
            "claim_findings": [],
        }
    )


def test_scene_batches_split_on_large_gap_and_size():
    blocks = [
        SubtitleBlock(1, "00:00:00,000", "00:00:01,000", "a", 0.0, 1.0),
        SubtitleBlock(2, "00:00:02,000", "00:00:03,000", "b", 2.0, 3.0),
        SubtitleBlock(3, "00:00:30,000", "00:00:31,000", "c", 30.0, 31.0),
    ]
    scenes = build_scene_batches(blocks, max_blocks=2, maximum_gap_seconds=15.0)
    assert [[block.number for block in scene] for scene in scenes] == [[1, 2], [3]]


def test_proxy_sample_is_fixed_and_stratified():
    decisions = [
        {"block_number": number, "source_quality_status": ("trusted", "suspect", "unusable")[number % 3]}
        for number in range(1, 301)
    ]
    first = stratified_proxy_sample(decisions, title_id="SAMPLE", sample_size=120)
    second = stratified_proxy_sample(list(reversed(decisions)), title_id="SAMPLE", sample_size=120)
    assert first == second
    assert len(first) == 120
    selected_statuses = {decisions[number - 1]["source_quality_status"] for number in first}
    assert selected_statuses == {"trusted", "suspect", "unusable"}


def test_evidence_ceiling_blocks_impossible_title_before_model_calls():
    quality = {
        number: {"source_quality_status": "trusted" if number <= 100 else "unusable"}
        for number in range(1, 299)
    }
    acoustic = {
        number: {"independent_source_families": ["a", "b"] if number in range(101, 135) else []}
        for number in range(1, 299)
    }
    result = evaluate_evidence_ceiling(
        title_id="SSIS-908",
        expected_blocks=list(range(1, 299)),
        source_quality=quality,
        acoustic=acoustic,
    )
    assert result["status"] == "fail"
    assert result["model_calls_allowed"] is False
    assert result["eligible_block_count"] == 134


class FakeQualityProvider:
    def run_structured(self, *, role, title_id, call_id, prompt, payload, schema, resume=True):
        model, effort = ROLE_POLICY[role]
        scene_id = payload["scene_id"]
        numbers = [row["block_number"] for row in payload["locked_blocks"]]
        if role.startswith("meaning-frame"):
            response = {"scene_id": scene_id, "frames": [frame(number) | {"confidence": "high", "evidence_refs": [], "reason": "supported"} for number in numbers]}
        elif role in {"translation-terra", "repair-terra"}:
            response = {
                "scene_id": scene_id,
                "blocks": [
                    {
                        "block_number": number,
                        "source_faithful_korean": f"멈춰 {number}",
                        "viewer_natural_korean": f"멈춰 {number}",
                        "conservative_source_faithful_korean": f"멈춰 {number}",
                        "conservative_viewer_natural_korean": f"멈춰 {number}",
                        "source_status": "accepted",
                        "viewer_status": "supported",
                        "recovery_state": "accepted_consensus",
                        "fallback_recovery_state": "accepted_consensus",
                        "rendered_slots": ["speech_act", "polarity", "action"],
                        "fallback_rendered_slots": ["speech_act", "polarity", "action"],
                        "uncertainty_codes": [],
                        "evidence_refs": [],
                        "reason": "supported",
                    }
                    for number in numbers
                ],
            }
        else:
            response = {
                "scene_id": scene_id,
                "reviews": [
                    {
                        "block_number": number,
                        "verdict": "accept",
                        "critical_slot_conflicts": [],
                        "unsupported_additions": [],
                        "naturalness_issues": [],
                        "claim_findings": [],
                        "fallback_blocking": False,
                        "reason": "supported",
                    }
                    for number in numbers
                ],
            }
        receipt = {
            "role": role,
            "requested_model": model,
            "reasoning_effort": effort,
            "ephemeral": True,
            "isolated_temporary_directory": True,
            "requested_model_verified_by_cli_invocation": True,
            "model_call_verified": True,
            "api_key_used": False,
            "sandbox": "read-only",
            "exit_code": 0,
            "thread_id": f"thread-{call_id}",
            "codex_cli_version": "codex-cli test",
            "request_sha256": "a" * 64,
            "prompt_sha256": "b" * 64,
            "evidence_sha256": "c" * 64,
            "schema_sha256": "d" * 64,
            "response_sha256": "e" * 64,
        }
        return response, receipt


class VocalizationQualityProvider(FakeQualityProvider):
    def run_structured(self, *, role, title_id, call_id, prompt, payload, schema, resume=True):
        if role.startswith("meaning-frame"):
            model, effort = ROLE_POLICY[role]
            scene_id = payload["scene_id"]
            numbers = [row["block_number"] for row in payload["locked_blocks"]]
            frames = []
            for number in numbers:
                row = frame(number)
                row.update(
                    {
                        "speech_act": "vocalization",
                        "question": None,
                        "polarity": None,
                        "command_strength": "none",
                        "action": None,
                        "intensity": None,
                        "tense_aspect": None,
                    }
                )
                frames.append(row | {"confidence": "high", "evidence_refs": [], "reason": "audible vocalization"})
            response = {"scene_id": scene_id, "frames": frames}
            receipt = {
                "role": role,
                "requested_model": model,
                "reasoning_effort": effort,
                "ephemeral": True,
                "isolated_temporary_directory": True,
                "requested_model_verified_by_cli_invocation": True,
                "model_call_verified": True,
                "api_key_used": False,
                "sandbox": "read-only",
                "exit_code": 0,
                "thread_id": f"thread-{call_id}",
                "codex_cli_version": "codex-cli test",
                "request_sha256": "a" * 64,
                "prompt_sha256": "b" * 64,
                "evidence_sha256": "c" * 64,
                "schema_sha256": "d" * 64,
                "response_sha256": "e" * 64,
            }
            return response, receipt
        return super().run_structured(
            role=role,
            title_id=title_id,
            call_id=call_id,
            prompt=prompt,
            payload=payload,
            schema=schema,
            resume=resume,
        )


class ConflictRetryQualityProvider(FakeQualityProvider):
    def __init__(self):
        self.frame_call_ids = []

    def run_structured(self, *, role, title_id, call_id, prompt, payload, schema, resume=True):
        if role.startswith("meaning-frame"):
            self.frame_call_ids.append(call_id)
            response, receipt = super().run_structured(
                role=role,
                title_id=title_id,
                call_id=call_id,
                prompt=prompt,
                payload=payload,
                schema=schema,
                resume=resume,
            )
            retry = "meaning-rerun" in call_id
            for row in response["frames"]:
                if int(row["block_number"]) == 1:
                    row["polarity"] = "negative" if retry or role.endswith("sol") else "positive"
            return response, receipt
        return super().run_structured(
            role=role,
            title_id=title_id,
            call_id=call_id,
            prompt=prompt,
            payload=payload,
            schema=schema,
            resume=resume,
        )


def test_end_to_end_quality_package_with_real_contracts(tmp_path):
    structure = tmp_path / "sample.ja.srt"
    structure.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nやめて\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\n待って\n",
        encoding="utf-8",
    )
    source_map = tmp_path / "source-quality-map.jsonl"
    acoustic = tmp_path / "block-acoustic-evidence.jsonl"
    source_rows = []
    acoustic_rows = []
    for number in (1, 2):
        source_rows.append(
            {
                "block_number": number,
                "source_quality_status": "trusted",
                "reason_codes": [],
            }
        )
        acoustic_rows.append(
            {
                "block_number": number,
                "transcripts": [],
                "evidence_refs": [],
                "independent_source_families": [],
            }
        )
    source_map.write_text("".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8")
    acoustic.write_text("".join(json.dumps(row) + "\n" for row in acoustic_rows), encoding="utf-8")
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    package = tmp_path / "package"
    result = run_codex_quality_title(
        title_id="SAMPLE",
        structure_path=structure,
        source_quality_map_path=source_map,
        acoustic_evidence_path=acoustic,
        audio_path=None,
        output_dir=package,
        provider=FakeQualityProvider(),
        prompt_dir=root / "prompts",
        schema_dir=root / "schemas",
        max_scene_blocks=20,
        max_repairs=2,
        resume=False,
    )
    assert result["status"] == "quality-gates-passed"
    validation = validate_codex_quality(package)
    assert validation["status"] == "pass"
    assert validation["accepted_rate"] == 1.0
    assert validation["safe_usable_rate"] == 1.0
    assert validation["recovery_state_counts"] == {"accepted_consensus": 2}


def test_end_to_end_vocalization_route_controls_rendering(tmp_path):
    structure = tmp_path / "sample.ja.srt"
    structure.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nあ\n\n",
        encoding="utf-8",
    )
    source_map = tmp_path / "source-quality-map.jsonl"
    source_map.write_text(
        json.dumps({"block_number": 1, "source_quality_status": "trusted", "reason_codes": []}) + "\n",
        encoding="utf-8",
    )
    acoustic = tmp_path / "block-acoustic-evidence.jsonl"
    acoustic_row = add_asr_fusion(
        {
            "block_number": 1,
            "start": "00:00:00,000",
            "end": "00:00:01,000",
            "transcripts": [
                {
                    "source_family": "whisper",
                    "text": "あ",
                    "utterance_id": "u1",
                    "alignment_scope": "utterance-timestamp",
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                },
                {
                    "source_family": "reazon",
                    "text": "あ",
                    "utterance_id": "u2",
                    "alignment_scope": "utterance-timestamp",
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                },
            ],
            "evidence_refs": ["utterance:u1:whisper", "utterance:u2:reazon"],
            "independent_source_families": ["reazon", "whisper"],
        }
    )
    acoustic.write_text(json.dumps(acoustic_row, ensure_ascii=False) + "\n", encoding="utf-8")
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    package = tmp_path / "package"
    result = run_codex_quality_title(
        title_id="SAMPLE",
        structure_path=structure,
        source_quality_map_path=source_map,
        acoustic_evidence_path=acoustic,
        audio_path=None,
        output_dir=package,
        provider=VocalizationQualityProvider(),
        prompt_dir=root / "prompts",
        schema_dir=root / "schemas",
        max_scene_blocks=20,
        max_repairs=0,
        resume=False,
    )
    assert result["status"] == "quality-gates-passed"
    decision = json.loads((package / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert decision["utterance_kind"] == "vocalization"
    assert decision["recovery_state"] == "vocalization"
    assert decision["viewer_natural_korean"] == "아…"


def test_end_to_end_conflict_rerun_is_block_local_and_re_fused(tmp_path, monkeypatch):
    structure = tmp_path / "sample.ja.srt"
    structure.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nやめて\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\n待って\n\n",
        encoding="utf-8",
    )
    source_map = tmp_path / "source-quality-map.jsonl"
    source_map.write_text(
        "".join(
            json.dumps({"block_number": number, "source_quality_status": "trusted", "reason_codes": []}) + "\n"
            for number in (1, 2)
        ),
        encoding="utf-8",
    )
    acoustic = tmp_path / "block-acoustic-evidence.jsonl"
    acoustic.write_text(
        "".join(
            json.dumps(
                {
                    "block_number": number,
                    "transcripts": [],
                    "evidence_refs": [],
                    "independent_source_families": [],
                }
            )
            + "\n"
            for number in (1, 2)
        ),
        encoding="utf-8",
    )

    def fake_rerun(*, blocks, **kwargs):
        result = {}
        for block in blocks:
            result[block.number] = add_asr_fusion(
                {
                    "block_number": block.number,
                    "start": block.start,
                    "end": block.end,
                    "transcripts": [
                        {
                            "source_family": "whisper",
                            "text": "やめて",
                            "utterance_id": f"retry-{block.number}-w",
                            "alignment_scope": "boundary-expanded-window-rerun",
                        },
                        {
                            "source_family": "reazon",
                            "text": "やめて",
                            "utterance_id": f"retry-{block.number}-r",
                            "alignment_scope": "boundary-expanded-window-rerun",
                        },
                    ],
                    "evidence_refs": [
                        f"asr:retry-{block.number}:whisper",
                        f"asr:retry-{block.number}:reazon",
                    ],
                    "independent_source_families": ["reazon", "whisper"],
                }
            )
        return result

    monkeypatch.setattr("translation_forensics.codex_quality.run_conflict_asr_rerun", fake_rerun)
    provider = ConflictRetryQualityProvider()
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    package = tmp_path / "package"
    result = run_codex_quality_title(
        title_id="SAMPLE",
        structure_path=structure,
        source_quality_map_path=source_map,
        acoustic_evidence_path=acoustic,
        audio_path=tmp_path / "dummy.wav",
        output_dir=package,
        provider=provider,
        prompt_dir=root / "prompts",
        schema_dir=root / "schemas",
        max_scene_blocks=20,
        max_repairs=0,
        resume=False,
    )
    assert result["status"] == "quality-gates-passed"
    assert sorted(call for call in provider.frame_call_ids if "meaning-rerun" in call) == [
        "scene-0001.meaning-rerun.sol",
        "scene-0001.meaning-rerun.terra",
    ]
    decisions = {
        int(row["block_number"]): row
        for row in (
            json.loads(line)
            for line in (package / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    assert decisions[1]["recovery_state"] == "recovered_context"
    assert decisions[1]["boundary_expansion_asr_rerun"] is True
    assert decisions[2]["boundary_expansion_asr_rerun"] is False
    agreements = [
        json.loads(line)
        for line in (package / "frame-agreements.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert agreements[0]["boundary_expansion_asr_rerun"] is True
    assert agreements[1]["boundary_expansion_asr_rerun"] is False
