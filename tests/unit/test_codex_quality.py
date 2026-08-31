from __future__ import annotations

import json
import hashlib

from jsonschema import Draft202012Validator

import translation_forensics.codex_quality as codex_quality
from translation_forensics.codex_exec_provider import ROLE_POLICY
from translation_forensics.codex_quality import (
    _build_translation_continuity_memory,
    _review_repair_eligible,
    build_scene_batches,
    compare_independent_frames,
    evaluate_codex_quality,
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
    first = stratified_proxy_sample(decisions, title_id="SSIS-908", sample_size=120)
    second = stratified_proxy_sample(list(reversed(decisions)), title_id="SSIS-908", sample_size=120)
    assert first == second
    assert len(first) == 120
    selected_statuses = {decisions[number - 1]["source_quality_status"] for number in first}
    assert selected_statuses == {"trusted", "suspect", "unusable"}


def test_translation_continuity_memory_is_not_evidence():
    memory = _build_translation_continuity_memory(
        [
            {
                "block_number": 7,
                "selected_frame": {"speaker": "speaker-1", "actor": "actor-1"},
                "utterance_kind": "lexical_speech",
                "source_faithful_korean": "이전 대사",
                "viewer_natural_korean": "자연스러운 이전 대사",
            }
        ]
    )
    assert memory[0]["memory_role"] == "continuity-only-not-evidence"
    assert memory[0]["evidence_refs"] == []
    assert memory[0]["speaker"] == "speaker-1"


def test_evidence_ceiling_blocks_impossible_title_before_model_calls():
    quality = {
        number: {"source_quality_status": "trusted" if number <= 100 else "unusable"}
        for number in range(1, 299)
    }
    acoustic = {
        number: {
            "independent_source_families": ["a", "b"] if number in range(101, 135) else [],
            "asr_fusion": {
                "state": "dual_agreement",
                "alignment_strength": "block-aligned",
                "risk_codes": [],
            }
            if number in range(101, 135)
            else {},
        }
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


def test_evidence_ceiling_uses_downstream_fusion_predicate():
    quality = {number: {"source_quality_status": "unusable"} for number in range(1, 4)}
    acoustic = {
        1: {"independent_source_families": ["a", "b"]},
        2: {
            "independent_source_families": ["a", "b"],
            "asr_fusion": {
                "state": "dual_conflict",
                "alignment_strength": "block-aligned",
                "risk_codes": ["dual-asr-source-conflict"],
            },
        },
        3: {
            "independent_source_families": ["a", "b"],
            "block_alignment_status": "single-block-expanded",
            "asr_fusion": {
                "state": "dual_compatible",
                "alignment_strength": "expanded",
                "risk_codes": [],
            },
        },
    }
    result = evaluate_evidence_ceiling(
        title_id="SSIS-908",
        expected_blocks=[1, 2, 3],
        source_quality=quality,
        acoustic=acoustic,
    )
    # Family names alone are not enough: a fusion decision is mandatory for
    # the production evidence ceiling.
    assert result["eligible_block_numbers"] == [3]


def test_unregistered_title_uses_strict_default_and_blocks_empty_evidence():
    result = evaluate_evidence_ceiling(
        title_id="UNREGISTERED",
        expected_blocks=[1, 2],
        source_quality={
            1: {"source_quality_status": "unusable"},
            2: {"source_quality_status": "unusable"},
        },
        acoustic={1: {}, 2: {}},
    )
    assert result["status"] == "fail"
    assert result["model_calls_allowed"] is False
    assert result["minimum_required_accepted_rate"] == 1.0
    assert result["gate_configured"] is False
    assert result["gate_policy"] == "unregistered-deny"


def test_unregistered_title_is_denied_even_with_full_evidence():
    result = evaluate_evidence_ceiling(
        title_id="UNREGISTERED",
        expected_blocks=[1],
        source_quality={1: {"source_quality_status": "trusted"}},
        acoustic={1: {}},
    )
    assert result["maximum_possible_accepted_rate"] == 1.0
    assert result["status"] == "fail"
    assert result["model_calls_allowed"] is False


def test_resume_recomputes_gate_policy_for_legacy_unregistered_package(tmp_path):
    structure = tmp_path / "structure.srt"
    structure.write_text("1\n00:00:00,000 --> 00:00:01,000\n原文\n", encoding="utf-8")
    source_map = tmp_path / "source-quality-map.jsonl"
    source_map.write_text(json.dumps({"block_number": 1, "source_quality_status": "unusable"}) + "\n", encoding="utf-8")
    acoustic = tmp_path / "acoustic.jsonl"
    acoustic.write_text(json.dumps({"block_number": 1, "transcripts": [], "evidence_refs": []}) + "\n", encoding="utf-8")
    package = tmp_path / "package"
    package.mkdir()

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (package / "manifest.json").write_text(
        json.dumps(
            {
                "title_id": "UNREGISTERED",
                "status": "quality-gates-passed",
                "inputs": {
                    "structure": {"sha256": digest(structure)},
                    "source_quality_map": {"sha256": digest(source_map)},
                    "acoustic_evidence": {"sha256": digest(acoustic)},
                },
                "model_call_count": 1,
            }
        ),
        encoding="utf-8",
    )

    result = run_codex_quality_title(
        title_id="UNREGISTERED",
        structure_path=structure,
        source_quality_map_path=source_map,
        acoustic_evidence_path=acoustic,
        audio_path=None,
        output_dir=package,
        provider=object(),
        prompt_dir=tmp_path,
        schema_dir=tmp_path,
        resume=True,
    )
    assert result["cache_hit"] is False
    assert result["status"] == "evidence-ceiling-failed"
    assert result["model_call_count"] == 0
    assert result["gate_policy"] == "unregistered-deny"


def test_registered_gate_hash_invalidates_resume_and_forged_gate_is_rejected(tmp_path, monkeypatch):
    structure = tmp_path / "structure.srt"
    structure.write_text("1\n00:00:00,000 --> 00:00:01,000\n原文\n", encoding="utf-8")
    source_map = tmp_path / "source-quality-map.jsonl"
    source_map.write_text(json.dumps({"block_number": 1, "source_quality_status": "unusable"}) + "\n", encoding="utf-8")
    acoustic = tmp_path / "acoustic.jsonl"
    acoustic.write_text(json.dumps({"block_number": 1, "transcripts": [], "evidence_refs": []}) + "\n", encoding="utf-8")
    package = tmp_path / "package"
    first = run_codex_quality_title(
        title_id="SSIS-908",
        structure_path=structure,
        source_quality_map_path=source_map,
        acoustic_evidence_path=acoustic,
        audio_path=None,
        output_dir=package,
        provider=object(),
        prompt_dir=tmp_path,
        schema_dir=tmp_path,
        resume=False,
    )
    assert first["cache_hit"] is False
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_hash = manifest["gate_config_sha256"]

    monkeypatch.setitem(
        codex_quality.TITLE_GATES,
        "SSIS-908",
        {"minimum_accepted_rate": 0.86, "minimum_safe_usable_rate": 0.85, "maximum_ellipsis_rate": 0.08},
    )
    resumed = run_codex_quality_title(
        title_id="SSIS-908",
        structure_path=structure,
        source_quality_map_path=source_map,
        acoustic_evidence_path=acoustic,
        audio_path=None,
        output_dir=package,
        provider=object(),
        prompt_dir=tmp_path,
        schema_dir=tmp_path,
        resume=True,
    )
    assert resumed["cache_hit"] is False
    assert resumed["gate_config_sha256"] != original_hash

    forged = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    forged["gate_configured"] = False
    manifest_path.write_text(json.dumps(forged), encoding="utf-8")
    validation = validate_codex_quality(package)
    assert validation["status"] == "fail"
    assert any("manifest gate_configured does not match" in error for error in validation["errors"])


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


class ProxyQualityProvider(FakeQualityProvider):
    def run_structured(self, *, role, title_id, call_id, prompt, payload, schema, resume=True):
        if role.startswith("proxy-evaluator"):
            model, effort = ROLE_POLICY[role]
            batch_id = payload["batch_id"]
            numbers = [int(row["block_number"]) for row in payload["blocks"]]
            response = {
                "batch_id": batch_id,
                "judgments": [
                    {
                        "block_number": number,
                        "winner": "tie",
                        "critical_error_side": "none",
                        "reason": "candidate is supported",
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
        return super().run_structured(
            role=role,
            title_id=title_id,
            call_id=call_id,
            prompt=prompt,
            payload=payload,
            schema=schema,
            resume=resume,
        )


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
        self.payloads = {}

    def run_structured(self, *, role, title_id, call_id, prompt, payload, schema, resume=True):
        self.payloads[call_id] = json.loads(json.dumps(payload, ensure_ascii=False))
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


class RepairThenAcceptQualityProvider(FakeQualityProvider):
    def __init__(self):
        self.call_ids = []

    def run_structured(self, *, role, title_id, call_id, prompt, payload, schema, resume=True):
        self.call_ids.append(call_id)
        response, receipt = super().run_structured(
            role=role,
            title_id=title_id,
            call_id=call_id,
            prompt=prompt,
            payload=payload,
            schema=schema,
            resume=resume,
        )
        if role == "critique-sol" and call_id.endswith(".0"):
            for review in response["reviews"]:
                review.update(
                    {
                        "verdict": "repair",
                        "unsupported_additions": ["location: unsupported detail"],
                        "claim_findings": [
                            {
                                "slot": "location",
                                "disposition": "omit",
                                "reason": "location is unsupported",
                            }
                        ],
                        "reason": "remove unsupported location claim",
                    }
                )
        elif role == "repair-terra":
            for block in response["blocks"]:
                block["source_faithful_korean"] = f"수리된 멈춰 {block['block_number']}"
                block["viewer_natural_korean"] = f"수리된 멈춰 {block['block_number']}"
                block["conservative_source_faithful_korean"] = f"수리된 멈춰 {block['block_number']}"
                block["conservative_viewer_natural_korean"] = f"수리된 멈춰 {block['block_number']}"
        return response, receipt


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
        title_id="SSIS-908",
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
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    manifest_schema = json.loads((root / "schemas" / "codex-quality-manifest.schema.json").read_text(encoding="utf-8"))
    evidence_schema = json.loads((root / "schemas" / "codex-quality-evidence-ceiling.schema.json").read_text(encoding="utf-8"))
    validation_schema = json.loads((root / "schemas" / "codex-quality-validation.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(manifest_schema).validate(json.loads((package / "manifest.json").read_text(encoding="utf-8")))
    evidence_path = package / "evidence-feasibility.json"
    if evidence_path.is_file():
        Draft202012Validator(evidence_schema).validate(json.loads(evidence_path.read_text(encoding="utf-8")))
    Draft202012Validator(validation_schema).validate(validation)


def test_partial_evidence_evaluation_runs_only_eligible_blocks_and_never_promotes(tmp_path):
    structure = tmp_path / "sample.ja.srt"
    structure.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n?꾠굙??n\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\n?꾠굙??n\n",
        encoding="utf-8",
    )
    source_map = tmp_path / "source-quality-map.jsonl"
    source_map.write_text(
        "".join(
            json.dumps(
                {
                    "block_number": number,
                    "source_quality_status": "trusted" if number == 1 else "unusable",
                    "reason_codes": [],
                }
            )
            + "\n"
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
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    package = tmp_path / "partial-package"
    result = run_codex_quality_title(
        title_id="SSIS-908",
        structure_path=structure,
        source_quality_map_path=source_map,
        acoustic_evidence_path=acoustic,
        audio_path=None,
        output_dir=package,
        provider=FakeQualityProvider(),
        prompt_dir=root / "prompts",
        schema_dir=root / "schemas",
        max_scene_blocks=20,
        max_repairs=0,
        resume=False,
        partial_evidence_evaluation=True,
    )
    assert result["status"] == "partial-evidence-candidate"
    assert result["partial_execution"] is True
    assert result["eligible_block_numbers"] == [1]
    assert result["ineligible_block_numbers"] == [2]
    validation = validate_codex_quality(package)
    assert validation["status"] == "partial"
    assert validation["metric_gate_passed"] is False
    assert validation["final_promotion_allowed"] is False
    assert validation["eligible_block_numbers"] == [1]
    manifest_schema = json.loads((root / "schemas" / "codex-quality-manifest.schema.json").read_text(encoding="utf-8"))
    validation_schema = json.loads((root / "schemas" / "codex-quality-validation.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(manifest_schema).validate(json.loads((package / "manifest.json").read_text(encoding="utf-8")))
    Draft202012Validator(validation_schema).validate(validation)
    manifest_path = package / "manifest.json"
    forged_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forged_manifest["eligible_block_numbers"] = [2]
    manifest_path.write_text(json.dumps(forged_manifest), encoding="utf-8")
    forged_validation = validate_codex_quality(package)
    assert forged_validation["status"] == "fail"
    assert any("eligible block list differs" in error for error in forged_validation["errors"])
    assert len((package / "model-call-receipts.jsonl").read_text(encoding="utf-8").splitlines()) > 0
    assert all(
        row["block_number"] == 1
        for row in (
            json.loads(line)
            for line in (package / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
        )
    )
    viewer_text = (package / "SSIS-908.viewer-natural-ko.autonomous-quality-v1.srt").read_text(encoding="utf-8")
    assert "…" in viewer_text


def test_partial_proxy_evaluation_is_diagnostic_and_keeps_full_denominator(tmp_path):
    structure = tmp_path / "sample.ja.srt"
    structure.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n?꾠굙??n\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\n?꾠굙??n\n",
        encoding="utf-8",
    )
    source_map = tmp_path / "source-quality-map.jsonl"
    source_map.write_text(
        "".join(
            json.dumps({"block_number": number, "source_quality_status": "trusted" if number == 1 else "unusable", "reason_codes": []}) + "\n"
            for number in (1, 2)
        ),
        encoding="utf-8",
    )
    acoustic = tmp_path / "block-acoustic-evidence.jsonl"
    acoustic.write_text(
        "".join(json.dumps({"block_number": number, "transcripts": [], "evidence_refs": [], "independent_source_families": []}) + "\n" for number in (1, 2)),
        encoding="utf-8",
    )
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    package = tmp_path / "partial-package"
    provider = ProxyQualityProvider()
    run_codex_quality_title(
        title_id="SSIS-908",
        structure_path=structure,
        source_quality_map_path=source_map,
        acoustic_evidence_path=acoustic,
        audio_path=None,
        output_dir=package,
        provider=provider,
        prompt_dir=root / "prompts",
        schema_dir=root / "schemas",
        max_scene_blocks=20,
        max_repairs=0,
        resume=False,
        partial_evidence_evaluation=True,
    )
    baseline = tmp_path / "baseline.srt"
    baseline.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n기준 대사\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\n기준 대사\n",
        encoding="utf-8",
    )
    evaluation = evaluate_codex_quality(
        package_dir=package,
        baseline=baseline,
        provider=provider,
        prompt_dir=root / "prompts",
        schema_dir=root / "schemas",
        sample_size=1,
        batch_size=1,
        resume=False,
    )
    assert evaluation["status"] == "partial-pass"
    assert evaluation["evaluation_scope"] == "eligible-blocks-only"
    assert evaluation["partial_evaluation"] is True
    assert evaluation["denominator_block_count"] == 2
    assert evaluation["eligible_block_count"] == 1
    assert evaluation["title_gate_passed"] is False
    assert evaluation["promotion_permanently_blocked"] is True
    evaluation_schema = json.loads((root / "schemas" / "codex-quality-proxy-evaluation.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(evaluation_schema).validate(evaluation)


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
        title_id="SSIS-908",
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
        title_id="SSIS-908",
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
    initial_locked = provider.payloads["scene-0001.meaning.terra"]["locked_blocks"]
    assert initial_locked[0]["source_text_evidence_ref"] == "source-srt:block-1"
    assert initial_locked[0]["asr_fusion"]["state"] == "empty"
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
    translation_payload = provider.payloads["scene-0001.translation.terra"]
    critique_payload = provider.payloads["scene-0001.critique.sol.0"]
    for captured in (translation_payload, critique_payload):
        locked = {int(row["block_number"]): row for row in captured["locked_blocks"]}
        assert "utterance:retry-1-w:whisper" in {
            ref
            for hypothesis in locked[1]["asr_fusion"]["family_hypotheses"]
            for ref in hypothesis["evidence_refs"]
        }
        assert locked[1]["asr_fusion"]["state"] == "dual_agreement"
        assert locked[2]["asr_fusion"].get("state", "empty") == "empty"
    agreements = [
        json.loads(line)
        for line in (package / "frame-agreements.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert agreements[0]["boundary_expansion_asr_rerun"] is True
    assert agreements[1]["boundary_expansion_asr_rerun"] is False


def test_end_to_end_repair_critique_runs_terra_repair_then_accept(tmp_path):
    structure = tmp_path / "sample.ja.srt"
    structure.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nやめて\n\n",
        encoding="utf-8",
    )
    source_map = tmp_path / "source-quality-map.jsonl"
    source_map.write_text(
        json.dumps({"block_number": 1, "source_quality_status": "trusted", "reason_codes": []}) + "\n",
        encoding="utf-8",
    )
    acoustic = tmp_path / "block-acoustic-evidence.jsonl"
    acoustic.write_text(
        json.dumps(
            {
                "block_number": 1,
                "transcripts": [],
                "evidence_refs": [],
                "independent_source_families": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provider = RepairThenAcceptQualityProvider()
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    package = tmp_path / "package"
    result = run_codex_quality_title(
        title_id="SSIS-908",
        structure_path=structure,
        source_quality_map_path=source_map,
        acoustic_evidence_path=acoustic,
        audio_path=None,
        output_dir=package,
        provider=provider,
        prompt_dir=root / "prompts",
        schema_dir=root / "schemas",
        max_scene_blocks=20,
        max_repairs=2,
        resume=False,
    )
    assert result["status"] == "quality-gates-passed"
    assert any("repair.terra.1" in call_id for call_id in provider.call_ids)
    decision = json.loads((package / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert decision["sol_final_verdict"] == "accept"
    assert len(decision["repair_history"]) == 2


def test_translation_payload_carries_prior_scene_continuity_only_memory(tmp_path):
    structure = tmp_path / "sample.ja.srt"
    structure.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n?꾠굙??n\n\n"
        "2\n00:00:10,000 --> 00:00:11,000\n?꾠굙??n\n",
        encoding="utf-8",
    )
    source_map = tmp_path / "source-quality-map.jsonl"
    source_map.write_text(
        "".join(json.dumps({"block_number": number, "source_quality_status": "trusted", "reason_codes": []}) + "\n" for number in (1, 2)),
        encoding="utf-8",
    )
    acoustic = tmp_path / "block-acoustic-evidence.jsonl"
    acoustic.write_text(
        "".join(json.dumps({"block_number": number, "transcripts": [], "evidence_refs": [], "independent_source_families": []}) + "\n" for number in (1, 2)),
        encoding="utf-8",
    )
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    provider = ConflictRetryQualityProvider()
    package = tmp_path / "continuity-package"
    result = run_codex_quality_title(
        title_id="SSIS-908",
        structure_path=structure,
        source_quality_map_path=source_map,
        acoustic_evidence_path=acoustic,
        audio_path=None,
        output_dir=package,
        provider=provider,
        prompt_dir=root / "prompts",
        schema_dir=root / "schemas",
        max_scene_blocks=1,
        maximum_scene_gap_seconds=1.0,
        max_repairs=0,
        resume=False,
    )
    assert result["status"] == "quality-gates-failed"
    memory = provider.payloads["scene-0002.translation.terra"]["continuity_memory"]
    assert memory and memory[-1]["block_number"] == 1
    assert memory[-1]["memory_role"] == "continuity-only-not-evidence"
    assert memory[-1]["evidence_refs"] == []
