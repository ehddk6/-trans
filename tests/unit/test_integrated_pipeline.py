from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from translation_forensics.integrated_pipeline import (
    CACHE_SCHEMA_VERSION,
    INTEGRATION_SCHEMA_VERSION,
    REVIEW_HOLD_TEXT,
    BundleBlockedError,
    CoverageError,
    ExtractRequest,
    GateResult,
    IntegrationPipelineError,
    JapaneseSubtitleBundle,
    ResumeRejected,
    UnitTranslation,
    append_review_decision,
    build_cache_identity,
    build_translation_inputs,
    cache_is_reusable,
    evaluate_bundle,
    load_review_decisions,
    load_translation_units,
    normalize_whitespace,
    package_dual_outputs,
    promote_staged_run,
    require_reusable_cache,
    select_extraction_backend,
    stage_run,
    transition_review_decision,
    write_translation_units,
)
from translation_forensics.srt import parse_srt


MEDIA_HASH = "a" * 64


def _write_transcript(path: Path) -> Path:
    rows = [
        {
            "utterance_id": "utt_000001",
            "start": 1.0,
            "end": 2.25,
            "text_raw": "今日は いい天気",
            "text_normalized": "この値は使わない",
            "source_segment_ids": [10],
            "words": [{"text": "今日は", "start": 1.0, "end": 1.4}],
            "warnings": [],
            "speaker": "speaker_a",
        },
        {
            "utterance_id": "utt_000002",
            "start": 2.5,
            "end": 3.5,
            "text_raw": "そうですね",
            "text_normalized": "viewerで短縮",
            "source_segment_ids": [11],
            "words": [],
            "warnings": ["review_required"],
        },
        {
            "utterance_id": "utt_000003",
            "start": 4.0,
            "end": 5.0,
            "text_raw": "おわりおわりおわり",
            "source_segment_ids": [12],
            "words": [],
            "warnings": ["possible_runaway_repetition"],
        },
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _bundle(**changes) -> JapaneseSubtitleBundle:
    value = JapaneseSubtitleBundle(
        run_version="extractor-1",
        media_sha256=MEDIA_HASH,
        backend="ensemble",
        artifacts={},
        valid=True,
        accepted=False,
    )
    return replace(value, **changes)


def _inputs(tmp_path: Path):
    transcript = _write_transcript(tmp_path / "transcript_ja.jsonl")
    return build_translation_inputs(transcript, tmp_path / "translation-inputs")


def _translations():
    return [
        UnitTranslation(
            unit_id="utt_000001",
            viewer_complete_ko="오늘은 날씨가 좋네.",
            source_faithful_ko="오늘은 좋은 날씨다.",
            viewer_natural_ko="오늘 날씨 좋네.",
            evidence_ids=("audio:1",),
        ),
        UnitTranslation(
            unit_id="utt_000002",
            viewer_complete_ko="그러게요.",
            source_faithful_ko="그렇네요.",
            viewer_natural_ko="그러게요.",
            evidence_ids=("audio:2",),
        ),
        UnitTranslation(
            unit_id="utt_000003",
            viewer_complete_ko="끝, 끝, 끝.",
            source_faithful_ko="끝, 끝, 끝.",
            viewer_natural_ko="이제 끝이야.",
            evidence_ids=("audio:3",),
        ),
    ]


def test_extract_request_requires_path_for_approved_reference(tmp_path: Path) -> None:
    with pytest.raises(IntegrationPipelineError, match="requires"):
        ExtractRequest(
            title_id="ADN-622",
            media=tmp_path / "ADN-622.mp4",
            project_root=tmp_path,
            reference_ja_approved=True,
        )


def test_backend_selection_never_uses_an_unapproved_reference(tmp_path: Path) -> None:
    reference = tmp_path / "approved.ja.srt"
    reference.write_text("approved reference", encoding="utf-8")
    automatic = ExtractRequest(
        title_id="ADN-622",
        media=tmp_path / "ADN-622.mp4",
        project_root=tmp_path,
        reference_ja=reference,
    )
    approved = replace(automatic, reference_ja_approved=True)
    forced_reference = replace(automatic, backend_policy="reference")

    assert select_extraction_backend(automatic) == "ensemble"
    assert select_extraction_backend(approved) == "reference"
    with pytest.raises(BundleBlockedError, match="explicitly approved"):
        select_extraction_backend(forced_reference)


def test_bundle_gates_are_independent_and_valid_false_stops_translation() -> None:
    invalid = evaluate_bundle(_bundle(valid=False))
    assert not invalid.translation_allowed
    assert not invalid.complete_draft_allowed

    recognition_review = evaluate_bundle(
        _bundle(recognition=GateResult("review_required", ("low_confidence",)))
    )
    assert recognition_review.translation_allowed
    assert recognition_review.complete_draft_allowed
    assert not recognition_review.candidate_generation_allowed
    assert recognition_review.review_required
    assert not recognition_review.release_allowed

    presentation_review = evaluate_bundle(
        _bundle(presentation=GateResult("review_required", ("line_length",)))
    )
    assert presentation_review.translation_allowed
    assert "presentation_review_required" in presentation_review.reasons

    recognition_failure = evaluate_bundle(
        _bundle(recognition=GateResult("failed", ("no_usable_speech",)))
    )
    assert not recognition_failure.translation_allowed

    inconsistent_accepted_review = evaluate_bundle(
        _bundle(
            accepted=True,
            recognition=GateResult("review_required", ("direct_listening_needed",)),
        )
    )
    assert inconsistent_accepted_review.complete_draft_allowed
    assert not inconsistent_accepted_review.candidate_generation_allowed


def test_build_translation_inputs_uses_only_text_raw_and_preserves_content(
    tmp_path: Path,
) -> None:
    transcript = _write_transcript(tmp_path / "transcript_ja.jsonl")
    (tmp_path / "viewer_ja.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n使ってはいけない\n", encoding="utf-8"
    )

    artifacts = build_translation_inputs(transcript, tmp_path / "out")
    blocks, _, _ = parse_srt(artifacts.translation_ja_srt)
    source_rows = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]

    assert artifacts.exact_after_whitespace_normalization
    assert normalize_whitespace("".join(block.text for block in blocks)) == normalize_whitespace(
        "".join(row["text_raw"] for row in source_rows)
    )
    assert "この値は使わない" not in artifacts.translation_ja_srt.read_text(encoding="utf-8")
    assert "使ってはいけない" not in artifacts.translation_ja_srt.read_text(encoding="utf-8")
    assert [unit.quality_status for unit in artifacts.units] == [
        "trusted",
        "suspect",
        "unusable",
    ]
    written_rows = [
        json.loads(line)
        for line in artifacts.translation_units_ja_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert written_rows[0]["text_raw"] == "今日は いい天気"
    assert written_rows[0]["source_segment_ids"] == [10]
    assert written_rows[0]["evidence_ids"] == ["transcript_ja.jsonl#L1"]


def test_translation_units_round_trip_and_reject_old_schema(tmp_path: Path) -> None:
    artifacts = _inputs(tmp_path)
    loaded = load_translation_units(artifacts.translation_units_ja_jsonl)
    assert loaded == artifacts.units

    copied = tmp_path / "copied-units.jsonl"
    assert write_translation_units(copied, loaded) == copied
    assert load_translation_units(copied) == loaded
    rows = [json.loads(line) for line in copied.read_text(encoding="utf-8").splitlines()]
    rows[0]["schema_version"] = "0"
    old = tmp_path / "old-units.jsonl"
    old.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ResumeRejected, match="schema"):
        load_translation_units(old)


def test_build_translation_inputs_rejects_viewer_or_invalid_bundle(tmp_path: Path) -> None:
    viewer = tmp_path / "viewer_ja.srt"
    viewer.write_text("not a transcript", encoding="utf-8")
    with pytest.raises(IntegrationPipelineError, match="presentation-only"):
        build_translation_inputs(viewer, tmp_path / "out")

    transcript = _write_transcript(tmp_path / "transcript_ja.jsonl")
    with pytest.raises(BundleBlockedError):
        build_translation_inputs(transcript, tmp_path / "other", bundle=_bundle(valid=False))


def test_build_translation_inputs_never_overwrites_existing_artifacts(tmp_path: Path) -> None:
    transcript = _write_transcript(tmp_path / "transcript_ja.jsonl")
    output = tmp_path / "out"
    build_translation_inputs(transcript, output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_translation_inputs(transcript, output)


def test_dual_outputs_require_complete_coverage(tmp_path: Path) -> None:
    units = _inputs(tmp_path).units
    with pytest.raises(CoverageError, match="cover every unit"):
        package_dual_outputs(units, _translations()[:-1], tmp_path / "package")
    broken = _translations()
    broken[1] = replace(broken[1], viewer_complete_ko="")
    with pytest.raises(CoverageError, match="empty"):
        package_dual_outputs(units, broken, tmp_path / "package-2")


def test_dual_outputs_separate_complete_draft_from_evidence_candidates(
    tmp_path: Path,
) -> None:
    units = _inputs(tmp_path).units
    approved = transition_review_decision(
        None,
        unit_id="utt_000002",
        status="approved",
        reviewer="human-a",
        reason="direct listening confirmed the warning",
        decided_at="2026-08-09T00:00:00+00:00",
    )
    artifacts = package_dual_outputs(
        units,
        _translations(),
        tmp_path / "package",
        review_decisions={"utt_000002": approved},
    )

    complete, _, _ = parse_srt(artifacts.viewer_complete_ko_srt)
    faithful, _, _ = parse_srt(artifacts.source_faithful_ko_srt)
    qa = json.loads(artifacts.qa_path.read_text(encoding="utf-8"))
    assert len(complete) == len(units) == 3
    assert all(block.text.strip() for block in complete)
    assert faithful[0].text == "오늘은 좋은 날씨다."
    assert faithful[1].text == "그렇네요."
    assert faithful[2].text == REVIEW_HOLD_TEXT
    assert artifacts.candidate_units == 2
    assert artifacts.held_units == ("utt_000003",)
    assert qa["viewer_complete_coverage"] == 1.0
    assert qa["verification_status"] == "not-demonstrated"
    assert not qa["final_promotion_allowed"]


def test_dual_packager_accepts_terra_decision_field_names(tmp_path: Path) -> None:
    units = _inputs(tmp_path).units
    decisions = [
        {
            "unit_id": value.unit_id,
            "source_faithful_korean": value.source_faithful_ko,
            "viewer_natural_korean": value.viewer_natural_ko,
            "evidence_ids": list(value.evidence_ids),
        }
        for value in _translations()
    ]
    artifacts = package_dual_outputs(units, decisions, tmp_path / "package")
    complete, _, _ = parse_srt(artifacts.viewer_complete_ko_srt)
    assert [block.text for block in complete] == [
        "오늘 날씨 좋네.",
        "그러게요.",
        "이제 끝이야.",
    ]


def test_candidate_requires_evidence_even_for_trusted_unit(tmp_path: Path) -> None:
    units = _inputs(tmp_path).units
    translations = _translations()
    translations[0] = replace(translations[0], evidence_ids=())
    artifacts = package_dual_outputs(units, translations, tmp_path / "package")
    faithful, _, _ = parse_srt(artifacts.source_faithful_ko_srt)
    assert faithful[0].text == REVIEW_HOLD_TEXT
    assert artifacts.candidate_units == 0


def test_unaccepted_bundle_holds_candidates_without_per_unit_human_approval(
    tmp_path: Path,
) -> None:
    units = _inputs(tmp_path).units
    approved = transition_review_decision(
        None,
        unit_id="utt_000002",
        status="approved",
        reviewer="human-a",
        reason="direct listening confirmed this unit",
    )
    artifacts = package_dual_outputs(
        units,
        _translations(),
        tmp_path / "package",
        bundle=_bundle(accepted=False),
        review_decisions={"utt_000002": approved},
    )
    faithful, _, _ = parse_srt(artifacts.source_faithful_ko_srt)
    assert [block.text for block in faithful] == [
        REVIEW_HOLD_TEXT,
        "그렇네요.",
        REVIEW_HOLD_TEXT,
    ]
    assert artifacts.candidate_units == 1


def test_recognition_review_gate_holds_candidates_even_if_bundle_says_accepted(
    tmp_path: Path,
) -> None:
    units = _inputs(tmp_path).units
    bundle = _bundle(
        accepted=True,
        recognition=GateResult("review_required", ("direct_listening_needed",)),
    )
    artifacts = package_dual_outputs(
        units,
        _translations(),
        tmp_path / "package",
        bundle=bundle,
    )
    faithful, _, _ = parse_srt(artifacts.source_faithful_ko_srt)
    assert all(block.text == REVIEW_HOLD_TEXT for block in faithful)
    assert artifacts.candidate_units == 0


def test_review_decision_transitions_are_append_only_and_terminal(tmp_path: Path) -> None:
    path = tmp_path / "review_decisions.jsonl"
    deferred = transition_review_decision(
        None,
        unit_id="utt_1",
        status="deferred",
        reviewer="reviewer",
        reason="needs direct listening",
        decided_at="2026-08-09T00:00:00+00:00",
    )
    append_review_decision(path, deferred)
    approved = transition_review_decision(
        deferred,
        unit_id="utt_1",
        status="approved",
        reviewer="reviewer",
        reason="direct listening completed",
        decided_at="2026-08-09T01:00:00+00:00",
    )
    append_review_decision(path, approved)

    assert load_review_decisions(path)["utt_1"] == approved
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    with pytest.raises(IntegrationPipelineError, match="terminal"):
        transition_review_decision(
            approved,
            unit_id="utt_1",
            status="rejected",
            reviewer="reviewer",
            reason="attempted reversal",
        )


def test_cache_identity_includes_stable_inputs_but_excludes_temp_wav() -> None:
    left = build_cache_identity(
        code_version="commit-a",
        schema_version="schema-a",
        media_sha256=MEDIA_HASH,
        options={"beam_size": 5, "temp_wav_path": "C:/Temp/a.wav"},
    )
    right = build_cache_identity(
        code_version="commit-a",
        schema_version="schema-a",
        media_sha256=MEDIA_HASH,
        options={"temp_wav_path": "D:/Elsewhere/b.wav", "beam_size": 5},
    )
    changed = build_cache_identity(
        code_version="commit-a",
        schema_version="schema-a",
        media_sha256=MEDIA_HASH,
        options={"beam_size": 6, "temp_wav_path": "C:/Temp/a.wav"},
    )

    assert left == right
    assert "temp_wav_path" not in left["options"]
    assert left["identity_sha256"] != changed["identity_sha256"]


def test_partial_and_old_cache_manifests_are_not_reusable(tmp_path: Path) -> None:
    identity = build_cache_identity(
        code_version="commit-a",
        schema_version="schema-a",
        media_sha256=MEDIA_HASH,
        options={},
    )
    complete = {
        "schema_name": "translation-forensics/integrated-cache",
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": "complete",
        "cache_identity": identity,
    }
    assert cache_is_reusable(complete, identity)
    assert not cache_is_reusable({**complete, "status": "partial"}, identity)
    assert not cache_is_reusable({**complete, "schema_version": "0"}, identity)
    with pytest.raises(ResumeRejected):
        require_reusable_cache({**complete, "status": "partial"}, identity)


def test_stage_resume_requires_matching_current_state_and_promotes_atomically(
    tmp_path: Path,
) -> None:
    identity = build_cache_identity(
        code_version="commit-a",
        schema_version="schema-a",
        media_sha256=MEDIA_HASH,
        options={"vad": False},
    )
    staged = stage_run(tmp_path, "run-001", cache_identity=identity)
    assert staged == tmp_path / ".partial" / "run-001"
    assert stage_run(tmp_path, "run-001", cache_identity=identity, resume=True) == staged
    (staged / "translation_ja.srt").write_text("verified", encoding="utf-8")

    with pytest.raises(BundleBlockedError):
        promote_staged_run(tmp_path, "run-001", verification_passed=False)
    promoted = promote_staged_run(
        tmp_path,
        "run-001",
        verification_passed=True,
        stage="machine-draft",
        pending_review_count=2,
        required_artifacts=("translation_ja.srt",),
    )
    assert promoted == tmp_path / "run-001"
    assert promoted.is_dir()
    assert not staged.exists()
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["run_id"] == "run-001"
    assert latest["pending_review_count"] == 2
    assert latest["cache_identity_sha256"] == identity["identity_sha256"]


def test_stage_resume_rejects_mismatched_identity_and_old_schema(tmp_path: Path) -> None:
    identity = build_cache_identity(
        code_version="commit-a",
        schema_version="schema-a",
        media_sha256=MEDIA_HASH,
        options={},
    )
    changed = build_cache_identity(
        code_version="commit-b",
        schema_version="schema-a",
        media_sha256=MEDIA_HASH,
        options={},
    )
    staged = stage_run(tmp_path, "run-001", cache_identity=identity)
    with pytest.raises(ResumeRejected):
        stage_run(tmp_path, "run-001", cache_identity=changed, resume=True)

    state_path = staged / "run-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["schema_version"] = "0"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ResumeRejected):
        stage_run(tmp_path, "run-001", cache_identity=identity, resume=True)


def test_promotion_requires_declared_artifacts(tmp_path: Path) -> None:
    identity = build_cache_identity(
        code_version="commit-a",
        schema_version=INTEGRATION_SCHEMA_VERSION,
        media_sha256=MEDIA_HASH,
        options={},
    )
    stage_run(tmp_path, "run-001", cache_identity=identity)
    with pytest.raises(BundleBlockedError, match="missing"):
        promote_staged_run(
            tmp_path,
            "run-001",
            verification_passed=True,
            required_artifacts=("translation_units_ja.jsonl",),
        )
