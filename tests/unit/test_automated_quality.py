from translation_forensics.automated_quality import audit_translation_decision, audit_translation_decisions


def test_audit_preserves_question_negation_and_numbers():
    result = audit_translation_decision(
        unit_id="u1",
        source_japanese="3人じゃない？",
        source_faithful_korean="3명 아니야?",
        viewer_natural_korean="3명 아닌가요?",
        source_quality_status="trusted",
        confidence="high",
    )
    assert result["status"] == "passed"
    assert result["question_preserved"] is True
    assert result["negation_preserved"] is True
    assert result["numeric_tokens_preserved"] is True


def test_audit_marks_unsafe_translation_as_fallback_without_hold_text():
    result = audit_translation_decision(
        unit_id="u2",
        source_japanese="行く？",
        source_faithful_korean="가.",
        viewer_natural_korean="가.",
        source_quality_status="suspect",
        confidence="high",
    )
    assert result["status"] == "fallback"
    assert "question_force_not_preserved" in result["hard_failures"]
    assert "검수 보류" not in " ".join(result["reasons"])


def test_audit_decisions_keeps_unit_order_and_evidence():
    units = [
        {"unit_id": "u1", "source_japanese": "こんにちは", "quality_status": "trusted"},
        {"unit_id": "u2", "source_japanese": "そこ。", "quality_status": "suspect"},
    ]
    decisions = [
        {
            "unit_id": "u1",
            "source_faithful_korean": "안녕하세요.",
            "viewer_natural_korean": "안녕하세요.",
            "confidence": "high",
            "review_required_reasons": [],
        },
        {
            "unit_id": "u2",
            "source_faithful_korean": "거기.",
            "viewer_natural_korean": "거기.",
            "confidence": "low",
            "review_required_reasons": [],
        },
    ]
    updated, records = audit_translation_decisions(units, decisions)
    assert [row["unit_id"] for row in updated] == ["u1", "u2"]
    assert all(row["automated_quality_evidence_id"].startswith("automated-quality:") for row in updated)
    assert [row["unit_id"] for row in records] == ["u1", "u2"]


def test_short_vocalization_high_compression_is_suppressed_only_without_semantic_guard():
    result = audit_translation_decision(
        unit_id="u3",
        source_japanese="あっ…",
        source_faithful_korean="앗…",
        viewer_natural_korean="앗…",
        source_quality_status="trusted",
        confidence="high",
        existing_reasons=["high_compression_ratio"],
    )
    assert result["status"] == "passed"
    assert result["suppressed_warnings"] == ["high_compression_ratio_short_vocalization"]
    assert "high_compression_ratio" not in result["reasons"]


def test_high_compression_is_not_suppressed_when_question_force_is_present():
    result = audit_translation_decision(
        unit_id="u4",
        source_japanese="あ？",
        source_faithful_korean="어?",
        viewer_natural_korean="어?",
        source_quality_status="trusted",
        confidence="high",
        existing_reasons=["high_compression_ratio"],
    )
    assert result["status"] == "fallback"
    assert result["suppressed_warnings"] == []
    assert "high_compression_ratio" in result["reasons"]


def test_recovery_classification_cannot_be_promoted_to_machine_verified():
    result = audit_translation_decision(
        unit_id="u5",
        source_japanese="ん",
        source_faithful_korean="응…",
        viewer_natural_korean="응…",
        source_quality_status="suspect",
        confidence="high",
        recovery_classification="FUNCTIONAL_RECOVERY",
        recovery_basis=["neighboring_turns"],
    )
    assert result["status"] == "fallback"
    assert result["recovery_classification"] == "FUNCTIONAL_RECOVERY"
    assert "recovery_classification_requires_machine_uncertain" in result["warnings"]
