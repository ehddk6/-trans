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


def test_audit_accepts_natural_korean_ordinals_and_cardinals():
    cases = [
        ("デビューから4本目?", "데뷔하고 네 번째 작품이죠?"),
        ("そうですね。2本目です。", "네, 두 번째예요."),
        ("まだ一回抜けてないの?", "아직 한 번도 안 빠졌어?"),
        ("1人分から始まった時", "한 사람 분부터 시작할 땐"),
        ("5キメラ同士", "다섯 키메라끼리"),
    ]
    for index, (source, translation) in enumerate(cases, 1):
        result = audit_translation_decision(
            unit_id=f"ordinal-{index}",
            source_japanese=source,
            source_faithful_korean=translation,
            viewer_natural_korean=translation,
            source_quality_status="trusted",
            confidence="high",
        )
        assert "numeric_token_not_preserved" not in result["hard_failures"]


def test_audit_ignores_lexical_kanji_and_embedded_rhetorical_questions():
    cases = [
        ("恥ずかしいんですけど。", "좀 부끄럽긴 한데요."),
        ("何でしょう?何も考えずに。", "뭐랄까, 아무 생각 없이요."),
        ("全く違う快感っていうんですか?を得られることができて。", "완전히 다른 쾌감을 느꼈어요."),
        ("多分今までの一面をお見せできるように。", "지금까지와 다른 모습을 보여드릴 수 있게."),
    ]
    for index, (source, translation) in enumerate(cases, 1):
        result = audit_translation_decision(
            unit_id=f"rhetorical-{index}",
            source_japanese=source,
            source_faithful_korean=translation,
            viewer_natural_korean=translation,
            source_quality_status="trusted",
            confidence="high",
        )
        assert result["hard_failures"] == []


def test_audit_accepts_natural_refusal_command_and_deictic_forms():
    cases = [
        ("いやー、まだまだですね。", "아뇨, 아직 멀었어요."),
        ("休憩してください。", "좀 쉬세요."),
        ("ちょっと待ってください。", "잠깐만요."),
        ("ここはいい?", "여긴 좋아?"),
    ]
    for index, (source, translation) in enumerate(cases, 1):
        result = audit_translation_decision(
            unit_id=f"natural-form-{index}",
            source_japanese=source,
            source_faithful_korean=translation,
            viewer_natural_korean=translation,
            source_quality_status="trusted",
            confidence="high",
        )
        assert result["hard_failures"] == []


def test_audit_keeps_momentary_aspect_as_a_targeted_semantic_guard():
    result = audit_translation_decision(
        unit_id="momentary",
        source_japanese="一瞬、上がってきたの?",
        source_faithful_korean="방금 올라온 거야?",
        viewer_natural_korean="방금 올라온 거야?",
        source_quality_status="trusted",
        confidence="high",
    )
    assert "momentary_aspect_not_preserved" in result["hard_failures"]


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
    assert result["status"] == "passed"
    assert result["suppressed_warnings"] == []
    assert "high_compression_ratio" in result["reasons"]


def test_high_compression_is_not_suppressed_for_kana_lexical_commands_or_locations():
    cases = [
        ("やって", "그래.", "request_command_not_preserved"),
        ("もっと", "됐어.", "continue_command_not_preserved"),
        ("ここ", "응.", "deictic_location_not_preserved"),
    ]
    for index, (source, translation, expected_failure) in enumerate(cases, 1):
        result = audit_translation_decision(
            unit_id=f"lexical-{index}",
            source_japanese=source,
            source_faithful_korean=translation,
            viewer_natural_korean=translation,
            source_quality_status="trusted",
            confidence="high",
            existing_reasons=["high_compression_ratio"],
        )
        assert result["status"] == "fallback"
        assert expected_failure in result["hard_failures"]
        assert result["suppressed_warnings"] == []
        assert "high_compression_ratio" in result["reasons"]


def test_permission_force_cannot_pass_as_plain_command_or_proposal():
    cases = [
        ("入れていいよ", "넣어."),
        ("してもいい？", "할까?"),
    ]
    for index, (source, translation) in enumerate(cases, 1):
        result = audit_translation_decision(
            unit_id=f"permission-{index}",
            source_japanese=source,
            source_faithful_korean=translation,
            viewer_natural_korean=translation,
            source_quality_status="trusted",
            confidence="high",
        )
        assert result["status"] == "fallback"
        assert "permission_not_preserved" in result["hard_failures"]
        assert result["permission_preserved"] is False


def test_permission_request_and_location_pass_when_preserved():
    cases = [
        ("入れていいよ", "넣어도 돼."),
        ("やって", "해 줘."),
        ("もっと", "더 해."),
        ("ここ", "여기."),
    ]
    for index, (source, translation) in enumerate(cases, 1):
        result = audit_translation_decision(
            unit_id=f"preserved-{index}",
            source_japanese=source,
            source_faithful_korean=translation,
            viewer_natural_korean=translation,
            source_quality_status="trusted",
            confidence="high",
        )
        assert result["status"] == "passed"
        assert result["hard_failures"] == []


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
    assert result["status"] == "passed"
    assert result["recovery_classification"] == "FUNCTIONAL_RECOVERY"
    assert "recovery_classification_requires_machine_uncertain" in result["warnings"]
