from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

CRITICAL_SLOTS = (
    "speech_act",
    "question",
    "polarity",
    "refusal_permission",
    "command_strength",
    "speaker",
    "actor",
    "action",
    "target",
    "location",
    "tense_aspect",
    "direction",
    "intensity",
)

MEANING_FLIPPING_SLOTS = frozenset(
    {
        "speech_act",
        "question",
        "polarity",
        "refusal_permission",
        "command_strength",
    }
)
STRUCTURAL_SLOTS = frozenset({"speaker", "actor", "action", "target"})
DESCRIPTIVE_SLOTS = frozenset({"location", "tense_aspect", "direction", "intensity"})

RECOVERY_STATES = frozenset(
    {
        "accepted_consensus",
        "accepted_partial",
        "recovered_context",
        "recovered_single_model",
        "minimal_speech_act",
        "vocalization",
        "abstained",
    }
)
PROVENANCE_STATES = frozenset(
    {
        "dual_agreement",
        "dual_compatible",
        "terra_only_supported",
        "sol_only_supported",
        "conflict_omitted",
        "conflict_blocking",
        "absent",
    }
)

_SELF_CONTAINED_MINIMAL_ACTS = frozenset(
    {"refusal", "prohibition", "response", "vocalization", "nonverbal"}
)
_SEMANTIC_SLOTS = frozenset(
    {"speech_act", "question", "polarity", "refusal_permission", "command_strength", "action"}
)


def normalize_slot_value(slot: str, value: Any) -> Any:
    """Normalize one frame slot without converting unknown values into evidence."""
    if value is None or isinstance(value, bool):
        normalized = value
    else:
        normalized = re.sub(r"\s+", " ", str(value).strip().casefold()) or None
    if normalized in {None, "unknown"}:
        return None
    if slot in {"refusal_permission", "command_strength"} and normalized == "none":
        return None
    if slot == "polarity" and normalized == "neutral":
        return None
    return normalized


def _fusion_family_refs(acoustic_record: Mapping[str, Any]) -> list[set[str]]:
    fusion = acoustic_record.get("asr_fusion") or {}
    hypotheses = fusion.get("family_hypotheses") or []
    result: list[set[str]] = []
    for row in hypotheses:
        if not isinstance(row, Mapping):
            continue
        refs = {
            str(ref).strip()
            for ref in (row.get("evidence_refs") or [])
            if str(ref).strip()
        }
        if refs:
            result.append(refs)
    return result


def _independent_support(
    frame: Mapping[str, Any],
    *,
    slot: str,
    acoustic_record: Mapping[str, Any],
    source_quality_record: Mapping[str, Any],
    block_number: int | None,
) -> tuple[bool, list[str], str]:
    support_map = frame.get("slot_support") or {}
    support = support_map.get(slot) if isinstance(support_map, Mapping) else None
    if not isinstance(support, Mapping):
        return False, [], ""

    basis = str(support.get("basis") or "none")
    cited = {
        str(ref).strip()
        for ref in (support.get("evidence_refs") or [])
        if str(ref).strip()
    }
    if basis == "source-text":
        if source_quality_record.get("source_quality_status") != "trusted":
            return False, [], basis
        source_ref = (
            f"source-srt:block-{block_number}"
            if block_number is not None
            else "source-srt:trusted-block"
        )
        # A model may not manufacture source-text corroboration merely by naming
        # the basis. The exact deterministic block reference must be cited.
        if source_ref not in cited:
            return False, [], basis
        return True, sorted(cited), basis

    if basis != "dual-asr":
        return False, [], basis

    fusion = acoustic_record.get("asr_fusion") or {}
    if fusion.get("state") not in {"dual_agreement", "dual_compatible"}:
        return False, [], basis

    alignment_strength = str(fusion.get("alignment_strength") or "none")
    if slot in MEANING_FLIPPING_SLOTS | STRUCTURAL_SLOTS:
        if alignment_strength != "block-aligned":
            return False, [], basis
    elif alignment_strength not in {"block-aligned", "expanded"}:
        return False, [], basis

    family_refs = _fusion_family_refs(acoustic_record)
    if len(family_refs) < 2:
        return False, [], basis
    if not cited or any(not (cited & refs) for refs in family_refs):
        return False, [], basis
    return True, sorted(cited), basis


def derive_slot_corroboration(
    terra_frame: Mapping[str, Any],
    sol_frame: Mapping[str, Any],
    *,
    acoustic_record: Mapping[str, Any],
    source_quality_record: Mapping[str, Any],
    block_number: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Find model-specific slot values independently corroborated by source evidence.

    This function never compares model identities by preference. It emits a
    model choice only when exactly one competing value has independently
    validated source-text or dual-family acoustic support.
    """

    result: dict[str, dict[str, Any]] = {}
    for slot in CRITICAL_SLOTS:
        terra_value = normalize_slot_value(slot, terra_frame.get(slot))
        sol_value = normalize_slot_value(slot, sol_frame.get(slot))
        if terra_value == sol_value:
            continue

        terra_supported, terra_refs, terra_basis = _independent_support(
            terra_frame,
            slot=slot,
            acoustic_record=acoustic_record,
            source_quality_record=source_quality_record,
            block_number=block_number,
        )
        sol_supported, sol_refs, sol_basis = _independent_support(
            sol_frame,
            slot=slot,
            acoustic_record=acoustic_record,
            source_quality_record=source_quality_record,
            block_number=block_number,
        )
        if terra_supported == sol_supported:
            continue
        if terra_supported:
            result[slot] = {
                "model": "terra",
                "independent": True,
                "evidence_refs": terra_refs,
                "basis": terra_basis,
            }
        else:
            result[slot] = {
                "model": "sol",
                "independent": True,
                "evidence_refs": sol_refs,
                "basis": sol_basis,
            }
    return result


def _corroborated_model(
    slot: str,
    corroboration: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[str | None, list[str]]:
    if not corroboration:
        return None, []
    record = corroboration.get(slot)
    if not isinstance(record, Mapping) or record.get("independent") is not True:
        return None, []
    model = str(record.get("model") or "").strip().casefold()
    refs = sorted({str(ref).strip() for ref in record.get("evidence_refs", []) if str(ref).strip()})
    if model not in {"terra", "sol"} or not refs:
        return None, []
    return model, refs


_OPPOSITE_DIRECTIONS = frozenset(
    {
        frozenset({"toward", "away"}),
        frozenset({"in", "out"}),
        frozenset({"up", "down"}),
        frozenset({"stop", "continue"}),
    }
)
_COMPLETION_ASPECTS = frozenset({"past", "completed"})
_NONCOMPLETION_ASPECTS = frozenset({"present", "future", "progressive"})
_FORCEFUL_SPEECH_ACTS = frozenset({"command", "prohibition", "refusal"})


def _conflict_is_render_blocking(
    slot: str,
    terra_values: Mapping[str, Any],
    sol_values: Mapping[str, Any],
) -> bool:
    """Promote descriptive conflicts only when they can reverse core meaning."""
    if slot in MEANING_FLIPPING_SLOTS or slot == "action":
        return True
    left = terra_values.get(slot)
    right = sol_values.get(slot)
    if left is None or right is None or left == right:
        return False
    if slot == "direction":
        return frozenset({str(left), str(right)}) in _OPPOSITE_DIRECTIONS
    if slot == "tense_aspect":
        pair = {str(left), str(right)}
        return bool(pair & _COMPLETION_ASPECTS and pair & _NONCOMPLETION_ASPECTS)
    if slot == "intensity" and {str(left), str(right)} == {"low", "high"}:
        # When both models already agree on explicit command strength, intensity
        # is redundant descriptive detail and can be omitted safely.
        left_strength = terra_values.get("command_strength")
        right_strength = sol_values.get("command_strength")
        if left_strength is not None and left_strength == right_strength:
            return False
        acts = {
            str(terra_values.get("speech_act") or ""),
            str(sol_values.get("speech_act") or ""),
        }
        refusal_permission = {
            str(terra_values.get("refusal_permission") or ""),
            str(sol_values.get("refusal_permission") or ""),
        }
        return bool(
            acts & _FORCEFUL_SPEECH_ACTS
            or refusal_permission & {"refusal", "permission", "prohibition"}
        )
    return False


def _slot_record(
    *,
    slot: str,
    terra_value: Any,
    sol_value: Any,
    corroboration: Mapping[str, Mapping[str, Any]] | None,
    blocking_conflict: bool,
) -> dict[str, Any]:
    if terra_value == sol_value and terra_value is not None:
        return {
            "terra_value": terra_value,
            "sol_value": sol_value,
            "selected_value": terra_value,
            "provenance": "dual_agreement",
            "disposition": "render",
            "evidence_refs": [],
            "confidence": "high",
            "rationale_code": "independent-model-agreement",
        }
    if terra_value is None and sol_value is None:
        return {
            "terra_value": None,
            "sol_value": None,
            "selected_value": None,
            "provenance": "absent",
            "disposition": "omit",
            "evidence_refs": [],
            "confidence": "low",
            "rationale_code": "unsupported-by-both-models",
        }

    corroborated_model, evidence_refs = _corroborated_model(slot, corroboration)
    if terra_value is not None and sol_value is not None and terra_value != sol_value:
        if corroborated_model == "terra":
            return {
                "terra_value": terra_value,
                "sol_value": sol_value,
                "selected_value": terra_value,
                "provenance": "terra_only_supported",
                "disposition": "render",
                "evidence_refs": evidence_refs,
                "confidence": "medium",
                "rationale_code": "independent-evidence-supports-terra",
            }
        if corroborated_model == "sol":
            return {
                "terra_value": terra_value,
                "sol_value": sol_value,
                "selected_value": sol_value,
                "provenance": "sol_only_supported",
                "disposition": "render",
                "evidence_refs": evidence_refs,
                "confidence": "medium",
                "rationale_code": "independent-evidence-supports-sol",
            }
        blocking = blocking_conflict
        return {
            "terra_value": terra_value,
            "sol_value": sol_value,
            "selected_value": None,
            "provenance": "conflict_blocking" if blocking else "conflict_omitted",
            "disposition": "block" if blocking else "omit",
            "evidence_refs": [],
            "confidence": "low",
            "rationale_code": (
                "uncorroborated-meaning-flipping-conflict"
                if blocking
                else "uncorroborated-conflict-omitted"
            ),
        }

    supplied_by = "terra" if terra_value is not None else "sol"
    supplied_value = terra_value if terra_value is not None else sol_value
    if corroborated_model == supplied_by:
        return {
            "terra_value": terra_value,
            "sol_value": sol_value,
            "selected_value": supplied_value,
            "provenance": f"{supplied_by}_only_supported",
            "disposition": "render",
            "evidence_refs": evidence_refs,
            "confidence": "medium",
            "rationale_code": f"independent-evidence-supports-{supplied_by}-coverage-gap",
        }
    return {
        "terra_value": terra_value,
        "sol_value": sol_value,
        "selected_value": None,
        "provenance": "absent",
        "disposition": "omit",
        "evidence_refs": [],
        "confidence": "low",
        "rationale_code": "uncorroborated-single-model-value",
    }


def _finding(
    code: str,
    severity: str,
    slots: Sequence[str],
    remediation: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "slots": list(slots),
        "remediation": remediation,
    }


def coherence_check(selected_frame: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a conservative frame and explicit coherence findings."""
    frame = {slot: selected_frame.get(slot) for slot in CRITICAL_SLOTS}
    findings: list[dict[str, Any]] = []

    speech_act = frame["speech_act"]
    question = frame["question"]
    if question is True and speech_act not in {None, "question", "request"}:
        findings.append(
            _finding(
                "question-speech-act-mismatch",
                "high",
                ("speech_act", "question"),
                "minimal-or-abstain",
            )
        )
    if speech_act == "question" and question is False:
        findings.append(
            _finding(
                "question-speech-act-mismatch",
                "high",
                ("speech_act", "question"),
                "minimal-or-abstain",
            )
        )

    refusal_permission = frame["refusal_permission"]
    compatible_acts = {
        "refusal": {"refusal", "response"},
        "permission": {"permission", "response"},
        "prohibition": {"prohibition", "command", "response"},
    }
    if refusal_permission in compatible_acts and speech_act not in compatible_acts[refusal_permission] | {None}:
        findings.append(
            _finding(
                "refusal-permission-speech-act-mismatch",
                "high",
                ("speech_act", "refusal_permission"),
                "minimal-or-abstain",
            )
        )

    command_strength = frame["command_strength"]
    if command_strength in {"suggestion", "request", "command", "urgent"}:
        if speech_act not in {None, "request", "command", "prohibition"}:
            findings.append(
                _finding(
                    "command-strength-speech-act-mismatch",
                    "medium",
                    ("speech_act", "command_strength"),
                    "omit-command-strength",
                )
            )
            frame["command_strength"] = None

    if speech_act in {"vocalization", "nonverbal"}:
        removable = [
            slot
            for slot in ("speaker", "actor", "action", "target", "location", "tense_aspect", "direction")
            if frame.get(slot) is not None
        ]
        if removable:
            findings.append(
                _finding(
                    "nonlexical-frame-retains-semantic-detail",
                    "medium",
                    tuple(removable),
                    "omit-semantic-detail",
                )
            )
            for slot in removable:
                frame[slot] = None

    if not any(frame.get(slot) is not None for slot in _SEMANTIC_SLOTS):
        findings.append(
            _finding(
                "empty-semantic-core",
                "high",
                tuple(sorted(_SEMANTIC_SLOTS)),
                "abstain",
            )
        )

    return frame, findings


def _minimal_speech_act_available(frame: Mapping[str, Any]) -> bool:
    speech_act = frame.get("speech_act")
    if speech_act in _SELF_CONTAINED_MINIMAL_ACTS:
        return True
    return bool(
        speech_act in {"statement", "question", "request", "command", "permission"}
        and frame.get("action") is not None
    )


def _recovery_state(
    *,
    selected_frame: Mapping[str, Any],
    conflicts: Sequence[str],
    coverage_gaps: Sequence[str],
    slot_provenance: Mapping[str, Mapping[str, Any]],
    render_blocking_conflicts: Sequence[str],
    coherence_findings: Sequence[Mapping[str, Any]],
    context_retried: bool,
) -> str:
    speech_act = selected_frame.get("speech_act")
    if speech_act in {"vocalization", "nonverbal"} and not render_blocking_conflicts:
        return "vocalization"

    high_findings = [finding for finding in coherence_findings if finding.get("severity") == "high"]
    if render_blocking_conflicts or high_findings:
        if (
            not any(slot in MEANING_FLIPPING_SLOTS for slot in render_blocking_conflicts)
            and _minimal_speech_act_available(selected_frame)
            and not high_findings
        ):
            return "minimal_speech_act"
        return "abstained"

    if any(
        record.get("provenance") in {"terra_only_supported", "sol_only_supported"}
        for record in slot_provenance.values()
    ):
        return "recovered_single_model"
    if context_retried:
        return "recovered_context"
    if not conflicts and not coverage_gaps:
        return "accepted_consensus"
    if _minimal_speech_act_available(selected_frame):
        return "accepted_partial"
    return "abstained"


def build_frame_agreement(
    terra_frame: Mapping[str, Any],
    sol_frame: Mapping[str, Any],
    *,
    context_retried: bool = False,
    corroboration: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a slot-aware arbitration record from two independent frames."""
    terra_values = {
        slot: normalize_slot_value(slot, terra_frame.get(slot))
        for slot in CRITICAL_SLOTS
    }
    sol_values = {
        slot: normalize_slot_value(slot, sol_frame.get(slot))
        for slot in CRITICAL_SLOTS
    }
    conflicts = [
        slot
        for slot in CRITICAL_SLOTS
        if terra_values[slot] is not None
        and sol_values[slot] is not None
        and terra_values[slot] != sol_values[slot]
    ]
    coverage_gaps = [
        slot
        for slot in CRITICAL_SLOTS
        if (terra_values[slot] is None) != (sol_values[slot] is None)
    ]

    slot_provenance = {
        slot: _slot_record(
            slot=slot,
            terra_value=terra_values[slot],
            sol_value=sol_values[slot],
            corroboration=corroboration,
            blocking_conflict=_conflict_is_render_blocking(
                slot,
                terra_values,
                sol_values,
            ),
        )
        for slot in CRITICAL_SLOTS
    }
    selected_frame = {
        slot: slot_provenance[slot]["selected_value"]
        for slot in CRITICAL_SLOTS
    }
    selected_frame, coherence_findings = coherence_check(selected_frame)

    # Coherence remediation may remove a selected value; reflect that in provenance.
    for slot in CRITICAL_SLOTS:
        record = slot_provenance[slot]
        if record["selected_value"] is not None and selected_frame[slot] is None:
            record["selected_value"] = None
            record["disposition"] = "omit"
            record["rationale_code"] = "omitted-by-coherence-check"

    render_blocking_conflicts = sorted(
        slot
        for slot, record in slot_provenance.items()
        if record["disposition"] == "block"
    )
    resolved_conflicts = sorted(
        slot
        for slot in conflicts
        if slot_provenance[slot]["provenance"] in {"terra_only_supported", "sol_only_supported"}
    )
    omitted_slots = sorted(
        slot
        for slot, record in slot_provenance.items()
        if record["selected_value"] is None
    )
    rendered_slots = sorted(
        slot
        for slot, record in slot_provenance.items()
        if record["selected_value"] is not None
    )
    uncertainty_codes = sorted(
        {
            str(record["rationale_code"])
            for record in slot_provenance.values()
            if record["confidence"] != "high"
        }
        | {
            str(finding["code"])
            for finding in coherence_findings
        }
    )
    state = _recovery_state(
        selected_frame=selected_frame,
        conflicts=conflicts,
        coverage_gaps=coverage_gaps,
        slot_provenance=slot_provenance,
        render_blocking_conflicts=render_blocking_conflicts,
        coherence_findings=coherence_findings,
        context_retried=context_retried,
    )

    return {
        "schema_name": "translation-forensics/codex-frame-agreement",
        "schema_version": "2",
        "critical_slot_conflicts": conflicts,
        "slot_coverage_gaps": coverage_gaps,
        "consensus_frame": {
            slot: terra_values[slot] if terra_values[slot] == sol_values[slot] else None
            for slot in CRITICAL_SLOTS
        },
        "selected_frame": selected_frame,
        "slot_provenance": slot_provenance,
        "rendered_slots": rendered_slots,
        "omitted_slots": omitted_slots,
        "render_blocking_conflicts": render_blocking_conflicts,
        "resolved_conflicts": resolved_conflicts,
        "coherence_findings": coherence_findings,
        "recovery_state": state,
        "uncertainty_codes": uncertainty_codes,
        "agreed": not conflicts,
        "fully_agreed": not conflicts and not coverage_gaps,
    }


def needs_context_rerun(agreement: Mapping[str, Any]) -> bool:
    """Retry only blocks whose remaining conflict prevents a safe full rendering."""
    return bool(agreement.get("render_blocking_conflicts"))


def compatibility_status(recovery_state: str) -> tuple[str, str]:
    if recovery_state in {"accepted_consensus", "accepted_partial", "recovered_context", "vocalization"}:
        return "accepted", "supported"
    if recovery_state in {"recovered_single_model", "minimal_speech_act"}:
        return "unresolved", "best_effort"
    return "abstained", "unrecoverable"


def _has_dual_acoustic(acoustic_record: Mapping[str, Any]) -> bool:
    families = set(acoustic_record.get("independent_source_families", []))
    if len(families) < 2:
        return False
    fusion = acoustic_record.get("asr_fusion")
    if not isinstance(fusion, Mapping):
        # Backward-compatible read path for existing v1 evidence.
        return True
    return (
        fusion.get("state") in {"dual_agreement", "dual_compatible"}
        and fusion.get("alignment_strength") in {"block-aligned", "expanded"}
        and not fusion.get("risk_codes")
    )


def graduated_source_status(
    agreement: Mapping[str, Any],
    source_quality_record: Mapping[str, Any],
    acoustic_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive compatibility statuses without using them as the arbitration source."""
    state = str(agreement.get("recovery_state") or "abstained")
    reason_codes = list(agreement.get("uncertainty_codes", []))
    quality = str(source_quality_record.get("source_quality_status") or "unusable")

    if quality in {"suspect", "unusable"} and not _has_dual_acoustic(acoustic_record):
        selected = agreement.get("selected_frame") or {}
        if state != "vocalization" and _minimal_speech_act_available(selected):
            state = "minimal_speech_act"
            reason_codes.append("damaged-source-without-dual-acoustic-evidence")
        elif state != "vocalization":
            state = "abstained"
            reason_codes.append("damaged-source-without-dual-acoustic-evidence")

    source_status, viewer_status = compatibility_status(state)
    return {
        "recovery_state": state,
        "source_status": source_status,
        "viewer_status": viewer_status,
        "uncertainty_codes": sorted(set(reason_codes)),
    }


def _candidate_text(row: Mapping[str, Any], *, conservative: bool) -> tuple[str, str]:
    if conservative:
        return (
            str(row.get("conservative_source_faithful_korean") or "").strip(),
            str(row.get("conservative_viewer_natural_korean") or "").strip(),
        )
    return (
        str(row.get("source_faithful_korean") or "").strip(),
        str(row.get("viewer_natural_korean") or "").strip(),
    )


def apply_graduated_evidence_gate(
    rows: Sequence[Mapping[str, Any]],
    agreements: Mapping[int, Mapping[str, Any]],
    source_quality: Mapping[int, Mapping[str, Any]],
    acoustic: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply arbitration outcomes while retaining only slot-safe candidates."""
    enforced: list[dict[str, Any]] = []
    for row in rows:
        number = int(row["block_number"])
        agreement = agreements[number]
        status = graduated_source_status(
            agreement,
            source_quality[number],
            acoustic[number],
        )
        output = dict(row)
        output.update(status)
        output.setdefault(
            "reason",
            "Translation is constrained by the slot-level arbitration record.",
        )

        state = status["recovery_state"]
        if state == "vocalization":
            controlled = str(
                agreement.get("controlled_nonlexical_korean") or ""
            ).strip()
            if controlled:
                output.update(
                    {
                        "source_faithful_korean": controlled,
                        "viewer_natural_korean": controlled,
                        "conservative_source_faithful_korean": controlled,
                        "conservative_viewer_natural_korean": controlled,
                        "source_status": "accepted",
                        "viewer_status": "supported",
                        "recovery_state": "vocalization",
                        "fallback_recovery_state": "vocalization",
                        "rendered_slots": ["speech_act"],
                        "fallback_rendered_slots": ["speech_act"],
                        "fallback_evidence_safe": True,
                        "reason": "controlled non-lexical rendering from routed acoustic evidence",
                    }
                )
                enforced.append(output)
                continue
            state = "abstained"
        use_conservative = state == "minimal_speech_act"
        allowed_slots = set(agreement.get("rendered_slots", []))
        primary_slots = {
            str(slot) for slot in output.get("rendered_slots", []) if str(slot)
        }
        fallback_slots = {
            str(slot)
            for slot in output.get("fallback_rendered_slots", [])
            if str(slot)
        }
        requested_fallback_state = str(
            output.get("fallback_recovery_state") or "minimal_speech_act"
        )
        primary_missing_claims = (
            state not in {"vocalization", "abstained"} and not primary_slots
        )
        fallback_missing_claims = (
            requested_fallback_state not in {"vocalization", "abstained"}
            and not fallback_slots
        )
        primary_overclaims = (
            not primary_slots.issubset(allowed_slots)
            or primary_missing_claims
        )
        fallback_overclaims = (
            not fallback_slots.issubset(allowed_slots)
            or fallback_missing_claims
        )
        fallback_source, fallback_viewer = _candidate_text(
            output,
            conservative=True,
        )
        fallback_evidence_safe = bool(
            not fallback_overclaims
            and fallback_source
            and fallback_viewer
            and fallback_source != "…"
            and fallback_viewer != "…"
            and requested_fallback_state != "abstained"
        )
        if fallback_overclaims:
            output.update(
                {
                    "conservative_source_faithful_korean": "…",
                    "conservative_viewer_natural_korean": "…",
                    "fallback_recovery_state": "abstained",
                    "fallback_rendered_slots": [],
                    "fallback_evidence_safe": False,
                }
            )
        else:
            output["fallback_evidence_safe"] = fallback_evidence_safe

        if primary_missing_claims:
            status["uncertainty_codes"] = sorted(
                set(status["uncertainty_codes"])
                | {"translation-missing-rendered-slots"}
            )
        if primary_overclaims:
            status["uncertainty_codes"] = sorted(
                set(status["uncertainty_codes"])
                | {"translation-claims-omitted-slot"}
            )
            if not fallback_overclaims:
                use_conservative = True
                state = requested_fallback_state
            else:
                state = "abstained"

        source_text, viewer_text = _candidate_text(
            output,
            conservative=use_conservative,
        )
        chosen_slots = fallback_slots if use_conservative else primary_slots
        if state == "abstained" or not source_text or not viewer_text:
            output.update(
                {
                    "source_faithful_korean": "…",
                    "viewer_natural_korean": "…",
                    "source_status": "abstained",
                    "viewer_status": "unrecoverable",
                    "recovery_state": "abstained",
                    "rendered_slots": [],
                    "reason": "; ".join(status["uncertainty_codes"])
                    or "no-safe-semantic-core",
                }
            )
        else:
            source_status, viewer_status = compatibility_status(state)
            output.update(
                {
                    "source_faithful_korean": source_text,
                    "viewer_natural_korean": viewer_text,
                    "source_status": source_status,
                    "viewer_status": viewer_status,
                    "recovery_state": state,
                    "rendered_slots": sorted(chosen_slots),
                    "uncertainty_codes": sorted(
                        set(output.get("uncertainty_codes", []))
                        | set(status["uncertainty_codes"])
                    ),
                }
            )
        enforced.append(output)
    return enforced

def apply_review_outcomes(
    rows: Sequence[Mapping[str, Any]],
    reviews: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply Sol reviews claim by claim, preferring a safe fallback over deletion."""
    by_number = {int(review["block_number"]): review for review in reviews}
    result: list[dict[str, Any]] = []
    for row in rows:
        number = int(row["block_number"])
        review = by_number[number]
        verdict = str(review.get("verdict") or "quarantine")
        findings = review.get("claim_findings") or []
        blocking_findings = [
            finding
            for finding in findings
            if isinstance(finding, Mapping)
            and finding.get("disposition") == "blocking"
        ]
        # v2 claim findings control disposition. The historical conflict list is
        # diagnostic and must not recreate block-level unanimity. For a legacy
        # response with no claim findings, retain the conservative behavior.
        legacy_blocking = not findings and bool(
            review.get("critical_slot_conflicts")
        )
        unsupported = bool(review.get("unsupported_additions"))
        primary_safe = (
            verdict == "accept"
            and not blocking_findings
            and not legacy_blocking
            and not unsupported
        )
        if primary_safe:
            result.append(dict(row))
            continue

        fallback_source, fallback_viewer = _candidate_text(
            row,
            conservative=True,
        )
        fallback_state = str(
            row.get("fallback_recovery_state") or "minimal_speech_act"
        )
        fallback_slots = {
            str(slot)
            for slot in row.get("fallback_rendered_slots", [])
            if str(slot)
        }
        fallback_allowed = (
            bool(fallback_source and fallback_viewer)
            and fallback_source != "…"
            and fallback_viewer != "…"
            and fallback_state != "abstained"
            and row.get("fallback_evidence_safe") is True
            and (
                bool(fallback_slots)
                or fallback_state == "vocalization"
            )
            and not bool(review.get("fallback_blocking"))
        )
        if fallback_allowed:
            recovered = dict(row)
            source_status, viewer_status = compatibility_status(
                fallback_state
            )
            recovered.update(
                {
                    "source_faithful_korean": fallback_source,
                    "viewer_natural_korean": fallback_viewer,
                    "source_status": source_status,
                    "viewer_status": viewer_status,
                    "recovery_state": fallback_state,
                    "rendered_slots": sorted(
                        {
                            str(slot)
                            for slot in row.get(
                                "fallback_rendered_slots",
                                [],
                            )
                            if str(slot)
                        }
                    ),
                    "uncertainty_codes": sorted(
                        set(row.get("uncertainty_codes", []))
                        | {"sol-selected-conservative-fallback"}
                    ),
                    "reason": (
                        "Sol selected conservative fallback: "
                        f"{review.get('reason', '')}"
                    ).strip(),
                }
            )
            result.append(recovered)
            continue

        result.append(
            {
                **row,
                "source_faithful_korean": "…",
                "viewer_natural_korean": "…",
                "source_status": "abstained",
                "viewer_status": "unrecoverable",
                "recovery_state": "abstained",
                "rendered_slots": [],
                "uncertainty_codes": sorted(
                    set(row.get("uncertainty_codes", []))
                    | {"sol-found-no-safe-candidate"}
                ),
                "reason": (
                    "Sol found no safe candidate: "
                    f"{review.get('reason', '')}"
                ).strip(),
            }
        )
    return result
