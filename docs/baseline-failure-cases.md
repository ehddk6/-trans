# Baseline failure cases

This ledger preserves known failure and stop conditions. It is not an error-rate estimate.

| ID | Trigger | Expected safe behavior | Evidence |
| --- | --- | --- | --- |
| F01 | More than one plausible input file for a role | Stop and require an explicit path | `test_discovery_stops_on_multiple_candidates` |
| F02 | Structure changes, Japanese residue, blank subtitle, or work tag | Fail validation/package | `test_validation_flags_japanese_placeholder_and_tags` |
| F03 | Same Whisper family produces several candidate texts | Count one source family, not independent votes | `test_asr_defaults_to_one_whisper_family_and_escalates_declared_slot_conflict` |
| F04 | Confirmed critical semantic slot conflict | Escalate; never auto-select a reading | `test_forensic_frames_escalate_critical_conflicts_and_do_not_double_count_family` |
| F05 | Media ends before a structure SRT block | Block audio clipping | `test_timeline_blocks_media_shorter_than_structure_and_does_not_claim_edit_identity` |
| F06 | Unreviewed evaluation summary or empty gold skeleton | Block evaluation/final package | `test_evaluation_package_rejects_empty_gold_or_unreviewed_summary` |
| F07 | Existing Korean carried over as the decision method | Reject strict application | `test_apply_translation_decisions_rejects_carryover_in_strict_mode` |

No listed case establishes translation quality; each only demonstrates a guardrail.
