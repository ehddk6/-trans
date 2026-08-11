# Integrated Noisy-ASR Recovery Contract

Use this contract for the Terra translation stage of `process-title`. It returns
one JSON decision per locked `translation_units_ja.jsonl` unit; it does not write
SRT, calculate QA, or grant final status.

## Goal

Translate reliable Japanese faithfully and recover only the broad utterance function
when the ASR is demonstrably corrupted. Preserve the supplied `unit_id`, timing, and
source boundaries. Produce both `source_faithful_korean` and `viewer_natural_korean`.

## Evidence and recovery

1. Treat the supplied Japanese source as the authority unless its quality state or
   ASR warnings demonstrate a systemic error.
2. Classify every unit as exactly one of:
   - `RELIABLE`: source meaning and force can be translated directly.
   - `FUNCTIONAL_RECOVERY`: only a broad function is supported by local turns,
     repetition, timing, or a supplied evidence record.
   - `UNRESOLVED`: source quality is `unusable` and no broad function is supported.
3. For `FUNCTIONAL_RECOVERY`, do not invent actions, body parts, relationships,
   locations, consent, or a spoken proposition. Record the concrete `recovery_basis`.
4. `[불명]` is allowed only for `UNRESOLVED` with `source_quality_status=unusable`.
   Never use an ellipsis or an empty string as a substitute for translation.

## Invariants

Preserve question force, negation, refusal, permission, stop/continue commands,
and numeric tokens when they are reliable source evidence. Do not add information
outside source and approved contextual evidence. A low-confidence decision remains
machine-uncertain; never present it as a final translation.

## Output shape

Return JSON matching `TRANSLATION_BATCH_SCHEMA`: one decision per requested unit,
with both Korean fields, confidence, uncertainty slots, reason codes, optional
`recovery_classification`, and optional `recovery_basis`. The pipeline owns SRT
rendering, deterministic QA, visual evidence receipts, and release status.
