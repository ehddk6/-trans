# Targeted machine retranslation v1

## Goal

Repair only the requested Japanese-to-Korean subtitle blocks in a machine-final package. Produce a replacement only when the Japanese source and local context support it. This is a machine revision, not a human approval or a claim of reference-answer equality.

## Evidence priority

1. Japanese reference SRT for the target block and its neighboring blocks
2. Local ASR candidates, only as supporting evidence; multiple Whisper profiles are one family, not independent votes
3. Existing Korean source/viewer drafts, only as text to improve or preserve—not as meaning evidence

## Decision rules

- Preserve question, negation, refusal/permission, speaker/agent, object, tense/aspect, direction, and intensity.
- Do not add sexual, relational, emotional, or visual detail absent from the supported text.
- Correct repeated end-card substitutions, placeholders, fragments, subject reversals, and malformed Korean only when the Japanese evidence supports a specific correction.
- When the Japanese source is corrupted, mixed-language, incomplete, or conflicts with ASR and context, return `decision: "hold"`. Do not invent a Korean sentence.
- `source_faithful_korean` must preserve the supported meaning. `viewer_natural_korean` may improve Korean only within the same meaning range.
- Use `confidence: "high"` only for clear Japanese evidence; use `medium` for context-dependent but supported reconstruction; use `low` only with `decision: "hold"`.

## Output

Read only the target blocks and their local context from the supplied workspace paths. Return exactly one JSON object conforming to the supplied JSON schema. Do not edit files, add Markdown, or include commentary outside the schema.

For each requested block, include one result object. `replace` must contain two non-empty Korean strings. `hold` must leave both Korean strings empty and explain the missing or conflicting evidence.
