# Targeted machine retranslation review v1

## Goal

Adversarially review proposed Korean replacements against the quoted Japanese evidence. This is a machine check; do not claim human review or reference-answer equality.

## Rules

- Keep a replacement only when it preserves the supported Japanese meaning, including polarity, question form, speaker/agent, object, tense/aspect, and intensity.
- Revert when the proposal adds unsupported detail, changes the subject/object, or turns a damaged source into a confident meaning.
- Do not repair wording in this pass. Judge only `keep` or `revert`.
- All evidence is embedded in the request. Do not use tools or files.

## Output

Return exactly one JSON object matching the supplied schema, with one review for every quoted candidate.
