# Scene translation benchmark

## CLI workflow

Initialize the ledger with the scene IDs being compared:

```powershell
python -m translation_forensics.cli init-scene-benchmark `
  --project-root . --benchmark-id scene-v2-pilot `
  --scene-id SAMPLE-001 --scene-id SAMPLE-002
```

Ingest B only from a user-supplied external output, then build a blinded pack. No command calls Gemini.

```powershell
python -m translation_forensics.cli ingest-external-baseline `
  --project-root . --input .\external-baseline.json `
  --output .\benchmark\external-baseline.validated.json

python -m translation_forensics.cli build-scene-blind-review-pack `
  --project-root . --block-v1 .\benchmark\block-v1.json `
  --scene-v2 .\benchmark\scene-v2.json `
  --external-baseline .\benchmark\external-baseline.validated.json `
  --seed 20260831 --output .\benchmark\blind-pack.json `
  --internal-key .\benchmark\blind-key.json
```

After human review, validate before deblinding and summarize with the separately held key:

```powershell
python -m translation_forensics.cli validate-scene-review `
  --project-root . --pack .\benchmark\blind-pack.json `
  --review .\benchmark\human-review.json

python -m translation_forensics.cli summarize-scene-benchmark `
  --project-root . --pack .\benchmark\blind-pack.json `
  --review .\benchmark\human-review.json `
  --internal-key .\benchmark\blind-key.json `
  --output .\benchmark\summary.json
```

This benchmark is a blinded, scene-level comparison:

- A: current `block_v1`
- B: user-supplied external baseline (for example Gemini)
- C: `scene_v2`

B is an empty slot until the user supplies validated outputs; the project does not call Gemini or invent its results. Candidate order is independently shuffled per scene, with evaluator pack and internal key kept separate.

The semantic panel may inspect Japanese and records error category/span/severity. The naturalness panel sees scene-level Korean with system identity hidden and records translationese, response flow, endings, register, and dialogue consistency. Human benchmark results are currently **not evaluated**.

Report semantic error, critical/major error, qualified naturalness wins/losses, ties, readability preference, translationese, and dialogue inconsistency separately. A candidate with a critical semantic error cannot win on naturalness. Do not collapse axes into one weighted score. Targets are: versus block_v1, qualified naturalness win rate ≥65%, loss ≤15%, zero new critical errors, major-error non-inferiority, and no readability disadvantage; versus an external baseline, 95% lower bound of decisive qualified wins >50%, semantic non-inferiority, and no readability disadvantage. Targets are goals, not achieved results.
