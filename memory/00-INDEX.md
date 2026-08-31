# memory/ — translation-forensics brain

Purpose: this folder is the durable memory for translation-forensics. Conversations forget; this folder does not. What is recorded here survives topic changes, session resets, and context compaction.

## File map

| File | What | Write rule |
|---|---|---|
| `DECISIONS.md` | Confirmed decisions | Append-only. Supersede protocol; never rewrite past entries |
| `OPEN-QUESTIONS.md` | Unresolved items and provisional readings | Close each row with the resolving decision or finding; never silently remove it |
| `SESSION-LOG.md` | What happened in each working session | Append, dated |
| `PRODUCT-TRUTH.md` | What the product demonstrably does | Evidence and check date only; keep implemented / not implemented / excluded separate |
| `CHECKPOINT.md` | Current thirty-second return point | Replace only after archiving the outgoing version |
| `checkpoints/` | Historical checkpoints | Append-only, timestamped filenames |
| `goal/` | Canonical goal terrain maps and atomic implementation skeletons | Version cuts; mark nodes filled only with attached evidence |
| `knowledge/` | Findings that passed the verification gate | Evidence, limits, date, and claim label required |
| `translation-memory.jsonl` | Approved translation memory records | Project schema and provenance rules apply |
| `scene-memory.jsonl` | Scene-level memory records | Project schema and provenance rules apply |
| `error-memory.jsonl` | Translation error memory records | Project schema and provenance rules apply |

## Operating principles

1. Record decisions and important facts in the same session in which they appear.
2. Distinguish user-confirmed decisions from AI-proposed or provisional readings.
3. Claims carry one of these labels: confirmed, observed, assumed, hearsay, or unknown.
4. External product claims require dated evidence in `PRODUCT-TRUTH.md`.
5. Register unresolved items instead of relying on conversational memory.
