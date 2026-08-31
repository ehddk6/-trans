# scene_v2 architecture (experimental)

`block_v1` remains the canonical, default `process-title` path. `scene_v2` is an opt-in design/implementation track and must be labelled `experimental-unbenchmarked` until a human benchmark exists.

## Contract

The intended pass order is: scene construction → semantic reconstruction → narrowly targeted visual observation → scene-level Korean dialogue realization → segmentation/projection → independent source-faithful generation → deterministic structure/semantic QA → semantic-drift critic → Korean-dialogue critic → targeted repair → repeat segmentation and both critics → unit decision adapter → existing dual-output packaging.

Viewer-natural generation and repair receive Japanese, semantic frames, and bounded scene context; they do not receive source-faithful Korean. The source-faithful artifact is an independent audit/debug output. Semantic and Korean-dialogue critics have separate roles, prompts, schemas, and artifacts; neither silently rewrites the other.

The final SRT remains structure-locked: same cue count, numbers, timestamps, order, UTF-8/LF, no empty cues, and every source unit consumed exactly once. Scene-level N:1/1:N alignment is projected back to those locked units.

Visual pixels are off in `off`/`metadata` modes. `targeted` may transfer only selected frames when unresolved speaker/addressee/deictic/on-screen-text/scene-continuity ambiguity has competing interpretations and major/critical translation impact; ordinary style or low confidence alone is not a trigger. Every transfer needs a receipt and hash.

The implemented `process-title --translation-architecture scene_v2` path is covered by synthetic fake-provider integration tests for its passes, resume behavior, artifacts, visual-policy boundary, and final SRT structure lock. It remains `experimental-unbenchmarked`: no human naturalness result, external Gemini baseline, or superiority claim is established.
