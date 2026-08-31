# Visual ablation

## CLI workflow

Build the experiment manifest without transferring pixels. The shuffled-image negative control needs its explicit guard and cannot be marked as a production output.

```powershell
python -m translation_forensics.cli build-visual-ablation-manifest `
  --project-root . --experiment-id scene-v2-visual-pilot `
  --scene-id SAMPLE-001 --scene-id SAMPLE-002 `
  --allow-negative-control `
  --output .\benchmark\visual-ablation-manifest.json

python -m translation_forensics.cli validate-visual-ablation-result `
  --project-root . --input .\benchmark\visual-result-a.json `
  --manifest .\benchmark\visual-ablation-manifest.json

python -m translation_forensics.cli summarize-visual-ablation `
  --project-root . --manifest .\benchmark\visual-ablation-manifest.json `
  --result .\benchmark\visual-result-a.json `
  --output .\benchmark\visual-ablation-summary.json
```

The experiment compares A text-only, B text+metadata, C text+targeted image, D text+broader scene visual context, and E shuffled/incongruent-image negative control. E is research-only and must never enter production output without an explicit experiment flag.

Measure semantic accuracy, speaker/addressee, deictic referent, on-screen text, continuity, unsupported visual additions, hallucination, Korean naturalness, cost, latency, and pixel-transfer count. Pixel transfer is opt-in and must be receipt-backed. Targeted visual is supported only if C improves B on the relevant subset without worsening the visual-unhelpful subset or increasing unsupported additions. If D is not significantly better than C, broader visual context is not recommended. No such result is currently established.
