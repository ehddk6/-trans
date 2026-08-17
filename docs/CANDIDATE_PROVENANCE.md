# 한국어 후보 provenance

`run-closed-world`는 파일명이 다른 후보를 자동으로 독립 계열로 계산하지 않는다.

후보 SRT와 같은 디렉터리에 `candidate-provenance.json`을 두거나, 후보 파일 옆에 `<name>.srt.provenance.json` 또는 `<name>.provenance.json`을 둔다.

```json
{
  "schema_name": "translation-forensics/candidate-provenance",
  "schema_version": "1",
  "candidates": [
    {
      "path": "TITLE.source-faithful-ko.model-a-v1.srt",
      "sha256": "<64자리 SHA-256>",
      "source_family": "model-a/run-20260728",
      "model": "model-a",
      "run_id": "run-20260728",
      "parent_sha256": "<부모 산출물 SHA-256>",
      "prompt_sha256": "<프롬프트 SHA-256>"
    }
  ]
}
```

규칙은 다음과 같다.

- SRT SHA-256이 provenance 기록과 다르면 해당 기록을 사용하지 않는다.
- 같은 `source_family`는 파일 수와 관계없이 하나의 계열이다.
- `source_family`가 없으면 모델·실행 ID·부모 해시·프롬프트 해시의 조합으로 계열을 만든다.
- 검증 가능한 provenance가 없는 생성 후보는 모두 `unprovenanced-korean-candidates` 하나로 합친다.
- 기존 한국어 후보는 `previous-korean`으로 별도 기록하지만, 생성 후보의 독립 합의를 대신하지 않는다.
- `accepted`에는 최소 두 개의 독립된 생성 후보 계열이 필요하다.
