# 합성 일본어→한국어 품질 회귀 제약

`tests/fixtures/translation-quality-regressions.json`은 실제 작품·사용자 자막을 포함하지 않는 19개 합성 사례다. 정답 문장 하나를 고정하는 대신, 통과 후보와 의도적 회귀 후보에 대해 보존·금지·상태·가독성 제약이 각각 작동하는지 검사한다.

자동 검사 범위는 존댓말·반말, 호칭, 용어·고유명사, 특정 정보 보존, unsupported inference, abstain, 독립 후보 계열 수, CPS다. 말투의 미묘한 자연스러움, 욕설의 문화적 등가, 서사상 관계 변화, 실제 청취 판단은 `human_review_required`로 남기며 자동 통과로 표현하지 않는다.

SSIS-908 Sol 수리 기록에서 재현된 화자 창작, 생략 대상 보충, 손상 단편 노출, 확인 화행 약화는 각각 `pilot_error_type`으로 연결했다. 최상위 `required_pilot_error_types` 중 하나라도 합성 사례에서 빠지면 회귀 묶음 전체가 실패한다.

```powershell
python -m translation_forensics.cli validate-quality-regressions `
  --input .\tests\fixtures\translation-quality-regressions.json
```

이 묶음은 외부 모델을 호출하거나 실제 번역 정확도 점수를 산출하지 않는다. 프롬프트·결정·일관성 계층을 변경했을 때 대표 실패를 다시 허용하지 않기 위한 회귀 계약이다.
