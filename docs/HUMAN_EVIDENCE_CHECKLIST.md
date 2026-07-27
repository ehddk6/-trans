# 사람 검수 증거 체크리스트

이 문서는 자동화가 대신할 수 없는 v2 완료 조건을 실제로 채우기 위한 작업 순서다. 모든 기록은 로컬 산출물에만 남기며, 민감한 미디어를 외부로 전송하지 않는다.

## 1. 시간축 담당자

각 작품에서 초·중·후반의 명확한 발화 또는 장면 앵커 3개 이상을 직접 확인해 다음 CSV를 만든다.

```text
anchor_id,srt_time_seconds,media_time_seconds,source
```

그 뒤 `validate-timeline`을 실행한다.

- `JUQ-439`, `JUQ-778`, `SSIS-575`, `SSIS-642`, `SSIS-652`는 현재 미디어가 SRT보다 짧아 다른 편집본 또는 올바른 미디어를 먼저 확인해야 한다.
- 나머지 8개 작품은 길이 범위는 충분하지만 앵커가 없어 `unresolved`다.
- 일정 오프셋이면 승인자·근거·범위를 담은 offset map을 별도로 기록한다. 자동 보정은 금지한다.

## 2. 일본어 청취·골드 담당자

`init-gold-record`으로 장면을 만들고 다음 순서를 따른다.

1. 두 청취자가 독립 전사를 작성한다.
2. 불일치가 있으면 조정자를 기록한다.
3. 허용 한국어 범위·금지 해석·오류 심각도를 입력한다.
4. 작품은 하나의 split에만 둔다. `sealed-test`는 개발에 재사용하지 않는다.
5. `validate-gold-record`와 `validate-gold-suite`을 통과시킨다.

## 3. 장면 검수 담당자

각 작품의 `intermediate/<title>.review-pack-v1/review.html`을 열어 원음·문맥·ASR 후보를 검토한다.

- `review-decisions.template.jsonl`의 각 블록에 결정과 evidence refs를 작성한다.
- `validate-review-decisions`로 빠진 블록·근거 없는 승인·중복 결정을 해결한다.
- 화자 상태, 음가 후보, semantic frame, hypothesis, 정렬 근거를 실제 증거가 있는 경우에만 채운다.

## 4. 번역·역검증 담당자

1. 구조 전체에 대해 source-faithful와 viewer-natural 결정을 완성한다.
2. `apply-translations --strict`를 통과시킨다.
3. 생성된 reverse semantic check 템플릿을 사람이 채운다.
4. `validate-reverse-check`에서 `meaning-flip`, `meaning-addition`, `unresolved`를 모두 해소하거나 보류한다.

## 5. 평가·감사 담당자

- 블라인드 A/B 패킷을 사람에게 배포하고, 결과를 `summarize-blind-review`으로 해제한다.
- P3/P4 감사 표본에 실제 `review_result`, 오류 유형, 심각도, 소요 시간을 입력한다.
- 완전한 사람 라벨과 조정된 골드 분모가 있을 때만 `evaluate-review-budget`로 5/10/20% 지표를 계산한다.

## 6. 최종 책임자

`validate-release-gate`에 gold suite, blind review, audit summary, metrics, 책임자 승인을 제공한다. 통과한 gate, 통과 시간축, 전체 블록 검수·직접 청취·근거 완결성이 모두 있을 때만 `package --stage final`을 실행한다.

이전 결과보다 개선되었다는 평가는 sealed-test 또는 블라인드 사람 판정 결과가 생긴 뒤에만 `improved`, `mixed`, `not-demonstrated`, `regressed` 중 하나로 기록한다.
