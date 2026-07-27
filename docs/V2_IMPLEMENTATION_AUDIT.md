# v2 구현 완료 감사

기준: `번역 포렌식 프로젝트 개선 구현 계획 v2`의 Phase 0–12와 폐쇄형 자동 검증 확장. 이 문서는 코드 경로와 실제 검증 증거를 분리한다. 사람 청취·판정이 없는 상태를 완료 또는 품질 개선으로 표시하지 않는다.

| Phase | 코드 구현 상태 | 주요 근거 | 실제 데이터/승격 상태 |
| --- | --- | --- | --- |
| 0. 현재 구현 감사 | 구현 | `current-implementation-audit.md`, `baseline-capability-matrix.csv`, `baseline-failure-cases.md` | 감사는 코드·테스트 근거이며 품질 평가는 아님 |
| 1. ID·스키마·재현성 | 구현 | `identity.py`, `migrate-identity`, `validate-identity`, 자동 package run manifest | 기존 산출물은 명시적 마이그레이션 전 legacy일 수 있음 |
| 2. 시간축 | 구현 | `validate-timeline`, 다중 앵커·승인 offset map, package gate | 실제 작품별 3개 앵커와 승인 맵은 아직 사람 입력 필요 |
| 3. 골드·평가 계약 | 구현 | gold record/suite, split 누출 검사, blind pack | adjudicated gold와 sealed-test 결과 없음 |
| 4. 의미 판정 | 구현 | semantic frames, hypothesis ledger, slot conflicts, speaker/phonetic templates | 음가·화자·정렬 판정은 사람 근거가 필요 |
| 5. Terra 이중 생성 | 구현 | versioned prompt contract, decision JSONL, strict apply | 모델 결과는 검수 전 후보이며 실제 품질 개선 미입증 |
| 6. 역의미 검증 | 구현 | `init-reverse-check`, `validate-reverse-check` | 사람 작성 역검증 레코드 없음 |
| 7. 검수 패킷 | 구현 | local review HTML, decision coverage/evidence validation, blind pack | 실제 검수 결정 없음 |
| 8. 능동 검수·감사 | 구현 | seeded sampling, audit summary, guarded 5/10/20% metrics | 완전 라벨·골드 분모 없이는 `not-demonstrated` |
| 9. 용어·메모리 | 구현 | terminology validation/conflicts, scoped memory ledgers | 승인 항목·작품 간 재사용 근거 없음 |
| 10. 근거 그래프 | 구현 | graph JSONL, provenance JSON, orphan CSV | 실제 final block 계보는 입력 결정·평가 근거가 있어야 완성 |
| 11. Sol 반증 | 구현 | counterexample-only record contract, transfer/cost/human-verdict guard | 이 환경에서 Sol 호출·독립 실험은 수행되지 않음 |
| 12. 릴리스·final | 구현 | release metric and gate validator, timeline/evaluation/release-gate package requirements | gold·blind·audit·책임자 승인 증거가 없어 `final` 차단 |
| 확장. 폐쇄형 자동 검증 | 구현·13개 작품 실행 | `run-closed-world`, `validate-closed-world`, `prove-quality-claim`, 전용 스키마·결정적 매니페스트 | 10,380블록 전부 판정, 870 accepted·9,510 abstained, 사람 정답 동일성은 `unidentifiable` |

## 검증 명령

```powershell
python -m pytest
python -m translation_forensics.cli --help
python -m translation_forensics.cli validate-prompt-contract `
  --manifest .\prompts\terra-semantic-translation-v1.manifest.json
```

최근 전체 회귀 결과: **60 passed**.

## 2026-07-26 실제 준비 산출물

- 13개 작품에 실제 일본어 SRT·MP3로 time validation 보고서를 기록했다. 실제 집계는 5개가 `media-too-short`, 8개가 다중 앵커 부재로 `unresolved`이며 어느 작품도 사람 검증 클립·`final` 승격 허용 상태가 아니다.
- 13개 작품에 로컬 review pack, speaker-state 템플릿, phonetic candidate 템플릿, evidence graph, P3/P4 audit sample, reverse semantic check 템플릿, audit summary를 생성했다.
- 각 audit summary는 사람 라벨이 없으므로 `not-demonstrated`다.
- 13개 작품에 early/middle/late 앵커 템플릿을 생성했다. `media_time_seconds`는 직접 확인 전 비어 있다.

## 완료와 미완료의 경계

v2의 코드·스키마·CLI·게이트 구현은 위 표의 `구현` 항목으로 검증한다. 그러나 계획의 최종 완료 기준 3, 4, 8, 10, 11, 14는 실제 미디어 앵커, 독립 일본어 직접 청취, 사람 블라인드 판정, 무작위 감사, 책임자 승인 없이는 충족될 수 없다. 이 저장소는 그러한 증거가 없는 `evaluation-validated`와 `final` 승격을 차단한다.
