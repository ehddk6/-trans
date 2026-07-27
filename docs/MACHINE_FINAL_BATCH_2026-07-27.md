# Machine-final 실행 보고서 — 2026-07-27

13개 작품의 자동 초안, 구조 검증, ASR 후보, 시간축 상태, 폐쇄형 품질 증명, 블록별 자동 초안 출처 원장을 묶어 machine-final 패키지를 생성했다.

## 결과

- 패키지: 13개
- 블록: 10,380개
- `machine_final_allowed`: 13개 모두 `true`
- `human_final_allowed`: 13개 모두 `false`
- 기존 `final_promotion_allowed`: 13개 모두 `false`
- `human_reference_equality`: 13개 모두 `unidentifiable`
- `100_percent_equal`: 13개 모두 `false`

## 산출물

각 작품의 패키지 경로:

`workspaces/<TITLE>/runs/2026-07-26-full-execution-v1/machine-final-v1/`

패키지는 두 재생용 SRT, 자동 초안 결정 원장, ASR 후보, 시간축 상태, 폐쇄형 증명, 해시 매니페스트와 machine-final 보고서를 포함한다.

`machine-final`은 사람이 검수했다는 주장을 하지 않는 완성된 기계 산출물이다. 사람 검수 기반 `final` 상태와 혼동되지 않도록 두 상태를 별도로 유지한다.
