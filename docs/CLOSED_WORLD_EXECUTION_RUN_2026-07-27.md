# Closed-world 실행 보고서 — 2026-07-27

실행 계보: `workspaces/<title>/runs/2026-07-26-full-execution-v1`

이번 실행은 13개 작품의 원본 일본어 SRT·음성·영상·기존 한국어 후보를 새 매니페스트로 등록하고, 구조/시간축 분석, P1/P2 음성 장면 추출, 로컬 Whisper ASR, 검토 컨텍스트, 로컬 검수 패키지, 폐쇄형 자동 판정을 순서대로 수행했다.

## 결과

- 작품 수: 13
- 전체 블록: 10,380
- 직접 후보를 확정본이 아닌 후보로만 연결한 판정:
  - `accepted`: 661 (6.37%)
  - `abstained`: 9,719 (93.63%)
- 모든 작품에서 결정 커버리지: 100%
- 모든 작품에서 패키지 구조/해시 검증: 통과
- 모든 작품에서 `final_promotion_allowed`: `false`
- 사람 기준 답안 동일성: `unidentifiable`

시간축 상태는 `media-too-short` 5개, `unresolved` 8개이며, 자동 결과를 사람 청취 검증으로 표시하지 않았다.

직접 후보 연결 패키지는 각 작품의 `closed-world/closed-world-with-direct-candidates-v1`에 있다. 보류 블록은 미확정 사유와 경쟁 후보를 JSONL에 남겼으며, 미리보기 SRT에만 포함했다.

이 보고서는 사람 정답과의 의미/문자열 동일성을 주장하지 않는다. 사람 청취·골드·블라인드 평가가 추가되기 전에는 `final` 패키징이 허용되지 않는다.
