# 자동 초안 완성본 — 2026-07-27

13개 작품의 전 블록을 재생 가능한 자동 초안 SRT로 생성했다.

## 범위

- 전체 블록: 10,380
- 빈 텍스트 블록: 0
- 직접 번역 source/viewer 후보: 각 7,535블록
- 단일 자동 후보: 각 1,955블록 (JUQ-778)
- 직접 번역 결정 후보: 각 887블록 (JUQ-439)
- 구조 정렬 기존 한국어 fallback: 각 3블록 (JUQ-439 원문 손상 구간)

## 산출물

각 작품의 산출물은 다음 경로에 있다.

`workspaces/<TITLE>/runs/2026-07-26-full-execution-v1/automatic-draft-v1/`

폴더에는 다음 파일이 있다.

- `*.source-faithful-ko.automatic-draft-v1.srt`
- `*.viewer-natural-ko.automatic-draft-v1.srt`
- `automatic-draft-decisions.jsonl` — 블록별 후보·fallback 출처
- `automatic-draft-report.json` — 빈 블록 및 출처 집계
- `automatic-draft-manifest.json` — 입력·출력 해시
- `structural-validation.json` — 구조/번호/타임코드 검증 결과

모든 작품의 구조 검증은 통과했다. 이 산출물은 **재생용 자동 초안**이며, 사람 검수나 사람 정답과의 동일성을 주장하지 않고 `final` 승격도 허용하지 않는다.
