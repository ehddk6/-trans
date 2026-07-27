# Targeted machine retranslation v3 — 2026-07-27

## 실행 범위

품질 평가에서 의미 오류·자리표시자·반복 치환·손상 원문 위험으로 지정된 76개 블록을 대상으로 `gpt-5.6-terra` 재번역 후보를 생성하고, 별도 기계 검토를 수행했다.

프롬프트는 일본어 기준 SRT, 앞뒤 문맥, 동일 Whisper 계열 ASR 후보, 기존 한국어를 입력으로 사용했다. 손상·비문·혼합 언어 원문은 한국어 의미를 창작하지 않고 `hold`로 판정하도록 했다.

## 결과

- 대상 블록: 76
- 기계 검토 후 유지한 교체: 16
- 근거 부족 보류 표기(`…`): 60
- 기계 검토에서 철회한 초기 교체: 2
  - SSIS-642 #151
  - SSIS-400 #215
- 13개 작품의 v3 SRT 구조·번호·타임코드 검증: 모두 통과
- 전체 테스트: 통과

## 산출물

각 작품의 v3 자막은 다음 경로에 있다.

`workspaces/<TITLE>/runs/2026-07-26-full-execution-v1/targeted-retranslation-v3/`

포함 파일:

- `*.source-faithful-ko.targeted-retranslation-v3.srt`
- `*.viewer-natural-ko.targeted-retranslation-v3.srt`
- `targeted-retranslation-decisions.jsonl`
- `targeted-retranslation-report.json`
- `structural-validation.json`

## 해석

v3은 이전의 잘못된 번역을 무조건 보존하지 않는다. 명확한 일본어 근거가 있는 블록만 교체했으며, 손상 원문은 실제 번역처럼 보이게 하지 않고 `…`로 표시했다.

따라서 v3은 기계적으로 더 정직한 재생용 결과이지만, 보류 60개가 남아 있으므로 `machine-final` 또는 사람 검수 기반 `final`로 승격되지 않는다.
