# Goal — `비디오` subtitle improvement

- Version: 0.3
- Opened: 2026-08-24
- State: complete

## Phase 0 — restatement and definition of done

사용자의 목표는 `C:\Users\ehddk\OneDrive\비디오`의 자연스러운 재생용 자막을 기준으로 유지하면서, `C:\Users\ehddk\OneDrive\Videos`의 정렬된 일본어·한국어 쌍을 근거로 일본어 잔존과 알려진 반복·오역 위험을 선택적으로 제거한 실제 개선본을 만드는 것이다.

완료 조건:

1. 원본 `비디오` 파일을 덮어쓰지 않고 모든 입력 SHA-256을 보존한다.
2. `Videos`와 공통인 111작품을 결정적으로 매핑하며 `SONE-054↔SONE-054-C_GG5`, `START-126↔START-126-UC`, `PRED-879` 다중본을 명시적으로 처리한다.
3. 공통 작품의 재생용 한국어 자막에서 일본어 잔존을 0으로 만든다.
4. `ABF-196`의 주기적 결말 문구와 `PRED-879`의 `わかった→아…` 같은 알려진 `Videos` 결함을 개선본에 재도입하지 않는다.
5. 변경하지 않은 파일은 원본과 byte-identical이고, 변경 파일은 번호·타임코드·빈 블록·인코딩을 검증한다.
6. 변경 원장과 작품별 전후 지표를 남기며, 자동·텍스트 기반 개선의 한계를 명시한다.

## Mobilization

| Branch | Needed | Already held | Gap → first move |
|---|---|---|---|
| 실제 대상 식별 | Windows 알려진 폴더와 공통 작품 매핑 | `비디오` 175 SRT, `Videos` 224 SRT를 직접 관찰 | alias와 다중본을 코드 계약으로 고정 |
| 잔존 제거 | 작품·큐별 일본어 문자와 안전한 대체 근거 | `translation_forensics.srt`, 정렬된 JA/KO 쌍, canonical quality 원칙 | overlap 사용 조건과 수동 override 경계 결정 |
| 자연스러움 보존 | 일괄 source-faithful 교체 방지 | 사용자 비교 결과, `비디오`의 병합 자막 | 변경 최소화 및 clean-control diff 측정 |
| 알려진 결함 차단 | 반복 boilerplate와 의미 있는 발화→신음 오역 탐지 | ABF-196·PRED-879 실물 예시, 기존 prompt 규칙 | 승격 금지 패턴과 검증 assertion 구현 |
| 안전한 산출 | 원본 보존·재현·롤백 | A-002, 기존 evidence-first 도구 | 형제 출력 폴더와 SHA/change ledger 생성 |

## Terrain map

- `observed` 2026-08-24: 실제 대상은 `C:\Users\ehddk\OneDrive\비디오`, 비교 기준은 `C:\Users\ehddk\OneDrive\Videos`다.
- `observed` 2026-08-24: `Videos`는 111개 작품 디렉터리에 224 SRT가 있다. 일반 작품은 JA/KO 1쌍이고 `PRED-879`만 두 쌍이다.
- `observed` 2026-08-24: 111작품이 대상에 매핑된다. exact 109개와 alias 2개다.
- `observed` 2026-08-24: 공통 재생용 파일 10개에 가나 2,175자, 전체 Japanese regex 2,558자가 남는다. `IPZZ-856`이 273/285블록과 가나 2,118자를 차지한다.
- `observed` 2026-08-24: 나머지 9개 파일의 잔존은 32블록이며 대부분 `ー`, `っ`, `・`, 일본식 점 또는 이름 한 글자다.
- `observed` 2026-08-24: `Videos/ABF-196`은 `ご視聴ありがとうございました`가 중간 구간에 반복되지만 대상 `비디오/ABF-196.srt`에는 해당 번역문이 없다.
- `observed` 2026-08-24: `Videos/PRED-879`의 첫 `わかった`는 `아…`이지만 대상 `비디오/PRED-879.srt`는 같은 초반 문맥을 `알았어`로 처리한다.
- `assumed`: `IPZZ-856`의 일본어 잔존 큐는 동일 타임라인의 `Videos` 한국어 overlap을 우선 사용하되, 결말 문구 hallucination은 제외하고 근거가 없거나 의미가 뒤집히는 경우에는 명시적 override를 사용한다.

## Skeleton v0.1

Status: `[x]` evidence attached, `[~]` in progress, `[ ]` named-unfilled.

### A. Baseline

- [x] A1. 실제 폴더·작품·SRT 수 확인 — read-only filesystem audit.
- [x] A2. 공통 111작품과 alias·PRED 다중본 확인 — title-directory mapping audit.
- [x] A3. 일본어 잔존 작품·큐 기준선 확인 — 10 files, 305 lines, 2,175 kana.
- [x] A4. ABF/PRED 알려진 결함을 양쪽 실물에서 재현 — direct SRT search.

### B. Design

- [x] B1. 원본 보존형 출력·manifest·change ledger 계약 — sibling output, input SHA guards, JSONL/JSON artifacts.
- [x] B2. IPZZ overlap 후보의 허용·거부·override 계약 — overlap score ≥0.5, boilerplate exclusion, 154 manual overrides after full candidate review.
- [x] B3. 작은 잔존 32블록의 의미·기호 정규화 표 — hash-guarded config entries.
- [x] B4. 정상 파일 byte-identity와 구조 보존 계약 — 165 clean-control hashes plus changed-file structure checks.

### C. Build

- [x] C1. 재현 가능한 개선 도구와 tests 작성 — module, CLI, config, 8 focused tests.
- [x] C2. 형제 출력 폴더에 전체 재생용 자막 세트 생성 — `C:\Users\ehddk\OneDrive\비디오_개선본_20260824`, 175 SRT.
- [x] C3. 작품별 change ledger와 QA report 생성 — `change_ledger.jsonl`, `qa_report.json`, `README.md`.

### D. Verification

- [x] D1. 공통 111작품 Japanese residual 0 — independent scan 2,558 → 0.
- [x] D2. 변경하지 않은 작품 byte-identical — 165 checked, mismatch 0.
- [x] D3. 변경 파일 SRT parse·번호·타임코드·빈 블록 검증 — all 10 pass; existing duplicate number retained without ambiguity.
- [x] D4. ABF 반복·PRED 의미 축소 미재도입 — both byte-identical; forbidden periodic phrase hits 0.
- [x] D5. 표적 tests, full pytest, compileall, output hash audit — 8 focused and 461 total tests pass; compileall pass.
- [x] D6. zero-context rehearsal와 checkpoint — round 1 stall fixed; fresh round 2 clean; checkpoint archived and replaced.

## Single next leaf

None — done-check passed.

## Named gaps

- 기존부터 파싱되지 않는 무수정 파일 3개(`MOON-057`, `PRED-488`, `SNOS-167`)의 향후 복구 여부.
- 개선본을 원본 `비디오`로 승격할지는 사용자 선택이며 이번 실행에서는 수행하지 않음.
- 실제 음성 직접 청취는 현재 범위에 없음.

## Done-check

모든 D leaf가 pass했다. 461 tests, compileall, independent output audit, hash audit, known-regression controls, fresh-executor rehearsal round 2가 통과했다. 일본어 문자가 0이라는 사실은 번역 정확도 증명이 아니며, 변경 원장에 `videos-overlap`, `manual-text-override`, `punctuation-normalization` 등 근거를 구분한다.

## Rehearsal log

- Round 1 — `not clean`: 결과 폴더만 받은 한국어 Windows 사용자 persona가 개선·원본 보존·잔여 오류는 확인했으나, 어떤 SRT를 MP4에 연결해야 하는지와 111개 비교 범위를 README만으로 확정하지 못했다.
- Fix: README에 exact-stem MP4→SRT 선택 규칙, 외부 자막 사용법, 변경 파일, 110개 비교 포함 영상/16개 비포함 영상을 명시했다. QA report에 126개 exact mapping과 111개 common mapping 전체를 추가했다.
- Round 2 — `clean`: fresh executor가 `IPZZ-856.mp4→IPZZ-856.srt`, `ABF-169.mp4→ABF-169.srt`를 추측 없이 선택하고, QA와 305행 ledger를 파싱했으며 blocking stall을 보고하지 않았다.

## Sub-foundations exposed

- target/source title identity — atomic.
- timeline overlap arbitration — not atomic → boilerplate exclusion, overlap selection, placeholder guard, override로 분리.
- subtitle quality proof — not atomic → structure, residual, known-regression, human/audio limits로 분리.
