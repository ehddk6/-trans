# Checkpoint — `비디오` subtitle improvement complete — 2026-08-24 23:45:40

## The story so far

`C:\Users\ehddk\OneDrive\비디오`의 원본은 보존하고, 검증된 175개 SRT 개선 세트를 `C:\Users\ehddk\OneDrive\비디오_개선본_20260824`에 만들었다. 10개 파일·305블록을 선택적으로 수정해 `Videos`와 매핑되는 111개 재생용 자막의 일본어 문자를 2,558자에서 0자로 줄였다. `ABF-196`과 `PRED-879`은 byte-identical control로 유지했고, 변경 원장·QA 보고서·사용 안내를 결과 폴더에 포함했다.

## Decided

- D-002: `Videos`의 정렬 장점과 `비디오`의 자연스러움을 결합해 일괄 교체가 아닌 선택적 개선을 수행한다.
- 원본 `비디오`는 덮어쓰지 않고 형제 결과 폴더를 만든다(A-002의 안전한 실행 해석).
- 일본어 문자 0은 의미 정확도의 완전한 증명이 아니며, 음성 직접 청취는 이번 범위에 포함하지 않는다.

## Waiting on the user

- None. 원본으로의 영구 승격은 요청되지 않아 수행하지 않았다.

## Next first action

Open `C:\Users\ehddk\OneDrive\비디오_개선본_20260824\README.md`, report the completed output path, and do not overwrite the original unless the user explicitly asks for promotion.

## Tried

- Source discovery initially treated `.freebuff` as a title; discovery now requires a directory containing SRTs.
- A strict duplicate-number guard stopped on an unrelated duplicate `704` in `START-126-UC`; the final guard rejects only ambiguous residual target numbers and preserves the legacy structure.
- Three untouched common files do not parse under strict SRT rules; residual auditing now scans them without silently repairing their timecodes and reports them explicitly.
- The first zero-context rehearsal stalled on playback-file selection. README and QA gained exact MP4→SRT mappings and comparison-scope lists; fresh round 2 was clean.
- Final evidence: 461 tests pass, compileall passes, independent 111-title residual scan is 2,558→0, and 165 unchanged hashes have 0 mismatches.
