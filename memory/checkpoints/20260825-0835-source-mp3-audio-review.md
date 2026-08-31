# Checkpoint — active subtitle corpus improved — 2026-08-25

## The story so far

사용자가 모든 재생용 자막을 `C:\Users\ehddk\OneDrive\비디오`로 모았다. 이 폴더와 `C:\Users\ehddk\OneDrive\Videos` 작품 폴더를 대조해 정본 125편을 개선·설치했다. 9편의 의미 번역 보정, 6편 131개 거부 표식의 원문 대응 복원, 14편 이상의 파싱·무효 길이·타임코드 역행 수리, `SNOS-361` 후반 498큐 로컬 ASR 재타이밍을 완료했다. 이어 125편 공통으로 잔류 시간·중복·미세 큐·줄바꿈을 정리했다.

최종 감사 결과는 파싱 오류 0, 일본어 잔존 0, `[안전상 번역 불가]` 0, 무효 길이 0, 타임코드 역행 0이다. 30초 초과 문장 큐는 966→16, 0.25초 미만 문장 큐는 2,523→400, 32자 초과 줄은 1,333→0, 3줄 이상 큐는 71→0, 인접 동일 문구는 3,604→1,869로 줄었다. 감사 위험 합계는 29,518→13,450이다.

## Decided

- D-002: `Videos`의 정렬 장점과 `비디오`의 자연스러움을 결합해 일괄 교체가 아닌 선택적 개선을 수행한다.
- D-003: 작품별 사진·음성을 함께 검증하고, 자산이 없을 때는 실제 원본 영상 또는 기존 로컬 음성에서만 생성한다.
- D-004: `C:\Users\ehddk\OneDrive\비디오`를 활성 재생 기준으로 삼고, 작업공간에서 검증·백업한 뒤 설치한다.
- 오프라인 OneDrive 영상을 강제 다운로드하거나 실제 장면이 아닌 가짜 사진을 증거로 만들지 않는다.
- 0.5초 이내에 붙은 동일 자막만 합치고 그보다 떨어진 반복 대사는 보존한다.

## Verified state

- 최종 감사: `workspaces/active-subtitle-audit-20260825-final-v3.json`
- 공통 가독성 보고서: `workspaces/corpus-readability-20260825/qa-report.json`
- 32자 레이아웃 보고서: `workspaces/corpus-readability-32char-20260825/qa-report.json`
- 500ms 중복 정리 보고서: `workspaces/corpus-dedupe-500ms-20260825/qa-report.json`
- 증거 자산: 실제 사진 21장, 음성 98개, 음성 보유 작품 33편.
- 추가 음성 보고서: `workspaces/workspace-audio-evidence-20260825.json`
- 전체 461개 테스트 및 변경 도구 py_compile 통과.
- 각 단계 설치 전 자막은 해당 작업공간 `backups/` 또는 작품별 `deliverable/*.before-*.srt`에 보존했다.

## Waiting on the user

- 필수 대기는 없다. 남은 92편의 사진·음성 기반 심층 의미 검토를 원하면 해당 MP4를 OneDrive에서 로컬로 내려받아야 한다.

## Next first action

추가 심층 검토를 요청받으면 최종 감사 위험 1위 `SNOS-295`의 MP4 로컬 상태부터 확인하고, 실제 장면 3장·음성 3개를 생성한 뒤 반복·미세 큐 중심으로 의미 검토한다.

## Tried

- 오프라인 MP4를 직접 여는 검사는 작품당 5~8GB 다운로드를 유발해 중단했다.
- 기존 작업공간을 재검색해 27편의 로컬 음성을 찾았고, 이미 증거가 있던 1편을 제외한 26편에 15초 대표 클립 77개를 생성했다.
- 남은 25개 겹침은 동시 발화 또는 기존 source 구조로 남겼고 무효 길이·역행은 아니다.
