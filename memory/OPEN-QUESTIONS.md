# OPEN QUESTIONS — registered, not remembered

Rule: an unresolved item remains here until a decision or verified finding closes it.

| ID | Question | Opened | Status |
|---|---|---|---|
| Q-001 | 현재 저장소에 사람 판정 sealed-test가 있는가? 없다면 자동화만으로 증명 가능한 품질 범위는 어디까지인가? | 2026-08-24 | resolved — F-001 |
| Q-002 | 8월 17일 v2 병합이 실제 번역 품질을 개선했는지 동일 입력·동일 평가 기준으로 재현 가능한 비교가 존재하는가? | 2026-08-24 | resolved — F-002 |

## Resolutions

- F-001 — `observed`, 2026-08-24: sealed human answers는 없다. `evaluation/gold/manifest.json`은 `empty-no-gold-answers`이고 pilot human review/adjudication은 완료되지 않았다. 자동화로 입증 가능한 범위는 구조·계약 불변성과 synthetic ASR-conflict detection뿐이며, 실제 번역 정확도는 포함하지 않는다.
- F-002 — `observed`, 2026-08-24: commit `0dddfe4`와 현재 작업트리의 동일 실제 영상·동일 blind human rubric 비교는 없다. 따라서 8월 17일 변경의 실제 번역 품질 향상은 `not-demonstrated`다. 이번 작업은 별도로 같은 synthetic fixture에서 bridge 전후의 conflict-blocking 개선만 재현했다.

## Readings in force — assumed, not decided

| ID | User's words (verbatim) | Our reading (`assumed`) | Breaks if wrong | Ends when | Relied on in |
|---|---|---|---|---|---|
| A-001 | “그럼 프로젝트를 더욱 개선하기 위해 프롬프트를 작성하고 구현계획을 작성하고 구현계획에 따라 목표를 수행해서 작업을 완료해 피상적인 접근말고 문제에 대해 구조적으로 깊은 이해를 통해 혁신적인 결과물을 제공해라” | `translation-forensics` Git 저장소를 개선 대상 프로젝트로 삼고, 외부 배포 자막은 검증 없이 덮어쓰지 않으면서 저장소 내부 코드·문서·테스트를 수정한다. | 사용자가 상위 폴더 전체나 실제 배포 자막의 즉시 교체를 뜻했다면 범위가 달라진다. | 사용자가 범위를 수정하거나 최종 구현 결과를 확인할 때 | `memory/goal/subtitle-quality-improvement.md`, 이번 구현 변경 |
| A-002 | “`비디오` 폴더의 자막을 개선해봐” | 원본 `C:\Users\ehddk\OneDrive\비디오`는 보존하고, 검증 가능한 형제 폴더에 개선본을 만든 뒤 차이를 보고한다. | 사용자가 원본 파일의 즉시 인플레이스 교체만을 원했다면 최종 배치 위치가 달라진다. | 개선본 전달 또는 사용자의 인플레이스 승격 요청 | `memory/goal/video-subtitle-improvement.md`, 개선본 출력 경로 |
| A-003 | “작품폴더에 보면 사진이랑 음성있을텐데, 없는 경우 만들어서 번역해봐” | 직전 개선에서 미검증 변경이 가장 많았던 `IPZZ-856`을 첫 시청각 재번역 대상으로 삼고, 검증된 절차를 나머지 작품에 순차 적용한다. | 사용자가 특정 다른 작품 또는 116개 전체의 즉시 완성을 우선했다면 실행 순서가 달라진다. | `IPZZ-856` 시청각 개선본 전달 또는 사용자의 우선순위 변경 | `memory/goal/multimodal-subtitle-translation.md` |
