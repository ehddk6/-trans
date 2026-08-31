# Checkpoint — scene_v2 natural-dialogue implementation — 2026-08-31 10:33

## The story so far

`block_v1` 기본 경로와 SRT 구조 계약을 보존한 채, opt-in `scene_v2`를 구현했다. 장면 구성·의미 복원·제한적 visual observation·자연 대사·독립 source-faithful·두 critic·repair·재감사·패키징을 분리했으며, fake-provider 통합 테스트로 visual 정책 경계·resume·산출물·구조 잠금을 확인했다. 현재 상태는 `experimental-unbenchmarked`이고, 사용자 제공 외부 baseline과 인간 블라인드 평가는 아직 없다.

## Decided

- D-010: 일본어 어휘가 있는 성인물 대사는 보수적 보류 표식 대신 자연스럽고 직접적인 완결 한국어로 번역한다.
- D-011: 화면용 자연 번역을 먼저 작성하고, 질문·부정·동의/거절·대상·방향·시제·수치·강도·근거 없는 추가 같은 실제 의미 오류만 별도로 검수한다.
- D-013: `block_v1`을 보존하면서 `scene_v2` 장면 단위 자연 대사 아키텍처를 opt-in으로 구현하고, 인간·외부 baseline 검증 전 우수성을 주장하지 않는다.

## Waiting on the user

실작품 `scene_v2` 결과에 대한 인간 블라인드 평가와 사용자 제공 외부 baseline 비교의 범위·입력을 사용자가 정해야 한다.

## Next first action

사용자가 검증용 작품과 입력을 제공하면 `process-title --translation-architecture scene_v2`를 별도 run으로 실행하고, `block_v1` 및 사용자 제공 외부 baseline과 블라인드 pack을 만든다.

## Tried

- 모든 줄에 독립 의미 검수를 호출하는 방식은 느리고, 낮은 검수 점수만으로 자연스러운 대사를 직역투로 평탄화했다.
- 비어 있는 역번역이나 낮은 검수 점수만으로는 의미 오류로 판정하지 않도록 바꿨다.
- `scene_v2`의 구현 여부를 문서 승인만으로 판단하려던 초안은 실제 fake-provider 통합 검증 결과로 교체했다.
