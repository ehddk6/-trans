# Canonical Evidence Bridge — implementation plan

- Date: 2026-08-24
- Prompt SoT: `prompts/canonical-evidence-bridge-agent-v1.md`
- Status: completed and verified on 2026-08-24; plan frozen before implementation
- Result: `docs/CANONICAL_EVIDENCE_BRIDGE_RESULT.md`

## Outcome

정본 `process-title`이 이미 생성된 독립 ASR 대안을 버리지 않고 품질 판단까지 운반하게 한다. 이 변경이 증명하는 것은 “관찰 가능한 ASR 의미 충돌을 자동 확정에서 차단하는 능력”이며, 사람 판정 번역 품질 향상은 별도 sealed 평가 전까지 `not-demonstrated`다.

## Design decisions

1. 공용 경계는 새 `evidence_bridge.py`다. subtitle-pipeline의 내부 객체나 `codex_quality`의 전체 파이프라인을 서로 직접 호출하지 않는다.
2. 원시 텍스트는 그대로 보존하고, fusion용 계열 이름만 별도 canonical family로 정규화한다.
3. bridge는 비대칭이다. 합의는 자동 승격을 높이지 않으며, 관찰된 dual conflict와 meaning-flip risk만 보류를 강화한다.
4. `TranslationUnit.source_evidence`는 additive 필드다. 기존 저장 단위에 필드가 없으면 `unavailable`로 읽는다.
5. 새 모델 호출은 추가하지 않는다. Terra와 Sol이 같은 bridge record를 보게 하고, 최종 차단은 deterministic gate가 소유한다.
6. 품질 비교는 사람 정확도 지표가 아니라 engineering safety detection으로 이름 붙인다.

## Atomic implementation sequence

### Phase 1 — restore a trustworthy baseline

1. `prompt_contract.py`에 UTF-8 text의 newline-canonical SHA-256을 정의한다.
2. LF 체크아웃 정책을 `.gitattributes`에 기록하고 CRLF/LF 동일성·내용 변경 거부 테스트를 추가한다.
3. `translation-decision.schema.json`의 필수 필드를 Terra manifest와 맞추고 unresolved/translated 예제를 다시 검증한다.
4. `codex_quality.py`의 입력 acoustic record를 항상 공용 bridge로 정규화한다.
5. `_scene_payload()`에 deterministic `source-srt:block-N` 참조를 노출한다.
6. conflict ASR backend를 실제 rerun 호출 내부에서 지연 생성하되 장면 간 재사용하고, 선택 dependency가 없는 mock 테스트를 통과시킨다.

Verification: 기존 실패 3개를 각각 표적 재실행하고, 수정 전 원인이 아닌 기대값 완화가 없음을 확인한다.

### Phase 2 — canonical evidence bridge

1. `evidence_bridge.py`에 다음 순수 함수를 구현한다.
   - ASR family alias canonicalization
   - raw record 보존형 acoustic normalization + `add_asr_fusion`
   - transcript row의 `asr_metrics.backend`, `qwen_alternatives`, timing, evidence IDs를 단위 evidence로 변환
   - fusion risk → critical slot/reason code 매핑
   - frame 없는 단계에서의 보수적 utterance route 요약
2. `TranslationUnit`에 `source_evidence`를 추가하고 build/write/load/resume 왕복을 검증한다.
3. `_unit_for_model()`과 Terra batch payload, Sol audit payload에 같은 record를 전달한다.
4. `automated_quality.audit_translation_decision()`이 bridge state·risk를 기록하고 dual conflict를 fallback으로 만든다.
5. process-title 통합 테스트에서 위험 단위는 `machine-uncertain`, 정상 단위는 기존 상태를 유지하는지 검증한다.

Verification: round-trip, same-family collapse, unknown-family fail-closed, polarity/question/stop-direction conflicts, clean/reference controls.

### Phase 3 — executable contrast evaluation

1. `tests/fixtures/source-evidence-contrast-cases.json`에 위험 최소쌍과 정상 대조군을 동결한다.
2. bridge 없는 기존 audit와 bridge 적용 audit를 동일 입력에 실행하는 평가 함수를 작성한다.
3. 결과에 `newly_blocked_hazards`, `missed_hazards`, `clean_control_regressions`, reason-code coverage를 기록한다.
4. 재현 결과를 `evaluation/engineering/canonical-evidence-bridge-v1.json`에 고정하되 사람 품질 지표와 구분한다.

Acceptance: 모든 위험 사례 차단, missed 0, clean regression 0. 표본 수와 fixture 한계를 보고서에 명시한다.

### Phase 4 — prompt, docs, and claim boundary

1. `integrated-noisy-asr-recovery-v1.md`에 `source_evidence`의 권위와 dual conflict 규칙을 추가한다.
2. README와 `INTEGRATED_PROCESS_TITLE.md`에 bridge 데이터 흐름·산출 상태·한계를 설명한다.
3. 기존 `QUALITY_REGRESSIONS.md`에 새 contrast suite가 증명하는 범위를 추가한다.
4. `PRODUCT-TRUTH.md`, goal skeleton, knowledge note, session log, checkpoint를 실제 테스트 증거로 갱신한다.

### Phase 5 — completion verification

1. 새 표적 테스트.
2. 관련 process-title/codex-quality/prompt-contract 테스트.
3. 전체 `python -m pytest -q`.
4. `python -m compileall -q src`.
5. `git diff --check`, 변경 파일/호출 그래프 감사.
6. zero-context rehearsal: 문서만 보고 다른 세션이 평가를 재실행할 수 있는지 확인.

## Rollback boundary

Bridge 필드는 additive이며 원시 transcript는 불변이다. 문제가 발견되면 quality gate의 bridge 소비를 끄더라도 저장된 `source_evidence`는 삭제하지 않는다. 배포 자막과 기존 workspace 산출물은 이 계획의 변경 대상이 아니다.

## Explicitly deferred

- 사람 직접 청취 sealed-test와 두 검수자 MQM 조정
- 별도 Korean-only 모델 역검사 호출
- 전체 영상에 대한 두 번째 ASR 패스
- 실제 배포 자막 교체

이 항목들은 가치가 없어서가 아니라 현재 증거·권한·비용 경계 밖이므로 별도 목표로 남긴다.
