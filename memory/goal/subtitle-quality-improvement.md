# Goal — canonical subtitle-quality improvement

- Version: 0.3
- Opened: 2026-08-24
- State: complete and verified

## Phase 0 — restatement and definition of done

사용자가 요구한 결과는 개선 아이디어 문서가 아니라, `translation-forensics`의 실제 정본 경로를 구조적으로 이해한 뒤 실행 프롬프트와 구현계획을 만들고, 그 계획을 코드·테스트·증거까지 수행해 닫는 것이다.

완료 조건:

1. 실행에 사용한 최종 프롬프트가 저장소의 단일 진실 원천(SoT)으로 남고, AUTO 모드 자기검토를 정확히 한 번 거친 흔적이 있다.
2. 현재 정본인 `process-title`과 2026-08-17 v2 복원 계층 사이의 관계가 코드와 문서에서 하나의 일관된 구조가 된다. 미연결을 유지한다면 의도와 경계가 명시되고, 연결한다면 실제 호출·산출물·게이트가 테스트로 증명된다.
3. Windows 체크아웃을 포함한 기준 테스트 실패가 원인별로 분리되고, 저장소가 재현 가능한 방식으로 수정된다.
4. 위험 의미 슬롯(질문, 부정, 숫자, 명령/중지, 거부/허용, 지시 위치)의 보존과 근거 없는 추가를 기존보다 강한 자동 회귀 평가가 검증한다.
5. 전체 `pytest`와 `compileall`이 통과하고, 핵심 변경은 표적 테스트와 실패 주입 테스트를 함께 통과한다.
6. 자동화가 입증한 범위와 입증하지 못한 실제 번역 품질·사람 동등성의 경계를 결과 문서와 제품 진실 파일에 명시한다.
7. 새 작업을 처음 보는 사람이 체크포인트와 문서만으로 검증 명령과 남은 한계를 재현할 수 있다.

## Claim ledger

| Claim | Label | Evidence | Limit |
|---|---|---|---|
| 새 작업의 정본 명령은 `process-title`이다. | observed | `README.md:34`, `docs/INTEGRATED_PROCESS_TITLE.md:3` | 문서와 현재 HEAD 기준 |
| v2 복원 모듈은 `codex_quality.py`와 `local_asr.py`에서 사용된다. | observed | 정적 import/call 검색과 단위 테스트 | 동적 호출 빈도는 측정하지 않음 |
| `process_title.py`는 v2 복원 모듈을 직접 호출하지 않는다. | observed | 현재 HEAD의 import/call 검색 | 정적 도달성 관찰이며 외부 래퍼는 제외 |
| 수정 전 Windows 기준선에서 전체 테스트는 3개 실패했다. | observed | 2026-08-24 최초 `python -m pytest -q` | 현재 작업트리에서는 수정되어 전체 통과 |
| 8월 17일 변경이 실제 번역 품질을 높였다. | unknown | 동일 입력의 사람 판정 before/after 없음 | 자동 테스트 통과 여부와 별개 |
| 현재 저장소는 사람 동등 품질을 입증한다. | unknown | adjudicated sealed-test 미발견 | 자동화만으로 승격 금지 |
| 현재 작업트리는 정본 경로에서 ASR critical conflict를 보존·보류한다. | observed | 통합 테스트, 9-case contrast, full suite | synthetic conflict detection이며 실제 번역 점수 아님 |
| 현재 작업트리 전체 테스트는 통과한다. | observed | 453 collected, 2026-08-24 `pytest -q` exit 0 | 현재 미커밋 작업트리 기준 |

## Mobilization — what is held and what is missing

| Branch | Needed | Already held | Gap → next move |
|---|---|---|---|
| 정본 실행 구조 | 실제 호출 그래프와 데이터 계약 | `process_title.py`, `integrated_translation.py`, 정본 문서 | v2 증거가 들어갈 수 있는 입력/산출물 접점 확정 |
| v2 복원 구조 | ASR fusion·slot arbitration·utterance routing의 보장 범위 | `asr_fusion.py`, `graduated_recovery.py`, `utterance_routing.py`, `codex_quality.py` | 정본 경로와 중복/단절을 API 단위로 해소 |
| 프롬프트 계약 | lean outcome-first 실행 계약과 위험 슬롯 규칙 | 운영 overlay, Terra/Sol 프롬프트, manifest | 구조 개선용 실행 프롬프트 SoT 작성 및 한 번 자기검토 |
| 평가 | 재현 가능한 before/after와 실패 주입 | 19개 정적 synthetic 사례, prompt contract 검사 | 대조쌍/변형 기반 의미 불변성 평가와 정본 통합 테스트 |
| 재현성 | 플랫폼 독립 테스트와 해시 무결성 | exact-byte SHA-256 manifest | CRLF 체크아웃 원인 확인 후 EOL 정책 또는 검증 계약 수정 |
| 외부 품질 증명 | 사람 청취·판정 sealed-test | 템플릿과 제한 문서만 존재 | 없는 증거를 만들지 말고 자동 증명 경계를 명시 |
| 전달 | 계획, 검증 명령, 재개 지점 | ballast memory scaffold | 계획 문서·체크포인트·제품 진실 갱신 및 zero-context rehearsal |

## Terrain map — current architecture

```text
official path
  process-title
    → Japanese evidence bundle / translation units
    → Terra translation
    → deterministic automated_quality
    → independent Sol audit
    → machine-final | machine-uncertain package

v2 recovery path
  local ASR evidence
    → ASR-family fusion
    → independent meaning frames
    → slot-level graduated recovery
    → utterance routing
    → codex-quality gates/package

evaluation path
  prompt manifest hash checks
  + 19 hand-authored synthetic quality cases
  + unit/integration tests
  − adjudicated gold
  − sealed human blind comparison
```

Primary bottleneck (`observed`): 안전 기능이 부족한 것이 아니라, 최신 복원 기능과 정본 실행·평가 경로가 분리되어 같은 품질 계약을 공유하지 않는 “split-brain” 구조다.

## Skeleton v0.1 — atomic leaves

Status key: `[x]` evidence attached, `[~]` in progress, `[ ]` not filled.

### A. Evidence baseline

- [x] A1. Git 정본과 최근 변경 식별 — `0dddfe4`, 2026-08-17.
- [x] A2. 정본 실행 경로 식별 — README와 통합 경로 문서.
- [x] A3. v2 실제 import/call 위치 식별 — `codex_quality.py`, `local_asr.py`.
- [x] A4. 전체 테스트 기준선 실행 — 3 failures observed.
- [x] A5. 세 실패의 독립 원인 분리 — text hash EOL 2건, eager optional ASR 초기화 1건.
- [x] A6. 사람 골드/봉인 평가 자산 존재 여부 확인 — `evaluation/gold/manifest.json`은 `empty-no-gold-answers`.

### B. Design contract

- [x] B1. 공유 경계를 additive `source_evidence`와 순수 `evidence_bridge.py`로 선택.
- [x] B2. 기존 필드 부재는 `unavailable`; conflict만 보류를 강화하는 비대칭 폴백 규칙 확정.
- [x] B3. 자동 주장은 conflict detection으로 제한하고 사람 번역 품질은 `not-demonstrated`로 확정.
- [x] B4. 위험 전수 차단·missed 0·정상 대조 신규 보류 0으로 판정식 확정.
- [x] B5. UTF-8 newline-canonical hash + 저장소 LF 정책으로 선택.

### C. Prompt and implementation plan

- [x] C1. 실존 전문가 프라이밍을 포함한 lean 실행 프롬프트 작성.
- [x] C2. AUTO 자기검토 정확히 1회 수행하고 변경점 기록.
- [x] C3. 최종 프롬프트를 `prompts/canonical-evidence-bridge-agent-v1.md`로 고정.
- [x] C4. 파일별 구현 순서·검증·롤백 경계를 `docs/CANONICAL_EVIDENCE_BRIDGE_IMPLEMENTATION_PLAN.md`로 고정.

### D. Implementation

- [x] D1. 재현성 기준선 결함 수정과 회귀 테스트 — newline-canonical hash, LF policy, lazy ASR.
- [x] D2. 공유 품질/복원 계약 구현 — `evidence_bridge.py`와 additive `source_evidence`.
- [x] D3. 정본 `process-title` 호출·산출물에 계약 연결 — unit/Terra/decision/Sol/audit 통합 테스트.
- [x] D4. 위험 의미 슬롯 대조/변형 평가 구현 — 위험 6, 정상 3, frozen exact report.
- [x] D5. CLI·schema·manifest·문서의 필요 변경 동기화 — schema v2/v3 및 계약 문서.

### E. Verification and release evidence

- [x] E1. 새 표적 단위 테스트 통과 — 33-test targeted set 포함.
- [x] E2. 실패 주입 테스트가 의도한 이유 코드로 실패 — ASR risk별 code와 `machine-uncertain` 확인.
- [x] E3. 전체 `pytest -q` 통과 — 453 collected, exit 0.
- [x] E4. `compileall` 통과 — `src tools`, exit 0.
- [x] E5. before/after 자동 평가 표 생성 — `evaluation/engineering/canonical-evidence-bridge-v1.json`.
- [x] E6. 사람 품질 주장을 `not-demonstrated`로 봉인 — result와 product truth에 기록.
- [x] E7. zero-context rehearsal, checkpoint, product truth, session log 갱신 — round 1 clean, archived/replaced checkpoint.

## Named-but-unfilled

- 실제 번역 품질을 판정할 인간 골드셋 — 이번 목표의 권한·증거 경계 밖으로 명시적 이월.

## Recheck triggers

이 골격은 다음 사건마다 재검토한다: 독립 감사 회수, 실패 원인 확정, 설계 선택, 구현 직후, 전체 테스트 직후. 새 모순이나 더 작은 근본 원인이 발견되면 버전을 올리고 변경 이유를 남긴다.

## Zero-context rehearsal log

### Round 1 — clean, observed 2026-08-24

- Persona: Python과 `pytest`를 사용할 수 있지만 이번 작업 세션과 변경 배경은 모르는 저장소 유지보수자.
- Deliverable: `docs/CANONICAL_EVIDENCE_BRIDGE_RESULT.md` 하나만 제공.
- Execution: 문서에 적힌 full pytest, compileall, 두 prompt contract, frozen evaluation exact match, `git diff --check`를 실제 실행.
- Result: 5개 명령 모두 exit 0. 453 tests, prompt contracts 2 pass, frozen report status pass. CRLF→LF 안내는 문서의 기대와 일치.
- Blocking stalls, guesses, or misreads: 없음.
- Fixes: 없음. 첫 라운드 clean이므로 리허설 종료.

### Round 2 — clean after broader compile command, observed 2026-08-24

- Trigger: 저장소 규칙에 맞춰 문서의 compile command를 `src tools`로 넓힌 뒤 fresh executor로 다시 실행.
- Result: full pytest, `compileall -q src tools`, 두 prompt contract, frozen evaluation exact match, diff check 모두 exit 0.
- Reported false alarm: 실행자는 문서가 명령 수를 잘못 셌다고 적었지만 해당 개수 표현은 문서에 존재하지 않는다. 실행 누락·추측·blocking stall은 없었다.
- Fixes: 없음. 문서의 실행 계약은 clean.
