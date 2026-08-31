# Canonical Evidence Bridge — implementation result

- Date: 2026-08-24
- Audience: Python과 `pytest`를 사용할 수 있지만 이번 작업 세션의 맥락은 모르는 `translation-forensics` 유지보수자
- Prompt SoT: `prompts/canonical-evidence-bridge-agent-v1.md`
- Implementation plan: `docs/CANONICAL_EVIDENCE_BRIDGE_IMPLEMENTATION_PLAN.md`

## Outcome

정본 `process-title`은 이제 subtitle ensemble의 primary ASR과 Qwen 대안을 단위별 `source_evidence`로 보존하고, Terra 입력·Sol 감사 입력·deterministic 품질 게이트까지 같은 record를 운반한다. 독립 계열 사이에서 의미 반전 위험이 관찰되면 자동 `machine-final` 승격을 막고 `machine-uncertain`으로 폴백한다.

이 결과가 입증하는 범위는 **관찰 가능한 ASR 의미 충돌을 놓치지 않고 보류하는 자동 안전성**이다. 실제 영상의 한국어 번역 품질 향상, 사람 동등성, 배포 자막 개선은 입증하지 않았다.

## Resulting data flow

```text
transcript_ja.jsonl
  └─ asr_metrics.backend + qwen_alternatives + raw evidence IDs
      └─ evidence_bridge.py
          ├─ raw values preserved
          ├─ ASR family aliases normalized
          ├─ same-family repeats collapsed
          └─ fusion state + critical meaning risks
              └─ TranslationUnit.source_evidence (schema v2)
                  ├─ Terra batch payload
                  ├─ translation decision
                  ├─ Sol audit payload
                  └─ deterministic quality gate
                      ├─ no critical conflict → existing decision path
                      └─ critical dual conflict → machine-uncertain fallback
```

`process-title` run schema는 v3이다. 이전 bridge 없는 부분 실행을 잘못 재개하지 않도록 process cache 경계를 올렸고, 저장된 TranslationUnit v1은 `source_evidence={}`로 읽을 수 있다. 새 v2 단위에서 bridge 필드가 빠지면 재개를 거부한다.

## Verified evidence

| Check | Observed result | Claim boundary |
|---|---:|---|
| Full test suite | 453 collected, `pytest -q` exit 0 | 코드·계약 회귀 검증 |
| Python compilation | `compileall -q src tools` exit 0 | 구문·import compilation |
| Prompt contracts | Terra + autonomous contracts pass | canonical UTF-8/LF hash와 schema 계약 |
| Frozen contrast evaluation | `status=pass`, exact JSON match | synthetic engineering evidence |
| Hazard cases | 6/6 newly blocked, 0 missed | 극성, 질문, 거절/허용, 중단/계속, 위/아래, 안/밖 |
| Clean controls | 0/3 regressions | exact, compatible, unavailable controls |
| Human-quality proof | `false` | 직접 청취·adjudicated gold 없음 |

수정 전 전체 테스트 실패 3건은 두 원인으로 분리됐다. prompt provenance의 exact-byte 해시가 Windows CRLF 체크아웃과 충돌한 2건, 선택적 ASR dependency를 mock 경로보다 먼저 생성한 1건이다. 수정은 기대값 완화가 아니라 newline-canonical text hash, 저장소 LF 정책, 실제 rerun 시점의 lazy backend 생성으로 이루어졌다.

## Reproduce from repository root

PowerShell에서 다음을 순서대로 실행한다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src tools
.\.venv\Scripts\python.exe -c "from pathlib import Path; from translation_forensics.prompt_contract import validate_prompt_contract; names=['terra-semantic-translation-v1.manifest.json','autonomous-subtitle-decision-v1.manifest.json']; results=[validate_prompt_contract(Path('prompts')/n) for n in names]; assert all(r['status']=='pass' for r in results), results; print('prompt-contracts: 2 pass')"
.\.venv\Scripts\python.exe -c "import json; from pathlib import Path; from translation_forensics.evidence_bridge_evaluation import evaluate_source_evidence_contrasts; p=Path('tests/fixtures/source-evidence-contrast-cases.json'); actual=evaluate_source_evidence_contrasts(p); frozen=json.loads(Path('evaluation/engineering/canonical-evidence-bridge-v1.json').read_text(encoding='utf-8')); assert actual == frozen; print('frozen-evaluation: exact match; status=', actual['status'])"
git diff --check
```

기대 결과:

- 모든 명령이 exit code 0으로 끝난다.
- prompt 계약은 `prompt-contracts: 2 pass`를 출력한다.
- 고정 평가는 `frozen-evaluation: exact match; status= pass`를 출력한다.
- `git diff --check`는 오류를 출력하지 않는다. Windows working-copy의 CRLF→LF 안내는 `.gitattributes` 정책에 따른 경고이며 diff 오류가 아니다.

## Failure injection covered by tests

- Whisper와 Qwen이 질문 여부를 다르게 인식한 `process-title` 단위는 Terra와 Sol에 동일한 conflict record가 전달되고 최종 `machine-uncertain`이 된다.
- 같은 Whisper 계열의 별칭·반복은 독립 투표로 증가하지 않는다.
- 출처를 알 수 없는 family label은 `unverified`로 접혀 합의 근거를 만들지 않는다.
- bridge 근거가 없는 v1 단위와 정상 대조군은 기존 의미 결정 상태를 바꾸지 않는다.
- prompt 파일의 CRLF/LF 표현은 같은 hash를 가지지만 실제 내용 변경은 거부된다.
- mock conflict rerun은 optional FasterWhisper/Reazon dependency를 설치하지 않아도 실행된다.

## Artifacts

- Bridge: `src/translation_forensics/evidence_bridge.py`
- Reproducible evaluator: `src/translation_forensics/evidence_bridge_evaluation.py`
- Frozen suite: `tests/fixtures/source-evidence-contrast-cases.json`
- Frozen report: `evaluation/engineering/canonical-evidence-bridge-v1.json`
- Canonical-path integration test: `tests/integration/test_process_title.py`
- Bridge unit tests: `tests/unit/test_evidence_bridge.py`

## Explicit limits and next proof

현재 `evaluation/gold/manifest.json`에는 sealed human answers가 없고, pilot human review와 adjudication도 완료되지 않았다. 따라서 “8월 17일보다 실제 번역이 더 정확해졌다”는 주장은 `not-demonstrated`다.

그 주장을 검증하려면 변경 전 commit과 현재 코드를 동일한 실제 영상 샘플에 실행하고, 소스 버전을 가린 두 명 이상의 일본어→한국어 검수자가 직접 청취한 뒤 MQM 오류와 critical meaning-flip을 조정 판정해야 한다. 이 작업은 새 외부 모델 호출·사람 검수·배포 권한이 필요한 별도 목표이며 이번 구현에는 포함하지 않았다.

## Rollback boundary

원시 transcript와 배포 자막은 수정하지 않았다. 문제가 생기면 deterministic gate의 bridge 소비를 비활성화할 수 있지만, 저장된 원시 `source_evidence`를 삭제하거나 이전 process cache를 강제 재사용하면 안 된다.
