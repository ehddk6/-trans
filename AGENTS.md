# translation-forensics — 프로젝트 분석 가이드

이 문서는 GPT / Codex가 `translation-forensics` 저장소를 분석하고 작업할 때 필요한 전체 구조와 규칙을 정의한다.

## 1. 프로젝트 개요

**목적**: 일본어 성인영상 SRT 자막을 한국어로 번역·검증·패키징하는 evidence-first 워크플로우.
**핵심 원칙**: 모든 결정은 실제 증거(원문 SRT, ASR, 타임라인 정렬, 음성)에 기반해야 하며, 추론이나 창작을 금지한다.
**버전**: 0.1.0 (Python >= 3.10)

## 2. 저장소 구조

```
translation-forensics/
├── src/translation_forensics/   # 핵심 Python 패키지 (43개 모듈)
│   ├── cli.py                   # CLI 진입점 (60+ 서브커맨드)
│   ├── srt.py                   # SRT 파싱/쓰기/비교
│   ├── autonomous_release.py    # A 경로: 자율 릴리스 (Terra/Sol/증거 기반)
│   ├── offline_hybrid.py        # B 경로: 로컬 병합 릴리스
│   ├── openai_provider.py       # OpenAI API 호출 프로바이더
│   ├── closed_world.py          # 폐쇄형 검증 (ASR + 결정 규칙)
│   ├── inferred_recovery.py     # 음성 기반 복구
│   ├── machine_final.py         # 기계 최종 패키징
│   ├── alignment.py             # 블록 정렬
│   ├── validation.py            # SRT 쌍 검증
│   └── ... (33개 추가 모듈)
├── prompts/                     # GPT 번역·비평 프롬프트 (Markdown)
├── schemas/                     # JSON Schema (결정·증거·비평·증명)
├── tools/                       # Codex 자율 릴리스 브릿지 스크립트
│   ├── codex_autonomous_release.py  # prepare + finalize 2단계
│   └── codex_gen3.py                # Gen 3 고품질 결정 생성기
├── vendor/                      # 외부 도구 (Subtitle Forensics, ASR Runner)
├── workspaces/<title>/          # 작품별 작업 디렉터리
│   ├── inputs/                  # 원본 입력 파일
│   ├── intermediate/            # 중간 산출물 (GPT direct, 큐, 결정)
│   ├── closed-world/            # 폐쇄형 검증 결과
│   ├── inferred-recovery-v4/    # 음성 복구 결과
│   └── autonomous-release-*/    # 자율 릴리스 패키지
├── docs/                        # 설계 문서 (WORKFLOW, INPUT_OUTPUT 등)
├── tests/                       # 단위/통합 테스트
├── scripts/                     # 작품별 GPT-5.6 direct 번역 스크립트
└── pyproject.toml               # 패키지 정의
```

## 3. 전체 워크플로우

```
[입력 단계]
  init-title → inspect → validate-timeline → analyze
       ↓
[오디오/ASR]
  prepare-audio → run-asr → ingest-asr → build-review-context
       ↓
[번역 초안 - 병렬]
  ├─ generate_machine_draft.py (Google Translate, 외부)
  ├─ generate_*gpt56_direct*.py (GPT-5.6 직접)
  └─ build_automatic_draft() (자동 병합, core)
       ↓
[결정적 검증]
  run-closed-world (ASR + 결정 규칙)
       ↓
[기계 최종]
  package-machine-final
       ↓
[정밀 재번역 - v2/v3]
  apply-targeted-retranslations
       ↓
[음성 기반 복구 - v4/v4.1]
  prepare-inference-audio → build-inference-context → apply-inferred-recovery
       ↓
[최종 출력 - 두 경로]
  A) autonomous-release (Codex/API 기반, 증거·비평·수리 루프)
  B) build-offline-hybrid (로컬 파일만, 우선순위 병합)
```

## 4. 핵심 모듈 역할

### CLI (`cli.py`)
- 60+ 서브커맨드, `python -m translation_forensics <command>` 로 실행
- 주요 명령: `init-title`, `inspect`, `analyze`, `prepare-audio`, `run-asr`, `run-closed-world`, `run-autonomous-release`, `build-offline-hybrid`, `validate`, `package`

### SRT (`srt.py`)
- `SubtitleBlock(number, start, end, text, start_seconds, end_seconds)` — 핵심 데이터 클래스
- `parse_srt(path)` → `(blocks, header_lines, raw_lines)` — SRT 파싱
- `write_srt(path, blocks)` — SRT 쓰기
- `compare_structure(ref, candidate)` → `{pass, errors, warnings}` — 구조 일치 검사
- 모든 시간은 `00:00:01,000` 형식 (밀리초)

### 자율 릴리스 (`autonomous_release.py`)
- 자율 릴리스의 핵심. Terra(결정 생성) + Sol(비평) + Repair Loop 구조
- API 필요 부분: `provider.generate_decisions()`, `provider.critique_decisions()`, `provider.transcribe_audio()`
- 로컬 가능 부분: `build_autonomous_evidence()`, `estimate_machine_alignment()`, `validate_autonomous_release()`
- Codex 브릿지 스크립트(`tools/codex_*)`가 API 부분을 Codex 추론으로 대체

### 오프라인 하이브리드 (`offline_hybrid.py`)
- 로컬 파일만으로 최종 SRT 생성. 우선순위: closed-world(1순위) → inferred-recovery(2순위) → machine-final(3순위)
- API 불필요, 네트워크 불필요

### 증거 기반 결정 (`asr_evidence.py`, `alignment.py`, `evidence_graph.py`)
- ASR 후보 통합, 블록 간 시간 기반 정렬, 증거-결정 연결 그래프

## 5. 데이터 형식

### 증거 번들 (evidence JSONL)
```json
{
  "block_number": 1,
  "japanese": {"evidence_id": "japanese-srt:1", "text": "おはようございます。"},
  "local_asr": [{"evidence_id": "local-asr:scene1:whisper", "text": "...", "profile": "large-v3"}],
  "local_context": [{"block_number": 1, "japanese_srt": "...", "is_target": true}],
  "previous_korean_candidate": {"text": "...", "alignment": "time-overlap", "confidence": "high"},
  "risk_codes": ["missing-japanese", "corrupt-japanese"],
  "timeline_compatible": false,
  "scene_id": null
}
```

### 결정 레코드 (decisions JSONL)
- 필수 필드: `title_id`, `block_number`, `source_faithful_korean`, `viewer_natural_korean`, `source_status`(accepted/abstained), `viewer_status`(supported/best_effort/unrecoverable), `confidence`(high/medium/low), `evidence_refs`, `semantic_slots`(12개 서브필드), `risk_codes`, `reason`
- `semantic_slots`: question, polarity, refusal_permission, command_strength, speaker, actor, action, target, location, tense_aspect, direction, intensity (모두 nullable)
- 스키마: `schemas/autonomous-decision.schema.json`

### SRT 파일 규칙
- UTF-8, LF 개행
- 구조 기준 SRT와 블록 수·번호·타임코드·순서 완전 일치
- 명명 규칙: `<title>.<role>.<stage>-v<N>.srt`
  - role: `source-faithful-ko` / `viewer-complete-ko` / `viewer-natural-ko`
  - stage: `closed-world-validated-v1` / `machine-final-v1` / `inferred-recovery-v4` / `autonomous-release-v1` / `codex-release-v1`

## 6. 작업 공간 규칙

```
workspaces/<title>/
├── metadata/project-manifest.json  # 작품 메타데이터 (입출력, 역할, 해시)
├── intermediate/                    # 중간 산출물
├── closed-world/closed-world-validated-v1/  # 폐쇄형 검증 결과
├── inferred-recovery-v4/           # 음성 복구 (v4)
├── inferred-recovery-v4.1/         # 음성 복구 (v4.1)
└── autonomous-release-codex-v1/    # Codex 자율 릴리스
```

- 매니페스트에서 구조(structure)와 일본어(ja) 경로 확인
- 이전 한국어(previous_ko)는 시간 정렬 후보, 번역 결과가 아님
- ASR 후보는 `asr-candidates.csv` 또는 `inferred-recovery-audio-v4/asr-candidates.csv`

## 7. 출시 경로 비교

| 항목 | A) autonomous-release | B) build-offline-hybrid |
|---|---|---|
| API 의존 | Terra/Sol GPT 호출 (또는 Codex 대체) | 없음 |
| 결정 방식 | 증거 기반 + AI 추론 | 우선순위 병합 (CW > IR > MF) |
| semantic_slots | 채움 (Terra 또는 Codex) | 없음 |
| QA/증명서 | 상세 (증거 그래프, 불확실성 맵) | 기본 |
| 추적성 | evidence_refs, reason, 모델 호출 기록 | 없음 |
| 처리 속도 | 느림 (증거 수집 + 결정 + 패키징) | 빠름 (파일 복사만) |

## 8. CLI 사용 패턴

```powershell
# 초기화
python -m translation_forensics init-title --title SSIS-575
python -m translation_forensics inspect --title SSIS-575

# 오디오/ASR
python -m translation_forensics prepare-audio --title SSIS-575
python -m translation_forensics run-asr --title SSIS-575
python -m translation_forensics ingest-asr --title SSIS-575

# 번역/검증
python -m translation_forensics run-closed-world --title SSIS-575
python -m translation_forensics package-machine-final --title SSIS-575

# 복구
python -m translation_forensics apply-targeted-retranslations --title SSIS-575
python -m translation_forensics prepare-inference-audio --title SSIS-575
python -m translation_forensics apply-inferred-recovery --title SSIS-575

# Codex 자율 릴리스
python tools/codex_autonomous_release.py prepare --title SSIS-575
# → Codex가 evidence.jsonl 읽고 decisions.jsonl 작성
python tools/codex_autonomous_release.py finalize --title SSIS-575
# 또는 단일 자동화:
python tools/codex_gen3.py  # 증거→결정→패키징 일괄

# 오프라인 하이브리드 (API 불필요)
python -m translation_forensics build-offline-hybrid --titles-file titles.txt --output offline-hybrid-v1

# 검증
python -m translation_forensics validate --title SSIS-575
python -m translation_forensics validate-autonomous-release --package <path>
```

## 9. Codex 작업 규칙

1. **증거 우선**: 모든 번역 결정은 `evidence.jsonl`의 증거에 기반. ASR·컨텍스트·이전 초안을 종합.
2. **금지**: 원문에 없는 행동·신체 부위·위치·감정·관계·강압·결과를 추론해 추가 금지.
3. **두 SRT**: source_faithful(원문 충실)과 viewer_natural(자연스러운 한국어)을 별도로 생성.
   - source_faithful: 일본어 구조와 의미를 보존한 직역
   - viewer_natural: 한국인이 실제로 말할 법한 자연스러운 대사
4. **완전 자동 금지**: 검증되지 않은 산출물을 final로 승격 금지. `human_final_allowed=False`.
5. **구조 불변**: 출력 SRT는 입력 structure SRT와 블록 구조가 완전히 일치해야 함.
6. **불확실성 표시**: 확실하지 않은 블록은 `source_status: abstained`, `viewer_natural_korean: …` 로 처리.
7. **semantic_slots**: 각 블록의 질문 여부, 극성, 명령 강도, 시제, 방향, 강도 등을 채울 것.
   - question: 일본어가 か/？/? 로 끝나거나 의문사로 시작하면 True
   - polarity: ない/ません 포함 시 negative, else positive
   - command_strength: statement/question/command/polite_command/request/suggestion
   - tense_aspect: present/past/progressive/future
8. **Codex 자율 릴리스**: `codex_gen3.py`로 전체 배치 처리. 수동 개입 불필요.

## 10. 제약 사항

- `openai` 패키지는 선택 의존성. `cloud` extra 필요: `pip install -e .[cloud]`
- 로컬 ASR: `faster-whisper` + CUDA (또는 `--cpu`)
- `deep-translator`는 Google Translate 무료 API용 (속도 제한 있음)
- 모든 SRT는 UTF-8 BOM (utf-8-sig), LF
- 기존 산출물 덮어쓰기 금지 (FileExistsError)
