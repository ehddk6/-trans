# autonomous-release 운영 안내

`autonomous-release`는 사람 개입 없이 시청 가능한 한국어 SRT를 만들되, 근거 충실 번역과 추론 번역을 분리하는 자동 배포 등급이다. 사람 검증 등급인 `final`과는 별도 계열이며 자동 산출물로 `final`을 승인할 수 없다.

## 설치와 사전 조건

```powershell
python -m pip install -e ".[dev,cloud]"
$env:OPENAI_API_KEY = "사용자 API 키"
```

실제 네트워크 실행은 `--allow-network`와 양수 `--max-cost-usd`를 모두 요구한다. 둘 중 하나라도 없거나 API 키·OpenAI SDK가 없으면 작품 출력 디렉터리를 만들기 전에 중단한다. 네트워크 사용은 Terra/Sol 호출과 필요한 위험 장면의 `gpt-4o-transcribe` 전사에 한정된다.

## 13개 작품 실행

먼저 파일 탐색과 실행 계획만 확인한다.

```powershell
python -m translation_forensics run-autonomous-release `
  --titles-file .\config\autonomous-titles.txt `
  --dry-run --json
```

실제 실행 예시는 다음과 같다.

```powershell
python -m translation_forensics run-autonomous-release `
  --titles-file .\config\autonomous-titles.txt `
  --allow-network --max-cost-usd 100 `
  --max-workers 2 --json
```

기존 로컬 ASR CSV가 있으면 재사용한다. 장면 파일은 있지만 ASR CSV가 없으면 로컬 Whisper `large-v3`를 오프라인 모드로 먼저 실행한다. 로컬 모델 다운로드까지 허용하려면 `--allow-local-model-download`를 명시한다. CPU 강제 실행은 `--cpu`, 중단된 실행 재개는 `--resume`을 사용한다.

## 결정 흐름

1. 원본 블록 수·번호·타임코드·순서를 잠근다.
2. 일본어 SRT, 기존 한국어 후보, 로컬 Whisper를 분리해 근거 묶음을 만든다.
3. 장면 오프셋의 단조 일관성과 음성 전사 존재 여부로 `machine-aligned` 구간을 정한다. SRT와 ASR 의미가 다르면 시간축 실패로 숨기지 않고 `audio-srt-conflict`로 기록한다.
4. 위험 구간만 독립 클라우드 ASR에 보낸다. 같은 Whisper 계열의 여러 프로필은 독립 투표로 세지 않는다.
5. Terra가 source/viewer 결정을 만들고, Sol이 위험·충돌·추론 블록만 `accept | repair | quarantine`으로 반증한다.
6. 최대 두 번 수리 후에도 통과하지 못한 source 결정은 `abstained`가 된다. viewer는 가장 좁은 추론 또는 `…`만 허용한다.
7. 구조·한국어·근거 ID·의미 슬롯·회귀·가독성·해시를 검증한 뒤에만 패키지를 완성한다.

## 작품별 산출물

- `*.source-faithful-ko.autonomous-release-v1.srt`
- `*.viewer-complete-ko.autonomous-release-v1.srt`
- `autonomous-decisions.jsonl`, `autonomous-critiques.jsonl`
- `autonomous-evidence.jsonl`, `evidence-graph.json`, `uncertainty-map.jsonl`
- `machine-alignment.json`, `title-memory.json`
- `model-call-manifest.jsonl`, `autonomous-release-manifest.json`
- `qa-report.json`, `autonomous-release-report.json`, `autonomous-proof.json`
- 실행에 사용한 `prompts/`, `schemas/`, `regressions/` 사본

패키지는 상대 경로와 SHA-256을 사용하므로 디렉터리를 이동해도 검증할 수 있다.

```powershell
python -m translation_forensics validate-autonomous-release `
  --package .\workspaces\IPX-998\autonomous-release-v1 --json

python -m translation_forensics prove-autonomous-claim `
  --package .\workspaces\IPX-998\autonomous-release-v1 --json
```

## 보장과 비보장

검증기는 viewer 블록 충족, SRT 구조 보존, source 전 블록 판정, accepted 결정의 근거 ID, 의미 슬롯 완전성, critical conflict 0건, 외부 호출 추적성, 패키지 해시를 검사한다. 캐시 재생은 동일 응답을 재사용하지만 라이브 모델 호출의 바이트 동일성은 주장하지 않는다.

사람 기준 답안이 없으면 `human_reference_equality`는 항상 `unidentifiable`, `100_percent_equal`은 항상 `false`다. 자동 결과에는 `human_reviewed`, `human_final_allowed`, `final_promotion_allowed`가 모두 `false`로 기록된다.

## 현재 환경 상태

2026-07-27 기준 13개 작품 10,380블록의 입력·구조·장면·로컬 ASR 탐색과 무변경 evidence shadow 검사가 통과했다. 세부 집계는 `docs/AUTONOMOUS_RELEASE_SHADOW_2026-07-27.json`에 있다. 라이브 실행은 현재 `OPENAI_API_KEY`와 `openai` SDK가 없어 사전 검사에서 정상적으로 차단된다. 키와 SDK가 준비되기 전에는 실제 Terra/Sol 결과와 13개 완성 패키지가 생성됐다고 간주하지 않는다.
