# translation-forensics

일본어 성인영상의 성인 출연자 발화 자막을 원문 범위 안에서 한국어로 복원·번역하고, 구조·문자·가독성·근거·회귀 검사를 수행하는 로컬 프로젝트다. 성적 표현을 자동 순화하거나 원문에 없는 행동·신체 부위·감정·관계를 창작하지 않는다.

## 구현 범위

이 프로젝트는 다음을 실제 코드로 제공한다.

- 입력 역할 탐색, 다중 후보 중단, SHA-256 매니페스트
- UTF-8/LF SRT 파싱과 구조 잠금·겹침 정렬
- 제공된 `subtitle_forensics_v1` 보존 및 실행 어댑터
- 제공된 `subtitle_audio_forensics_runner_v1` 보존 및 음성·ASR 어댑터
- 구조 기반 review queue fallback (기존 학습 입력이 없을 때만 사용하며 모델 결과로 가장하지 않음)
- P1/P2 장면 준비, 체크포인트 ASR 결과 수집, prompted 결과의 낮은 증거 등급
- 장면 검토 컨텍스트, evidence ledger, uncertainty map, change log 생성
- 최종 SRT의 구조·문자·가독성·중복·회귀 검사와 새 버전 패키징

`vendor/`는 원본 구현 보존 영역이다. 기존 Subtitle Forensics의 자체 테스트는 110개 검사·실패 0으로 제공 ZIP에서 확인했으며, 통합 어댑터는 원본 코드를 덮어쓰지 않는다.

## 설치

텍스트 작업은 Python 3.10 이상으로 실행할 수 있다.

```powershell
cd "C:\경로\translation-forensics"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m translation_forensics.cli doctor --project-root . --json
```

ASR가 필요하면 프로젝트 루트에서 `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` 후 `.INSTALL_ASR.ps1`을 실행한다. `ffmpeg -version`이 먼저 동작해야 한다. 스크립트가 있는 폴더로 이동하기 어렵다면 `& "C:\경로\translation-forensics\INSTALL_ASR.ps1"`처럼 전체 경로로 실행한다. 현재 PowerShell 프로세스에서만 실행 정책을 바꾼다.

GPU는 `nvidia-smi`가 실제로 확인될 때만 CUDA/float16을 선택한다. 그 외에는 `run-asr --cpu`로 CPU/int8을 사용한다. 모델 파일은 `faster-whisper` 실행 시 내려받아질 수 있으며, 이 저장소에는 모델을 포함하지 않는다.

## 작품 폴더

```powershell
python -m translation_forensics.cli init-title --project-root . --title SAMPLE
```

생성된 `workspaces/SAMPLE/inputs/`에 다음 표준 이름을 넣는다.

```text
SAMPLE.structure.srt
SAMPLE.ja.srt
SAMPLE.previous-ko.srt        # 선택
SAMPLE.mp3                    # 또는 지원 영상
SAMPLE.photos.zip             # 선택
```

후보가 여러 개이면 자동으로 고르지 않는다. `inspect`나 `prepare-audio`에 명시적 경로를 전달한다.

## 전체 흐름

```powershell
python -m translation_forensics.cli inspect --project-root . --title SAMPLE
python -m translation_forensics.cli analyze --project-root . --title SAMPLE
python -m translation_forensics.cli prepare-audio --project-root . --title SAMPLE
python -m translation_forensics.cli run-asr --project-root . --title SAMPLE --cpu
python -m translation_forensics.cli ingest-asr --project-root . --title SAMPLE --source .\SAMPLE.work_audio.zip
python -m translation_forensics.cli build-review-context --project-root . --title SAMPLE
```

`analyze`는 원본 학습 입력이 함께 제공된 경우에만 기존 Subtitle Forensics를 실행한다. 그렇지 않으면 `structure_fallback` 큐를 만들고 그 사실을 `forensics-run.json`에 기록한다. 이는 자동 번역이나 오역 확정이 아니다.

Codex 또는 사람이 장면 의미를 검토해 두 SRT를 만든 뒤:

```powershell
python -m translation_forensics.cli validate --project-root . --title SAMPLE `
  --source-faithful .\draft\SAMPLE.source-faithful-ko.srt `
  --viewer-natural .\draft\SAMPLE.viewer-natural-ko.srt `
  --output .\workspaces\SAMPLE\intermediate\validation.json

python -m translation_forensics.cli package --project-root . --title SAMPLE `
  --ja .\workspaces\SAMPLE\inputs\SAMPLE.ja.srt `
  --source-faithful .\draft\SAMPLE.source-faithful-ko.srt `
  --viewer-natural .\draft\SAMPLE.viewer-natural-ko.srt `
  --stage text-crosschecked
```

패키징은 기존 결과를 덮어쓰지 않고 다음 `vN`을 사용한다. 자동 검사는 의미 판정을 대체하지 않으며, `final` 상태는 전체 블록 검수·필요한 증거·QA 통과를 사람이 확인한 뒤에만 사용한다.

## 산출물과 검증 상태

최종 폴더에는 원문 충실본, 감상용 자연본, change log, evidence ledger, uncertainty map, ASR 장면 판정, scene map, regression check, QA 보고서와 실제 검토가 완료된 경우의 의미 프레임·가설 원장·화자 상태·정렬 근거·MQM·역번역·평가 산출물을 새 버전으로 복사한다. 상태는 `structure-validated`, `text-crosschecked`, `audio-asr-crosschecked`, `audio-human-verified`, `evaluation-validated`, `final`을 사용한다. 다중 ASR만으로 `audio-human-verified`를 부여하지 않는다.

## 문제 해결

- `후보가 여러 개라 자동 선택하지 않았습니다`: `--structure`, `--ja`, `--review-queue`, `--audio` 등 명시적 경로를 준다.
- `ffmpeg를 찾지 못했습니다`: `ffmpeg -version`을 실행하고 PATH를 확인한다.
- ASR 모델 로드 실패: `run-asr --cpu`로 재시도하거나 `faster-whisper` 설치와 모델 캐시를 확인한다.
- PowerShell 경로 오류: 프로젝트 루트로 `cd`한 뒤 실행하거나 명령에 전체 경로를 사용한다.
- `ExecutionPolicy`: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`만 사용한다.
- `final` 패키징 실패: 먼저 `validate`를 실행하고 빈 자막·일본어·작업용 태그·구조 차이를 해결한다.

## 한계

위험 모델은 자동 오역 확정기가 아니며, 기존 모델은 ABF-303 내부 그룹 교차검증 결과다. 사진은 자동 판독하지 않는다. 같은 Whisper 모델의 여러 패스는 독립 증거가 아니다. 원음 직접 청취·외부 골드 세트·실제 작품 샘플은 이 작업에 제공되지 않았다.

## 의미 번역 단계

기존 한국어 자막을 시간 정렬한 `build-korean-draft`는 번역 완료가 아니다. 실제 한국어 번역은 `build-translation-queue`로 만든 블록별 결정 레코드를 `apply-translations --strict`로 적용해야 한다. 이 단계는 일본어 의미, 앞뒤 문맥, ASR, 기존 후보를 분리하고 `source-faithful`과 `viewer-natural`을 별도로 생성한다. 자세한 규격은 `docs/TRANSLATION_DECISIONS.md`를 참조한다.

## 차세대 포렌식 기록과 평가

```powershell
# 정답을 넣지 않는 평가 골격을 한 번만 생성
python -m translation_forensics.cli init-gold --project-root .

# 번역 큐에서 명시적 미검수 semantic frame 템플릿만 생성
python -m translation_forensics.cli init-forensic-records --project-root . --title SAMPLE

# 사람이 채운 frame/hypothesis 원장의 형식과 critical 충돌을 검사
python -m translation_forensics.cli validate-forensic-records `
  --frames .\workspaces\SAMPLE\intermediate\SAMPLE.semantic-frames.reviewed-v1.jsonl `
  --hypotheses .\workspaces\SAMPLE\intermediate\SAMPLE.hypothesis-ledger.reviewed-v1.jsonl

# 사람이 작성한 자막 전용 MQM 오류 원장 검사
python -m translation_forensics.cli validate-mqm --input .\workspaces\SAMPLE\intermediate\SAMPLE.mqm-errors.reviewed-v1.csv
```

`init-forensic-records`의 결과는 `unreviewed` 템플릿이며 가설은 비어 있다. 이 명령은 해석·번역·검증 결과를 만들지 않는다. 가설은 실제 증거가 있을 때만 기록하고, critical 슬롯 충돌은 해결 또는 `unresolved` 처리 전에는 최종 상태로 승격하지 않는다. 연구 적용 근거는 `docs/research-findings.md`에, 현재 구현 감사는 `docs/current-system-audit.md`에 있다.
