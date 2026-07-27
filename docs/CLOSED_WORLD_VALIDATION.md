# 폐쇄형 자동 검증

`closed-world-validated`는 사람 정답·직접 청취·외부 서비스 없이 로컬에 이미 있는 일본어 SRT, ASR, 기존 한국어 후보와 번역 초안을 보수적으로 교차 검사하는 별도 등급이다.

이 등급은 `audio-human-verified`, `evaluation-validated`, `final`의 하위 대체물이 아니다. 사람 증거 필드를 만들지 않으며 `final_promotion_allowed`는 항상 `false`다.

## 실행

작품의 `project-manifest.json`과 기존 중간 산출물이 있으면 다음 명령 하나로 실행한다.

```powershell
python -m translation_forensics.cli run-closed-world `
  --project-root . `
  --title SAMPLE
```

후보 SRT를 직접 지정하려면 `--candidate-srt`를 여러 번 사용한다. 기본 명령은 네트워크를 호출하거나 모델을 다운로드하지 않는다. 기존 `review-context.jsonl`과 로컬 ASR이 있으면 재사용한다. ASR CSV가 없고 로컬 장면 클립이 있으면 캐시된 faster-whisper 모델을 오프라인 모드로 실행하며, 캐시나 클립이 없으면 해당 근거 없이 더 보수적으로 판정한다.

ASR 모델이 없을 때 다운로드까지 허용하려면 다음처럼 명시한다.

```powershell
python -m translation_forensics.cli run-closed-world `
  --project-root . `
  --title SAMPLE `
  --allow-model-download
```

이 경우 결과는 `workspaces/SAMPLE/network-enabled/`에 별도로 저장되고, 매니페스트에 다운로드 허용 사실을 기록한다. 다운로드한 모델은 실행 도구일 뿐 외부 정답이나 사람 검수 근거가 아니므로, `final` 승격 조건은 변하지 않는다.

검증과 품질 주장 감사:

```powershell
python -m translation_forensics.cli validate-closed-world `
  --package .\workspaces\SAMPLE\closed-world\closed-world-validated-v1

python -m translation_forensics.cli prove-quality-claim `
  --package .\workspaces\SAMPLE\closed-world\closed-world-validated-v1
```

## 판정 규칙

- 모든 구조 블록은 `accepted` 또는 `abstained` 중 하나여야 한다.
- `accepted`에는 생성 후보 계열과 별도 후보 계열의 합의가 필요하다. 기존 한국어 자막 하나만으로 승인하지 않는다.
- 질문, 극성, 요청, 참여자, 표현 강도의 명시적 표면 의미가 뒤집힌 후보는 탈락한다.
- 짧은 발화에서 행위자나 대상 생략으로 복수 해석이 남으면 보류한다.
- 같은 Whisper 모델의 여러 profile은 `whisper-family` 하나로만 센다.
- ASR과 일본어 SRT가 충돌하거나 후보 계열이 충분히 합의하지 않으면 보류한다.
- 보류 블록은 번역문을 비우고 미리보기 SRT에 `[미확정]`을 표시한다. 이 SRT는 `final` 산출물이 아니다.

## 보장과 비보장

패키지는 다음을 검증한다.

- 블록 수·번호·타임코드·순서 보존
- UTF-8/LF 미리보기
- 전 블록 판정 커버리지
- 승인 블록의 명시적 표면 의미 규칙 준수
- 사람 증거 미생성 및 `final` 게이트 미우회
- 입력·산출물 해시와 동일 경로 재실행의 바이트 재현성

사람 정답이 없으면 `human_reference_equality`는 `unidentifiable`이다. `prove-quality-claim`은 사람 기준본을 별도로 주더라도 일치율을 측정값으로만 보고하며 `100_percent_equal`을 `true`로 만들지 않는다.

## 산출물

작품별 `closed-world/closed-world-validated-v1/`에 다음 파일을 만든다.

- `observations.jsonl`
- `semantic-lattice.jsonl`
- `decisions.jsonl`
- `source-faithful.preview.srt`
- `viewer-natural.preview.srt`
- `unresolved.jsonl`
- `machine-timeline.json`
- `candidate-inventory.json`
- `summary.json`
- `proof.json`
- `run-manifest.json`
- `validation.json`
