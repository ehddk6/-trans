# Subtitle Forensics 입출력 규격 v1

## 작품별 입력

권장 파일명:

- `<작품명>.structure.srt` — 블록 수·번호·타임코드·순서 기준
- `<작품명>.ja.srt` — 일본어 기준본 또는 기존 일본어 ASR
- `<작품명>.previous-ko.srt` — 기존 한국어 후보
- `<작품명>.mp3` 또는 영상 파일 — 로컬 음성 분석 입력
- `<작품명>.photos.zip` — 타임스탬프 사진
- `<작품명>.review-queue.<단계>-vN.csv` — Subtitle Forensics 검수 대기열

파일명이 다르더라도 공용 실행기는 review queue와 음성 파일을 자동 탐색한다. 같은 폴더에 후보가 여러 개면 명시적 경로를 사용한다.

## Review queue 필수 열

- `block_number`
- `timecode` (`HH:MM:SS,mmm --> HH:MM:SS,mmm`)
- `review_band` (`P1`, `P2` 등)

권장 열:

- `source_japanese`
- `provisional_korean`
- `confidence`
- `uncertain_scope`
- `required_evidence`
- `forensics_risk`
- `priority_action`

## `review-scenes.csv`

- `scene_id`
- `start_time`
- `end_time`
- `duration_sec`
- `block_numbers`
- `highest_band`
- `original_audio`
- `dialogue_audio`
- `context_japanese`
- `blocks_json`

`blocks_json`에는 해당 장면에 포함된 구조 기준 블록 정보가 들어간다.

## `asr-candidates.csv`

- `scene_id`
- `model`
- `profile`
- `audio`
- `text`
- `avg_logprob`
- `no_speech_prob`
- `language_probability`
- `error`

`profile` 값:

- `original_unbiased`
- `dialogue_unbiased`
- `original_no_vad`
- `original_prompted`

## 로컬 산출물 ZIP

파일명:

- `<작품명>.work_audio.zip`

필수 포함:

- `manifest.json`
- `review-scenes.csv`
- `asr-candidates.csv`
- `run-summary.json`

선택 포함:

- `clips/*.original.wav`
- `clips/*.dialogue.wav`

## 최종 프로젝트 산출물

- `<작품명>.source-faithful-ko.<단계>-vN.srt`
- `<작품명>.viewer-natural-ko.<단계>-vN.srt`
- `<작품명>.change-log.<단계>-vN.csv`
- `<작품명>.evidence-ledger.<단계>-vN.csv`
- `<작품명>.uncertainty-map.<단계>-vN.csv`
- `<작품명>.asr-scene-verdicts.<단계>-vN.csv`
- `<작품명>.scene-map.<단계>-vN.json`
- `<작품명>.regression-check.<단계>-vN.json`
- `<작품명>.qa-report.<단계>-vN.md`

모든 SRT는 UTF-8, LF로 저장한다. 원본과 이전 결과물을 덮어쓰지 않는다.
