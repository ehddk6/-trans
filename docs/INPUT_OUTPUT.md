# 입출력 규격

## 작품 입력

- `<title>.structure.srt`
- `<title>.ja.srt`
- `<title>.previous-ko.srt`
- `<title>.mp3` 또는 영상
- `<title>.photos.zip`
- `<title>.review-queue.<stage>-vN.csv`
- `<title>.work_audio.zip`
- `<title>.asr-candidates.csv`

후보가 여러 개이면 자동 선택하지 않고 `--input` 등 명시적 경로를 요구한다.

## CSV

`review-queue` 필수 열: `block_number`, `timecode`, `review_band`.

`review-scenes.csv`: `scene_id`, `start_time`, `end_time`, `duration_sec`, `block_numbers`, `highest_band`, `original_audio`, `dialogue_audio`, `context_japanese`, `blocks_json`.

`asr-candidates.csv`: `scene_id`, `model`, `profile`, `audio`, `text`, `avg_logprob`, `no_speech_prob`, `language_probability`, `error`.

## 최종 산출물

- `<title>.source-faithful-ko.<stage>-vN.srt`
- `<title>.viewer-natural-ko.<stage>-vN.srt`
- `<title>.change-log.<stage>-vN.csv`
- `<title>.evidence-ledger.<stage>-vN.csv`
- `<title>.uncertainty-map.<stage>-vN.csv`
- `<title>.asr-scene-verdicts.<stage>-vN.csv`
- `<title>.scene-map.<stage>-vN.json`
- `<title>.regression-check.<stage>-vN.json`
- `<title>.qa-report.<stage>-vN.md`

SRT는 UTF-8/LF이며 구조 기준본과 블록 수·번호·타임코드·순서가 완전히 일치해야 한다.
