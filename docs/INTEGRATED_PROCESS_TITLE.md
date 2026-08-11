# 일본어 영상→한국어 자막 통합 실행 계약

`process-title`은 새 작업의 단일 정본 명령이다. 기본 `--quality-policy automated`는 사람 승인 없이 질문·부정·숫자·명령·일본어 잔존·빈 번역을 자동 검사하고, Terra 결과를 별도 Sol 감사기가 `pass/fail/unknown`과 근거로 판정한다. 감사기는 번역문을 수정하지 않으며, 실패·미확정 단위는 원문 충실 번역으로 보수적 폴백한다. 기존 `run`, 정적 번역표, 자동 초안 재포장 경로는 호환·분석 용도로 남지만 통합 산출물로 승격하지 않는다.

## 입력 선택

일본어 근거는 다음 순서 중 하나만 사용한다.

1. `--japanese-bundle`: `bundle_verification.json`이 있는 기존 번들을 읽기 전용으로 재사용한다.
2. `--reference-ja --reference-ja-approved`: 한글 우세 여부를 검사한 뒤 reference backend를 사용한다.
3. 둘 다 없으면 VAD를 끈 Whisper `large-v3-turbo` 전체 전사와 Qwen 제한 검증을 실행한다.

번역 원문은 항상 `transcript_ja.jsonl`의 `text_raw`다. 반복 축약·표시 최적화가 적용된 `viewer_ja.srt`는 번역 입력이 아니다. `translation_ja.srt`는 공백 정규화 후 원문 내용이 100% 일치할 때만 생성된다.

## 게이트

- `valid=false` 또는 구조/인식/정렬 실패: 실행 중단
- 인식·정렬 `review_required`: 완전 초안과 검수 큐만 허용
- 표시 경고: 번역은 허용하지만 배포·final은 보류
- `accepted=false`: `automated` 정책에서는 유효한 번들을 대상으로 자동 품질 게이트를 적용하고, `legacy` 정책에서만 단위별 사람 승인으로 후보를 열 수 있다.
- 단위 상태는 `trusted`, `suspect`, `unusable`로 별도 기록

## 한국어 산출물

- `outputs/viewer_complete_ko.srt`: 모든 단위를 포함한 자동 검증 초안
- `outputs/source_faithful_ko.srt`: 자동 게이트를 통과하거나 보수적 폴백된 원문 충실 출력
- `outputs/viewer_natural_ko.srt`: 자동 게이트를 통과한 자연스러운 시청 출력

`automated` 정책은 `[검수 보류]`를 생성하지 않는다. deterministic hard fail은 Sol pass로 뒤집을 수 없고, Sol `fail`·`unknown`·사용량 제한은 source-faithful 자체의 hard gate를 다시 거친 뒤 폴백한다. 모든 단위가 deterministic와 Sol 감사를 통과하면 `machine-final`, 폴백·감사 불가가 있으면 `machine-uncertain`으로 승격한다. `legacy` 정책은 기존 사람 승인·보류 계약을 유지한다.

## 시각 문맥

파일명의 `sub_XXXX`는 무시하고 마지막 초 단위 timestamp만 사용한다. 프레임은 SHA-256으로 중복 제거하고 단위 시작·중앙·끝에 가장 가까운 고유 이미지를 최대 3장 고른다. `--legacy-captures`가 없거나 비어 있으면 선택된 시각 문맥 단위에 대해 FFmpeg로 대표 프레임을 자동 생성한다. 생성 프레임은 실행 번들의 `generated-frames/`와 `capture_index.jsonl`에 `source=auto-generated`로 기록되며, `--no-auto-capture-frames`로 끌 수 있다. 전체 영상을 무차별 캡처하지 않고 번역 모호성이 있는 선택 단위만 캡처한다.

`metadata`는 픽셀을 전송하지 않는다. `targeted`는 번역 확신도가 낮고 화자·상대·지시 위치·화면 문자·장면 연속성 중 하나가 실제로 모호한 단위만 전송한다. 각 호출은 경로, SHA-256, 선정 이유, 전송 장수, Codex 호출 영수증을 `visual_context.jsonl`과 `model_call_receipts.jsonl`에 남긴다. 화면에만 보이는 행동·신체 부위·관계를 발화로 추가해서는 안 된다. critical 의미 슬롯은 자동 수정하지 않고 source-faithful 폴백을 유지한다.

## 캐시와 재개

정규화 오디오는 미디어 SHA-256 아래의 16 kHz mono PCM WAV 한 개를 Whisper와 Qwen이 공유한다. 캐시 identity에는 코드, 스키마, 옵션, 미디어 해시가 포함되며 임시 WAV 경로는 제외된다. 실행은 `integrated/.partial/<run-id>`에 쓴 뒤 QA가 통과할 때만 승격하고 `integrated/latest.json`을 갱신한다. `--resume`은 동일 identity의 완전한 캐시와 부분 실행만 재사용한다.

## 주요 감사 파일

- `input_manifest.json`: 원본 경로·크기·mtime·SHA-256
- `japanese_bundle.json`: backend, 독립 게이트, 아티팩트 해시
- `translation-input/translation_units_ja.jsonl`: 원문·시각·단어 시각·경고·근거
- `translation_decisions.jsonl`: Terra/Sol 기계 결정
- `automated_quality.jsonl`: 자동 의미 보존 검사·점수·폴백 사유
- `automated_quality.jsonl`: deterministic 결과, Sol `pass/fail/unknown`, 역번역 보조 증거, 감사 호출 ID
- `visual_context.jsonl`: 프레임 선택·허용 슬롯·전송 영수증
- `review_queue.jsonl`, `review_decisions.jsonl`: 보류 사유와 사람 결정
- `qa_report.json`: 커버리지, 일본어 잔존, 반복, 질문/부정, 영수증 검사
- `run_manifest.json`, `latest.json`: 실행 상태와 단일 최신 포인터
