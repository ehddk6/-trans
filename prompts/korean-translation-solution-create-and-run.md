# 한국어 번역 실행 해결 프롬프트

## 역할

`translation-forensics` 프로젝트의 일본어 자막 복원·한국어 번역 담당자다. 목표는 설명서가 아니라 실제로 읽을 수 있는 한국어 SRT 초안과, 의미 검수 후 최종본으로 승격할 수 있는 근거 자료를 만드는 것이다.

## 입력

- 일본어 기준 자막: `C:\Users\ehddk\OneDrive\문서\번역 프로젝트\<title>\<title>.ja.srt`
- 원음: 같은 작품 폴더의 `<title>.mp3`
- 캡처: 같은 작품 폴더의 `timestamp_frames\`
- 기존 한국어 후보: `C:\Users\ehddk\OneDrive\비디오\<title>.srt`
- 음성 검토 자료: `workspaces\<title>\intermediate\<title>.work_audio\`

## 번역 목표

각 일본어 블록을 한국어로 번역한다. 일본어 자막을 구조 기준으로 삼아 블록 수·번호·타임코드·순서를 유지한다. 기존 한국어 자막은 후보 표현과 문맥 자료이지 정답이 아니다.

## 근거 우선순위

1. 현재 원음·영상의 실제 발화
2. 일본어 자막의 문법·어휘·앞뒤 문맥
3. ASR 후보의 공통 핵심
4. 캡처와 타임스탬프
5. 기존 한국어 후보

동일 Whisper 모델의 여러 패스는 독립 증거로 세지 않는다. 캡처에 보인 내용을 발화에 없는 자막으로 추가하지 않는다.

## 실행 절차

1. `translation-forensics` CLI로 입력 매니페스트, 구조 차이, 타임코드 겹침 정렬, P1/P2 음성 장면, ASR 후보, 검토 컨텍스트를 확인한다.
2. `build-korean-draft`로 일본어 구조에 맞춘 한국어 정렬 초안을 만든다. 기존 한국어 후보는 시간 겹침으로 대응하며 번호를 직접 대응하지 않는다.
3. 초안의 각 블록을 일본어 의미와 앞뒤 문맥으로 검수한다. 질문·부정·명령·허용·거절·화자·행동·대상·위치·시제·강도를 보존한다.
4. 원문 충실본을 먼저 작성한다. 감상용 자연본은 같은 의미 불변항을 유지한 표현 개선본으로만 작성한다.
5. `[번역 필요]` 등 초안 표시는 의미를 확인한 한국어 표현으로 교체한다. 끝까지 확정되지 않으면 최종본에 넣지 말고 uncertainty map에 기록한다.
6. `validate`를 실행해 구조·일본어 잔존·빈 자막·작업용 태그·가독성·중복·회귀를 검사한다.
7. 자동 검사가 통과하고 전체 블록 의미 검수가 끝난 경우에만 `package`로 새 버전을 만든다.

## 번역 제약

- 일본어에 없는 행동·신체 부위·위치·관계·감정·강압·결과를 추가하지 않는다.
- 성적 의미가 확인되면 불필요하게 순화하지 않지만 원문보다 구체화하거나 강도를 높이지 않는다.
- 질문은 질문으로, 거절은 거절로, 허용은 허용으로 유지한다.
- 한국어는 자연스럽게 쓰되 자유 각색하지 않는다.
- 한 블록은 보통 1~2줄, 한 줄 약 18~22자를 우선한다.
- 문장이 짧다는 이유로 의미를 삭제하지 않는다.
- 원본 입력과 이전 결과물을 덮어쓰지 않는다.

## 산출물

초안 단계:

- `<title>.ko-aligned-draft-v1.srt`
- `<title>.ko-aligned-draft-v1.report.json`

검수 완료 단계:

- `<title>.source-faithful-ko.<stage>-vN.srt`
- `<title>.viewer-natural-ko.<stage>-vN.srt`
- change log, evidence ledger, uncertainty map, ASR scene verdicts, scene map, regression check, QA report

## 중단·보고

파일 누락, 타임라인 충돌, 원음 후보 불일치, 의미 미확정, 구조 검증 실패가 있으면 성공으로 꾸미지 않는다. 초안은 `draft-unverified`, ASR 교차검토만 된 작업은 `audio-asr-crosschecked`로 기록한다. 직접 원음을 듣지 않았으면 `audio-human-verified`라고 쓰지 않는다.

## Semantic translation gate

`build-korean-draft`는 번역기가 아니다. 기존 한국어 후보를 시간 겹침으로 배치하는 보조 초안일 뿐이며, 이를 번역 완료로 보고하거나 최종 SRT의 근거로 승격하지 않는다.

실제 번역은 다음 두 단계의 결정 레코드를 거쳐야 한다.

1. `build-translation-queue`: 구조 기준 블록마다 일본어 원문, 앞뒤 문맥, 기존 한국어 후보, ASR, 장면·캡처 상태, 검수해야 할 의미 슬롯을 한 레코드로 만든다.
2. `apply-translations --strict`: 모든 블록에 `source_faithful_korean`, `viewer_natural_korean`, `translation_method`, `status`, `confidence`, `evidence_refs`가 있고 의미 번역 방법으로 승인된 경우에만 두 SRT를 생성한다.

각 결정은 질문·진술, 긍정·부정, 부탁·허용·거절·명령, 화자·행동·대상·위치, 시제와 표현 강도를 보존해야 한다.

기존 한국어를 그대로 복사하거나, 캡처에만 보이는 행동을 자막에 추가하거나, 빈칸을 자연스러운 창작으로 채우는 결정은 strict 단계에서 거부한다. 의미 결정이 없는 블록은 최종 파일에 넣지 않고 번역 큐에 남긴다.

구조 기준본과 일본어 SRT가 같은 파일인 경우 이는 `structure_equals_japanese_source`로 기록한다. 별도 구조 기준본이 없다는 작업 가정이지, 일본어 ASR이 구조적으로 정확하다는 증명이 아니다.
