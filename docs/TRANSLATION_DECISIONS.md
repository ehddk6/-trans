# 의미 번역 결정 계층

이 프로젝트에서 기존 한국어 자막은 번역 결과가 아니라 시간 정렬용 후보다. 일본어 SRT와 기존 한국어 SRT의 블록 수가 달라도 기존 자막을 그대로 복사하지 않는다.

## 1. 번역 큐 생성

```powershell
python -m translation_forensics.cli build-translation-queue `
  --project-root . --title SAMPLE `
  --structure .\workspaces\SAMPLE\inputs\SAMPLE.structure.srt `
  --ja .\workspaces\SAMPLE\inputs\SAMPLE.ja.srt `
  --previous-ko .\workspaces\SAMPLE\inputs\SAMPLE.previous-ko.srt `
  --review-context .\workspaces\SAMPLE\intermediate\review-context.jsonl `
  --review-queue .\workspaces\SAMPLE\intermediate\SAMPLE.review-queue.structure-fallback-v4.csv `
  --capture-index .\workspaces\SAMPLE\metadata\capture-index.json
```

출력은 `SAMPLE.translation-queue-v1.jsonl`이다. 각 블록에는 일본어, 시간 정렬된 기존 한국어 후보, 앞뒤 문맥, ASR 후보, P1/P2 등급, 캡처 상태, 보존해야 할 의미 슬롯이 들어간다.

## 2. 결정 레코드 작성

번역 담당자는 각 레코드에 다음 값을 추가한다.

- `source_faithful_korean`: 일본어 의미를 가장 충실하게 보존한 한국어
- `viewer_natural_korean`: 의미를 바꾸지 않고 영상 자막 호흡에 맞춘 한국어
- `translation_method`: `semantic_review_from_japanese`, `semantic_review_with_asr`, `semantic_review_with_context`, `human_verified_semantic_review` 중 하나
- `translation_model`: 고정값 `gpt-5.6-terra`. 번역 큐·결정 템플릿·적용 CLI는 이 값 외의 모델을 받지 않으며, ASR 모델에는 적용되지 않는다.
- `status`: `translated`, `reviewed`, `approved` 중 하나
- `confidence`: `high`, `medium`, `low`
- `evidence_refs`: 실제로 사용한 근거의 ID
- `uncertain_slots`, `review_note`: 남은 불확실성

`previous_korean_copy`, `aligned_previous`, 빈칸, `[번역 필요]`, 일본어 잔존은 최종 결정으로 사용할 수 없다.

## 3. 두 SRT 생성

```powershell
python -m translation_forensics.cli apply-translations `
  --project-root . --title SAMPLE `
  --structure .\workspaces\SAMPLE\inputs\SAMPLE.structure.srt `
  --translation-queue .\workspaces\SAMPLE\intermediate\SAMPLE.translation-queue-v1.jsonl `
  --decisions .\workspaces\SAMPLE\intermediate\SAMPLE.translation-decisions.jsonl `
  --source-output .\workspaces\SAMPLE\intermediate\SAMPLE.source-faithful-ko.text-crosschecked-v1.srt `
  --viewer-output .\workspaces\SAMPLE\intermediate\SAMPLE.viewer-natural-ko.text-crosschecked-v1.srt `
  --strict
```

`--strict`는 모든 구조 블록의 결정, 한국어 문장, 의미 번역 방법, 승인 상태, 확신도, 일본어 잔존·작업 표식 부재를 검사한다. 추가로 각 결정의 `evidence_refs`가 같은 블록의 원본 translation queue에 실제로 선언되어 있는지 대조한다. 하나라도 부족하거나 임의의 근거 ID가 있으면 두 SRT를 생성하지 않는다.

translation queue에 `consistency_context.conflicts`가 있으면 결정은 해당 consistency ID를 `consistency_conflicts`에 기록해야 한다. 이는 충돌을 해결했다는 뜻이 아니라, 충돌을 숨기지 않고 검수 흐름으로 보냈다는 뜻이다. confirmed consistency 항목을 실제로 적용한 경우에만 `consistency_refs`에 해당 ID를 기록한다.

적용 보고서에도 사용 모델이 기록된다. 과거의 외부 기계번역 초안은 이 정책의 산출물이 아니며, 검증 또는 최종 승격 대상이 아니다.

작품을 여러 차례 나누어 검수할 때는 완료된 블록만 별도 JSONL로 저장한 뒤 병합한다.

```powershell
python -m translation_forensics.cli merge-translation-decisions `
  --base .\workspaces\SAMPLE\intermediate\SAMPLE.translation-decisions-template-v1.jsonl `
  --reviewed .\workspaces\SAMPLE\intermediate\SAMPLE.translation-decisions-reviewed-p1p2-v1.jsonl `
  --output .\workspaces\SAMPLE\intermediate\SAMPLE.translation-decisions-progress-v1.jsonl
```

병합 결과에 미검수 블록이 하나라도 있으면 `apply-translations --strict`가 실패한다.

## 3-1. 불확실성 중심 검수 순서

```powershell
python -m translation_forensics.cli build-uncertainty-review-queue `
  --project-root . --title SAMPLE `
  --decisions .\workspaces\SAMPLE\intermediate\SAMPLE.translation-decisions-progress-v1.jsonl `
  --translation-queue .\workspaces\SAMPLE\intermediate\SAMPLE.translation-queue-v1.jsonl
```

출력 큐는 미확정·보류 상태, low evidence strength, 불확실 슬롯, 근거 참조 누락·불일치, consistency ledger 충돌·미확정 항목, 기존 P1/P2·낮은 포렌식 근거, source/viewer 자동 QA 문제를 이유별로 보여 준다. 단일 점수만으로 의미 위험을 숨기지 않으며, `critical → high → medium`과 블록 번호 순으로 정렬한다.

## 4. 최종 패키징 게이트

`final` 패키지는 의미 번역 결정 파일, 전체 블록 검수, 직접 원음 청취, 근거·QA 완료가 모두 필요하다. ASR만으로 직접 청취 상태를 부여하지 않는다. 캡처 파일이 존재해도 화면 의미를 실제로 판독하지 않았다면 `capture_semantic_interpretation=false`로 남긴다.

## 5. 의미 프레임과 가설 원장

번역 담당자는 semantic frame에서 `speaker`, `addressee`, `speech_act`, `polarity`, `interrogative`, `request_strength`, `permission`, `prohibition`, `action`, `actor`, `target`, `body_part`, `location`, `direction`, `temporal_state`, `completion_state`, `result`, `emotion`, `sexual_semantic_class`, `register`를 슬롯별로 작성한다. 각 값은 `{ "value": ..., "state": "null|unknown|confirmed|contradicted", "evidence_refs": [...] }` 구조를 사용한다.

`confirmed`는 실제 근거가 있는 값만, `unknown`은 정보가 있을 수 있으나 확정할 수 없는 값만 사용한다. high-risk 발화에 상충하는 해석이 생기면 `hypothesis_id`, `semantic_frame`, `supported_by`, `contradicted_by`, `unsupported_specificity`, `status`를 가진 후보를 기록한다. 무근거 후보를 형식상 둘 만들지 않는다.

극성·의문·요청 강도·허용/금지·행위자·대상·위치·완료·성적 의미·화자가 서로 충돌하면 `validate-forensic-records`가 escalation을 보고한다. 동일 Whisper profile의 여러 패스는 `whisper-family` 하나로만 계산한다.

## 6. MQM과 평가 게이트

`mqm-errors` CSV는 최소 `block_number,error_type,severity,evidence_refs,review_status` 열을 가진다. 오류 유형은 정확성, 참여자, 행동/성적 의미, 대화, 자막 자연성, 가독성, 증거/검증 오류를 포함하며 전체 목록은 `schemas/mqm-error.schema.json`과 `src/translation_forensics/mqm.py`에 있다.

`evaluation-validated`와 `final`에는 `evaluation-summary` 및 gold 또는 blind review 근거가 필요하다. `evaluation-validated`/`final`에는 검증된 semantic frame·hypothesis ledger가 필요하고, MQM 원장이 있다면 critical 오류가 남아 있으면 패키징이 중단된다. 실제 평가가 없는 경우에는 해당 산출물을 만들지 않고 `not-demonstrated`로 보고한다.
