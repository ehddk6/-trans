# 현재 시스템 감사 (2026-07-26)

## 확인 범위와 방법

원본 공용 자료 5종과 `vendor/` 보존본, 통합 CLI, 단위·통합 테스트, 작품별 `workspaces/` 산출물을 읽었다. 원본 ZIP·입력 SRT·기존 한국어 결과는 수정하지 않았다. 이 문서는 코드/파일 확인 결과이지 성능 주장이 아니다.

## 실제 구성과 데이터 흐름

`inspect`가 입력 역할을 확정하고 SHA-256 매니페스트와 구조 잠금을 만든다. `analyze`는 제공 Forensics 실행 조건이 갖춰졌을 때만 vendor를 호출하고, 그렇지 않으면 `structure_fallback` 검토 큐를 만든다. `prepare-audio`/`run-asr`/`ingest-asr`는 P1/P2 장면과 Whisper 계열 후보를 연결한다. `build-translation-queue`는 일본어·기존 한국어·ASR·주변 문맥을 한 레코드로 모으고, `apply-translations --strict`는 사람이 채운 결정 JSONL을 두 SRT에 적용한다. `validate`와 `package`는 구조·문자·가독성·상태 게이트를 검사한다.

| 요소 | 실제 역할 | 출력/한계 |
| --- | --- | --- |
| `vendor/subtitle_forensics_v1` | 위험 우선순위화 | ABF-303 학습 입력이 없으면 실행 불가. fallback은 모델 결과가 아님 |
| `vendor/subtitle_audio_forensics_runner_v1` | P1/P2 클립·Whisper 후보 | 동일 Whisper 다중 패스는 독립 증거가 아님 |
| `src/translation_forensics` | 안전한 입력 탐색·구조 잠금·어댑터·패키징 | 의미를 자동 확정하지 않음 |
| `schemas/` | 결정/검토 큐 계약 | 기존 v1에는 명시적 가설·MQM 계약이 없었음 |
| `workspaces/` | 작품별 중간 산출물 | 일부 작품은 P1/P2만 처리되어 전체 완료를 뜻하지 않음 |

## 테스트와 검증 가능성

`tests/unit/test_core.py`는 SRT 구조, 중복 입력 중단, fallback 표기, ASR 에스컬레이션, 상태 전이, 결정 JSONL strict 적용을 검사한다. `tests/integration/test_cli_flow.py`는 새 프로젝트의 `init-title → inspect → analyze` 흐름을 검사한다. 제공 vendor 자체 테스트와 통합 테스트는 코드 동작의 근거일 뿐, 번역 정확도·ASR 정확도·작품 간 일반화의 근거는 아니다.

## 확인된 한계와 미증명 주장

- `avg_logprob`, `no_speech_prob`, 문자열 유사도는 검토 우선순위 신호이며 오역 확정 기준이나 보정된 확률이 아니다.
- P1/P2 표본만으로 전체 작품 recall 또는 전체 자막 품질을 주장할 수 없다. P3/P4 무작위 감사가 필요하다.
- 현재 제공 자료에는 독립 골드 정답, 블라인드 평가, 일본어 직접 청취 판정이 없다. 따라서 기존보다 **개선됨**은 아직 증명되지 않았다.
- 장면/화자 상태와 사진은 근거 메타데이터일 뿐 자동 의미 판독 또는 실제 인물 식별이 아니다.
- forced alignment 실패는 의미 판정이 아니라 정렬 불확실성으로 기록해야 한다.

## 차세대 추가 계층

`forensic_model.py`와 새 스키마는 의미 프레임의 `null`/`unknown`/`confirmed`/`contradicted` 구분, source-family 단위 독립성, critical slot 충돌 escalation, 가설 원장 검증을 제공한다. `init-forensic-records`가 만드는 레코드는 명시적으로 `unreviewed`이며 번역 또는 검증 완료가 아니다.

## 2026-07-26 implementation delta

`validate-evidence-artifact` now validates reviewer-authored speaker-state,
alignment-evidence, and backtranslation-check artifacts. ASR rows without an
explicit family are assigned `whisper-family`; multiple Whisper profiles cannot
be counted as independent. Only explicit reviewer-supplied confirmed slot
claims may trigger semantic conflict escalation. This is a validation guard,
not an ASR semantic parser or a quality-improvement result. See
`docs/FINAL_STATUS_REPORT.md` for the full evidence boundary and experiment
status.
