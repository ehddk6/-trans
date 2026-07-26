# 공용 자료 검사 결과와 명세 차이

## 확인한 자료

작업 디렉터리에서 공용 자료 5종을 모두 확인했다. 원본 ZIP은 수정하지 않았고, 별도 검사 디렉터리에 풀어 목록·코드·README·자체 테스트를 확인한 뒤 `vendor/`에 보존했다. 파일별 SHA-256과 크기는 `references/source-audit.json`에 기록했다.

## Subtitle Forensics v1

- 제공 버전: `1.0.0`
- 핵심 파일: `subtitle_forensics.py`, `self_test.py`, `risk_model.joblib`, `model_metrics.json`
- CLI: `--root`, `--output`
- 자체 테스트: 110 checks, 0 failures
- import: `joblib`, `numpy`, `pandas`, `scipy`, `scikit-learn`
- 출력: 작품별 evidence ledger, uncertainty map, review queue, review packets, scene map, style profile, QA 보고서

문서 명세의 작품별 표준 입력과 달리 제공 구현의 `run()`은 ABF-303 학습 SRT와 변경 CSV를 특정 경로에서 요구하고, 여러 작품을 한 번에 분석한다. 제공 자료에는 해당 학습 입력이 없으므로 통합 `analyze`는 그 조건이 충족될 때만 vendor CLI를 호출한다. 그렇지 않으면 `forensics-run.json`에 사유를 기록하고 `structure_fallback` 큐를 만든다. 이 fallback은 위험 모델 결과가 아니다.

또한 vendor의 자체 `parse_srt`와 출력은 통합 프로젝트의 최종 SRT 규격·검증 상태·버전 정책을 모두 보장하지 않으므로, 통합 검증기를 별도로 둔다. vendor 파일은 변경하지 않았다.

## Audio Runner v1

- 핵심 파일: `run_forensic_audio.py`, `prepare_review_pack.py`, `run_asr_adaptive.py`
- 의존성: `ffmpeg`, `faster-whisper`
- 기본 프로필: `original_unbiased`, `dialogue_unbiased`
- 조건부 프로필: `original_no_vad`, `original_prompted`
- 체크포인트 키: `(scene_id, model, profile)`

제공 실행기는 프로젝트 폴더에서 queue·음성 후보를 찾고, 후보가 하나면 자동 선택한다. 통합 프로젝트는 입력 역할의 후보가 두 개 이상이면 더 엄격하게 중단하며, adapter가 명시적 경로를 전달한다. 제공 실행기는 산출 ZIP을 다시 만들 때 기존 ZIP을 교체할 수 있으므로, 통합 adapter는 중간 작업 폴더가 이미 채워져 있으면 `--force` 없이는 중단한다.

문서 명세의 `ingest-asr`, 검증 상태 전이, prompted 결과의 증거 등급, 안전한 ZIP 경로 검사, 최종 버전 증가·덮어쓰기 방지는 통합 계층에서 추가했다.

## 문서와 코드의 공통 차이

- 제공 문서는 의미 판정·한국어 번역을 Codex 계층에 두지만 vendor에는 해당 LLM 루프가 없다. 통합 프로젝트도 번역을 자동 생성하지 않고, 사람이 의미를 확정한 SRT를 `validate`·`package`에 입력으로 요구한다.
- 사진 연결은 제공 구현과 통합 계층 모두 근거 경로일 뿐 자동 판독이 아니다.
- 동일 Whisper 모델의 다중 패스는 통합 evidence ledger에서 독립 증거로 세지 않는다.
- `final` 상태는 자동으로 승격하지 않으며 전체 블록 검수·직접 청취·증거 완료 플래그를 명시적으로 요구한다.
