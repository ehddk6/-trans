# Subtitle Forensics v1.0.0

불완전한 자막·사진·수정 이력을 결합해 **오류를 자동 확정하는 것이 아니라, 깊은 검수가 필요한 블록을 증거 기반으로 순위화**하는 구현체입니다.

## 구현된 핵심

1. **잠재 대본 위험 모델**: ABF-303의 검증 변경 522개를 학습 라벨로 사용합니다.
2. **시간·문맥 특징**: 자막 길이, 초당 글자 수, 앞뒤 문장, 말투 변화, 반복과 질문–응답 신호를 봅니다.
3. **오류 기억**: 지금까지의 `before → after` 변경 이력을 모아 유사한 과거 오류를 탐색합니다.
4. **다차원 근거표**: 원문 의미·장면 적합·화자·한국어 자연도·환각 위험을 분리합니다.
5. **장면 단위 분석**: 긴 공백과 전환 신호를 이용해 블록을 장면 묶음으로 나눕니다.
6. **회귀 검사**: `[불명]`, 일본어 잔존, 3줄, 42자 초과, 숫자 환각, 알려진 오역 패턴을 검사합니다.
7. **이중 검수 큐**: 절대 위험 T1~T4와 작품 내부 상대 우선순위 P1~P4를 함께 제공합니다.
8. **4단계 절대 위험 등급**:
   - T1: 원문·원음·사진으로 전면 재복원
   - T2: 후보 3개 생성 후 증거 기반 판정
   - T3: 화자·문맥·자연스러움 검토
   - T4: 구조검사 통과 시 유지

## 모델 보정 결과

- 검증 방식: 5-fold StratifiedGroupKFold, contiguous 50-block groups
- ROC-AUC: 0.647
- Average Precision: 0.520
- 재현율 중심 F2: 0.785
- Precision / Recall: 0.433 / 0.985
- 임계값: 0.145

> 주의: 같은 작품 내부의 그룹 교차검증입니다. 다른 작품에서의 정확도를 뜻하지 않습니다.

## 분석 대상

- ABF-169: 기준본 — `ABF-169.final-natural-verified-ko.v7(1).srt`
- ABF-196: 사진 대조 1차 교정본 — `ABF-196.photo-guided-pass1-ko.v1.srt`
- ABF-303: 원문·사진 검증본 — `ABF-303.final-photo-verified-retranslated-ko.v1.srt`
- ABF-305: 사진 대조 1차 교정본 — `ABF-305.photo-guided-pass1-ko.v1.srt`
- ABF-328: 사진 대조 1차 교정본 — `ABF-328.photo-guided-pass1-ko.srt`
- ABF-337: 사진 대조 1차 교정본 — `ABF-337.photo-guided-pass1-ko.srt`
- ABF-338: 사진 대조 1차 교정본 — `ABF-338.photo-guided-pass1-ko.srt`
- ADN-503: 화면 영문·사진 검증본 — `ADN-503.photo-english-verified-ko.v1.srt`
- ADN-622: 사진 대조 1차 교정본 — `ADN-622.photo-guided-pass1-ko.srt`
- ADN-746: 사진 대조 1차 교정본 — `ADN-746.photo-guided-pass1-ko.srt`
- ADN-789: 사진 대조 1차 교정본 — `ADN-789.photo-guided-pass1-ko.srt`
- SONE-785: 사진 대조 1차 교정본 — `SONE-785.photo-guided-pass1-ko.srt`
- SSIS-400: 사진 대조 1차 교정본 — `SSIS-400.photo-guided-pass1-ko.srt`
- SSIS-575: 사진 대조 1차 교정본 — `SSIS-575.photo-guided-pass1-ko.srt`
- SSIS-642: 사진 대조 1차 교정본 — `SSIS-642.photo-guided-pass1-ko.srt`
- SSIS-652: 사진 대조 재번역본 — `SSIS-652.final-photo-guided-retranslated-ko.v1.srt`
- SSIS-908: 사진 대조 재번역본 — `SSIS-908.final-photo-guided-retranslated-ko.v1.srt`

## 주요 출력

각 작품 폴더:

- `*.evidence-ledger.csv`: 모든 블록의 증거와 위험 구성요소
- `*.uncertainty-map.csv`: 위험 점수 순 정렬
- `*.top-review-queue.csv`: 상대·절대 위험을 결합한 상위 200개 검수 큐
- `*.review-packets.jsonl`: 앞뒤 문맥·사진·오류 기억을 묶은 상위 100개 장면 패킷
- `*.scene-map.json`: 시간축 장면 묶음
- `*.style-profiles.json`: 실제 화자가 아닌 말투 유형 후보
- `*.qa-report.md/json`: 구조검사와 위험 요약

전체 폴더:

- `portfolio-priority-queue.csv`: 모든 작품을 합친 우선 검수 큐
- `portfolio-summary.csv`: 작품별 위험도 요약
- `risk_model.joblib`: 재사용 가능한 모델
- `model_metrics.json`: 모델 보정 결과
- `regression-memory.csv`: 과거 오류 기억

## 실행

```bash
python subtitle_forensics.py \
  --root /mnt/data \
  --output /mnt/data/subtitle_forensics_v1
```

## 중요한 한계

이 버전은 **사진 파일을 블록에 연결하지만 사진 의미를 자동 생성하지 않습니다.** 영상 원음·일본어 N-best ASR·시각 캡션을 추가하면 HLGR 전체 구조로 확장할 수 있습니다. 현재 결과는 오류 확정표가 아니라 가장 효율적으로 깊게 검수할 순서를 제시합니다.
