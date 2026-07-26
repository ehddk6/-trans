# 연구·공식 문서 검토와 적용 판단

## 채택

- Whisper 구현은 낮은 `avg_logprob`와 높은 `no_speech_prob`를 디코딩 fallback 신호로 사용한다. 따라서 두 값은 **검토 우선순위/에스컬레이션 신호**로만 보관하고, 정확도 확률·단일 임계값·의미 확정에는 쓰지 않는다. [OpenAI Whisper `transcribe.py`](https://github.com/openai/whisper/blob/main/whisper/transcribe.py)
- forced alignment는 단어/구간 시간 정보와 비발화 주석을 다루는 정렬 기법이다. token 시간과 block overlap, boundary warning을 별도 산출물로 두되 정렬 실패를 번역 의미의 증거로 해석하지 않는다. [Montreal Forced Aligner 문서](https://montreal-forced-aligner.readthedocs.io/en/v1.0/)
- MQM은 Accuracy의 addition/omission/mistranslation 등 원인 분류와 severity를 제공한다. 프로젝트는 이를 자막 특화 오류(화자·화행·가독성·증거 과장)로 확장하되, 핵심 축은 MQM의 정확성/언어 규범/스타일/디자인 원칙에 맞춘다. [MQM Typology](https://themqm.org/error-types-2/typology/), [MQM scoring models](https://themqm.org/error-types-2/the-mqm-scoring-models/)
- 문맥을 가진 전문 인간 평가가 자동 지표나 단문 단위 판정보다 중요하다는 결과를 반영해, 장면 단위 Q/A·화자·말투 검토와 블라인드 골드 평가를 분리한다. [Freitag et al., 2021](https://arxiv.org/abs/2104.14478)

## 채택하지 않음

- Whisper confidence나 문자열 유사도로 ASR/번역을 자동 확정하지 않는다. 이는 디코더 내부 신호이지 보정된 의미 정확도가 아니다.
- 서로 다른 Whisper profile을 독립 증거로 더하지 않는다. `whisper-family` 하나로 묶는다. 독립 ASR, 사람 청취, 일본어 기준본, forced alignment 등은 별도 family만 센다.
- 자동 MT 지표나 모델 심사 점수로 최종 품질을 선포하지 않는다. WMT 평가도 인간 평가를 gold standard로 둔다. [WMT 2026 MTEval task](https://www2.statmt.org/wmt26/mteval-task.html)
- reference 한 문장만으로 문맥 번역의 우열이나 인간 동등성을 주장하지 않는다. 문맥·평가자·reference 설계가 결론을 바꿀 수 있다. [Toral, 2020](https://arxiv.org/abs/2005.05738)

## 실험 설계

작품 단위 leave-one-title-out을 기본으로 하고, 불가능하면 같은 작품 내 지표는 일반화 성능으로 표기하지 않는다. P1/P2 우선 검토의 recall은 P3/P4 무작위 감사 표본으로 보정·보고한다. 보고 지표는 PR-AUC, review budget 5/10/20%의 recall·precision, critical-error recall, title-level generalization, calibration error, false-negative 분석이다. 골드 또는 블라인드 인간 평가가 없으면 결과 판정은 `not-demonstrated`다.
