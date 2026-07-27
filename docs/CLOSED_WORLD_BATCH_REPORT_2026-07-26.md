# 폐쇄형 자동 검증 배치 보고서 — 2026-07-26

## 결과

13개 작품의 `run-closed-world` 실행, 전용 패키지 검증, 품질 주장 감사, 동일 경로 재실행을 완료했다.

- 총 블록: 10,380
- `accepted`: 870 (8.3815%)
- `abstained`: 9,510 (91.6185%)
- 결정 커버리지: 100%
- 패키지 검증 실패: 0
- 재실행 불일치 또는 덮어쓰기 충돌: 0
- `human_reference_equality`: 전 작품 `unidentifiable`
- `100_percent_equal`: 전 작품 `false`
- `final_promotion_allowed`: 전 작품 `false`

| 작품 | 전체 | 승인 | 보류 | 승인율 | 기계 시간축 |
| --- | ---: | ---: | ---: | ---: | --- |
| ADN-622 | 651 | 62 | 589 | 9.52% | `machine-aligned` |
| ADN-746 | 386 | 79 | 307 | 20.47% | `machine-aligned` |
| IPX-998 | 832 | 177 | 655 | 21.27% | `machine-aligned` |
| JUQ-439 | 890 | 0 | 890 | 0.00% | `machine-aligned-partial` |
| JUQ-778 | 1,955 | 0 | 1,955 | 0.00% | `machine-aligned-partial` |
| JUQ-811 | 827 | 15 | 812 | 1.81% | `machine-aligned-partial` |
| MDON-065 | 533 | 33 | 500 | 6.19% | `machine-aligned-partial` |
| SONE-785 | 858 | 46 | 812 | 5.36% | `machine-aligned-partial` |
| SSIS-400 | 577 | 31 | 546 | 5.37% | `machine-aligned-partial` |
| SSIS-575 | 1,628 | 61 | 1,567 | 3.75% | `machine-aligned-partial` |
| SSIS-642 | 641 | 131 | 510 | 20.44% | `machine-aligned-partial` |
| SSIS-652 | 304 | 89 | 215 | 29.28% | `insufficient-machine-anchors` |
| SSIS-908 | 298 | 146 | 152 | 48.99% | `insufficient-machine-anchors` |

## 해석

승인율은 번역 정확도 추정치가 아니다. 승인된 블록은 로컬 후보 계열 합의와 구현된 명시적 의미 슬롯 규칙을 통과한 범위이며, 보류 블록은 자동 판단 범위를 넘어선다.

`machine-aligned`는 로컬 Whisper와 일본어 SRT가 초·중·후반에서 일치한 기계 정렬 상태다. `machine-aligned-partial`과 `insufficient-machine-anchors`는 전체 시간축의 사람 검증을 뜻하지 않는다.

원래 시간축 보고서의 실제 상태는 `media-too-short` 5개, 다중 앵커 부재 `unresolved` 8개다.

## 산출물

각 작품의 결과는 다음 디렉터리에 있다.

```text
workspaces/<작품>/closed-world/closed-world-validated-v1/
```

`source-faithful.preview.srt`와 `viewer-natural.preview.srt`는 구조 확인용 미리보기다. `[미확정]` 블록을 포함하므로 `final` 자막이 아니다. 상세 보류 사유는 `unresolved.jsonl`, 보장 범위는 `proof.json`, 재현 해시는 `run-manifest.json`에서 확인한다.
