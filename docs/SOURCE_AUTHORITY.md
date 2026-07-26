# 자료 권위와 충돌 처리

충돌 시 다음 순서로 판단한다.

1. 실제 증거와 구조 기준본
2. 번역 프롬프트 v6의 의미·품질·하드 제약
3. 프로젝트 지침
4. `subtitle-audio-crosscheck-method-v1.md`
5. `subtitle-forensics-io-schema-v1.md`
6. Subtitle Forensics의 위험도와 자동 검사 결과
7. 로컬 ASR 실행기의 자동 판정

실제 증거는 직접 확인한 원음·영상, 명확한 일본어 원문·ASR, 다중 ASR의 공통 핵심, 독립 대체 ASR, 앞뒤 문맥, 타임스탬프 화면을 뜻한다. 위험 점수·P1/P2·추가 ASR 패스는 최종 의미 판정이 아니다.

`vendor/subtitle_forensics_v1`은 검수 순서 결정기이며 자동 번역기나 오역 확정기로 취급하지 않는다. `style-profiles.json`은 실제 화자 식별 결과가 아니다.

`vendor/subtitle_audio_forensics_runner_v1`은 원음 증거와 일본어 후보를 만드는 실행기다. 다중 ASR만으로 `audio-human-verified`를 부여하지 않는다.
