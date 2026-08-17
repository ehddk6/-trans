# 번역 프롬프트 아키텍처

| 자료 | 실제 역할 | 호출·검증 경로 |
| --- | --- | --- |
| `AGENTS.md`, `references/translation-prompt-v6.txt` | 프로젝트 수준 번역 원칙 | Codex·사람 작업의 상위 지침, Python provider가 직접 주입하지 않음 |
| `prompts/batch-adult-srt-translation-run.md` | 작업자용 배치 라우팅 안내 | 문서 프롬프트; CLI가 직접 실행하지 않음 |
| `prompts/terra-semantic-translation-v1.md` | 블록별 source-faithful → viewer-natural 결정 계약 | `validate-prompt-contract`; 외부 호출은 저장소가 수행하지 않음 |
| `prompts/integrated-noisy-asr-recovery-v1.md` | v6·batch·Terra의 process-title 관련 의미·근거·복원 원칙을 좁혀 고정한 운영 overlay | `process-title` Terra 호출이 본문을 직접 로드하고 실행 영수증·cache identity에 SHA-256을 결속 |
| `prompts/autonomous-subtitle-decision-v1.md`, critic | provider 기반 autonomous-release 후보·반증 계약 | `run-autonomous-release`가 prompt·manifest·schema를 로드 |
| `docs/OFFLINE_HYBRID_PROMPT_v1.md` | offline-hybrid preview의 과거 설계 기록 | 운영 런타임 프롬프트가 아님 |
| `tools/codex_autonomous_release.py` | Codex 작성 결정의 bridge 검증·draft 패키징 | 별도 프롬프트 파일을 호출하지 않으며 human-final을 금지 |

Terra 계약은 translation queue의 `consistency_context`를 입력으로 받는다. source-faithful이 근거·불확실성·일관성 충돌을 먼저 기록하고, viewer-natural은 같은 의미 범위 안에서만 자막화를 수행한다. 현재 블록과 무관한 장편 원장 전체는 전달하지 않는다.

프롬프트 manifest의 해시는 파일 무결성·계약 버전을 확인한다. 모델이 실제로 해당 규칙을 따랐는지나 인간 번역 품질을 증명하지는 않는다.
