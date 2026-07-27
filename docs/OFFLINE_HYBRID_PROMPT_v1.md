# API 호출 없는 하이브리드 번역 파이프라인 개선 프롬프트

Role/Context:
이 프로젝트는 일본어 성인 영상의 자막을 검증 및 복원하는 파이프라인이다.
사용자는 외부 네트워크 호출(API 키, 과금) 없이, 이미 존재하는 로컬 결과물(`closed-world`, `machine-final`, `inferred-recovery`)만 조합하여 최종 `autonomous-release`와 유사한 하이브리드 완전 자동 완성본을 얻고자 한다.
이 문서는 이를 구현하기 위해 기존 파이프라인을 어떻게 수정해야 할지 지시하는 실행 프롬프트이다.

# Goal
기존 `run-autonomous-release` 워크플로우를 수정하거나 새로운 CLI 명령(`run-offline-hybrid-release`)을 만들어, API 호출 없이 13개 작품의 전체 블록을 커버하는 `source-faithful` 및 `viewer-complete` SRT 패키지를 생성한다.

# Inputs and Source Priority
1. `closed-world-validated` 결과 (최우선 순위: 로컬 음성과 SRT만으로 확정된 가장 안전한 번역)
2. `inferred-recovery-v4` 결과 (차순위: 손상되거나 누락된 블록을 문맥으로 복구한 번역)
3. `machine-final-v1` 결과 (최후 순위: 의미 검증은 통과하지 못했지만, 시청을 위해 화면에 띄울 수 있는 기계 번역 초안)

# Success Criteria
- 13개 작품 모두에 대해 네트워크 호출 없이 `source-faithful`과 `viewer-complete` 자막이 생성되어야 한다.
- `OPENAI_API_KEY`나 인터넷 연결이 없어도 동작해야 한다.
- 각 블록의 번역 출처가 `closed-world`, `inferred-recovery`, `machine-final` 중 어디에서 왔는지 추적 가능해야 한다.

# Constraints and Decision Rules
- API 호출 코드를 완전히 제거하지 말고, 네트워크 우회(bypass) 모드로 작동할 수 있게 하거나 전용 오프라인 병합 스크립트를 작성한다.
- `source-faithful`은 확실한 근거가 있는 `closed-world`와 `inferred-recovery` 결과만 포함하고, 나머지는 비운다.
- `viewer-complete`는 시청의 흐름이 끊기지 않게 `machine-final`의 기계 번역이라도 채워 넣는다. 단, 완전히 복구 불가능한 경우에만 `…`을 쓴다.

# Output
1. API 호출 없는 13개 작품 병합 처리를 실행하는 파이썬 스크립트 작성 (`scripts/build_offline_hybrid_release.py`)
2. 기존 CLI(`src/translation_forensics/cli.py`)에 `build-offline-hybrid` 명령어 추가
3. 실행 후 13개 작품에 대한 `offline-hybrid-v1` 패키지 생성
