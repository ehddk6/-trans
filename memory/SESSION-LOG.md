# SESSION LOG — append, dated

## 2026-08-31

- D-013에 따라 scene_v2 문서 계약을 추가·동기화했다. block_v1 기본/롤백, scene_v2 experimental-unbenchmarked, pass 순서, 구조 잠금, source-faithful와 viewer-natural 분리, critics 분리, targeted visual 정책을 명시했다.
- benchmark 문서에 user-supplied 외부 baseline, blinded A/B/C, human benchmark not evaluated, 성공 목표를 기록했다. visual ablation A–E와 production negative-control guard를 기록했다.
- 문서 초안 단계에서는 실제 코드/CLI 구현을 아직 확인하지 않았고, 이후 구현·합성 통합 검증을 반영해 PRODUCT-TRUTH를 수정했다. Gemini·인간 우수성은 계속 미검증이다.
- D-013에 따라 실제 `scene_v2` 코드·CLI·schema·평가 원장을 구현했다. fake-provider 기반 통합 테스트는 `off`·`metadata`의 픽셀 전송 0, 정확한 targeted trigger만의 관찰 영수증, resume·산출물 해시·SRT 구조 잠금을 확인했다. 이는 합성 엔지니어링 검증이며 인간 자연스러움 또는 Gemini 비교 결과가 아니다.

One short section per working session: what was worked on, what was decided, and what remains.

---

## 2026-08-24

- 사용자의 구조적 프로젝트 개선 목표를 D-001로 기록했다.
- 저장소 상태와 기존 ballast 계층을 회수했다. 기존 번역·장면·오류 메모리 JSONL 세 파일은 보존했다.
- 프로젝트 감사, 개선 프롬프트, 구현계획, 실제 구현, 동일 조건 검증을 하나의 목표로 시작했다.
- 정본 `process-title`과 8월 17일 v2 복원 계층이 ASR 의미 근거를 공유하지 않는 split-brain 경계를 1차 병목으로 확인했다.
- GPT-5.6 구현 프롬프트를 AUTO 자기검토 정확히 1회 후 고정하고, 단계·검증·롤백이 있는 구현계획을 동결했다.
- canonical evidence bridge, TranslationUnit schema v2, process schema v3, Terra/Sol/deterministic gate 연결을 구현했다. 원시 ASR 근거는 보존하고 critical conflict만 비대칭으로 보류하도록 했다.
- Windows newline-canonical prompt hash, legacy schema 필드 일치, v2 acoustic 입력 재정규화, source SRT ref, lazy optional-ASR 초기화를 함께 수정했다.
- 위험 6건·정상 3건 최소대조 평가를 고정했다. bridge는 위험 6/6을 새로 보류했고 missed 0, clean regression 0이었다. 이 결과는 synthetic engineering safety이며 human quality proof가 아니다.
- 전체 453개 테스트, compileall, 두 prompt contract, frozen-report exact match, diff check를 통과했다.
- 대화 맥락이 없는 유지보수자 persona의 zero-context 리허설 1회차가 blocking stall 없이 clean 통과했다.
- 이전 체크포인트를 `memory/checkpoints/20260824-201027-structural-subtitle-quality-improvement.md`로 보존하고, 현재 체크포인트를 완료 상태와 다음 human-proof 경계로 교체했다.
- 저장소 규칙에 맞춰 compile 검증을 `src tools`로 확대했다. 변경된 인수 문서의 fresh-executor 리허설 2회차도 모든 명령 exit 0, blocking stall 없이 clean이었다.
- 실제 `C:\Users\ehddk\OneDrive\비디오` 175개 SRT와 `Videos` 111작품을 다시 매핑하고, 원본 보존형 개선 목표 D-002를 수행했다.
- 10개 파일·305블록을 변경했다. `IPZZ-856`은 273개 잔존 블록 중 119개를 안전한 동시간대 source KO로 채우고 154개를 수동 보정했으며, 다른 9개 파일은 32개 표기·음역 잔존만 최소 수정했다.
- 결과를 `C:\Users\ehddk\OneDrive\비디오_개선본_20260824`에 생성했다. 공통 111자막의 Japanese regex 문자는 2,558→0, 무수정 165개 해시 불일치 0, ABF/PRED control byte-identical, 금지 반복 문구 hit 0이다.
- 기존부터 규격 오류가 있던 `MOON-057`, `PRED-488`, `SNOS-167`은 범위 확장을 피하려고 수정하지 않고 README/QA에 명시했다.
- 전체 461 tests와 temp pycache compileall이 통과했다. 자동 source-overlap 129건을 전수 텍스트 검토해 10건을 수동 override로 승격했다.
- zero-context rehearsal 1회차는 재생용 SRT 선택 안내 부족으로 not clean이었다. exact MP4→SRT 매핑과 범위 목록을 추가한 뒤 fresh round 2가 blocking stall 없이 clean 통과했다.

## 2026-08-25

- 활성 기준 `C:\Users\ehddk\OneDrive\비디오`의 정본 125편을 전수 감사했다.
- `SONE-436`, `SSIS-513`을 포함해 거부 표식 131개를 일본어 원문·기존 한국어 대응본으로 복원했다.
- `MOON-057`, `PRED-488`, `SNOS-167`, `START-626` 및 추가 타임코드 오류 작품을 해시 보호·백업 후 수리했다.
- `SNOS-361`의 00:48 이후 붕괴한 타임라인 498큐를 로컬 `large-v3-turbo` ASR 301개 앵커로 재구축했다.
- 125편 공통으로 1,055개 장시간 잔류 큐를 읽기 시간으로 축소하고, 2,707개 인접 동일 큐를 단계적으로 병합했으며, 32자·2줄 레이아웃과 미세 큐 표시 시간을 정리했다.
- 최종 구조 상태는 파싱 오류·일본어·거부 표식·무효 길이·타임코드 역행 모두 0이다.
- 실제 영상이 로컬인 7편에 사진 21장·음성 21개를 만들고, 기존 작업공간 음성을 재사용해 26편에 대표 음성 77개를 추가했다.
- 최종 461개 테스트와 변경 도구 컴파일 검증을 통과했다.
- `작품` 원본 MP3를 근거로 1~3차 총 24편을 의미 대조했다. 3차 8편의 90개 위험 구간에서 확실한 오류 13개를 수정해, 누적 46개를 안전한 블록·타임코드 패치로 반영했다.
- 3차 후 활성 `비디오`의 실제 SRT 164개 파싱 오류는 0이며, 정본 125편 감사도 파싱 오류·일본어 잔존·거부 표식·무효 길이·타임코드 역행이 모두 0이다. 감사 산출물은 `workspaces/active-subtitle-audit-20260825-audio-aware-batch3-final.json`에 남겼다.
- 4차는 원본 MP3가 있는 미검토 고위험 8편·95개 음성 구간을 대조해 8개를 수정했다. 1~4차 합계는 32편·359개 구간·54개 수정이다.
- 최종 감사에서 실제 `비디오` SRT 166개는 모두 파싱됐고, 정본 125편은 파싱 오류·일본어 잔존·거부 표식·무효 길이·타임코드 역행이 모두 0이었다. 기존 시간 겹침 25개는 새로 만들지 않았고 그대로 남았다. 산출물은 `workspaces/active-subtitle-audit-20260825-audio-aware-batch4-final.json`에 남겼다.
- 사용자가 “진행해 그럼”이라고 확인해 D-005로 추가 고위험 원본 MP3 검토를 계속하기로 했다. 5차 8편·89개 구간에서 확실한 오류 14개를 수정했고, 1~5차 합계는 40편·448개 구간·68개 수정이다.
- 5차 후 실제 SRT 166개는 모두 파싱됐고, 정본 125편의 파싱 오류·일본어 잔존·거부 표식·무효 길이·타임코드 역행은 모두 0이다. 감사 산출물은 `workspaces/active-subtitle-audit-20260825-audio-aware-batch5-final.json`에 남겼다.
- 6차 8편·67개 위험 구간에서 원본 MP3 ASR을 우선 근거로 사용해 20개를 수정했다. 특히 `PRED-879`은 참조 자막의 알려진 오역 가능성 때문에 원본 MP3와 일치하지 않는 참조 텍스트를 수정 근거로 삼지 않았다. 1~6차 합계는 48편·515개 구간·88개 수정이다.
- 6차 후 실제 SRT 166개는 모두 파싱됐고, 정본 125편의 파싱 오류·일본어 잔존·거부 표식·무효 길이·타임코드 역행은 모두 0이다. 감사 산출물은 `workspaces/active-subtitle-audit-20260825-audio-aware-batch6-final.json`에 남겼다.
- 7차는 `MIHD-005`, `PRED-526`, `OFES-056`, `SONE-666`, `START-285`, `ABF-273`, `JUR-629`, `SNOS-256`의 원본 MP3 위험 구간 67개를 대조했다. 근거가 명확한 16개를 수정했고, 특히 반복된 시청 종료 문구와 `[불명]` 잔존, 한 단어로 축약된 긴 대사를 실제 음성에 맞췄다. 첫 번째 동일 문구로 잘못 적용될 수 있었던 패치는 모두 즉시 원복한 뒤 큐 번호·타임코드 기준으로 재적용했다. 1~7차 누계는 56편·582개 구간·104개 수정이며, 최종 감사에서 실제 SRT 166개 파싱 오류 0, 정본 125편의 파싱 오류·일본어 잔존·거부 표식·무효 길이·타임코드 역행 0, 기존 겹침 25개 유지다. 감사 산출물은 `workspaces/active-subtitle-audit-20260825-audio-aware-batch7-final.json`에 남겼다.
- 8차는 `MIHD-002`, `SNOS-243`, `SNOS-134`, `SONE-968`, `ABF-338`, `IPX-998`, `MDON-065`, `SNOS-229`의 원본 MP3 위험 구간 67개를 대조했다. 45개 후보 중 음성·참조·문맥이 모두 충분한 5개만 수정했고, 나머지는 이미 적절하거나 근거가 충돌해 보류했다. 1~8차 누계는 64편·649개 구간·109개 수정이며, 최종 감사에서 실제 SRT 166개 파싱 오류 0, 정본 125편의 파싱 오류·일본어 잔존·거부 표식·무효 길이·타임코드 역행 0, 기존 겹침 25개 유지다. 감사 산출물은 `workspaces/active-subtitle-audit-20260826-audio-aware-batch8-final.json`에 남겼다.
- 9차는 `PRED-830`, `START-126-UC`, `FNS-190`, `SNOS-059`, `JUR-070`, `ABF-364`, `IPZZ-544`, `SONE-081`의 원본 MP3 위험 구간 68개를 대조했다. 29개 후보 중 음성·문맥이 충분한 8개만 수정했고, 참조가 없거나 짧은 큐에서 근거가 모호한 후보는 보류했다. 1~9차 누계는 72편·717개 구간·117개 수정이며, 최종 감사에서 실제 SRT 166개 파싱 오류 0, 정본 125편의 파싱 오류·일본어 잔존·거부 표식·무효 길이·타임코드 역행 0, 기존 겹침 25개 유지다. 감사 산출물은 `workspaces/active-subtitle-audit-20260826-audio-aware-batch9-final.json`에 남겼다.
- 10차는 `SNOS-148`, `MFYD-142`, `JUQ-958`, `PRED-863`, `SSIS-642`, `MDON-089`, `ATID-677`, `SIRO-5561`의 원본 MP3 위험 구간 62개를 대조했다. 41개 후보 중 7개만 수정했고, 긴 구간에서 ASR가 일부 발화만 포착한 후보는 과잉 번역하지 않았다. 1~10차 누계는 80편·779개 구간·124개 수정이며, 최종 감사에서 실제 SRT 166개 파싱 오류 0, 정본 125편의 파싱 오류·일본어 잔존·거부 표식·무효 길이·타임코드 역행 0, 기존 겹침 25개 유지다. 감사 산출물은 `workspaces/active-subtitle-audit-20260826-audio-aware-batch10-final.json`에 남겼다.
- 11차는 `ABF-328`, `WAAA-666`, `ADN-746`, `SONE-101`, `ADN-622`, `IPZZ-881`, `MOON-057`, `SONE-054-C_GG5`의 원본 MP3 위험 구간 47개를 대조했다. 19개 후보 중 6개만 수정했고, 한 큐에 이미 적절한 인접 자막이 있는 중복 후보는 보류했다. 1~11차 누계는 88편·826개 구간·130개 수정이며, 최종 감사에서 실제 SRT 166개 파싱 오류 0, 정본 125편의 파싱 오류·일본어 잔존·거부 표식·무효 길이·타임코드 역행 0, 기존 겹침 25개 유지다. 감사 산출물은 `workspaces/active-subtitle-audit-20260826-audio-aware-batch11-final.json`에 남겼다.
- 12차는 `SSIS-908`, `ADN-789`, `ABF-169`, `HMN-872`, `MIDA-688`, `SONE-785`, `ABF-337`, `IPZZ-856`의 위험 큐 36개를 대조했고, 명확한 웃음 1개만 수정했다. 13차는 위험 큐가 남은 5편·20개 구간에서 3개를 수정했으며, 위험 큐가 없던 3편은 추출하지 않았다. 14차는 마지막 위험 큐 6편·8개를 대조했으나 추가 수정은 없었다. 이로써 제목이 정확히 맞는 원본 MP3 112편의 범위가 종료됐다. 실제 ASR 대조는 107편·890개 구간, 누적 수정은 134개다. 최종 감사는 `workspaces/active-subtitle-audit-20260826-source-mp3-review-final.json`에 남겼고, 실제 SRT 166개 파싱 오류 0 및 정본 125편의 일본어 잔존·거부 표식·무효 길이·타임코드 역행 0을 확인했다. 기존 겹침 25개는 변경하지 않았다.

## 2026-08-26 (재생 품질 마무리)

- 사용자 확인 D-006에 따라 기존 시간 겹침 25건을 원문·문맥 기준으로 수리했다. 최종 활성 감사에서 파싱 오류·일본어 잔존·거부 표식·무효 길이·타임코드 역행·시간 겹침은 모두 0이었다.
- 로컬 MP4 9편에 SRT를 입힌 실제 재생 프레임을 만들고 시각 검사했다. 빠른 탐색의 타임코드 초기화 문제를 `-copyts`로 해결했으며, 9/9에서 자막 표시가 정상이고 `GNI-007`, `HMN-720` 수리 구간도 중복 표시되지 않았다.
- `작품`의 중첩 폴더에서 `IPZZ-722.mp3`를 추가로 찾아 원본 음성 보유 작품 수를 112편에서 113편으로 정정했다. 감사 도구는 중첩 음성 경로를 인식하도록 보완하고 통합 실행으로 검증했다.
- `MIMK-284`, `SONE-061`, `SONE-107`, `OFJE-620-A`, `OFJE-620-B`, `SNOS-074`, `START-626`은 작품 폴더에 원본 음성이 없고 활성 MP4도 온라인 전용이라 음성 기반 의미 검토를 보류했다.

## 2026-08-26 (로컬 MP4 음성 검토)

- 사용자 승인 D-007에 따라 원본 MP3가 없던 7편의 OneDrive MP4를 이 장치에 고정해 로컬 사용 가능 상태로 만들었다. 완료 후 자동 확인 작업은 중지했다.
- 위험 큐가 있었던 3편 8개 구간만 `large-v3-turbo`로 전사했다. `OFJE-620-A`, `OFJE-620-B`, `SNOS-074`, `START-626`은 위험 큐가 없어 ASR을 만들지 않았다.
- `MIMK-284` 4개와 `SONE-061` 1개, 총 5개를 실제 음성에 맞게 수정했다. `MIMK-284` 1034와 `SONE-107` 351은 현재 번역을 유지했고, 잡음 의심 ASR은 수정 근거로 사용하지 않았다.
- 최종 감사에서 정본 125편의 파싱 오류·일본어 잔존·거부 표식·무효 길이·타임코드 역행·시간 겹침은 모두 0이었다. `비디오`의 SRT 166개도 모두 파싱됐다.
