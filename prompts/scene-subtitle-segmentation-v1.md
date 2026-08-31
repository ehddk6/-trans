# Scene subtitle segmentation v1

장면 대사 turn을 구조가 잠긴 원본 unit cue로 투영한다. 의미를 번역하거나 새 대사를 만들지 않는다. 입력의 원본 unit 순서, 시간, 문자 예산, 읽기 속도와 structure lock을 지킨다.

- 각 원본 unit에 정확히 하나의 projection을 내고, 누락·중복·순서 변경·빈 text·일본어 잔존을 허용하지 않는다.
- 한 문장을 여러 cue에 나눌 때 조사, 보조용언, 말끝을 부자연스럽게 고립시키지 않는다. 같은 전체 문장을 여러 cue에 통째로 복사하지 않는다.
- 질문과 대답 순서 및 의미의 시간적 위치를 바꾸지 않는다. 안전한 분리가 불가능하면 입력의 alignment group과 문자 예산에 맞추어 가장 작은 범위에서만 재분배한다.

설명 없이 `scene-subtitle-segmentation-v1` 스키마의 JSON 객체 하나만 출력한다.
