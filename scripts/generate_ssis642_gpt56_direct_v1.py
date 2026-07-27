"""Create the SSIS-642 GPT-5.6 direct Japanese-to-Korean draft.

Only the Japanese SRT and model-authored Korean decisions below are used.
No former Korean subtitle, external translation API, or local MT engine is
read.  Blocks 161--173 have unusable Japanese subtitle text; exact-window,
local faster-whisper ASR was used only for those blocks.  No human reviewed
the audio.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from translation_forensics.semantic_translation import (
    apply_translation_decisions,
    validate_translation_decisions,
)
from translation_forensics.srt import parse_srt
from translation_forensics.translation_model import DEFAULT_TRANSLATION_MODEL
from translation_forensics.validation import validate_pair, write_validation_report


ROOT = Path(__file__).resolve().parents[1]
TITLE = "SSIS-642"
SOURCE = ROOT.parent / TITLE / f"{TITLE}.ja.srt"
OUT = ROOT / "workspaces" / TITLE / "intermediate"
STEM = f"{TITLE}.gpt56-direct-v1"


T = {
    1: "시청해 주셔서 감사합니다", 2: "자", 3: "죄송합니다", 4: "미안하다",
    5: "너한테도 한턱내고 싶었는데", 6: "마침 가진 게 없어서", 7: "미안하다",
    8: "네, 별일 아닙니다", 9: "실례하겠습니다", 10: "어라", 11: "반은 빌려줄게",
    12: "괜찮습니다", 13: "그러지 마", 14: "뭐야", 15: "가테킨 가글은 감기에 걸렸대",
    16: "너도 할래", 17: "괜찮습니다", 18: "아, 그래", 19: "다음 주 출장이라던데",
    20: "음", 21: "싫다", 22: "왜", 23: "모토 키브 씨도 같이 가는 거잖아",
    24: "한턱 얻어먹겠네", 25: "그 사람은 늘 더치페이라 기대 안 해", 26: "그렇구나",
    27: "엄청 짜구나", 28: "음", 29: "그런데", 30: "짜기만 한 게 아니라 더러워",
    31: "알아, 알아", 32: "들어 본 적 있어", 33: "그거잖아",
    34: "밥 먹고 이쑤시개로 쑤시는 거", 35: "그 뒤에 차로 헹구는 정도야",
    36: "음", 37: "기분 나빠지니까 그만해", 38: "그리고", 39: "침 묻혀서 책장 넘긴다고 들었어",
    40: "진짜 그만해, 살려줘", 41: "미안, 미안. 더는 안 할게", 42: "아, 출장 우울하네",
    43: "회사원의 비애지", 44: "아, 그건 그렇고", 45: "오늘 같이 가자", 46: "쉿",
    47: "그런 말은 사내에서도 하지 말자고 했잖아", 48: "응", 49: "아무도 안 듣잖아",
    50: "논점이 다르잖아", 51: "들리느냐 마느냐 얘기가 아니라", 52: "말하느냐 마느냐의 문제야",
    53: "아, 상담도 잘 마무리됐고", 54: "술이 맛있네요, 부장님", 55: "한낮부터 마시는 술은 맛있군",
    56: "음", 57: "음", 58: "음", 59: "음", 60: "아, 상담도 잘 마무리됐고",
    61: "술이 맛있네요, 부장님", 62: "한낮부터 마시는 술은 맛있군", 63: "음", 64: "음", 65: "음",
    66: "아, 상담도 잘 마무리됐고 술이 맛있네요, 부장님", 67: "한낮부터 마시는 술은 맛있군",
    68: "아, 아이테 군", 69: "마랑이나 오크라 같은 건 모르겠는데 그 끈적한 요리면 좀 시켜 둬",
    70: "네", 71: "부장님은 여전히 끈적하고 찐득한 걸 좋아하시네요", 72: "하하하, 맞아",
    73: "아이테 군, 내가 왜 끈적한 걸 좋아하는지 알아?", 74: "면역력 증강 같은 건가요?",
    75: "면역력 증강이라, 아쉽군", 76: "그것도 그렇긴 한데", 77: "끈적한 걸 먹고 몸에서 확 내보내는 거지",
    78: "확", 79: "모르겠어?", 80: "정자야, 정자", 81: "너", 82: "그건 성희롱 발언입니다",
    83: "잘 들어, 확이라는 건 비유야", 84: "알겠어?", 85: "일에 대한 열정이 확 나온다, 그런 뜻이야",
    86: "그런 뜻이야", 87: "실례하겠습니다", 88: "맞아", 89: "그래, 이걸로 가자", 90: "네",
    91: "입 좀 벌려", 92: "침", 93: "보여 봐", 94: "침이요?", 95: "응, 침이야",
    96: "자, 입 벌려 봐", 97: "상사 명령이야", 98: "빨리", 99: "자, 자", 100: "자", 101: "자, 자",
    102: "혀를 이렇게 위아래로 움직여", 103: "카, 카, 카, 카", 104: "더, 자", 105: "더 세게, 자",
    106: "음, 음, 음", 107: "자네", 108: "그러니까 그런 건 성희롱이야", 109: "두 번째",
    110: "실례하겠습니다", 111: "난 그런 외설적인 말을 하는 게 아니야", 112: "그녀가 끈적한 식재료를 먹었는지",
    113: "다시 입 벌려", 114: "이렇게 해서, 내밀어", 115: "그래, 위아래로 움직여", 116: "에 하고",
    117: "음 하고", 118: "더 내밀어 봐, 혀", 119: "더", 120: "좋네", 121: "더, 더", 122: "더",
    123: "누누누누누누 하고 있어", 124: "진단 결과, 그녀의 타액은 충분히 끈적합니다. 끈적한 식재료를 먹고 있다는 뜻이야",
    125: "그러니까 침도 끈적한 거야", 126: "끈적하기만 하면 안 돼, 묽게 해야 해", 127: "끈적해. 묽게 하지 마",
    128: "세게", 129: "입 벌려 봐. 혀 내밀고", 130: "아직도 끈적하네", 131: "묽게 하지 마",
    132: "묽게, 묽게 해야 해. 건강해져야지", 133: "성희롱 같은 건 안 돼", 134: "성희롱 같은 것도 힘들잖아",
    135: "말씀하신 대로네요", 136: "자, 못 참겠지", 137: "자, 이번엔, 이번엔", 138: "세게", 139: "세게",
    140: "세게 말이야", 141: "차와 술은 묽어", 142: "잘 마시네", 143: "내 술이 묽다는 게 무슨 뜻이야?",
    144: "묽게", 145: "묽게, 끈적끈적, 묽게", 146: "자, 자", 147: "네",
    148: "다마가 계속된 게 아니라, 방 두 개를 예약한 줄 알았는데요", 149: "어떻게 방 하나 더 부탁드릴 수 없을까요?",
    150: "네", 151: "아이비아인가", 152: "뭐, 떨어져서 자면 괜찮지 않을까?", 153: "응",
    154: "방 하나가 더 빈다는 얘기였어요", 155: "아, 그래", 156: "난 온천 다녀올게", 157: "꽤 마셨는데 괜찮아?",
    158: "같이 가 줄까?", 159: "혼욕", 160: "괜찮습니다",
    222: "오십", 223: "육십육", 224: "십일", 225: "십이", 226: "십이", 227: "십삼", 228: "십이",
    229: "십이", 230: "십삼", 231: "십사", 232: "십사", 233: "너", 234: "잠깐 손으로",
    235: "너", 236: "너", 237: "너", 238: "너", 239: "너", 240: "너", 241: "음", 242: "응",
    243: "오", 244: "응", 245: "윽", 246: "응", 247: "응", 248: "응", 249: "응",
    250: "정말 순종적이구나", 251: "천한 표정 지어", 252: "피스톤처럼", 253: "윽", 254: "윽",
    255: "아", 256: "윽", 257: "윽", 258: "윽", 259: "윽", 260: "윽", 261: "응, 뭐 해?",
    262: "뭐라고?", 263: "응", 264: "아하응", 265: "와, 뭐 해?", 266: "안", 267: "응", 268: "응",
    269: "응", 270: "응", 271: "응", 272: "어디야?", 273: "나한테 좀 기분 좋지?", 274: "응",
    275: "응", 276: "좋은 소리 내잖아", 277: "이렇게야?", 278: "이렇게야?", 279: "이리 와",
    280: "아파", 281: "더 세게", 282: "아아", 283: "아아아아아", 284: "아", 285: "아",
    286: "으으", 287: "으으", 288: "아", 289: "아", 290: "아", 291: "아",
    292: "착한 아이잖아. 이런 데까지 하게 만드는 거야", 335: "보지 젤이 멈추질 않아",
    336: "계속 나오고 있어, 보지 젤이", 337: "이제 큰일이야", 338: "보지 젤이 멈추질 않네",
    339: "침이 멈추질 않아", 340: "보지 젤", 341: "보지 젤", 342: "보지 젤", 343: "네", 344: "네",
    345: "아, 역시 한 명 더는 안 되나요?", 346: "알겠습니다", 347: "제가 무리한 부탁을 드려 죄송합니다",
    348: "네", 349: "네", 350: "그럼 부탁드립니다", 351: "그렇다는 거야", 352: "아이비아",
    353: "확정이네", 354: "난 만화카페에서 잘게", 355: "이런 시골에서?", 356: "보지 젤",
    357: "아, 잠깐만", 358: "그 술", 359: "나한테 줘", 360: "네 침과", 361: "또 그 술이 마시고 싶어",
    362: "자", 363: "아", 364: "아", 365: "더, 자", 366: "축하해", 367: "축하해", 368: "응",
    379: "미인의 침은 맛있군", 380: "네 침은 딱 알맞게 질척거리는구나", 381: "응", 382: "너도 목마르지?",
    383: "이거라도 물어", 384: "그건 싫어요", 385: "싫다는 게 아니야, 괜찮으니까 자",
    386: "내 성기를 침범벅으로 만들어", 387: "그래", 388: "우와", 389: "이봐", 390: "이봐", 391: "이봐",
    392: "이봐", 393: "이봐", 394: "이봐", 395: "그래", 396: "정말 네 끈적한 침을",
    397: "실린더에 침을 잔뜩 묻혀", 398: "응", 399: "하지만 들고, 들고, 자", 400: "응",
    401: "실린더에 침을 잔뜩 묻혀", 402: "오", 403: "정말 네 끈적한 침이 감겨들어",
    404: "정말 네 끈적한 침이 감겨서", 405: "단단해졌어", 406: "응", 407: "응", 408: "응",
    409: "조금은 서비스해", 410: "응", 411: "자, 흔들어", 412: "응", 413: "너도 잘하고 있어",
    414: "응", 415: "자", 416: "뭐 하는 거야", 417: "성기도 제대로 핥아, 자", 418: "오", 419: "응",
    420: "자", 421: "자", 422: "그래, 자", 423: "더 깊이 물어, 자", 424: "이렇게",
    425: "이리 와, 자", 426: "자", 427: "자", 428: "그래, 자", 429: "자", 430: "자",
    431: "이리 와, 자", 432: "자", 433: "자", 434: "자", 435: "그래, 더 깊이 물어야 해",
    436: "이렇게야, 이렇게", 437: "이리 와, 자", 438: "자", 439: "응", 440: "응", 441: "괜찮으니까, 자",
    442: "잠깐 뗄게", 443: "그래", 444: "더 깊이", 445: "응", 446: "응", 447: "그래, 자", 448: "그래",
    449: "자, 마셔 봐", 450: "빨아", 451: "자", 452: "자, 빨아", 453: "자, 끝 쪽에서도 나와",
    454: "이쪽도, 자", 487: "이 상황이 좋은 거잖아", 488: "자", 489: "자",
    490: "더 깊이 물어, 이렇게, 자", 491: "자", 492: "이렇게야, 자", 493: "자", 494: "자", 495: "자",
    496: "이렇게", 501: "더 깊이 물어", 502: "그래", 503: "오", 504: "오",
    640: "시청", 641: "시청해 주셔서 감사합니다",
}

for number in range(161, 174):
    T[number] = "시청해 주셔서 감사합니다"
for number in range(174, 222):
    T[number] = "안녕히 주무세요"
for number in range(293, 335):
    T[number] = "안녕히 주무세요"
for number in range(369, 379):
    T[number] = "나"
for number in range(455, 487):
    T[number] = "응"
for number in range(497, 501):
    T[number] = "응"
for number in range(505, 515):
    T[number] = "응"
for number in range(515, 640):
    T[number] = "시청해 주셔서 감사합니다"

ASR_BLOCKS = set(range(161, 174))
UNCERTAIN_BLOCKS = {
    15, 23, 35, 54, 68, 81, 92, 103, 105, 108, 114, 116, 123, 124, 141,
    148, 151, 154, 222, 234, 241, 250, 251, 252, 264, 266, 279, 280, 281,
    292, 335, 336, 338, 340, 341, 342, 352, 356, 379, 380, 383, 386, 397,
    401, 405, 411, 413, 417, 423, 435, 449, 453, 487, 490, 501,
}
JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uff66-\uff9f]")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def main() -> None:
    blocks, _, _ = parse_srt(SOURCE)
    expected = set(range(1, len(blocks) + 1))
    if len(blocks) != 641 or set(T) != expected:
        raise SystemExit(f"translation map mismatch: missing={sorted(expected-set(T))}, extra={sorted(set(T)-expected)}")
    if any(not text.strip() or JAPANESE.search(text) for text in T.values()):
        raise SystemExit("translation map has empty text or Japanese residue")

    decisions: list[dict] = []
    for block in blocks:
        n = block.number
        asr_assisted = n in ASR_BLOCKS
        uncertain = n in UNCERTAIN_BLOCKS
        decisions.append({
            "block_number": n,
            "source_japanese": block.text,
            "previous_korean": "",
            "source_faithful_korean": T[n],
            "viewer_natural_korean": T[n],
            "translation_method": "semantic_review_with_asr" if asr_assisted else "semantic_review_from_japanese",
            "translation_model": DEFAULT_TRANSLATION_MODEL,
            "status": "translated",
            "confidence": "medium" if asr_assisted or uncertain else "high",
            "uncertain_slots": ["source_transcription_damaged"] if asr_assisted or uncertain else [],
            "evidence_refs": (["japanese_srt", "local_asr_exact_window"] if asr_assisted else ["japanese_srt", "neighboring_context"]),
            "review_note": (
                "일본어 자막 텍스트가 파손되어 정확한 SRT 시간창의 로컬 faster-whisper ASR만 보조로 사용했습니다. 사람 음성 검수는 하지 않았습니다."
                if asr_assisted else
                "일본어 자막의 전사가 일부 불명확하거나 축약되어 인접한 일본어 문맥으로 보수적으로 옮겼습니다."
                if uncertain else ""
            ),
            "provenance": "gpt56_direct_reasoning_only",
        })

    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints: list[str] = []
    for completed in (100, 200, 300, 400, 500, 600, len(decisions)):
        path = OUT / f"{STEM}.checkpoint-{completed:03d}.decisions.jsonl"
        write_jsonl(path, decisions[:completed])
        checkpoints.append(str(path))
    decisions_path = OUT / f"{STEM}.decisions.jsonl"
    write_jsonl(decisions_path, decisions)

    source_out = OUT / f"{STEM}.source-faithful.srt"
    viewer_out = OUT / f"{STEM}.viewer-natural.srt"
    apply_report_path = OUT / f"{STEM}.apply-report.json"
    apply_report = apply_translation_decisions(SOURCE, decisions_path, source_out, viewer_out, apply_report_path, strict=True)
    decision_validation = validate_translation_decisions(SOURCE, decisions_path, strict=True)
    pair_validation = validate_pair(SOURCE, source_out, viewer_out, project_root=ROOT)
    validation_path = OUT / f"{STEM}.validation.json"
    write_validation_report(validation_path, pair_validation)
    failure = apply_report.get("status") != "text-crosschecked" or decision_validation.get("status") != "pass" or pair_validation.get("status") == "fail"
    report = {
        "title": TITLE,
        "status": "complete-gpt56-direct-draft" if not failure else "failed",
        "translation_scope": "Direct Japanese-to-Korean GPT-5.6 reasoning only; no former Korean subtitle, external MT API, or local MT engine was read or used.",
        "human_audio_verification": False,
        "blocks": len(blocks),
        "source": str(SOURCE),
        "decisions": str(decisions_path),
        "source_faithful": str(source_out),
        "viewer_natural": str(viewer_out),
        "checkpoints": checkpoints,
        "asr_assisted_blocks": sorted(ASR_BLOCKS),
        "source_uncertainty_blocks": sorted(UNCERTAIN_BLOCKS),
        "local_asr_note": "Blocks 161-173 used only exact SRT-window local faster-whisper transcription; it returned the repeated ending phrase. No human audio verification was performed.",
        "decision_validation": decision_validation,
        "pair_validation": pair_validation,
        "validation_artifact": str(validation_path),
        "final_promotion_allowed": False,
        "promotion_note": "Text-crosschecked direct draft only. It is not human-verified or promoted to final.",
    }
    (OUT / f"{STEM}.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
