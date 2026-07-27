"""Generate SSIS-652 GPT-5.6 direct Japanese-to-Korean draft artifacts.

This deliberately uses only the Japanese SRT plus the translations encoded
below by the reviewing model.  It does not read a prior Korean subtitle,
machine-translation output, or an external service.
"""

from __future__ import annotations

import json
from pathlib import Path

from translation_forensics.semantic_translation import (
    apply_translation_decisions,
    validate_translation_decisions,
)
from translation_forensics.srt import parse_srt
from translation_forensics.translation_model import DEFAULT_TRANSLATION_MODEL
from translation_forensics.validation import validate_pair


ROOT = Path(__file__).resolve().parents[1]
TITLE = "SSIS-652"
SOURCE = ROOT.parent / TITLE / f"{TITLE}.ja.srt"
INTERMEDIATE = ROOT / "workspaces" / TITLE / "intermediate"
STEM = f"{TITLE}.gpt56-direct-v1"

# Literal/direct translations decided from the Japanese source and its local
# neighboring subtitle context.  The two Japanese strings that are visibly
# corrupted in the source but have unambiguous local ASR support are marked in
# ASR_SUPPORTED_SOURCES below; no audio was listened to by a human.
TRANSLATIONS = {
    "お疲れ様です": "수고하셨습니다.",
    "初めまして": "처음 뵙겠습니다.",
    "お名前いいですか": "성함 여쭤봐도 될까요?",
    "あ、変えてくるわ": "아, 갈아입고 올게요.",
    "M-Oくんを攻めてたんですけど": "M남을 리드했는데요.",
    "すごい気持ちやさそうにしてくれてたし": "정말 기분 좋아 보이기도 했고요.",
    "射精も何回もさせられたので": "사정도 여러 번 하게 했으니까요.",
    "良かった": "좋았어요.",
    "今日の企画内容は": "오늘 기획 내용은",
    "最近S-1のお泊り物": "요즘 S-1의 숙박물은",
    "すごい厳しい": "정말 엄격해요.",
    "攻めるんじゃなくて": "일방적으로 리드하는 게 아니라",
    "相手の反応も見ながら": "상대 반응도 보면서",
    "攻めれたと思う": "리드할 수 있었다고 생각해요.",
    "気を付けてました": "신경 썼어요.",
    "時間をかける": "시간을 들이고,",
    "強弱をつける": "강약을 조절하고,",
    "相手が気持ちよく": "상대가 기분 좋게",
    "気を付けてくれないと意味がないなって思ってるので": "배려하지 않으면 의미가 없다고 생각해서요.",
    "ちょっと会話しつつ": "대화도 조금 하면서",
    "攻めることがこだわりです": "리드하는 게 제 고집이에요.",
    "水族館に行って": "수족관에 가서",
    "チューとかしたんですけど": "키스도 했는데요.",
    "もうちょっとしたかったんです": "조금 더 하고 싶었어요.",
    "いっぱいさせるさせた後に": "많이 하게 한 뒤에",
    "フォローバー行って": "팔로우 바에 가서",
    "二人で体洗ったんですけど": "둘이서 몸을 씻었는데요.",
    "本人はもう無理って言ってたんですけど": "본인은 이제 더는 못 하겠다고 했는데요.",
    "その後も出してくれたので": "그 뒤에도 해줬으니까요.",
    "それは成功したかなと": "그건 성공한 것 같아요.",
    "なるほど": "그렇군요.",
    "最近はガンマン浴遊とか": "요즘은 암반욕 같은 거요.",
    "サウナに行くことです": "사우나에 가는 거예요.",
    "野菜とフルーツです": "채소와 과일이에요.",
    "誰の食べ物はないです": "싫어하는 음식은 없어요.",
    "このタイプは優しくて": "좋아하는 타입은 다정하고",
    "面白くて話を聞いてくれる人です": "재미있고 제 얘기를 들어주는 사람이에요.",
    "高校2年生": "고등학교 2학년 때요.",
    "どこでしたんですか?": "어디서 했어요?",
    "彼氏の": "남자친구의",
    "当時の彼氏のお家です": "당시 남자친구 집에서요.",
    "友達のカップルがいるそばでした": "친구 커플이 있는 옆에서였어요.",
    "自分": "저요.",
    "今の仕事を長く続けてられるように": "지금 일을 오래 계속할 수 있도록요.",
    "笑ってたら嘘やん": "웃고 있으면 거짓말 같잖아요.",
    "嘘ではないです": "거짓말 아니에요.",
    "なんかいっぱいあります": "여러 가지가 있어요.",
    "まず買ってくれてありがとうございます": "우선 사 주셔서 감사합니다.",
    "攻めの作品": "리드하는 작품이에요.",
    "他にも撮ってるんですけど": "다른 것도 찍었는데요.",
    "スム入ってる作品なので": "S적인 면이 들어간 작품이라서요.",
    "私の今まで撮ってきた中では": "제가 지금까지 찍어 온 것 중에서는",
    "珍しいかな": "드문 편일 거예요.",
    "嬉しいですね": "기쁘네요.",
    "そういうのも見てほしいですし": "그런 것도 봐 주셨으면 하고요.",
    "見てる人も私に攻められてる": "보는 분들도 저에게 리드당하는",
    "気分になってくれたら嬉しいです": "기분이 들어 주시면 좋겠어요.",
    "ありがとうございました": "감사합니다.",
    "本当今日は": "정말 오늘은",
    "その辺長いこと": "그쪽도 오래도록",
    "これからも頑張りましょう": "앞으로도 힘내 봐요.",
    "はい": "네.",
    "頑張ります": "열심히 할게요.",
    "たぶきありがとう": "타부키, 고마워.",
    "相沢です": "아이자와입니다.",
    "今日楓": "오늘은 카에데",
    "フワさんにお会いするんですが": "후와 씨를 만나는데요.",
    "会ったことないんです": "만난 적은 없어요.",
    "すごい緊張しますね": "정말 긴장되네요.",
    "どんな人なんだろうなって": "어떤 분일까 싶어서요.",
    "この辺で待ち合わせしてるんです": "이 근처에서 만나기로 했어요.",
    "ありがとうございます": "감사합니다.",
    "すいません": "죄송합니다.",
    "お待たせしてないんです": "기다리게 하진 않았죠?",
    "あ、全然全然": "아, 전혀요.",
    "大丈夫です": "괜찮아요.",
    "お待たせしてるんです": "기다리게 했네요.",
    "今日は": "오늘은",
    "企画聞いてますか?": "기획 들으셨어요?",
    "外で一回": "밖에서 한 번",
    "映して": "찍고,",
    "ただ": "그냥",
    "ごまびあげるよ": "보상 줄게요.",
    "みたいな": "같은",
    "感じで": "느낌으로요.",
    "よろしくお願いします": "잘 부탁드립니다.",
    "うん": "응.",
    "できたくさん見たりしてるんです": "꽤 많이 보고 있어요.",
    "ああ": "아아.",
    "そうなんです": "그래요.",
    "前4年ぶりから": "전에는 4년 만이어서요.",
    "あ、このぐらい?": "아, 이 정도요?",
    "じゃあ楽しみですね": "그럼 기대되네요.",
    "楽しみです": "기대돼요.",
    "じゃあいい": "그럼 됐어요.",
    "どうも": "고마워요.",
    "改めて見るとすごいスタイル": "다시 보니 정말 몸매가 좋네요.",
    "めっちゃスタイルいいですね": "정말 몸매가 좋으시네요.",
    "すぐ良いですね": "정말 좋네요.",
    "足長いです": "다리가 길어요.",
    "長い?": "길어요?",
    "この建物にあるんですか?": "이 건물에 있는 건가요?",
    "そうですね": "그렇네요.",
    "あっち側にエレベーターがあって": "저쪽에 엘리베이터가 있어서",
    "で、上まで行って": "위까지 올라가서",
    "気づく感じ": "알게 되는 느낌이에요.",
    "ペンギンが一緒": "펭귄이 같이",
    "ペンギンが一緒に": "펭귄이 함께",
    "可愛くないです": "귀엽지 않아요?",
    "ペンギン可愛い": "펭귄 귀여워요.",
    "歩き方可愛い": "걷는 모습이 귀여워요.",
    "どうもが始まります": "이제 시작합니다.",
    "ちょっと歩き": "조금 걸어서",
    "一回": "한 번",
    "このプラーは": "이 플랜은",
    "前回に連絡しております": "미리 연락드렸습니다.",
    "よりもがいは": "그것보다 더는",
    "もうちょっと": "조금 더",
    "びっくりした": "깜짝 놀랐어요.",
    "ご視聴ありがとうございました": "시청해 주셔서 감사합니다.",
    "ご視聴": "시청",
}

ASR_SUPPORTED_SOURCES = {"最近はガンマン浴遊とか", "誰の食べ物はないです", "お待たせしてないんです"}
AMBIGUOUS_SOURCES = {
    "最近S-1のお泊り物",
    "フォローバー行って",
    "スム入ってる作品なので",
    "その辺長いこと",
    "たぶきありがとう",
    "今日楓",
    "ごまびあげるよ",
    "できたくさん見たりしてるんです",
    "前4年ぶりから",
    "気づく感じ",
    "どうもが始まります",
    "よりもがいは",
}


def main() -> None:
    blocks, _, _ = parse_srt(SOURCE)
    missing = sorted({block.text for block in blocks if block.text not in TRANSLATIONS})
    if missing:
        raise SystemExit(f"unmapped Japanese source text: {missing!r}")

    decisions = []
    for block in blocks:
        source = block.text
        method = "semantic_review_with_asr" if source in ASR_SUPPORTED_SOURCES else "semantic_review_from_japanese"
        note = ""
        uncertainty: list[str] = []
        if source in ASR_SUPPORTED_SOURCES:
            note = "원문 표기가 훼손되어 기존 로컬 ASR 후보와 앞뒤 일본어 문맥을 사용함. 사람 청취 검증은 하지 않음."
            uncertainty = ["human_audio_verification_not_performed"]
        elif source in AMBIGUOUS_SOURCES:
            note = "원문 표기가 불완전하거나 고유명사/축약어가 불명확함. 일본어 표기와 문맥만으로 보수적으로 번역함."
            uncertainty = ["source_wording_ambiguous"]
        decisions.append(
            {
                "block_number": block.number,
                "source_japanese": source,
                "source_faithful_korean": TRANSLATIONS[source],
                "viewer_natural_korean": TRANSLATIONS[source],
                "translation_method": method,
                "translation_model": DEFAULT_TRANSLATION_MODEL,
                "status": "translated",
                "confidence": "medium",
                "uncertain_slots": uncertainty,
                "evidence_refs": ["japanese_srt", "neighboring_context"] + (["local_asr_existing"] if source in ASR_SUPPORTED_SOURCES else []),
                "review_note": note,
                "provenance": "gpt56_direct_reasoning_only",
            }
        )

    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = []
    for completed in (76, 152, 228, len(decisions)):
        checkpoint = INTERMEDIATE / f"{STEM}.checkpoint-{completed:03d}.decisions.jsonl"
        with checkpoint.open("w", encoding="utf-8", newline="\n") as handle:
            for record in decisions[:completed]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        checkpoint_paths.append(str(checkpoint))
    decisions_path = INTERMEDIATE / f"{STEM}.decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in decisions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    source_output = INTERMEDIATE / f"{STEM}.source-faithful.srt"
    viewer_output = INTERMEDIATE / f"{STEM}.viewer-natural.srt"
    apply_report = INTERMEDIATE / f"{STEM}.apply-report.json"
    applied = apply_translation_decisions(SOURCE, decisions_path, source_output, viewer_output, apply_report, strict=True)
    decision_validation = validate_translation_decisions(SOURCE, decisions_path, strict=True)
    pair_validation = validate_pair(SOURCE, source_output, viewer_output)
    report = {
        "title": TITLE,
        "status": "complete-gpt56-direct-draft" if applied["status"] == "text-crosschecked" and decision_validation["status"] == "pass" and not pair_validation.get("errors") else "failed",
        "translation_scope": "direct Japanese-to-Korean reasoning; no prior Korean subtitles or external/local MT used",
        "human_audio_verification": False,
        "source": str(SOURCE),
        "decisions": str(decisions_path),
        "source_faithful": str(source_output),
        "viewer_natural": str(viewer_output),
        "blocks": len(blocks),
        "decision_checkpoints": checkpoint_paths,
        "asr_supported_blocks": [record["block_number"] for record in decisions if record["translation_method"] == "semantic_review_with_asr"],
        "ambiguous_source_blocks": [record["block_number"] for record in decisions if record["uncertain_slots"] and record["translation_method"] != "semantic_review_with_asr"],
        "decision_validation": decision_validation,
        "pair_validation": pair_validation,
    }
    (INTERMEDIATE / f"{STEM}.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
