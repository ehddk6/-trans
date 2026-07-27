from __future__ import annotations

import json
import platform
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .alignment import align_by_overlap
from .asr_evidence import normalize_japanese, text_similarity
from .manifest import sha256_file
from .srt import JAPANESE_RE, SubtitleBlock, compare_structure, parse_srt, write_srt


CLOSED_WORLD_STAGE = "closed-world-validated"
ENGINE_VERSION = "1"
ABSTAIN_MARKER = "[미확정]"

_PLACEHOLDER_RE = re.compile(r"\[(?:번역 필요|불명|미확정|검토 필요)\]|번역 불가|알아들을 수 없")
_HANGUL_RE = re.compile(r"[가-힣]")
_JA_QUESTION_RE = re.compile(r"[?？]|(?:か|かな|の)\s*[。.!！]?$")
_KO_QUESTION_RE = re.compile(r"[?？]|(?:습니까|나요|인가요|일까요|할까|했니|하니)\s*[.!。]?$")
_JA_NEGATION_RE = re.compile(r"(?:ない|なかった|ません|ぬ|ず|じゃない|ではない|だめ|駄目|嫌|いや|やめ)")
_KO_NEGATION_RE = re.compile(r"(?:^|[\s,])(안|못)\s|않|없|말[아지]|아니|싫|그만|금지")
_JA_REQUEST_RE = re.compile(r"(?:ください|下さい|くれない|てくれ|てもら|なさい|お願い)")
_KO_REQUEST_RE = re.compile(r"(?:주세요|해\s*줘|줘[.!?]?$|하세요|해라|부탁|줄래|주겠)")
_JA_PREDICATE_END_RE = re.compile(r"(?:る|う|く|す|つ|ぬ|む|ぶ|た|て|ない|たい|ます|です|だ|ね|よ|ろ)\s*[。.!！?？]?$")
_JA_EXPLICIT_PARTICIPANT_RE = re.compile(
    r"(?:私|僕|俺|あなた|君|彼|彼女|母|父|姉|妹|兄|弟|先生|店長|社長|優君|"
    r"[\u3040-\u30ff\u3400-\u9fff]{1,10}(?:は|が|を|に|へ|から|と))"
)
_AMBIGUITY_EXEMPT_RE = re.compile(r"^(?:ありがとう|ありがとうございます|ごめん|ごめんなさい|はい|いいえ|うん|そう|大丈夫|おはよう|こんにちは|こんばんは)[。.!！]?$")

_INTENSITY_PAIRS = {
    "もっと": ("더", "좀 더"),
    "すごく": ("정말", "아주", "굉장히"),
    "強く": ("세게", "강하게"),
    "激しく": ("격하게", "거칠게"),
    "ゆっくり": ("천천히",),
    "早く": ("빨리", "어서"),
    "優しく": ("부드럽게", "다정하게"),
}
_KO_INTENSITY_WORDS = tuple(sorted({word for words in _INTENSITY_PAIRS.values() for word in words}, key=len, reverse=True))
_PARTICIPANT_GROUPS = (
    (("私", "僕", "俺"), ("나", "저", "내가", "제가")),
    (("あなた", "君", "お前"), ("너", "당신", "네가", "자기")),
    (("彼女",), ("그녀", "여자친구")),
    (("彼",), ("그는", "그가", "그를", "남자친구")),
    (("母", "お母さん", "母親"), ("엄마", "어머니")),
    (("父", "お父さん", "父親"), ("아빠", "아버지")),
    (("姉",), ("누나", "언니", "누이")),
    (("兄",), ("형", "오빠", "오라버니")),
    (("妹",), ("여동생",)),
    (("弟",), ("남동생",)),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _canonical_jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _write_idempotent(path: Path, text: str) -> None:
    data = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"기존 폐쇄형 산출물과 새 결과가 달라 덮어쓰지 않습니다: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_srt_idempotent(path: Path, blocks: list[SubtitleBlock]) -> None:
    if path.exists():
        temporary = path.with_name(path.name + ".closed-world-check.tmp")
        try:
            write_srt(temporary, blocks)
            if path.read_bytes() != temporary.read_bytes():
                raise FileExistsError(f"기존 폐쇄형 SRT와 새 결과가 달라 덮어쓰지 않습니다: {path}")
        finally:
            temporary.unlink(missing_ok=True)
        return
    write_srt(path, blocks)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} JSONL 레코드는 객체여야 합니다.")
        rows.append(value)
    return rows


def _normalize_korean(text: str) -> str:
    return re.sub(r"[\s.,!?！？。、…~～\"'“”‘’\-]+", "", text or "").lower()


def _korean_similarity(left: str, right: str) -> float:
    a, b = _normalize_korean(left), _normalize_korean(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _candidate_family(path: Path, *, previous_path: Path | None) -> str:
    if previous_path and path.resolve() == previous_path.resolve():
        return "previous-korean"
    name = path.stem.lower()
    if "ko-aligned-draft" in name or "previous" in name:
        return "previous-korean"
    for suffix in (
        ".source-faithful",
        ".viewer-natural",
        ".source-faithful-ko",
        ".viewer-natural-ko",
        "-source-faithful",
        "-viewer-natural",
    ):
        name = name.replace(suffix, "")
    name = re.sub(r"\.(?:srt|preview)$", "", name)
    return re.sub(r"[^a-z0-9._-]+", "-", name).strip("-") or "local-candidate"


def _candidate_role(path: Path) -> str:
    name = path.name.lower()
    if "source-faithful" in name:
        return "source-faithful"
    if "viewer-natural" in name:
        return "viewer-natural"
    if "aligned-draft" in name or "previous" in name:
        return "previous"
    return "candidate"


def _aligned_text_map(reference: list[SubtitleBlock], candidate: list[SubtitleBlock]) -> dict[int, tuple[str, str]]:
    by_number = {block.number: block for block in candidate}
    result: dict[int, tuple[str, str]] = {}
    for alignment in align_by_overlap(reference, candidate):
        if alignment.candidate_number is None or alignment.candidate_number not in by_number:
            continue
        result[alignment.source_number] = (by_number[alignment.candidate_number].text, alignment.confidence)
    return result


def _semantic_slots(japanese: str) -> dict[str, Any]:
    normalized = normalize_japanese(japanese)
    interrogative = bool(_JA_QUESTION_RE.search(japanese.strip()))
    negative = bool(_JA_NEGATION_RE.search(japanese))
    request = bool(_JA_REQUEST_RE.search(japanese))
    explicit_participant = bool(_JA_EXPLICIT_PARTICIPANT_RE.search(japanese))
    location = bool(re.search(r"(?:で|へ|から|まで)", japanese))
    if re.search(r"(?:た|だった|でした|ていた)\s*[。.!！?？]?$", japanese):
        temporal = "past"
    elif re.search(r"(?:つもり|予定|だろう|でしょう|なる|ます)\s*[。.!！?？]?$", japanese):
        temporal = "future-or-nonpast"
    else:
        temporal = "unknown"
    intensity = [token for token in _INTENSITY_PAIRS if token in japanese]
    return {
        "interrogative": {"value": interrogative, "state": "surface-confirmed"},
        "polarity": {"value": "negative" if negative else "positive", "state": "surface-confirmed"},
        "request_strength": {"value": "request" if request else "none-observed", "state": "surface-confirmed"},
        "speaker": {"value": None, "state": "unknown"},
        "actor": {"value": "explicit" if explicit_participant else None, "state": "surface-confirmed" if explicit_participant else "unknown"},
        "target": {"value": "explicit" if re.search(r"(?:を|に|へ|から)", japanese) else None, "state": "surface-confirmed" if re.search(r"(?:を|に|へ|から)", japanese) else "unknown"},
        "location": {"value": "explicit" if location else None, "state": "surface-confirmed" if location else "unknown"},
        "temporal_state": {"value": temporal, "state": "surface-confirmed" if temporal != "unknown" else "unknown"},
        "intensity": {"value": intensity, "state": "surface-confirmed" if intensity else "null"},
        "action": {"value": "predicate-observed" if _JA_PREDICATE_END_RE.search(japanese.strip()) else None, "state": "surface-confirmed" if _JA_PREDICATE_END_RE.search(japanese.strip()) else "unknown"},
    }


def _is_ambiguous_omission(japanese: str) -> bool:
    normalized = normalize_japanese(japanese)
    if not normalized or _AMBIGUITY_EXEMPT_RE.fullmatch(japanese.strip()):
        return False
    if len(normalized) > 18:
        return False
    return bool(_JA_PREDICATE_END_RE.search(japanese.strip())) and not bool(_JA_EXPLICIT_PARTICIPANT_RE.search(japanese))


def _contains_korean_term(text: str, term: str) -> bool:
    if len(term) > 1:
        return term in text
    return bool(re.search(rf"(?:^|[\s,]){re.escape(term)}(?:는|가|를|에게|한테|도|와|랑|$|[\s,.!?])", text))


def _candidate_text_errors(japanese: str, korean: str) -> list[str]:
    errors: list[str] = []
    korean = korean.strip()
    if not korean:
        return ["empty-candidate"]
    if JAPANESE_RE.search(korean):
        errors.append("japanese-remains")
    if _PLACEHOLDER_RE.search(korean):
        errors.append("work-marker")
    if not _HANGUL_RE.search(korean) and korean not in {"…", "...", "♪", "♪♪"}:
        errors.append("no-korean-text")
    if normalize_japanese(japanese) and korean in {"…", "...", "♪", "♪♪"}:
        errors.append("content-omitted")

    ja_question = bool(_JA_QUESTION_RE.search(japanese.strip()))
    ko_question = bool(_KO_QUESTION_RE.search(korean.strip()))
    if ja_question != ko_question:
        errors.append("question-flip")

    ja_negative = bool(_JA_NEGATION_RE.search(japanese))
    ko_negative = bool(_KO_NEGATION_RE.search(korean))
    if ja_negative != ko_negative:
        errors.append("polarity-flip")

    if _JA_REQUEST_RE.search(japanese) and not _KO_REQUEST_RE.search(korean):
        errors.append("request-omitted")

    source_intensity = {ja: ko for ja, ko in _INTENSITY_PAIRS.items() if ja in japanese}
    for ja_word, ko_words in source_intensity.items():
        if not any(word in korean for word in ko_words):
            errors.append(f"intensity-omitted:{ja_word}")
    if not source_intensity and any(word in korean for word in _KO_INTENSITY_WORDS):
        errors.append("intensity-added")

    for japanese_words, korean_words in _PARTICIPANT_GROUPS:
        matched = next((word for word in korean_words if _contains_korean_term(korean, word)), None)
        if matched and not any(word in japanese for word in japanese_words):
            errors.append(f"participant-added:{matched}")
    return sorted(set(errors))


def _review_context(path: Path | None) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    if not path or not path.exists():
        return {}, []
    by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    scenes: list[dict[str, Any]] = []
    for value in _read_jsonl(path):
        blocks = [row for row in value.get("blocks", []) if isinstance(row, dict)]
        asr_rows = [row for row in value.get("asr_candidates", []) if isinstance(row, dict)]
        scene = {
            "scene_id": str(value.get("scene_id", "")),
            "context_japanese": str(value.get("context_japanese", "")),
            "block_numbers": [int(row["block_number"]) for row in blocks if str(row.get("block_number", "")).isdigit()],
            "asr_candidates": asr_rows,
        }
        scenes.append(scene)
        for number in scene["block_numbers"]:
            for row in asr_rows:
                enriched = dict(row)
                enriched["source_family"] = str(row.get("source_family") or "whisper-family")
                enriched["scope_block_count"] = len(scene["block_numbers"])
                enriched["scene_id"] = scene["scene_id"]
                by_block[number].append(enriched)
    return dict(by_block), scenes


def _asr_summary(japanese: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[str(row.get("source_family") or "whisper-family")].append(row)
    family_records: list[dict[str, Any]] = []
    conflicts: list[str] = []
    for family, family_rows in sorted(families.items()):
        unique_texts = sorted({str(row.get("text", "")).strip() for row in family_rows if str(row.get("text", "")).strip()})
        comparable = [row for row in family_rows if int(row.get("scope_block_count", 1)) == 1 and str(row.get("text", "")).strip()]
        similarities = [text_similarity(japanese, str(row.get("text", ""))) for row in comparable]
        if similarities and max(similarities) < 0.70:
            conflicts.append(f"asr-srt-conflict:{family}")
        family_records.append({
            "source_family": family,
            "profiles": sorted({str(row.get("profile", "")) for row in family_rows}),
            "unique_texts": unique_texts,
            "independent_evidence_count": 1,
            "best_srt_similarity": round(max(similarities), 6) if similarities else None,
        })
    return {
        "families": family_records,
        "independent_family_count": len(family_records),
        "same_family_duplicates_collapsed": max(0, len(rows) - len(family_records)),
        "conflicts": sorted(set(conflicts)),
    }


def _machine_timeline(
    scenes: list[dict[str, Any]],
    *,
    block_count: int,
    supplied_timeline: dict[str, Any] | None,
) -> dict[str, Any]:
    anchors: list[dict[str, Any]] = []
    for scene in scenes:
        numbers = scene.get("block_numbers") or []
        source = str(scene.get("context_japanese", ""))
        rows = [row for row in scene.get("asr_candidates", []) if str(row.get("profile", "")) in {"original_unbiased", "dialogue_unbiased"}]
        texts = [str(row.get("text", "")).strip() for row in rows if str(row.get("text", "")).strip()]
        if not numbers or len(texts) < 2:
            continue
        if text_similarity(texts[0], texts[1]) < 0.75:
            continue
        similarity = max(text_similarity(source, text) for text in texts)
        if similarity < 0.75:
            continue
        block_number = round(sum(numbers) / len(numbers))
        anchors.append({
            "scene_id": scene.get("scene_id"),
            "block_number": block_number,
            "relative_position": round(block_number / max(1, block_count), 6),
            "srt_asr_similarity": round(similarity, 6),
            "source_family": "whisper-family",
        })
    bands = {
        "early": [row for row in anchors if row["relative_position"] <= 0.34],
        "middle": [row for row in anchors if 0.34 < row["relative_position"] < 0.67],
        "late": [row for row in anchors if row["relative_position"] >= 0.67],
    }
    supplied_status = str((supplied_timeline or {}).get("status", "missing"))
    full_media_coverage = supplied_status != "media-too-short"
    has_three_bands = all(bands.values())
    if has_three_bands and full_media_coverage:
        status = "machine-aligned"
    elif anchors:
        status = "machine-aligned-partial"
    else:
        status = "insufficient-machine-anchors"
    return {
        "schema_name": "translation-forensics/machine-timeline",
        "schema_version": "1",
        "status": status,
        "supplied_timeline_status": supplied_status,
        "full_media_coverage": full_media_coverage,
        "machine_alignment_is_human_verification": False,
        "anchors": anchors,
        "anchor_bands": {name: len(rows) for name, rows in bands.items()},
        "note": "ASR 기반 정렬은 사람 직접 청취 또는 승인된 오프셋 맵을 대체하지 않습니다.",
    }


def _candidate_inventory(
    reference: list[SubtitleBlock],
    candidate_paths: list[Path],
    *,
    previous_path: Path | None,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    inventory: list[dict[str, Any]] = []
    by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_paths: set[Path] = set()
    for path in sorted((value.resolve() for value in candidate_paths), key=lambda value: str(value).lower()):
        if path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)
        blocks, encoding, newline = parse_srt(path)
        family = _candidate_family(path, previous_path=previous_path)
        role = _candidate_role(path)
        aligned = _aligned_text_map(reference, blocks)
        inventory.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "family": family,
            "role": role,
            "encoding": encoding,
            "newline": newline,
            "blocks": len(blocks),
        })
        for number, (text, confidence) in aligned.items():
            by_block[number].append({
                "path": str(path),
                "family": family,
                "role": role,
                "text": text,
                "alignment_confidence": confidence,
            })
    return inventory, dict(by_block)


def _decide_block(
    block: SubtitleBlock,
    japanese: str,
    candidates: list[dict[str, Any]],
    asr_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    slots = _semantic_slots(japanese)
    asr = _asr_summary(japanese, asr_rows)
    reasons: list[str] = []
    if not japanese.strip():
        reasons.append("missing-japanese-source")
    if _is_ambiguous_omission(japanese):
        reasons.append("ambiguous-omitted-participant")
    reasons.extend(asr["conflicts"])

    evaluated: list[dict[str, Any]] = []
    valid_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        errors = _candidate_text_errors(japanese, str(candidate.get("text", "")))
        row = {**candidate, "validation_errors": errors, "candidate_status": "rejected" if errors else "eligible"}
        evaluated.append(row)
        if not errors:
            valid_by_family[str(candidate["family"])].append(row)

    representatives: dict[str, dict[str, Any]] = {}
    for family, rows in valid_by_family.items():
        ordered = sorted(rows, key=lambda row: (row["role"] != "source-faithful", row["role"] == "previous", row["path"]))
        representatives[family] = ordered[0]
    generated_families = sorted(family for family in representatives if family != "previous-korean")
    if not generated_families:
        reasons.append("no-generated-korean-candidate")
    if len(representatives) < 2:
        reasons.append("insufficient-candidate-families")

    representative_rows = [representatives[family] for family in sorted(representatives)]
    pair_similarities: list[float] = []
    for index, left in enumerate(representative_rows):
        for right in representative_rows[index + 1:]:
            pair_similarities.append(_korean_similarity(str(left["text"]), str(right["text"])))
    if pair_similarities and min(pair_similarities) < 0.72:
        reasons.append("candidate-family-disagreement")

    source_row: dict[str, Any] | None = None
    viewer_row: dict[str, Any] | None = None
    if generated_families:
        chosen_family = generated_families[0]
        family_rows = valid_by_family[chosen_family]
        source_row = next((row for row in family_rows if row["role"] == "source-faithful"), None) or family_rows[0]
        viewer_row = next((row for row in family_rows if row["role"] == "viewer-natural"), None) or source_row
    else:
        chosen_family = None

    status = "accepted" if not reasons and source_row and viewer_row else "abstained"
    decision = {
        "schema_name": "translation-forensics/closed-world-decision",
        "schema_version": "1",
        "block_number": block.number,
        "timecode": f"{block.start} --> {block.end}",
        "source_japanese": japanese,
        "status": status,
        "source_faithful_korean": str(source_row["text"]).strip() if status == "accepted" and source_row else "",
        "viewer_natural_korean": str(viewer_row["text"]).strip() if status == "accepted" and viewer_row else "",
        "selected_generated_family": chosen_family if status == "accepted" else None,
        "candidate_family_count": len(representatives),
        "candidate_families": sorted(representatives),
        "asr_independent_family_count": asr["independent_family_count"],
        "abstention_reasons": sorted(set(reasons)) if status == "abstained" else [],
        "human_reviewed": False,
        "direct_human_listening": False,
        "final_promotion_allowed": False,
    }
    observation = {
        "schema_name": "translation-forensics/closed-world-observations",
        "schema_version": "1",
        "block_number": block.number,
        "timecode": decision["timecode"],
        "source_japanese": japanese,
        "candidate_translations": evaluated,
        "asr": asr,
        "human_evidence_present": False,
    }
    lattice = {
        "schema_name": "translation-forensics/semantic-lattice",
        "schema_version": "1",
        "block_number": block.number,
        "slots": slots,
        "alternatives": ["participant-omitted"] if _is_ambiguous_omission(japanese) else [],
        "conflicts": sorted(set([*asr["conflicts"], *(["candidate-family-disagreement"] if "candidate-family-disagreement" in reasons else [])])),
        "resolution": status,
    }
    return decision, observation, lattice


def _core_validation(
    structure_path: Path,
    package_dir: Path,
    *,
    require_proof: bool,
    require_manifest: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    reference, _, _ = parse_srt(structure_path)
    decisions_path = package_dir / "decisions.jsonl"
    observations_path = package_dir / "observations.jsonl"
    lattice_path = package_dir / "semantic-lattice.jsonl"
    source_preview_path = package_dir / "source-faithful.preview.srt"
    viewer_preview_path = package_dir / "viewer-natural.preview.srt"
    unresolved_path = package_dir / "unresolved.jsonl"
    required = [decisions_path, observations_path, lattice_path, source_preview_path, viewer_preview_path, unresolved_path]
    if require_proof:
        required.append(package_dir / "proof.json")
    if require_manifest:
        required.append(package_dir / "run-manifest.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {"status": "fail", "errors": [f"필수 산출물 누락: {path}" for path in missing]}

    decisions = _read_jsonl(decisions_path)
    observations = _read_jsonl(observations_path)
    lattices = _read_jsonl(lattice_path)
    unresolved = _read_jsonl(unresolved_path)
    by_number: dict[int, dict[str, Any]] = {}
    for row in decisions:
        try:
            number = int(row.get("block_number"))
        except (TypeError, ValueError):
            errors.append(f"잘못된 block_number: {row.get('block_number')!r}")
            continue
        if number in by_number:
            errors.append(f"중복 decision block {number}")
        by_number[number] = row
        status = row.get("status")
        if status not in {"accepted", "abstained"}:
            errors.append(f"block {number}: status는 accepted 또는 abstained여야 합니다.")
        if row.get("human_reviewed") is not False or row.get("direct_human_listening") is not False:
            errors.append(f"block {number}: 사람 증거를 자동 생성할 수 없습니다.")
        if row.get("final_promotion_allowed") is not False:
            errors.append(f"block {number}: 폐쇄형 결정은 final 승격을 허용할 수 없습니다.")
        if status == "accepted":
            source = str(row.get("source_faithful_korean", ""))
            viewer = str(row.get("viewer_natural_korean", ""))
            if not source or not viewer:
                errors.append(f"block {number}: accepted 번역문이 비었습니다.")
            for issue in _candidate_text_errors(str(row.get("source_japanese", "")), source):
                errors.append(f"block {number}: source 후보 위반 {issue}")
            for issue in _candidate_text_errors(str(row.get("source_japanese", "")), viewer):
                errors.append(f"block {number}: viewer 후보 위반 {issue}")
            if int(row.get("candidate_family_count", 0)) < 2:
                errors.append(f"block {number}: accepted에는 두 후보 계열이 필요합니다.")
            if not row.get("selected_generated_family") or row.get("selected_generated_family") == "previous-korean":
                errors.append(f"block {number}: 기존 한국어 후보만으로 accepted할 수 없습니다.")
            if row.get("abstention_reasons"):
                errors.append(f"block {number}: accepted에 abstention_reasons가 남았습니다.")
        elif status == "abstained":
            if row.get("source_faithful_korean") or row.get("viewer_natural_korean"):
                errors.append(f"block {number}: abstained 번역문은 비워야 합니다.")
            if not row.get("abstention_reasons"):
                errors.append(f"block {number}: abstained 사유가 없습니다.")

    expected = {block.number for block in reference}
    actual = set(by_number)
    for number in sorted(expected - actual):
        errors.append(f"decision 누락 block {number}")
    for number in sorted(actual - expected):
        errors.append(f"구조에 없는 decision block {number}")
    if len(observations) != len(reference):
        errors.append("observation 수가 구조 블록 수와 다릅니다.")
    if len(lattices) != len(reference):
        errors.append("semantic lattice 수가 구조 블록 수와 다릅니다.")
    abstained_numbers = {number for number, row in by_number.items() if row.get("status") == "abstained"}
    unresolved_numbers = {int(row["block_number"]) for row in unresolved if str(row.get("block_number", "")).isdigit()}
    if abstained_numbers != unresolved_numbers:
        errors.append("unresolved.jsonl과 abstained 결정 집합이 다릅니다.")

    for preview_path, field in ((source_preview_path, "source_faithful_korean"), (viewer_preview_path, "viewer_natural_korean")):
        preview, encoding, newline = parse_srt(preview_path)
        structure_report = compare_structure(reference, preview)
        if not structure_report["pass"]:
            errors.append(f"{preview_path.name}: 구조가 기준본과 다릅니다.")
        if encoding not in {"utf-8", "utf-8-sig"} or newline != "LF":
            errors.append(f"{preview_path.name}: UTF-8/LF가 아닙니다.")
        for block in preview:
            row = by_number.get(block.number)
            if not row:
                continue
            expected_text = str(row.get(field, "")) if row.get("status") == "accepted" else ABSTAIN_MARKER
            if block.text != expected_text:
                errors.append(f"{preview_path.name} block {block.number}: decision과 내용이 다릅니다.")

    proof_path = package_dir / "proof.json"
    if proof_path.exists():
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        if proof.get("100_percent_equal") is not False:
            errors.append("proof는 100_percent_equal=true를 주장할 수 없습니다.")
        if proof.get("human_reference_equality") not in {"unidentifiable", "measured-not-proven"}:
            errors.append("proof의 human_reference_equality 값이 잘못되었습니다.")

    manifest_path = package_dir / "run-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("stage") != CLOSED_WORLD_STAGE:
            errors.append("run manifest stage가 closed-world-validated가 아닙니다.")
        for record in [*manifest.get("inputs", []), *manifest.get("artifacts", [])]:
            path = Path(str(record.get("path", "")))
            if not path.exists():
                errors.append(f"manifest 파일이 없습니다: {path}")
            elif sha256_file(path) != record.get("sha256"):
                errors.append(f"manifest 해시 불일치: {path}")

    accepted = sum(row.get("status") == "accepted" for row in decisions)
    abstained = sum(row.get("status") == "abstained" for row in decisions)
    return {
        "schema_name": "translation-forensics/closed-world-validation",
        "schema_version": "1",
        "status": "pass" if not errors else "fail",
        "stage": CLOSED_WORLD_STAGE,
        "blocks": len(reference),
        "accepted": accepted,
        "abstained": abstained,
        "decision_coverage": (accepted + abstained) / len(reference) if reference else 0.0,
        "accepted_rate": accepted / len(reference) if reference else 0.0,
        "abstained_rate": abstained / len(reference) if reference else 0.0,
        "errors": errors,
        "final_promotion_allowed": False,
    }


def validate_closed_world_package(structure_path: Path, package_dir: Path) -> dict[str, Any]:
    return _core_validation(structure_path.resolve(), package_dir.resolve(), require_proof=True, require_manifest=True)


def prove_quality_claim(
    structure_path: Path,
    package_dir: Path,
    *,
    human_reference_path: Path | None = None,
) -> dict[str, Any]:
    validation = validate_closed_world_package(structure_path, package_dir)
    comparison: dict[str, Any] | None = None
    equality = "unidentifiable"
    if human_reference_path:
        reference_blocks, _, _ = parse_srt(human_reference_path)
        viewer_blocks, _, _ = parse_srt(package_dir / "viewer-natural.preview.srt")
        human_map = {block.number: block.text for block in reference_blocks}
        decisions = _read_jsonl(package_dir / "decisions.jsonl")
        accepted = [row for row in decisions if row.get("status") == "accepted"]
        exact = sum(human_map.get(int(row["block_number"])) == row.get("viewer_natural_korean") for row in accepted)
        comparison = {
            "human_reference_path": str(human_reference_path.resolve()),
            "human_reference_sha256": sha256_file(human_reference_path),
            "accepted_blocks_compared": len(accepted),
            "exact_string_matches": exact,
            "exact_string_match_rate": exact / len(accepted) if accepted else 0.0,
            "structure_comparison": compare_structure(reference_blocks, viewer_blocks),
        }
        equality = "measured-not-proven"
    return {
        "schema_name": "translation-forensics/closed-world-proof",
        "schema_version": "1",
        "status": "pass" if validation["status"] == "pass" else "fail",
        "stage": CLOSED_WORLD_STAGE,
        "human_reference_equality": equality,
        "100_percent_equal": False,
        "guarantees": {
            "decision_coverage_is_complete": validation.get("decision_coverage") == 1.0,
            "structure_and_timecodes_preserved": validation["status"] == "pass",
            "accepted_explicit_slot_invariant_violations": len(validation.get("errors", [])),
            "same_family_asr_deduplicated": True,
            "human_evidence_not_synthesized": True,
            "final_gate_not_bypassed": True,
        },
        "not_guaranteed": [
            "관측되지 않은 사람 정답과의 의미 또는 문자열 동일성",
            "자동 표면 규칙이 포착하지 못하는 화용·화자·대상·맥락 의미",
            "직접 청취 기반 시간축 또는 발화 검증",
        ],
        "validation": validation,
        "human_reference_comparison": comparison,
        "final_promotion_allowed": False,
    }


def _deterministic_manifest(
    *,
    title: str,
    project_root: Path,
    inputs: list[tuple[str, Path]],
    artifacts: list[Path],
    network_model_download_permitted: bool = False,
) -> dict[str, Any]:
    input_records = [{"role": role, "path": str(path.resolve()), "sha256": sha256_file(path)} for role, path in inputs]
    artifact_records = [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in artifacts]
    fingerprint_payload = {
        "engine_version": ENGINE_VERSION,
        "title": title,
        "inputs": [(record["role"], record["sha256"]) for record in input_records],
        "artifacts": [(Path(record["path"]).name, record["sha256"]) for record in artifact_records],
    }
    if network_model_download_permitted:
        fingerprint_payload["network_model_download_permitted"] = True
    import hashlib

    run_id = "closed-world-" + hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    manifest = {
        "schema_name": "translation-forensics/closed-world-run-manifest",
        "schema_version": "1",
        "run_id": run_id,
        "title_id": title,
        "stage": CLOSED_WORLD_STAGE,
        "engine_version": ENGINE_VERSION,
        "translation_forensics_version": __version__,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "network_used": False,
        "human_labels_used": False,
        "inputs": input_records,
        "artifacts": artifact_records,
        "final_promotion_allowed": False,
    }
    if network_model_download_permitted:
        manifest["network_model_download_permitted"] = True
        manifest["network_evidence_used"] = False
        manifest["note"] = "네트워크는 ASR 실행 모델 다운로드에만 허용되며, 외부 정답·사람 증거·최종 판정 근거로 사용되지 않습니다."
    return manifest


def run_closed_world(
    *,
    title: str,
    project_root: Path,
    structure_path: Path,
    japanese_path: Path,
    output_dir: Path,
    candidate_paths: list[Path],
    previous_path: Path | None = None,
    review_context_path: Path | None = None,
    asr_path: Path | None = None,
    timeline_validation_path: Path | None = None,
    network_model_download_permitted: bool = False,
) -> dict[str, Any]:
    structure_path = structure_path.resolve()
    japanese_path = japanese_path.resolve()
    output_dir = output_dir.resolve()
    previous_path = previous_path.resolve() if previous_path and previous_path.exists() else None
    review_context_path = review_context_path.resolve() if review_context_path and review_context_path.exists() else None
    asr_path = asr_path.resolve() if asr_path and asr_path.exists() else None
    timeline_validation_path = timeline_validation_path.resolve() if timeline_validation_path and timeline_validation_path.exists() else None

    reference, _, _ = parse_srt(structure_path)
    japanese_blocks, _, _ = parse_srt(japanese_path)
    japanese_map = _aligned_text_map(reference, japanese_blocks)
    all_candidates = list(candidate_paths)
    if previous_path:
        all_candidates.append(previous_path)
    inventory, candidates_by_block = _candidate_inventory(reference, all_candidates, previous_path=previous_path)
    asr_by_block, scenes = _review_context(review_context_path)
    supplied_timeline = json.loads(timeline_validation_path.read_text(encoding="utf-8")) if timeline_validation_path else None
    machine_timeline = _machine_timeline(scenes, block_count=len(reference), supplied_timeline=supplied_timeline)

    decisions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    lattices: list[dict[str, Any]] = []
    source_preview: list[SubtitleBlock] = []
    viewer_preview: list[SubtitleBlock] = []
    for block in reference:
        japanese = japanese_map.get(block.number, ("", "unresolved"))[0]
        decision, observation, lattice = _decide_block(
            block,
            japanese,
            candidates_by_block.get(block.number, []),
            asr_by_block.get(block.number, []),
        )
        decisions.append(decision)
        observations.append(observation)
        lattices.append(lattice)
        source_text = decision["source_faithful_korean"] if decision["status"] == "accepted" else ABSTAIN_MARKER
        viewer_text = decision["viewer_natural_korean"] if decision["status"] == "accepted" else ABSTAIN_MARKER
        source_preview.append(SubtitleBlock(block.number, block.start, block.end, source_text, block.start_seconds, block.end_seconds))
        viewer_preview.append(SubtitleBlock(block.number, block.start, block.end, viewer_text, block.start_seconds, block.end_seconds))

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "observations": output_dir / "observations.jsonl",
        "semantic_lattice": output_dir / "semantic-lattice.jsonl",
        "decisions": output_dir / "decisions.jsonl",
        "source_preview": output_dir / "source-faithful.preview.srt",
        "viewer_preview": output_dir / "viewer-natural.preview.srt",
        "unresolved": output_dir / "unresolved.jsonl",
        "machine_timeline": output_dir / "machine-timeline.json",
        "candidate_inventory": output_dir / "candidate-inventory.json",
        "summary": output_dir / "summary.json",
        "proof": output_dir / "proof.json",
        "manifest": output_dir / "run-manifest.json",
        "validation": output_dir / "validation.json",
    }
    _write_idempotent(paths["observations"], _canonical_jsonl(observations))
    _write_idempotent(paths["semantic_lattice"], _canonical_jsonl(lattices))
    _write_idempotent(paths["decisions"], _canonical_jsonl(decisions))
    _write_srt_idempotent(paths["source_preview"], source_preview)
    _write_srt_idempotent(paths["viewer_preview"], viewer_preview)
    unresolved = [row for row in decisions if row["status"] == "abstained"]
    _write_idempotent(paths["unresolved"], _canonical_jsonl(unresolved))
    _write_idempotent(paths["machine_timeline"], _canonical_json(machine_timeline))
    _write_idempotent(paths["candidate_inventory"], _canonical_json({"schema_version": "1", "candidates": inventory}))

    accepted = len(decisions) - len(unresolved)
    summary = {
        "schema_name": "translation-forensics/closed-world-summary",
        "schema_version": "1",
        "title_id": title,
        "stage": CLOSED_WORLD_STAGE,
        "blocks": len(decisions),
        "accepted": accepted,
        "abstained": len(unresolved),
        "decision_coverage": 1.0 if decisions else 0.0,
        "accepted_rate": accepted / len(decisions) if decisions else 0.0,
        "abstained_rate": len(unresolved) / len(decisions) if decisions else 0.0,
        "machine_timeline_status": machine_timeline["status"],
        "human_reference_equality": "unidentifiable",
        "100_percent_equal": False,
        "final_promotion_allowed": False,
    }
    if network_model_download_permitted:
        summary["network_model_download_permitted"] = True
    _write_idempotent(paths["summary"], _canonical_json(summary))

    preflight = _core_validation(structure_path, output_dir, require_proof=False, require_manifest=False)
    proof = {
        "schema_name": "translation-forensics/closed-world-proof",
        "schema_version": "1",
        "status": preflight["status"],
        "stage": CLOSED_WORLD_STAGE,
        "human_reference_equality": "unidentifiable",
        "100_percent_equal": False,
        "guarantees": {
            "decision_coverage_is_complete": preflight.get("decision_coverage") == 1.0,
            "structure_and_timecodes_preserved": preflight["status"] == "pass",
            "accepted_explicit_slot_invariant_violations": len(preflight.get("errors", [])),
            "same_family_asr_deduplicated": True,
            "human_evidence_not_synthesized": True,
            "final_gate_not_bypassed": True,
        },
        "not_guaranteed": [
            "관측되지 않은 사람 정답과의 의미 또는 문자열 동일성",
            "자동 표면 규칙이 포착하지 못하는 화용·화자·대상·맥락 의미",
            "직접 청취 기반 시간축 또는 발화 검증",
        ],
        "final_promotion_allowed": False,
    }
    if network_model_download_permitted:
        proof["network_model_download_permitted"] = True
        proof["not_guaranteed"].append("네트워크로 내려받은 모델 버전의 장기 가용성 또는 공급자 재현성")
    _write_idempotent(paths["proof"], _canonical_json(proof))

    manifest_inputs: list[tuple[str, Path]] = [("structure", structure_path), ("japanese", japanese_path)]
    if previous_path:
        manifest_inputs.append(("previous-korean-candidate", previous_path))
    if review_context_path:
        manifest_inputs.append(("review-context", review_context_path))
    if asr_path:
        manifest_inputs.append(("asr-candidates", asr_path))
    if timeline_validation_path:
        manifest_inputs.append(("timeline-validation", timeline_validation_path))
    for path in sorted({value.resolve() for value in candidate_paths if value.exists()}, key=lambda value: str(value).lower()):
        manifest_inputs.append(("korean-candidate", path))
    manifest_artifacts = [
        paths["observations"], paths["semantic_lattice"], paths["decisions"], paths["source_preview"],
        paths["viewer_preview"], paths["unresolved"], paths["machine_timeline"],
        paths["candidate_inventory"], paths["summary"], paths["proof"],
    ]
    manifest = _deterministic_manifest(
        title=title,
        project_root=project_root,
        inputs=manifest_inputs,
        artifacts=manifest_artifacts,
        network_model_download_permitted=network_model_download_permitted,
    )
    _write_idempotent(paths["manifest"], _canonical_json(manifest))
    validation = validate_closed_world_package(structure_path, output_dir)
    _write_idempotent(paths["validation"], _canonical_json(validation))
    if validation["status"] != "pass":
        raise RuntimeError("폐쇄형 패키지 자체 검증이 실패했습니다: " + "; ".join(validation["errors"][:5]))
    return {
        **summary,
        "status": "closed-world-validated",
        "output_dir": str(output_dir),
        "files": {name: str(path) for name, path in paths.items()},
        "validation": validation,
    }
