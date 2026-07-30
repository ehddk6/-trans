from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .asr_evidence import text_similarity
from .srt import SubtitleBlock, parse_srt


SOURCE_STATUSES = ("trusted", "suspect", "unusable")
OUTRO_RE = re.compile(r"(?:ご視聴|御視聴).{0,8}(?:ありがとう|有難う)|チャンネル登録|高評価")
MOJIBAKE_RE = re.compile(r"(?:�|\ufffd|(?:縺|譁|蜿|繧|莨|驥).*(?:縺|譁|蜿|繧|莨|驥))")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PUNCT_RE = re.compile(r"[\s\u3000。、！？!?…・〜～ー―—\-]+")
SHORT_VOCALIZATIONS = {
    "あ", "ああ", "う", "うう", "え", "ええ", "お", "おお", "ん", "うん", "はい",
    "いや", "やだ", "だめ", "そう", "ね", "ねえ", "よ", "ほら", "もっと", "痛い",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_source_text(text: str) -> str:
    return PUNCT_RE.sub("", text or "").casefold()


def is_legitimate_short_repetition(text: str) -> bool:
    normalized = normalize_source_text(text)
    if normalized in SHORT_VOCALIZATIONS:
        return True
    if not normalized or len(normalized) > 5:
        return False
    return bool(re.fullmatch(r"[ぁ-ゖァ-ヺゝ-ヿ]+", normalized))


def _consecutive_runs(blocks: list[SubtitleBlock]) -> dict[int, tuple[int, int, str]]:
    result: dict[int, tuple[int, int, str]] = {}
    cursor = 0
    while cursor < len(blocks):
        normalized = normalize_source_text(blocks[cursor].text)
        end = cursor + 1
        while end < len(blocks) and normalize_source_text(blocks[end].text) == normalized:
            end += 1
        length = end - cursor
        if normalized:
            for position in range(cursor, end):
                result[blocks[position].number] = (length, blocks[cursor].number, normalized)
        cursor = end
    return result


def _read_block_asr(path: Path | None) -> dict[int, list[dict[str, Any]]]:
    if path is None:
        return {}
    result: dict[int, list[dict[str, Any]]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"ASR evidence row {line_number} must be an object")
        try:
            block_number = int(value.get("block_number"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ASR evidence row {line_number} lacks block_number") from exc
        transcripts = value.get("transcripts")
        if not isinstance(transcripts, list):
            raise ValueError(f"ASR evidence row {line_number} transcripts must be an array")
        cleaned: list[dict[str, Any]] = []
        for item in transcripts:
            if not isinstance(item, dict):
                continue
            family = str(item.get("source_family") or "").strip()
            text = str(item.get("text") or "").strip()
            if family and text:
                cleaned.append(
                    {
                        "source_family": family,
                        "text": text,
                        "block_aligned": bool(item.get("block_aligned", True)),
                    }
                )
        result[block_number] = cleaned
    return result


def audit_source_blocks(
    *,
    title_id: str,
    japanese_path: Path,
    asr_evidence_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    blocks, encoding, newline = parse_srt(japanese_path)
    if not blocks:
        raise ValueError("Japanese SRT has no blocks")
    runs = _consecutive_runs(blocks)
    normalized_counts = Counter(normalize_source_text(block.text) for block in blocks if normalize_source_text(block.text))
    asr_by_block = _read_block_asr(asr_evidence_path)
    end_seconds = max(block.end_seconds for block in blocks)
    ending_zone = end_seconds * 0.97
    records: list[dict[str, Any]] = []

    for block in blocks:
        text = block.text.strip()
        normalized = normalize_source_text(text)
        run_length, run_start, _ = runs.get(block.number, (1, block.number, normalized))
        frequency = normalized_counts.get(normalized, 0)
        reasons: list[str] = []
        acoustic_families: set[str] = set()
        asr_source_similarity: float | None = None
        asr_family_agreement: float | None = None

        if not text:
            reasons.append("empty-source")
        if CONTROL_RE.search(text) or MOJIBAKE_RE.search(text):
            reasons.append("broken-characters")
        if OUTRO_RE.search(text) and block.start_seconds < ending_zone:
            reasons.append("mid-program-outro")
        short_repeat = is_legitimate_short_repetition(text)
        if not short_repeat and run_length >= 5:
            reasons.append("long-consecutive-repeat")
        if not short_repeat and frequency >= max(8, int(len(blocks) * 0.05)):
            reasons.append("high-global-reuse")

        transcripts = asr_by_block.get(block.number, [])
        by_family_parts: dict[str, list[str]] = {}
        for transcript in transcripts:
            by_family_parts.setdefault(transcript["source_family"], []).append(transcript["text"])
        by_family = {family: " ".join(parts) for family, parts in by_family_parts.items()}
        acoustic_families = set(by_family_parts)
        block_aligned = bool(transcripts) and all(bool(row.get("block_aligned")) for row in transcripts)
        if by_family and block_aligned:
            similarities = [text_similarity(text, candidate) for candidate in by_family.values()]
            asr_source_similarity = round(max(similarities), 4)
            if len(by_family) >= 2:
                values = list(by_family.values())
                agreements = [
                    text_similarity(values[left], values[right])
                    for left in range(len(values))
                    for right in range(left + 1, len(values))
                ]
                asr_family_agreement = round(max(agreements), 4) if agreements else None
                if asr_family_agreement is not None and asr_family_agreement >= 0.75 and asr_source_similarity < 0.35:
                    reasons.append("dual-asr-source-conflict")

        hard_reasons = {
            "empty-source",
            "broken-characters",
            "mid-program-outro",
            "dual-asr-source-conflict",
        }
        if run_length >= 8 and not short_repeat:
            hard_reasons.add("long-consecutive-repeat")
        if frequency >= max(20, int(len(blocks) * 0.15)) and not short_repeat:
            hard_reasons.add("high-global-reuse")
        if any(reason in hard_reasons for reason in reasons):
            status = "unusable"
        elif reasons:
            status = "suspect"
        else:
            status = "trusted"

        records.append(
            {
                "schema_name": "translation-forensics/source-quality-record",
                "schema_version": "1",
                "title_id": title_id,
                "block_number": block.number,
                "start": block.start,
                "end": block.end,
                "source_text": text,
                "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "source_quality_status": status,
                "reason_codes": sorted(set(reasons)),
                "consecutive_run_length": run_length,
                "consecutive_run_start_block": run_start,
                "global_reuse_count": frequency,
                "legitimate_short_repetition": short_repeat,
                "acoustic_source_families": sorted(acoustic_families),
                "asr_source_similarity": asr_source_similarity,
                "asr_family_agreement": asr_family_agreement,
            }
        )

    counts = Counter(record["source_quality_status"] for record in records)
    unique_count = len({normalize_source_text(block.text) for block in blocks if normalize_source_text(block.text)})
    longest_run = max((record["consecutive_run_length"] for record in records), default=0)
    report = {
        "schema_name": "translation-forensics/source-quality-report",
        "schema_version": "1",
        "title_id": title_id,
        "status": "audited",
        "source": {
            "path": str(japanese_path.expanduser().resolve()),
            "sha256": _sha256(japanese_path),
            "encoding": encoding,
            "newline": newline,
        },
        "block_count": len(blocks),
        "status_counts": {name: counts[name] for name in SOURCE_STATUSES},
        "unique_normalized_text_count": unique_count,
        "lexical_diversity": round(unique_count / len(blocks), 6),
        "longest_consecutive_repeat": longest_run,
        "dual_asr_evidence_present": bool(asr_by_block),
        "asr_evidence": (
            {"path": str(asr_evidence_path.expanduser().resolve()), "sha256": _sha256(asr_evidence_path)}
            if asr_evidence_path is not None
            else None
        ),
        "translation_input_policy": {
            "trusted": "Japanese may be used with scene context.",
            "suspect": "Acoustic evidence is required before acceptance.",
            "unusable": "Japanese text is structure-only; two independent ASR families are required before acceptance.",
        },
        "human_equal": False,
        "human_final": False,
        "final_promotion_allowed": False,
    }
    return report, records


def write_source_quality_audit(
    *,
    title_id: str,
    japanese_path: Path,
    output_dir: Path,
    asr_evidence_path: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    report_path = output_dir / "source-quality-report.json"
    map_path = output_dir / "source-quality-map.jsonl"
    if resume and report_path.is_file() and map_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_asr_hash = _sha256(asr_evidence_path) if asr_evidence_path is not None else None
        recorded_asr_hash = (report.get("asr_evidence") or {}).get("sha256")
        if (
            report.get("source", {}).get("sha256") == _sha256(japanese_path)
            and recorded_asr_hash == expected_asr_hash
        ):
            return {**report, "cache_hit": True, "output": str(output_dir)}
    existing = [path for path in (report_path, map_path) if path.exists()]
    if existing and not resume:
        raise FileExistsError("Source quality outputs already exist: " + ", ".join(str(path) for path in existing))
    report, records = audit_source_blocks(
        title_id=title_id,
        japanese_path=japanese_path,
        asr_evidence_path=asr_evidence_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    report["quality_map"] = {
        "path": map_path.name,
        "sha256": _sha256(map_path),
        "records": len(records),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return {**report, "cache_hit": False, "output": str(output_dir)}
