from __future__ import annotations

from collections.abc import Mapping
import re

from .models import SourceSegment


_KANA = re.compile(r"[\u3040-\u30ff\uff66-\uff9f]")
_HANGUL = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def reference_language_diagnostics(rows: list[Mapping[str, object]]) -> dict[str, object]:
    """Detect only high-confidence incompatible Korean subtitle sources.

    Japanese dialogue can legitimately contain long kanji-only spans, Latin
    names, numbers, or very little kana.  Those cases remain ``undetermined``
    instead of being rejected.  A substantial Hangul majority, however, is
    strong evidence that a translated Korean SRT was supplied where the
    Japanese-source pipeline requires Japanese text.
    """
    text = "".join(str(row.get("text", "")) for row in rows)
    kana = len(_KANA.findall(text))
    hangul = len(_HANGUL.findall(text))
    han = len(_HAN.findall(text))
    hangul_dominant = hangul >= 20 and hangul >= max(20, kana * 2)
    if hangul_dominant:
        status = "incompatible_hangul_dominant"
    elif kana or han:
        status = "japanese_like"
    else:
        status = "undetermined"
    return {
        "status": status,
        "kana_characters": kana,
        "han_characters": han,
        "hangul_characters": hangul,
        "compatible": not hangul_dominant,
    }


def require_japanese_reference(rows: list[Mapping[str, object]]) -> dict[str, object]:
    diagnostics = reference_language_diagnostics(rows)
    if not diagnostics["compatible"]:
        raise ValueError(
            "The reference SRT is Hangul-dominant and appears to be a Korean translation; "
            "this pipeline requires Japanese source subtitles."
        )
    return diagnostics


def segments_from_reference_rows(rows: list[Mapping[str, object]]) -> list[SourceSegment]:
    """Turn an existing SRT into source-preserving, non-overlapping segments.

    Reference timing is retained whenever possible.  A later cue is moved only
    when it overlaps the previous one, and that change is retained in the
    audit metadata instead of changing any Japanese source text.
    """
    segments: list[SourceSegment] = []
    previous_end = 0.0
    for index, row in enumerate(rows):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        original_start = float(row["start"])
        original_end = float(row["end"])
        start = max(0.0, original_start, previous_end)
        end = max(original_end, start + 0.01)
        warnings: list[str] = ["timestamp_precision_limited"]
        metadata: dict[str, object] = {
            "backend": "reference",
            "decision": "reference_primary",
            "reference_id": str(row.get("id", index + 1)),
        }
        if start != original_start or end != original_end:
            warnings.append("reference_timing_adjusted")
            metadata["reference_time_original"] = [original_start, original_end]
        segments.append(SourceSegment(
            id=len(segments), start=start, end=end, text=text,
            warnings=warnings, metadata=metadata,
        ))
        previous_end = end
    return segments
