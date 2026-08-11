from __future__ import annotations

import re
import unicodedata

_CLOSERS = "。！？?！？」』】〉》"
_PARTICLES = {"は", "が", "を", "に", "へ", "と", "で", "の", "も", "や", "か", "ね", "よ", "ぞ", "さ", "な", "まで", "より", "から", "ので", "けど"}
_AUXILIARY = {"です", "ます", "だ", "ない", "たい", "た", "て", "れる", "られる", "せる", "させる", "いる", "ある", "んだ", "んです", "じゃない", "わけだ", "ことになる"}
_RUNAWAY_REPETITION = re.compile(r"(.{1,3})\1{4,}")
_NONLEXICAL_VOCALISATION = frozenset("あいうえおぁぃぅぇぉんはひふへほねっー〜～")
_NONLEXICAL_LENGTHENERS = frozenset("ぁぃぅぇぉっー〜～")
_COMPACTABLE_RUNAWAY = re.compile(
    r"(?P<unit>[あいうえおぁぃぅぇぉんはひふへほねっー〜～]{1,3}?)(?:[、,?？\s]*(?P=unit)){4,}[、,?？\s]*"
)
_HIGH_CONFIDENCE_RUNAWAY = re.compile(
    r"(?P<unit>.{1,4}?)(?P<separator>[、,?？!！\s]*)(?P=unit)(?:(?P=separator)(?P=unit)){3,}(?P=separator)?"
)


def has_runaway_repetition(text: str) -> bool:
    return bool(_RUNAWAY_REPETITION.search(re.sub(r"\s+", "", text)))


def _is_compactable_vocalisation(unit: str) -> bool:
    if not unit or any(character not in _NONLEXICAL_VOCALISATION for character in unit):
        return False
    return len(unit) == 1 or unit == "ねえ" or any(character in _NONLEXICAL_LENGTHENERS for character in unit)


def compact_runaway_repetition(text: str) -> str:
    """Condense extreme repeated vocalisation units for viewer display only."""
    return _COMPACTABLE_RUNAWAY.sub(
        lambda match: f"{match.group(1)}…" if _is_compactable_vocalisation(match.group(1)) else match.group(0),
        text,
    )


def compact_high_confidence_runaway_repetition(text: str) -> str:
    """Condense any five-plus unit loop after the ASR quality checks prove it suspect.

    This is intentionally separate from :func:`compact_runaway_repetition`.
    The caller must require a model-side ``high_compression_ratio`` warning;
    ordinary repeated dialogue is never changed by this helper alone.
    """
    return _HIGH_CONFIDENCE_RUNAWAY.sub(lambda match: f"{match.group('unit')}…", text)


def is_compacted_vocalisation(text: str) -> bool:
    return bool(re.fullmatch(r"[あいうえおぁぃぅぇぉんはひふへほねっー〜～]+…", re.sub(r"\s+", "", text)))


def display_width(text: str) -> float:
    """Japanese full-width characters count as one; ASCII-like characters as half."""
    return sum(1.0 if unicodedata.east_asian_width(ch) in "WFA" else 0.5 for ch in text if not ch.isspace())


def normalize_japanese(text: str, mode: str = "conservative") -> str:
    if mode in {"strict", "layout"}:
        return text.strip()
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r"\s*([、。！？!?])\s*", r"\1", text)
    text = re.sub(r"([。！？])\1+", r"\1", text)
    if mode == "viewer":
        text = compact_runaway_repetition(text)
    # Repetition is only flagged by the ASR quality checks. It is not removed here:
    # a literal repeat can be an intentional spoken utterance.
    return text


def content_signature(text: str) -> str:
    """Return a comparison form that ignores only display whitespace."""
    return re.sub(r"\s+", "", text)


def is_bad_boundary(previous: str, following: str) -> bool:
    previous, following = previous.strip(), following.strip()
    if not previous or not following:
        return False
    if following in _PARTICLES or following in _AUXILIARY:
        return True
    if previous[-1:] in "0123456789" and following[:1] in "年月日時分秒円個本人回%％":
        return True
    if previous[-1:] in "、・" or following[:1] in "、。！？!?」』】":
        return True
    # Japanese inflections and short dangling fragments are poor subtitle boundaries.
    if len(following) <= 2 and not following.endswith(tuple(_CLOSERS)):
        return True
    return False


def boundary_kind(previous: str, following: str) -> str:
    if previous.rstrip().endswith(tuple(_CLOSERS)):
        return "sentence_end"
    if following.lstrip().startswith(("でも", "そして", "だから", "しかし", "けれど", "ただ")):
        return "connective"
    if previous.rstrip().endswith(("、", "…", "―")):
        return "clause_or_breath"
    return "word_alignment"


def wrap_two_lines(text: str, target: int, absolute: int) -> list[str]:
    """Wrap without changing text, preferring punctuation and Japanese phrase boundaries."""
    if display_width(text) <= absolute:
        return [text]
    candidates = [i for i, ch in enumerate(text, 1) if ch in "、。！？!? 」"]
    candidates += list(range(1, len(text)))
    fitting = [
        index for index in candidates
        if display_width(text[:index]) <= absolute and display_width(text[index:]) <= absolute
    ]
    pool = fitting or candidates
    best = min(
        pool,
        key=lambda i: (
            max(display_width(text[:i]), display_width(text[i:])),
            abs(display_width(text[:i]) - target),
            i,
        ),
    )
    left, right = text[:best].rstrip(), text[best:].lstrip()
    if not left or not right:
        return [text]
    return [left, right]
