"""Deterministic review samples for Japanese subtitle quality checks.

The review artifacts intentionally copy timing and text from the supplied
recognition results.  They are evidence for human review, not another
normalisation or subtitle-generation step.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from typing import Any


_CATEGORY_ORDER = (
    "engine_disagreement",
    "whisper_recovery",
    "runaway_repetition",
    "presentation_limit_exception",
    "long_cue",
    "silence_boundary",
)
_DEFAULT_WINDOW_SECONDS = 20.0
_DEFAULT_TARGET_WINDOWS = 30
_CATEGORY_QUOTA = 6

_CATEGORY_LABELS = {
    "engine_disagreement": "엔진 불일치",
    "whisper_recovery": "Whisper 보완 후보",
    "runaway_repetition": "반복·환각 의심",
    "presentation_limit_exception": "표시 제한 예외",
    "long_cue": "긴 자막 큐",
    "silence_boundary": "무음 경계",
    "deterministic_baseline": "균등 기준 표본",
}
_REASON_LABELS = {
    "explicit_disagreement": "엔진 불일치 기록",
    "qwen_recovery": "Qwen 보완 기록",
    "runaway_repetition_review": "반복 의심 검수",
    "whisper_disagreement": "Whisper 불일치",
    "whisper_candidate": "Whisper 보완 후보",
    "final_cue_silence_signal": "최종 자막 무음 신호",
    "declared_adjacent_silence": "인접 무음 기록",
    "source_gap": "원문 사이 무음",
    "duration_over_7_seconds": "7초 초과",
    "deterministic_baseline": "균등 기준 표본",
}
_QUALITY_REASON_LABELS = {
    "remaining_viewer_runaway_repetition": "시청용 자막에 남은 반복",
    "runaway_repetition_source_evidence": "원문 반복 의심 근거",
    "possible_silence_hallucination": "무음 구간 환각 의심",
    "possible_repetition": "반복 의심",
    "line_limit_unavoidable": "줄 길이 제한 불가피",
    "duration_limit_unavoidable": "7초 제한 적용 불가피",
    "short_duration_review_required": "0.8초 미만 수동 검수",
    "word_text_mismatch": "단어 정렬과 원문 불일치",
    "word_alignment_review_required": "단어 타임스탬프 정렬 검수 필요",
}
_STATUS_LABELS = {
    "passed": "자동 게이트 통과",
    "review_required": "수동 검수 필요",
    "failed": "구조 검증 실패",
}


def _ui_label(value: Any, labels: Mapping[str, str]) -> str:
    return labels.get(str(value), str(value))


def _value(item: Any, *names: str, default: Any = None) -> Any:
    """Read a field from either pipeline dataclasses or JSON-like input."""

    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(item: Any) -> str:
    value = _value(
        item,
        "text",
        "text_raw",
        "text_normalized",
        "transcript",
        "content",
        "whisper_text",
        "qwen_text",
        default="",
    )
    return "" if value is None else str(value)


def _identifier(item: Any, fallback: int) -> str:
    value = _value(item, "id", "cue_id", "segment_id", "utterance_id", "event_id", default=fallback)
    return str(value)


def _json_safe(value: Any) -> Any:
    """Turn common ASR / numpy-like metadata into stable JSON values."""

    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _row(item: Any, fallback_id: int, role: str) -> dict[str, Any] | None:
    """Make a compact, immutable review row from a cue, segment, or dict."""

    start = _number(_value(item, "start", "start_time", "begin", default=None), float("nan"))
    end = _number(_value(item, "end", "end_time", "finish", default=None), float("nan"))
    if start != start or end != end:  # NaN without importing math.
        return None
    if end < start:
        start, end = end, start

    row: dict[str, Any] = {
        "id": _identifier(item, fallback_id),
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(max(0.0, end - start), 3),
        "text": _text(item),
        "role": role,
    }
    metadata: dict[str, Any] = {}
    for key in (
        "category",
        "event_type",
        "decision",
        "reason",
        "warnings",
        "split_reasons",
        "merge_reasons",
        "adjacent_silence",
        "qwen_text",
        "whisper_text",
        "confidence",
        "probability",
        "avg_logprob",
        "no_speech_prob",
    ):
        value = _value(item, key, default=None)
        if value is not None and value != "":
            metadata[key] = _json_safe(value)
    source_metadata = _value(item, "metadata", default=None)
    if isinstance(source_metadata, Mapping):
        metadata.update(_json_safe(source_metadata))
    if metadata:
        row["metadata"] = metadata
    return row


def _normalise(items: Iterable[Any] | None, role: str) -> list[dict[str, Any]]:
    rows = [_row(item, index, role) for index, item in enumerate(items or (), 1)]
    return sorted((row for row in rows if row is not None), key=lambda row: (row["start"], row["end"], row["id"]))


def _overlaps(row: Mapping[str, Any], start: float, end: float) -> bool:
    row_start = _number(row.get("start"))
    row_end = _number(row.get("end"))
    if row_start == row_end:
        return start <= row_start <= end
    return row_start < end and row_end > start


def _event_text(row: Mapping[str, Any]) -> str:
    parts = [str(row.get("text", ""))]
    metadata = row.get("metadata", {})
    if isinstance(metadata, Mapping):
        parts.extend(str(metadata.get(key, "")) for key in ("category", "event_type", "decision", "reason", "warnings"))
    return " ".join(parts).casefold()


def _tagged(row: Mapping[str, Any], *needles: str) -> bool:
    haystack = _event_text(row)
    return any(needle in haystack for needle in needles)


def _signal(category: str, row: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "category": category,
        "start": _number(row["start"]),
        "end": _number(row["end"]),
        "reason": reason,
        "evidence": _json_safe(row),
    }


def _presentation_limit_codes(row: Mapping[str, Any]) -> list[str]:
    """Return viewer limits that could not be met without altering timing/text."""

    metadata = row.get("metadata", {})
    warnings = metadata.get("warnings", []) if isinstance(metadata, Mapping) else []
    if isinstance(warnings, str):
        warnings = [warnings]
    warning_set = {str(warning) for warning in warnings}
    return [
        code for code in (
            "line_limit_unavoidable",
            "duration_limit_unavoidable",
            "short_duration_review_required",
        )
        if code in warning_set
    ]


def _presentation_limit_exceptions(final_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep every unresolved display exception visible, not only the sampled six."""

    exceptions: list[dict[str, Any]] = []
    for row in final_rows:
        codes = _presentation_limit_codes(row)
        if not codes:
            continue
        exceptions.append({
            "cue_id": row["id"],
            "start": row["start"],
            "end": row["end"],
            "duration": row["duration"],
            "codes": codes,
            "text": row["text"],
        })
    return exceptions


def _stable_key(seed: str, category: str, signal: Mapping[str, Any]) -> tuple[str, float, float, str]:
    evidence = signal.get("evidence", {})
    encoded = json.dumps(_json_safe(evidence), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = sha256(f"{seed}|{category}|{encoded}".encode("utf-8")).hexdigest()
    return digest, _number(signal["start"]), _number(signal["end"]), encoded


def _source_silence_signals(source_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    previous: Mapping[str, Any] | None = None
    for row in source_rows:
        metadata = row.get("metadata", {})
        adjacent = _number(metadata.get("adjacent_silence"), 0.0) if isinstance(metadata, Mapping) else 0.0
        if adjacent >= 1.0:
            signals.append(_signal("silence_boundary", row, "declared_adjacent_silence"))
        if previous is not None:
            gap = _number(row["start"]) - _number(previous["end"])
            if gap >= 1.0:
                boundary = {
                    "id": f"gap_{previous['id']}_{row['id']}",
                    "start": previous["end"],
                    "end": row["start"],
                    "text": "",
                    "category": "silence_boundary",
                    "reason": "source_gap",
                    "gap_seconds": round(gap, 3),
                    "left_text": previous.get("text", ""),
                    "right_text": row.get("text", ""),
                }
                signals.append(_signal("silence_boundary", boundary, "source_gap"))
        previous = row
    return signals


def _signals(
    final_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    whisper_rows: Sequence[Mapping[str, Any]],
    disagreement_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result = {category: [] for category in _CATEGORY_ORDER}

    for row in disagreement_rows:
        result["engine_disagreement"].append(_signal("engine_disagreement", row, "explicit_disagreement"))
    for row in source_rows:
        if _tagged(row, "qwen_recovery"):
            result["whisper_recovery"].append(_signal("whisper_recovery", row, "qwen_recovery"))
        if _tagged(row, "possible_runaway_repetition", "possible_periodic_repetition"):
            result["runaway_repetition"].append(_signal("runaway_repetition", row, "runaway_repetition_review"))
    for row in whisper_rows:
        if _tagged(row, "disagreement", "review_required", "engine_conflict"):
            result["engine_disagreement"].append(_signal("engine_disagreement", row, "whisper_disagreement"))
        elif _tagged(row, "whisper_primary"):
            continue
        elif _tagged(row, "whisper_recovery"):
            result["whisper_recovery"].append(_signal("whisper_recovery", row, "whisper_candidate"))
        elif not _tagged(row, "whisper_verification"):
            # Backward-compatible behaviour for older callers that supply an
            # unannotated Whisper recovery candidate.
            result["whisper_recovery"].append(_signal("whisper_recovery", row, "whisper_candidate"))

    for row in final_rows:
        limit_codes = _presentation_limit_codes(row)
        if limit_codes:
            result["presentation_limit_exception"].append(
                _signal("presentation_limit_exception", row, ",".join(limit_codes))
            )
        elif _number(row["end"]) - _number(row["start"]) > 7.0:
            result["long_cue"].append(_signal("long_cue", row, "duration_over_7_seconds"))
        if _tagged(row, "silence", "long_gap"):
            result["silence_boundary"].append(_signal("silence_boundary", row, "final_cue_silence_signal"))

    result["silence_boundary"].extend(_source_silence_signals(source_rows))
    return result


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _window_start_for_signal(
    signal: Mapping[str, Any],
    window_seconds: float,
    max_start: float,
    used_starts: set[float],
) -> tuple[float, bool]:
    """Choose a unique window that still contains the evidence interval."""

    anchor_start = _number(signal["start"])
    anchor_end = max(anchor_start, _number(signal["end"]))
    lower = max(0.0, anchor_end - window_seconds)
    upper = min(max_start, anchor_start)
    if lower > upper:
        # A malformed out-of-range event should still be visible at the nearest edge.
        lower = upper = _clamp(anchor_start - 5.0, 0.0, max_start)
    preferred = _clamp(anchor_start - 5.0, lower, upper)
    offsets = [0.0]
    step = 0.5
    for index in range(1, int((upper - lower) / step) + 2):
        offsets.extend((-index * step, index * step))
    for offset in offsets:
        candidate = _clamp(preferred + offset, lower, upper)
        key = round(candidate, 3)
        if key not in used_starts:
            used_starts.add(key)
            return key, False
    return round(preferred, 3), True


def _baseline_signal(seed: str, index: int, video_duration: float, window_seconds: float) -> dict[str, Any]:
    max_start = max(0.0, video_duration - window_seconds)
    digest = sha256(f"{seed}|baseline|{index}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / ((1 << 64) - 1)
    start = max_start * fraction
    return {
        "category": "deterministic_baseline",
        "start": start + window_seconds / 2,
        "end": start + window_seconds / 2,
        "reason": "deterministic_baseline",
        "evidence": {"baseline_index": index},
    }


def _window_rows(rows: Sequence[Mapping[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [_json_safe(row) for row in rows if _overlaps(row, start, end)]


def build_review_manifest(
    final_cues: Iterable[Any],
    qwen_source: Iterable[Any],
    whisper_candidates: Iterable[Any] | None = None,
    engine_disagreements: Iterable[Any] | None = None,
    external_reference: Iterable[Any] | None = None,
    *,
    video_duration: float | None = None,
    video_path: str | Path | None = None,
    source_evidence_label: str = "Qwen/원문 근거",
    quality_gate: Mapping[str, Any] | None = None,
    window_seconds: float = _DEFAULT_WINDOW_SECONDS,
    target_windows: int = _DEFAULT_TARGET_WINDOWS,
    seed: str = "japanese-subtitle-review-v1",
) -> dict[str, Any]:
    """Create a deterministic, source-preserving human-review manifest.

    ``qwen_source`` accepts either :class:`Cue`, :class:`SourceSegment`, or
    transcript JSONL-shaped dictionaries.  ``whisper_candidates`` and
    ``engine_disagreements`` are deliberately permissive so the ensemble
    runner can pass its audit rows without a conversion layer.
    """

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if target_windows <= 0:
        raise ValueError("target_windows must be positive")

    final_rows = _normalise(final_cues, "final")
    source_rows = _normalise(qwen_source, "qwen_source")
    whisper_rows = _normalise(whisper_candidates, "whisper_evidence")
    disagreement_rows = _normalise(engine_disagreements, "engine_disagreement")
    external_rows = _normalise(external_reference, "external_reference")
    all_rows = final_rows + source_rows + whisper_rows + disagreement_rows + external_rows
    inferred_duration = max((float(row["end"]) for row in all_rows), default=0.0)
    effective_duration = max(inferred_duration, _number(video_duration, 0.0))
    max_start = max(0.0, effective_duration - window_seconds)

    grouped = _signals(final_rows, source_rows, whisper_rows, disagreement_rows)
    presentation_exceptions = _presentation_limit_exceptions(final_rows)
    selected: list[dict[str, Any]] = []
    used_starts: set[float] = set()
    category_counts = {category: 0 for category in _CATEGORY_ORDER}
    for category in _CATEGORY_ORDER:
        candidates = sorted(grouped[category], key=lambda signal: _stable_key(seed, category, signal))
        for signal in candidates:
            if category_counts[category] >= _CATEGORY_QUOTA or len(selected) >= target_windows:
                break
            start, duplicate = _window_start_for_signal(signal, window_seconds, max_start, used_starts)
            selected.append({**signal, "window_start": start, "duplicate_window_unavoidable": duplicate})
            category_counts[category] += 1

    baseline_index = 0
    # A generously sized deterministic pool avoids duplicate positions on normal
    # feature-length videos while retaining a defined outcome for very short ones.
    while len(selected) < target_windows:
        signal = _baseline_signal(seed, baseline_index, effective_duration, window_seconds)
        baseline_index += 1
        start, duplicate = _window_start_for_signal(signal, window_seconds, max_start, used_starts)
        selected.append({**signal, "window_start": start, "duplicate_window_unavoidable": duplicate})

    selected.sort(key=lambda signal: (signal["window_start"], signal["category"], _stable_key(seed, signal["category"], signal)))
    windows: list[dict[str, Any]] = []
    for index, signal in enumerate(selected, 1):
        start = float(signal["window_start"])
        end = min(effective_duration, start + window_seconds)
        windows.append({
            "window_id": f"review_{index:03}",
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(max(0.0, end - start), 3),
            "category": signal["category"],
            "selection_reason": signal["reason"],
            "duplicate_window_unavoidable": signal["duplicate_window_unavoidable"],
            "selection_evidence": signal["evidence"],
            "rows": {
                "qwen_source": _window_rows(source_rows, start, end),
                "whisper_evidence": _window_rows(whisper_rows + disagreement_rows, start, end),
                "final": _window_rows(final_rows, start, end),
                "external_reference": _window_rows(external_rows, start, end),
            },
        })

    return {
        "schema_version": "1.0",
        "seed": seed,
        "window_seconds": window_seconds,
        "video_duration_seconds": round(effective_duration, 3),
        "video_path": str(Path(video_path).resolve()) if video_path else None,
        "source_evidence_label": source_evidence_label,
        "quality_gate": _json_safe(quality_gate) if quality_gate is not None else None,
        "target_windows": target_windows,
        "input_counts": {
            "final": len(final_rows),
            "qwen_source": len(source_rows),
            "whisper_evidence": len(whisper_rows),
            "engine_disagreements": len(disagreement_rows),
            "external_reference": len(external_rows),
        },
        "stratification": {
            "per_category_quota": _CATEGORY_QUOTA,
            "category_counts": category_counts,
            "baseline_count": sum(window["category"] == "deterministic_baseline" for window in windows),
        },
        "unresolved_presentation_exceptions": presentation_exceptions,
        "windows": windows,
    }


def write_review_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write a human-readable, deterministic manifest without altering cues."""

    path.write_text(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _time_range(row: Mapping[str, Any]) -> str:
    return f"{_number(row.get('start')):.3f} – {_number(row.get('end')):.3f}"


def _cell(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return ""
    metadata = row.get("metadata", {})
    note = ""
    if metadata:
        note = f"<div class='meta'>{escape(json.dumps(metadata, ensure_ascii=False, sort_keys=True))}</div>"
    return f"<div class='time'>{escape(_time_range(row))}</div><div class='text'>{escape(str(row.get('text', '')))}</div>{note}"


def render_review_report(manifest: Mapping[str, Any]) -> str:
    """Render one portable HTML document from :func:`build_review_manifest`."""

    video_path = manifest.get("video_path")
    video_uri = Path(str(video_path)).resolve().as_uri() if video_path else ""
    source_evidence_label = escape(str(manifest.get("source_evidence_label", "Qwen/원문 근거")))
    quality_gate = manifest.get("quality_gate")
    quality_summary = ""
    if isinstance(quality_gate, Mapping):
        gate_status = escape(_ui_label(quality_gate.get("status", "unknown"), _STATUS_LABELS))
        reasons = quality_gate.get("reasons", [])
        if isinstance(reasons, list):
            reason_text = ", ".join(
                f"{_ui_label(item.get('code', 'unknown'), _QUALITY_REASON_LABELS)} ({item.get('count', 0)})"
                for item in reasons if isinstance(item, Mapping)
            ) or "없음"
        else:
            reason_text = "잘못된 형식"
        quality_summary = (
            f"<section class='quality'><h2>품질 게이트: {gate_status}</h2>"
            f"<p>남은 검수 사유: {escape(reason_text)}.</p></section>"
        )
    video_summary = (
        f"<p>원본 영상: <a href='{escape(video_uri, quote=True)}'>플레이어로 열기</a>. "
        "각 검수 구간에서도 해당 시각의 영상을 바로 재생할 수 있습니다.</p>"
        if video_uri else ""
    )
    sections: list[str] = []
    for window in manifest.get("windows", []):
        rows = window.get("rows", {})
        columns = [
            list(rows.get("qwen_source", [])),
            list(rows.get("whisper_evidence", [])),
            list(rows.get("final", [])),
            list(rows.get("external_reference", [])),
        ]
        body = "".join(
            "<tr>" + "".join(f"<td>{_cell(column[index] if index < len(column) else None)}</td>" for column in columns) + "</tr>"
            for index in range(max((len(column) for column in columns), default=0))
        ) or "<tr><td colspan='4' class='empty'>이 검수 구간과 겹치는 행이 없습니다.</td></tr>"
        evidence = escape(json.dumps(window.get("selection_evidence", {}), ensure_ascii=False, sort_keys=True, indent=2))
        duplicate_note = " <span class='warning'>중복 구간 불가피</span>" if window.get("duplicate_window_unavoidable") else ""
        category_label = escape(_ui_label(window.get("category", "baseline"), _CATEGORY_LABELS))
        reason_label = escape(_ui_label(window.get("selection_reason", ""), _REASON_LABELS))
        video_control = ""
        if video_uri:
            seek_uri = f"{video_uri}#t={float(window.get('start', 0)):.3f}"
            video_control = (
                "<details class='video'><summary>이 검수 구간 영상 재생</summary>"
                f"<video controls preload='none' src='{escape(seek_uri, quote=True)}'></video>"
                f"<p><a href='{escape(seek_uri, quote=True)}'>{float(window.get('start', 0)):.3f}초부터 열기</a>"
                f" (검수 구간 종료: {float(window.get('end', 0)):.3f}초).</p></details>"
            )
        sections.append(
            f"<section><h2>{escape(str(window.get('window_id', '검수')))} · "
            f"{float(window.get('start', 0)):.3f} – {float(window.get('end', 0)):.3f}</h2>"
            f"<p><b>{category_label}</b> · {reason_label}{duplicate_note}</p>"
            f"<details><summary>선정 근거 보기</summary><pre>{evidence}</pre></details>"
            f"{video_control}"
            f"<table><thead><tr><th>{source_evidence_label}</th><th>Whisper 근거/주 인식</th>"
            "<th>최종 자막</th><th>외부 참조 자막</th></tr></thead>"
            f"<tbody>{body}</tbody></table></section>"
        )
    exceptions = list(manifest.get("unresolved_presentation_exceptions", []))
    exception_section = ""
    if exceptions:
        exception_rows = "".join(
            "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in (
                str(item.get("cue_id", "")),
                _time_range(item),
                str(item.get("duration", "")),
                ", ".join(_ui_label(code, _QUALITY_REASON_LABELS) for code in item.get("codes", [])),
                str(item.get("text", "")),
            )) + "</tr>"
            for item in exceptions
        )
        exception_section = """<section><h2>해결되지 않은 표시 예외</h2>
<p>원문을 바꾸거나 근거 없는 시간을 만들지 않고는 표시 제한을 만족할 수 없는 큐입니다. 결과를 수락하기 전에 모든 행을 확인하세요.</p>
<table><thead><tr><th>큐</th><th>시간</th><th>초</th><th>사유</th><th>최종 자막</th></tr></thead><tbody>""" + exception_rows + "</tbody></table></section>"
    title = "일본어 자막 검수"
    return f"""<!doctype html>
<html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>{title}</title>
<style>
body{{font-family:system-ui,sans-serif;line-height:1.45;margin:1.5rem;color:#202124;background:#fafafa}}
section{{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:1rem;margin:1rem 0}}
h1,h2{{margin:.15rem 0 .6rem}} h2{{font-size:1.05rem}} table{{border-collapse:collapse;width:100%;table-layout:fixed}}
th,td{{border:1px solid #d0d7de;padding:.55rem;vertical-align:top;text-align:left}} th{{background:#f6f8fa}}
.time{{font:12px ui-monospace,monospace;color:#57606a;margin-bottom:.25rem}} .text{{white-space:pre-wrap;overflow-wrap:anywhere}}
.meta,pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:11px ui-monospace,monospace;color:#57606a}} .warning{{color:#9a6700}} .empty{{text-align:center;color:#57606a}}
.video{{margin:.75rem 0}} video{{display:block;max-width:100%;width:34rem;background:#111}}
.quality{{border-left:4px solid #9a6700}}
</style></head><body><h1>{title}</h1>
<p>검수 표본 {len(manifest.get('windows', []))}개 · 각 {escape(str(manifest.get('window_seconds', _DEFAULT_WINDOW_SECONDS)))}초</p>
{video_summary}{quality_summary}{exception_section}{''.join(sections) or '<p>생성된 검수 구간이 없습니다.</p>'}</body></html>"""


def write_review_report(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write the self-contained side-by-side human-review report."""

    path.write_text(render_review_report(manifest), encoding="utf-8")


def write_review_artifacts(
    manifest_path: Path,
    report_path: Path,
    final_cues: Iterable[Any],
    qwen_source: Iterable[Any],
    whisper_candidates: Iterable[Any] | None = None,
    engine_disagreements: Iterable[Any] | None = None,
    external_reference: Iterable[Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build and write both review artifacts, returning the exact manifest."""

    manifest = build_review_manifest(
        final_cues,
        qwen_source,
        whisper_candidates,
        engine_disagreements,
        external_reference,
        **kwargs,
    )
    write_review_manifest(manifest_path, manifest)
    write_review_report(report_path, manifest)
    return manifest
