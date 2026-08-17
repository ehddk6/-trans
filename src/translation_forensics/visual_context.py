from __future__ import annotations

"""Deterministic, evidence-safe selection of visual context frames.

This module indexes local image files and prepares attachment metadata only.  It
does not open a network connection or invoke a model.  A caller that performs
an external transfer is responsible for replacing the initial ``not-sent``
receipt with the provider's actual call receipt.
"""

import hashlib
import json
import math
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ALLOWED_VISUAL_SLOTS = (
    "speaker",
    "addressee",
    "deictic_location",
    "on_screen_text",
    "scene_continuity",
)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
TRANSFER_MODES = frozenset({"metadata-only", "targeted"})

_TITLE_CODE_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{2,10})-(\d{2,6})(?![A-Z0-9])", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(r"(?P<seconds>\d+(?:\.\d+)?)s(?=(?:[_.-]|$))", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class VisualFrame:
    path: Path
    timestamp_seconds: float
    sha256: str


@dataclass(frozen=True, slots=True)
class SelectedVisualFrame:
    frame: VisualFrame
    selection_reason: str
    anchor_seconds: float
    distance_seconds: float


def parse_capture_timestamp(path: str | Path) -> float:
    """Parse the seconds suffix in names such as ``sub_0231_7181.1s.jpg``.

    The subtitle-like ``sub_0231`` portion is deliberately ignored.  If more
    than one seconds token is present, the final token is used because legacy
    captures place the actual frame timestamp immediately before the suffix.
    """

    matches = list(_TIMESTAMP_RE.finditer(Path(path).name))
    if not matches:
        raise ValueError(f"capture filename has no seconds timestamp: {Path(path).name}")
    seconds = float(matches[-1].group("seconds"))
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"capture timestamp must be finite and non-negative: {Path(path).name}")
    return seconds


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sort_key(frame: VisualFrame) -> tuple[float, str]:
    return frame.timestamp_seconds, str(frame.path).casefold()


def deduplicate_frames(frames: Iterable[VisualFrame]) -> list[VisualFrame]:
    """Return one deterministic representative for each pixel SHA-256."""

    unique: list[VisualFrame] = []
    seen_hashes: set[str] = set()
    for frame in sorted(frames, key=_frame_sort_key):
        if frame.sha256 in seen_hashes:
            continue
        seen_hashes.add(frame.sha256)
        unique.append(frame)
    return unique


def index_legacy_captures(capture_dir: str | Path, *, recursive: bool = False) -> list[VisualFrame]:
    """Index timestamped legacy images and remove byte-identical duplicates."""

    root = Path(capture_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths = root.rglob("*") if recursive else root.iterdir()
    frames: list[VisualFrame] = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            timestamp = parse_capture_timestamp(path)
        except ValueError:
            continue
        frames.append(VisualFrame(path=path.resolve(), timestamp_seconds=timestamp, sha256=sha256_file(path)))
    return deduplicate_frames(frames)


def _unit_anchors(start_seconds: float, end_seconds: float, max_frames: int) -> list[tuple[str, float]]:
    middle = start_seconds + ((end_seconds - start_seconds) / 2)
    if max_frames <= 0:
        return []
    if max_frames == 1:
        return [("nearest_to_unit_middle", middle)]
    if max_frames == 2:
        return [("nearest_to_unit_start", start_seconds), ("nearest_to_unit_end", end_seconds)]
    return [
        ("nearest_to_unit_start", start_seconds),
        ("nearest_to_unit_middle", middle),
        ("nearest_to_unit_end", end_seconds),
    ]


def select_nearest_frames(
    frames: Iterable[VisualFrame],
    *,
    start_seconds: float,
    end_seconds: float,
    max_frames: int = 3,
) -> list[SelectedVisualFrame]:
    """Select unique frames nearest the unit's start, middle, and end anchors."""

    if not math.isfinite(start_seconds) or not math.isfinite(end_seconds):
        raise ValueError("unit timestamps must be finite")
    if start_seconds < 0 or end_seconds < start_seconds:
        raise ValueError("unit timestamps must satisfy 0 <= start_seconds <= end_seconds")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 0:
        raise ValueError("max_frames must be a non-negative integer")

    pool = sorted(frames, key=_frame_sort_key)
    selected: list[SelectedVisualFrame] = []
    used_hashes: set[str] = set()
    for reason, anchor in _unit_anchors(start_seconds, end_seconds, max_frames):
        eligible = [frame for frame in pool if frame.sha256 not in used_hashes]
        if not eligible:
            break
        nearest = min(
            eligible,
            key=lambda frame: (
                abs(frame.timestamp_seconds - anchor),
                frame.timestamp_seconds,
                str(frame.path).casefold(),
            ),
        )
        used_hashes.add(nearest.sha256)
        selected.append(
            SelectedVisualFrame(
                frame=nearest,
                selection_reason=reason,
                anchor_seconds=anchor,
                distance_seconds=abs(nearest.timestamp_seconds - anchor),
            )
        )
    return selected


def _title_codes(value: str | Path) -> set[str]:
    text = str(value).upper()
    return {f"{match.group(1).upper()}-{match.group(2)}" for match in _TITLE_CODE_RE.finditer(text)}


def title_code_matches(expected_title: str, candidate: str | Path) -> bool:
    """Match one exact hyphenated work code without fuzzy prefix correction.

    Ambiguous strings containing multiple title codes are rejected.  In
    particular, ``ABP-169`` never matches ``ABF-169``.
    """

    expected_codes = _title_codes(expected_title)
    candidate_codes = _title_codes(candidate)
    return len(expected_codes) == 1 and len(candidate_codes) == 1 and expected_codes == candidate_codes


def _visual_slots(values: Mapping[str, Any] | None) -> dict[str, Any]:
    supplied = dict(values or {})
    unsupported = sorted(set(supplied) - set(ALLOWED_VISUAL_SLOTS))
    if unsupported:
        raise ValueError(f"unsupported visual slots: {', '.join(unsupported)}")
    slots = {name: supplied.get(name) for name in ALLOWED_VISUAL_SLOTS}
    try:
        json.dumps(slots, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("visual slot values must be JSON-serializable") from exc
    return slots


def _selected_frame_record(selected: SelectedVisualFrame) -> dict[str, Any]:
    return {
        "path": str(selected.frame.path),
        "timestamp_seconds": selected.frame.timestamp_seconds,
        "sha256": selected.frame.sha256,
        "selection_reason": selected.selection_reason,
        "anchor_seconds": selected.anchor_seconds,
        "distance_seconds": selected.distance_seconds,
    }


def build_visual_context_record(
    *,
    title_id: str,
    unit_id: str,
    start_seconds: float,
    end_seconds: float,
    frames: Iterable[VisualFrame],
    visual_slots: Mapping[str, Any] | None = None,
    transfer_mode: str = "metadata-only",
    max_frames: int = 3,
) -> dict[str, Any]:
    """Build one JSONL-ready visual-context record with an honest receipt."""

    if transfer_mode not in TRANSFER_MODES:
        raise ValueError(f"transfer_mode must be one of {sorted(TRANSFER_MODES)}")
    selected = select_nearest_frames(
        frames,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        max_frames=max_frames,
    )
    record: dict[str, Any] = {
        "schema_name": "translation-forensics/visual-context",
        "schema_version": "1",
        "title_id": title_id,
        "unit_id": unit_id,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "visual_slots": _visual_slots(visual_slots),
        "frames": [_selected_frame_record(item) for item in selected],
        "transfer_mode": transfer_mode,
        "external_transfer_receipt": {
            "status": "not-sent",
            "external_transfer": False,
            "pixel_transfer_count": 0,
            "transferred_frame_sha256": [],
            "provider": None,
            "request_id": None,
        },
    }
    # Fail here instead of writing a malformed JSONL row later.
    json.dumps(record, ensure_ascii=False)
    return record


def prepare_model_attachments(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return hash-verified local descriptors without transmitting image bytes."""

    if record.get("transfer_mode") == "metadata-only":
        return []
    if record.get("transfer_mode") != "targeted":
        raise ValueError("visual record has an unsupported transfer_mode")

    attachments: list[dict[str, str]] = []
    for frame in record.get("frames", []):
        if not isinstance(frame, Mapping):
            raise ValueError("visual record frame must be an object")
        path = Path(str(frame.get("path", ""))).expanduser().resolve()
        digest = str(frame.get("sha256", ""))
        reason = str(frame.get("selection_reason", ""))
        if not path.is_file():
            raise FileNotFoundError(path)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid frame SHA-256: {digest}")
        if sha256_file(path) != digest:
            raise ValueError(f"frame changed after visual-context indexing: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        attachments.append(
            {
                "path": str(path),
                "mime_type": mime_type,
                "sha256": digest,
                "selection_reason": reason,
            }
        )
    return attachments
