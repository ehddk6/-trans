from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from subtitle_pipeline.text import wrap_two_lines

from .srt import JAPANESE_RE, SRTError, SubtitleBlock, compare_structure, has_japanese, parse_srt, write_srt


BANNED_SOURCE_JA = (
    "ご視聴ありがとうございました",
    "次の動画でお会いしましょう",
    "おやすみなさい",
)
BANNED_SOURCE_KO = (
    "시청해 주셔서 감사",
    "다음 영상에서 만나",
    "잘 자요",
    "협조해 주셔서 감사",
    "수고하셨습니다",
)
FORBIDDEN_OUTPUT_KO = (
    "시청해 주셔서 감사",
    "다음 영상에서 만나",
)


class ImprovementError(RuntimeError):
    """Raised when an input or output violates the improvement contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def overlap_seconds(left: SubtitleBlock, right: SubtitleBlock) -> float:
    return max(0.0, min(left.end_seconds, right.end_seconds) - max(left.start_seconds, right.start_seconds))


def overlap_score(target: SubtitleBlock, source: SubtitleBlock) -> float:
    overlap = overlap_seconds(target, source)
    if overlap <= 0:
        return 0.0
    target_ratio = overlap / target.duration if target.duration else 0.0
    source_ratio = overlap / source.duration if source.duration else 0.0
    return max(target_ratio, source_ratio)


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def select_overlap_translation(
    target: SubtitleBlock,
    source_ja: list[SubtitleBlock],
    source_ko: list[SubtitleBlock],
    *,
    minimum_score: float = 0.5,
) -> tuple[str | None, list[int]]:
    """Select deduplicated KO fragments with strong temporal overlap.

    Source JA and KO are required to have identical structure. Known periodic
    boilerplate and any KO fragment that still contains Japanese are excluded.
    """
    structure = compare_structure(source_ja, source_ko)
    if not structure["pass"]:
        raise ImprovementError(f"JA/KO source structure mismatch: {structure['issues'][:3]}")

    fragments: list[str] = []
    source_blocks: list[int] = []
    seen: set[str] = set()
    for ja_block, ko_block in zip(source_ja, source_ko):
        if overlap_score(target, ja_block) < minimum_score:
            continue
        ja_text = ja_block.text.replace("\n", " ").strip()
        ko_text = ko_block.text.replace("\n", " ").strip()
        if not ko_text or has_japanese(ko_text):
            continue
        if _contains_any(ja_text, BANNED_SOURCE_JA) or _contains_any(ko_text, BANNED_SOURCE_KO):
            continue
        normalized = " ".join(ko_text.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        fragments.append(normalized)
        source_blocks.append(ja_block.number)
    return (" ".join(fragments) if fragments else None), source_blocks


def _replacement_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if not lines:
        raise ImprovementError("replacement text is empty")
    if len(lines) == 2:
        return "\n".join(lines)
    flattened = " ".join(lines)
    return "\n".join(wrap_two_lines(flattened, 22, 42))


def residual_stats(blocks: Iterable[SubtitleBlock]) -> dict[str, int]:
    block_count = 0
    line_count = 0
    character_count = 0
    for block in blocks:
        block_has_residual = False
        for line in block.lines:
            matches = JAPANESE_RE.findall(line)
            if matches:
                line_count += 1
                character_count += len(matches)
                block_has_residual = True
        if block_has_residual:
            block_count += 1
    return {"blocks": block_count, "lines": line_count, "characters": character_count}


def residual_file_stats(path: Path) -> dict[str, int]:
    """Count Japanese-regex residue without requiring an otherwise valid SRT.

    A few untouched playback files contain pre-existing malformed timecodes. The
    residue audit must still cover them without silently repairing their timing.
    """
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    raw_blocks = re.split(r"\n\s*\n", text.strip("\n")) if text.strip() else []
    block_count = 0
    line_count = 0
    character_count = 0
    for raw_block in raw_blocks:
        block_has_residual = False
        for line in raw_block.splitlines():
            matches = JAPANESE_RE.findall(line)
            if matches:
                line_count += 1
                character_count += len(matches)
                block_has_residual = True
        if block_has_residual:
            block_count += 1
    return {"blocks": block_count, "lines": line_count, "characters": character_count}


def discover_common_targets(
    source_root: Path,
    target_root: Path,
    aliases: dict[str, str],
) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    title_dirs = sorted(
        path for path in source_root.iterdir() if path.is_dir() and any(path.glob("*.srt"))
    )
    for title_dir in title_dirs:
        target_name = aliases.get(title_dir.name, f"{title_dir.name}.srt")
        target = target_root / target_name
        if not target.is_file():
            raise ImprovementError(f"mapped target is missing: {title_dir.name} -> {target}")
        mapping[title_dir.name] = target
    if not mapping:
        raise ImprovementError(f"no title directories found under source root: {source_root}")
    return mapping


def _load_config(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ImprovementError(f"could not load config: {path}") from exc
    if data.get("schema_version") != 1 or not isinstance(data.get("files"), dict):
        raise ImprovementError("unsupported improvement config")
    return data


def _override_map(file_config: dict[str, Any]) -> dict[int, dict[str, str]]:
    overrides: dict[int, dict[str, str]] = {}
    default_method = str(file_config.get("default_method", "manual-text-override"))
    for raw_number, raw_value in file_config.get("overrides", {}).items():
        number = int(raw_number)
        if isinstance(raw_value, str):
            overrides[number] = {"text": raw_value, "method": default_method}
        elif isinstance(raw_value, dict) and isinstance(raw_value.get("text"), str):
            overrides[number] = {
                "text": raw_value["text"],
                "method": str(raw_value.get("method", default_method)),
                "reason": str(raw_value.get("reason", "")),
            }
        else:
            raise ImprovementError(f"invalid override for block {number}")
    return overrides


def _improve_file(
    *,
    input_path: Path,
    output_path: Path,
    file_config: dict[str, Any],
    source_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_hash = str(file_config["sha256"]).lower()
    actual_hash = sha256_file(input_path)
    if actual_hash != expected_hash:
        raise ImprovementError(
            f"input hash mismatch for {input_path.name}: expected {expected_hash}, got {actual_hash}"
        )

    blocks, _, _ = parse_srt(input_path)
    residual_numbers = {block.number for block in blocks if has_japanese(block.text)}
    overrides = _override_map(file_config)
    residual_number_counts = {
        number: sum(block.number == number and has_japanese(block.text) for block in blocks)
        for number in residual_numbers
    }
    ambiguous_residuals = sorted(number for number, count in residual_number_counts.items() if count != 1)
    if ambiguous_residuals:
        raise ImprovementError(
            f"residual block numbers are ambiguous in {input_path.name}: {ambiguous_residuals}"
        )
    unknown_overrides = sorted(set(overrides) - residual_numbers)
    if unknown_overrides:
        raise ImprovementError(f"override blocks are no longer residual in {input_path.name}: {unknown_overrides}")

    mode = str(file_config.get("mode", "manual"))
    source_ja: list[SubtitleBlock] = []
    source_ko: list[SubtitleBlock] = []
    if mode == "hybrid-overlap":
        source_ja, _, _ = parse_srt(source_root / str(file_config["source_ja"]))
        source_ko, _, _ = parse_srt(source_root / str(file_config["source_ko"]))
        source_structure = compare_structure(source_ja, source_ko)
        if not source_structure["pass"]:
            raise ImprovementError(f"source pair structure mismatch for {input_path.name}")
    elif mode != "manual":
        raise ImprovementError(f"unsupported mode for {input_path.name}: {mode}")

    if mode == "manual" and residual_numbers != set(overrides):
        missing = sorted(residual_numbers - set(overrides))
        raise ImprovementError(f"manual overrides do not cover every residual in {input_path.name}: {missing}")

    ledger: list[dict[str, Any]] = []
    replacements: dict[int, str] = {}
    for block in blocks:
        if not has_japanese(block.text):
            continue
        number = block.number
        if number in overrides:
            choice = overrides[number]
            replacement = _replacement_text(choice["text"])
            method = choice["method"]
            source_blocks: list[int] = []
            reason = choice.get("reason", "")
        else:
            candidate, source_blocks = select_overlap_translation(
                block,
                source_ja,
                source_ko,
                minimum_score=float(file_config.get("minimum_overlap_score", 0.5)),
            )
            if candidate is None:
                raise ImprovementError(f"no safe overlap or override for {input_path.name} block {number}")
            replacement = _replacement_text(candidate)
            method = "videos-overlap"
            reason = "strong time overlap; boilerplate and Japanese-bearing source cues excluded"
        if has_japanese(replacement):
            raise ImprovementError(f"replacement still contains Japanese: {input_path.name} block {number}")
        replacements[number] = replacement
        ledger.append(
            {
                "file": input_path.name,
                "block": number,
                "timecode": f"{block.start} --> {block.end}",
                "before": block.text,
                "after": replacement,
                "method": method,
                "source_blocks": source_blocks,
                "reason": reason,
            }
        )

    improved = [replace(block, text=replacements.get(block.number, block.text)) for block in blocks]
    if any(not block.text.strip() for block in improved):
        raise ImprovementError(f"empty block after improvement: {input_path.name}")
    if residual_stats(improved)["characters"]:
        raise ImprovementError(f"Japanese residual remains after improvement: {input_path.name}")
    structure = compare_structure(blocks, improved)
    if not structure["pass"]:
        raise ImprovementError(f"structure changed in memory: {input_path.name}")

    write_srt(output_path, improved)
    reparsed, encoding, newline = parse_srt(output_path)
    output_structure = compare_structure(blocks, reparsed)
    if not output_structure["pass"]:
        raise ImprovementError(f"written structure changed: {input_path.name}")
    return ledger, {
        "file": input_path.name,
        "input_sha256": actual_hash,
        "output_sha256": sha256_file(output_path),
        "changed_blocks": len(ledger),
        "residual_before": residual_stats(blocks),
        "residual_after": residual_stats(reparsed),
        "structure_pass": True,
        "empty_blocks": sum(not block.text.strip() for block in reparsed),
        "encoding": encoding,
        "newline": newline,
    }


def run_improvement(
    *,
    target_root: Path,
    source_root: Path,
    output_root: Path,
    config_path: Path,
    resume_incomplete: bool = False,
    refresh_generated: bool = False,
) -> dict[str, Any]:
    target_root = target_root.resolve()
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not target_root.is_dir() or not source_root.is_dir():
        raise ImprovementError("target and source roots must both exist")
    completed_output = output_root.exists() and (output_root / "qa_report.json").exists()
    if output_root.exists() and not (resume_incomplete or refresh_generated):
        raise ImprovementError(f"output path already exists: {output_root}")
    if completed_output and not refresh_generated:
        raise ImprovementError(f"refusing to replace a completed output: {output_root}")
    if output_root in (target_root, source_root):
        raise ImprovementError("output root must be separate from both input roots")

    config = _load_config(config_path)
    aliases = {str(key): str(value) for key, value in config.get("title_aliases", {}).items()}
    common = discover_common_targets(source_root, target_root, aliases)
    input_files = sorted(target_root.rglob("*.srt"))
    if not input_files:
        raise ImprovementError(f"no SRT files found in target root: {target_root}")
    input_hashes = {path.relative_to(target_root).as_posix(): sha256_file(path) for path in input_files}

    configured = config["files"]
    configured_names = set(configured)
    existing_names = {path.relative_to(target_root).as_posix() for path in input_files}
    missing_configured = sorted(configured_names - existing_names)
    if missing_configured:
        raise ImprovementError(f"configured target files are missing: {missing_configured}")

    if output_root.exists():
        if completed_output:
            try:
                previous_report = json.loads((output_root / "qa_report.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ImprovementError("could not verify the existing generated output") from exc
            expected_roots = (str(target_root), str(source_root), str(output_root))
            observed_roots = (
                previous_report.get("target_root"),
                previous_report.get("source_root"),
                previous_report.get("output_root"),
            )
            if observed_roots != expected_roots:
                raise ImprovementError("existing generated output belongs to different roots")
        allowed_metadata = {"README.md", "change_ledger.jsonl", "qa_report.json"} if completed_output else set()
        unexpected_partial_files = [
            path.relative_to(output_root).as_posix()
            for path in output_root.rglob("*")
            if path.is_file()
            and path.relative_to(output_root).as_posix() not in existing_names | allowed_metadata
        ]
        if unexpected_partial_files:
            raise ImprovementError(
                f"incomplete output contains unexpected files: {unexpected_partial_files[:5]}"
            )

    output_root.mkdir(parents=True, exist_ok=resume_incomplete or refresh_generated)
    for input_path in input_files:
        relative = input_path.relative_to(target_root)
        output_path = output_root / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)

    ledger: list[dict[str, Any]] = []
    changed_reports: list[dict[str, Any]] = []
    for relative_name in sorted(configured):
        input_path = target_root / Path(relative_name)
        output_path = output_root / Path(relative_name)
        file_ledger, file_report = _improve_file(
            input_path=input_path,
            output_path=output_path,
            file_config=configured[relative_name],
            source_root=source_root,
        )
        ledger.extend(file_ledger)
        changed_reports.append(file_report)

    after_input_hashes = {path.relative_to(target_root).as_posix(): sha256_file(path) for path in input_files}
    if input_hashes != after_input_hashes:
        raise ImprovementError("one or more original target files changed during the run")

    output_files = sorted(output_root.rglob("*.srt"))
    output_names = {path.relative_to(output_root).as_posix() for path in output_files}
    if output_names != existing_names:
        raise ImprovementError("output SRT inventory does not match the input inventory")
    unchanged_names = sorted(existing_names - configured_names)
    byte_identity_failures = [
        name for name in unchanged_names if sha256_file(target_root / name) != sha256_file(output_root / name)
    ]
    if byte_identity_failures:
        raise ImprovementError(f"unchanged outputs are not byte-identical: {byte_identity_failures[:5]}")

    common_before: dict[str, dict[str, int]] = {}
    common_after: dict[str, dict[str, int]] = {}
    preexisting_parse_failures: list[dict[str, str]] = []
    for title, input_path in common.items():
        relative = input_path.relative_to(target_root)
        output_path = output_root / relative
        common_before[title] = residual_file_stats(input_path)
        common_after[title] = residual_file_stats(output_path)
        try:
            parse_srt(input_path)
        except SRTError as exc:  # Preserve and report unrelated malformed legacy files.
            preexisting_parse_failures.append(
                {"title": title, "file": input_path.name, "error": f"{type(exc).__name__}: {exc}"}
            )
    residual_before = {
        key: sum(item[key] for item in common_before.values()) for key in ("blocks", "lines", "characters")
    }
    residual_after = {
        key: sum(item[key] for item in common_after.values()) for key in ("blocks", "lines", "characters")
    }
    if residual_after["characters"] != 0:
        raise ImprovementError(f"Japanese residual remains in common playback files: {residual_after}")

    controls: dict[str, Any] = {}
    for name in ("ABF-196.srt", "PRED-879.srt"):
        if name not in existing_names:
            raise ImprovementError(f"known-regression control is missing: {name}")
        identical = sha256_file(target_root / name) == sha256_file(output_root / name)
        controls[name] = {"byte_identical": identical}
        if not identical:
            raise ImprovementError(f"known-regression control changed: {name}")

    forbidden_hits: list[dict[str, str]] = []
    for name in ("ABF-196.srt", "IPZZ-856.srt"):
        text = (output_root / name).read_text(encoding="utf-8-sig")
        for pattern in FORBIDDEN_OUTPUT_KO:
            if pattern in text:
                forbidden_hits.append({"file": name, "pattern": pattern})
    if forbidden_hits:
        raise ImprovementError(f"forbidden source boilerplate was introduced: {forbidden_hits}")

    common_relative_names = {
        path.relative_to(target_root).as_posix() for path in common.values()
    }
    video_files = sorted(
        path for path in target_root.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4"
    )
    playback_matches: list[dict[str, str]] = []
    missing_playback_srt: list[str] = []
    for video_path in video_files:
        video_relative = video_path.relative_to(target_root)
        srt_relative = video_relative.with_suffix(".srt")
        if srt_relative.as_posix() in existing_names:
            playback_matches.append(
                {"video": video_relative.as_posix(), "srt": srt_relative.as_posix()}
            )
        else:
            missing_playback_srt.append(video_relative.as_posix())
    comparison_covered_video_srt = sorted(
        match["srt"] for match in playback_matches if match["srt"] in common_relative_names
    )
    outside_comparison_video_srt = sorted(
        match["srt"] for match in playback_matches if match["srt"] not in common_relative_names
    )

    ledger_path = output_root / "change_ledger.jsonl"
    ledger_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ledger),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_root": str(target_root),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "config": str(config_path.resolve()),
        "inventory": {
            "input_srt_files": len(input_files),
            "output_srt_files": len(output_files),
            "common_playback_files": len(common),
            "configured_changed_files": len(configured_names),
            "unchanged_byte_identical_files": len(unchanged_names),
            "unchanged_byte_identity_failures": byte_identity_failures,
        },
        "playback_selection": {
            "rule": "For each MP4, load the ordinary SRT with the exact same relative stem; do not select .ja.srt or .viewer_ja.srt auxiliaries.",
            "video_files": len(video_files),
            "exact_name_matches": len(playback_matches),
            "missing_exact_match": missing_playback_srt,
            "matches": playback_matches,
            "comparison_covered_video_srt": comparison_covered_video_srt,
            "outside_videos_comparison_video_srt": outside_comparison_video_srt,
        },
        "common_playback_mapping": [
            {
                "source_title": title,
                "target_srt": path.relative_to(target_root).as_posix(),
            }
            for title, path in sorted(common.items())
        ],
        "changes": {
            "changed_blocks": len(ledger),
            "files": changed_reports,
        },
        "common_playback_japanese_residual": {
            "before": residual_before,
            "after": residual_after,
        },
        "known_regression_controls": controls,
        "forbidden_boilerplate_hits": forbidden_hits,
        "preexisting_common_parse_failures": preexisting_parse_failures,
        "original_input_hashes_preserved": input_hashes == after_input_hashes,
        "limitations": [
            "Japanese-character absence is not proof of semantic translation accuracy.",
            "This run used subtitle text and timing evidence only; it did not listen to the source audio.",
            "Only the playback files mapped from the Videos title directories were evaluated for zero residual.",
        ],
    }
    (output_root / "qa_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_root / "README.md").write_text(
        "# 비디오 자막 개선본\n\n"
        f"- 원본: `{target_root}`\n"
        f"- 비교 근거: `{source_root}`\n"
        f"- SRT 수: {len(output_files)}개\n"
        f"- 수정 파일/블록: {len(configured_names)}개 / {len(ledger)}블록\n"
        f"- 공통 재생용 자막 일본어 문자: {residual_before['characters']} → {residual_after['characters']}\n"
        f"- 원본 파일 변경: 없음\n\n"
        "## 재생할 자막 고르기\n\n"
        "이 폴더에는 자막만 있습니다. 플레이어에서 외부 자막을 열고, 재생 중인 MP4와 파일명이 정확히 같은 일반 `.srt`를 선택하세요. "
        "예: `IPZZ-856.mp4` → `IPZZ-856.srt`. `.ja.srt`, `.viewer_ja.srt`처럼 일본어·검토용 접미사가 붙은 파일은 재생용 한국어 자막이 아닙니다.\n\n"
        f"현재 원본 폴더의 MP4 {len(video_files)}개는 모두 같은 이름의 SRT가 있습니다. 그중 {len(comparison_covered_video_srt)}개는 `Videos` 비교 범위에 포함됐고, "
        f"나머지 {len(outside_comparison_video_srt)}개는 비교 근거가 없어 원본 그대로 복사했습니다. 정확한 파일 목록은 `qa_report.json`의 "
        "`playback_selection`과 `common_playback_mapping`에서 확인할 수 있습니다.\n\n"
        "원본 폴더의 자막은 자동 교체하지 않았습니다. 한 작품씩 시험할 때는 이 폴더의 SRT를 플레이어에서 직접 불러오세요. "
        "영구 교체는 기존 SRT를 백업한 뒤 같은 이름의 개선본을 복사하는 별도 작업입니다.\n\n"
        "## 변경 범위\n\n"
        f"수정된 파일은 {', '.join(sorted(configured_names))}입니다. 나머지 {len(unchanged_names)}개 SRT는 원본과 바이트 단위로 같습니다.\n\n"
        f"기존부터 SRT 규격 오류가 있던 무수정 파일은 {len(preexisting_parse_failures)}개입니다: "
        f"{', '.join(item['file'] for item in preexisting_parse_failures) or '없음'}.\n\n"
        "변경 세부 내용은 `change_ledger.jsonl`, 자동 검증 결과는 `qa_report.json`을 확인하세요. "
        "일본어 문자가 없어졌다는 사실만으로 번역 의미의 완전한 정확성이 입증되지는 않으며, 이번 작업에는 음성 직접 청취가 포함되지 않았습니다.\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
