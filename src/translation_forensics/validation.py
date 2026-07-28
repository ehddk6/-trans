from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .srt import JAPANESE_RE, SRTError, SubtitleBlock, compare_structure, parse_srt


DEFAULT_CONFIG = {
    "max_lines": 2,
    "preferred_line_chars": 22,
    "max_line_chars_warning": 42,
    "preferred_cps_min": 12,
    "preferred_cps_max": 16,
    "placeholder_patterns": ["[불명]", "번역 불가", "알아들을 수 없"],
    "work_tag_patterns": ["[수정]", "[추정]", "[음성]", "[검토]", "[번역자]"],
    "broken_char_patterns": ["�", "\ufffd"],
}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: str
    message: str
    block: int | None = None
    detail: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ValidationReport:
    path: str
    status: str
    issues: list[ValidationIssue]
    checked_items: list[str]
    unverified_items: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "status": self.status,
            "issues": [issue.as_dict() for issue in self.issues],
            "checked_items": self.checked_items,
            "unverified_items": self.unverified_items,
            "error_count": sum(issue.severity == "error" for issue in self.issues),
            "warning_count": sum(issue.severity == "warning" for issue in self.issues),
        }


def load_config(project_root: Path | None = None) -> dict[str, object]:
    config = dict(DEFAULT_CONFIG)
    if project_root:
        path = project_root / "config" / "validation.json"
        if path.exists():
            try:
                config.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
    return config


def _issue(code: str, severity: str, message: str, block: int | None = None, **detail: object) -> ValidationIssue:
    return ValidationIssue(code, severity, message, block, detail or None)


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _overlap_blocks(blocks: list[SubtitleBlock]) -> set[int]:
    result: set[int] = set()
    previous_end = -1.0
    for block in blocks:
        if block.start_seconds < previous_end:
            result.add(block.number)
        previous_end = max(previous_end, block.end_seconds)
    return result


def _block_structure_issues(blocks: list[SubtitleBlock], *, baseline_overlap_blocks: set[int]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    numbers = [block.number for block in blocks]
    if len(numbers) != len(set(numbers)):
        issues.append(_issue("duplicate_number", "error", "블록 번호가 중복됩니다."))
    expected = list(range(1, len(numbers) + 1))
    if numbers != expected:
        issues.append(_issue("number_sequence", "error", "블록 번호가 1부터 연속하지 않습니다.", detail={"actual": numbers[:20], "expected_prefix": expected[:20]}))
    previous_end = -1.0
    for block in blocks:
        if block.end_seconds < block.start_seconds:
            issues.append(_issue("reversed_time", "error", "종료 시각이 시작 시각보다 빠릅니다.", block.number))
        if block.start_seconds < previous_end:
            if block.number in baseline_overlap_blocks:
                issues.append(_issue("baseline_overlap", "warning", "구조 기준본에 이미 존재하는 시간 겹침입니다.", block.number))
            else:
                issues.append(_issue("time_order", "error", "구조 기준본에 없던 시간 겹침 또는 역전이 생겼습니다.", block.number))
        previous_end = max(previous_end, block.end_seconds)
    return issues


def validate_srt_file(path: Path, *, reference: list[SubtitleBlock] | None = None, project_root: Path | None = None) -> ValidationReport:
    issues: list[ValidationIssue] = []
    checked = ["SRT 파싱", "블록 번호·타임코드 구조", "기준본 겹침 구분", "UTF-8", "LF", "빈 자막", "일본어 잔존", "작업용 태그", "가독성", "연속 중복", "회귀 후보"]
    unverified: list[str] = []
    config = load_config(project_root)
    try:
        blocks, encoding, newline = parse_srt(path)
    except (OSError, SRTError) as exc:
        return ValidationReport(str(path), "fail", [_issue("parse", "error", str(exc))], checked, unverified)
    if encoding != "utf-8":
        issues.append(_issue("encoding", "error", "UTF-8 무 BOM 형식이 아닙니다.", encoding=encoding))
    if newline != "LF":
        issues.append(_issue("newline", "error", "LF 줄바꿈이 아닙니다."))
    baseline_overlaps = _overlap_blocks(reference) if reference is not None else _overlap_blocks(blocks)
    issues.extend(_block_structure_issues(blocks, baseline_overlap_blocks=baseline_overlaps))
    if reference is not None:
        diff = compare_structure(reference, blocks)
        if not diff["pass"]:
            issues.append(_issue("structure_mismatch", "error", "구조 기준본과 블록 수·번호·타임코드가 다릅니다.", detail=diff))

    placeholder_patterns = [str(x) for x in config.get("placeholder_patterns", [])]
    work_tags = [str(x) for x in config.get("work_tag_patterns", [])]
    broken_patterns = [str(x) for x in config.get("broken_char_patterns", [])]
    previous_text = ""
    seen: dict[str, int] = {}
    for block in blocks:
        text = block.text
        if not text.strip():
            issues.append(_issue("empty_text", "error", "빈 자막입니다.", block.number))
        if JAPANESE_RE.search(text):
            issues.append(_issue("japanese_residue", "error", "한국어 최종 자막에 일본어 문자가 남아 있습니다.", block.number))
        for pattern in placeholder_patterns:
            if pattern in text:
                issues.append(_issue("placeholder", "error", f"금지된 불확실성 표기가 남아 있습니다: {pattern}", block.number))
        for pattern in work_tags:
            if pattern in text:
                issues.append(_issue("work_tag", "error", f"작업용 태그가 남아 있습니다: {pattern}", block.number))
        if any(pattern in text for pattern in broken_patterns):
            issues.append(_issue("broken_character", "error", "깨진 문자가 있습니다.", block.number))
        if re.search(r" {2,}|\t", text):
            issues.append(_issue("abnormal_space", "warning", "비정상적인 공백 또는 탭이 있습니다.", block.number))
        if re.search(r"[!?！？。]{2,}|\.{3,}", text):
            issues.append(_issue("repeated_punctuation", "warning", "문장부호가 연속 반복됩니다.", block.number))
        if len(block.lines) > int(config.get("max_lines", 2)):
            issues.append(_issue("too_many_lines", "error", "자막이 3줄 이상입니다.", block.number))
        for line in block.lines:
            if len(line) > int(config.get("max_line_chars_warning", 42)):
                issues.append(_issue("long_line", "warning", "한 줄이 42자를 초과합니다.", block.number, length=len(line)))
        chars = len(re.sub(r"\s+", "", text))
        cps = chars / block.duration if block.duration > 0 else float("inf")
        if cps > float(config.get("preferred_cps_max", 16)):
            issues.append(_issue("high_cps", "warning", "초당 문자 수가 권장 범위를 초과합니다.", block.number, cps=round(cps, 2)))
        if block.duration >= 1.0 and chars < 2:
            issues.append(_issue("fragment_candidate", "warning", "의미 없는 반쪽 문장·고립 표현 후보입니다.", block.number))
        normalized = _normalized(text)
        seen[normalized] = seen.get(normalized, 0) + 1
        if previous_text and normalized and normalized == previous_text:
            issues.append(_issue("consecutive_duplicate", "warning", "앞 블록과 동일한 번역이 연속됩니다.", block.number))
        previous_text = normalized
        if text.rstrip().endswith(".") and "?" in text:
            issues.append(_issue("question_regression_candidate", "warning", "질문과 평서 표기가 섞인 후보입니다.", block.number))
    for value, count in seen.items():
        if value and count >= 3:
            issues.append(_issue("reused_candidate", "warning", "같은 번역 문장이 반복 재사용됩니다.", detail={"text": value, "count": count}))
    status = "fail" if any(issue.severity == "error" for issue in issues) else "warning" if issues else "pass"
    return ValidationReport(str(path), status, issues, checked, unverified)


def _safe_parse(path: Path) -> list[SubtitleBlock] | None:
    try:
        blocks, _, _ = parse_srt(path)
        return blocks
    except (OSError, SRTError):
        return None


def validate_pair(reference_path: Path, source_faithful: Path, viewer_natural: Path, *, project_root: Path | None = None) -> dict[str, object]:
    try:
        reference, _, _ = parse_srt(reference_path)
    except (OSError, SRTError) as exc:
        return {"status": "fail", "reference_error": str(exc), "source_faithful": None, "viewer_natural": None, "structure_same": False}
    source_report = validate_srt_file(source_faithful, reference=reference, project_root=project_root)
    viewer_report = validate_srt_file(viewer_natural, reference=reference, project_root=project_root)
    status = "fail" if "fail" in {source_report.status, viewer_report.status} else "warning" if "warning" in {source_report.status, viewer_report.status} else "pass"
    source_blocks = _safe_parse(source_faithful)
    viewer_blocks = _safe_parse(viewer_natural)
    structure_same = bool(
        source_blocks is not None
        and viewer_blocks is not None
        and compare_structure(reference, source_blocks)["pass"]
        and compare_structure(reference, viewer_blocks)["pass"]
    )
    return {
        "status": status,
        "reference": str(reference_path),
        "source_faithful": source_report.as_dict(),
        "viewer_natural": viewer_report.as_dict(),
        "structure_same": structure_same,
    }


def write_validation_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
