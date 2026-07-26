from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
import textwrap
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

SYSTEM_VERSION = "1.0.0"

SRT_BLOCK_RE = re.compile(
    r"(?ms)^\s*(\d+)\s*\n"
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*\n"
    r"(.*?)(?=\n\s*\n\s*\d+\s*\n|\Z)"
)
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
NUMBER_RE = re.compile(r"\d")
ELLIPSIS_RE = re.compile(r"(?:\.{3,}|…+)")
KOREAN_WORD_RE = re.compile(r"[가-힣]+|[A-Za-z]+|\d+")

KNOWN_REGRESSION_PATTERNS = {
    "placeholder": [r"\[불명\]", r"번역\s*불가", r"알아들을\s*수\s*없"],
    "formal_scene_mistranslation": [r"수고하셨습니다", r"쉬세요", r"안녕히\s*가세요"],
    "counter_classifier_error": [r"몇\s*마리"],
    "meta_hallucination": [r"보너스\s*장면", r"두\s*번째\s*자세", r"세\s*번째\s*자세"],
    "known_odd_phrase": [r"기후가\s*껄끄", r"수고하셨어요"],
}

QUESTION_WORDS = ["누구", "언제", "어디", "왜", "어떻게", "얼마", "몇 명", "몇 번", "뭐", "무엇"]
NUMBER_WORDS = [
    "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉", "열",
    "하나", "둘", "셋", "넷", "몇", "없", "많", "적",
]
NARRATION_ENDINGS = ["했다", "였다", "였다.", "되었다", "있었다", "않았다", "같았다", "버렸다"]
FORMAL_ENDINGS = ["습니다", "세요", "인가요", "나요", "까요", "예요", "이에요", "드립니다", "합니다"]
CASUAL_ENDINGS = ["해", "야", "지", "네", "거야", "했어", "좋아", "괜찮아", "맞아", "아니야"]
REACTION_TOKENS = ["아", "하아", "응", "어", "으", "앗", "좋아", "잠깐", "싫어", "아파"]
SPECIFIC_ACTION_TOKENS = [
    "핥", "넣", "빼", "안쪽", "엉덩", "가슴", "혀", "젖", "움직", "키스", "씻", "샤워",
]


@dataclass
class SubtitleBlock:
    number: int
    start: str
    end: str
    start_seconds: float
    end_seconds: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.001, self.end_seconds - self.start_seconds)

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines() or [""]


@dataclass
class AssetRecord:
    title: str
    srt_path: Path
    manifest_path: Path | None
    changes_path: Path | None
    stage: str


@dataclass
class ModelBundle:
    vectorizer: TfidfVectorizer
    scaler: StandardScaler
    classifier: LogisticRegression
    numeric_feature_names: list[str]
    threshold: float
    metrics: dict[str, Any]


def tc_to_seconds(value: str) -> float:
    h, m, rest = value.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def seconds_to_tc(value: float) -> str:
    value = max(0.0, value)
    total_ms = int(round(value * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(path: Path) -> list[SubtitleBlock]:
    text = path.read_text(encoding="utf-8-sig", errors="strict").replace("\r\n", "\n")
    blocks: list[SubtitleBlock] = []
    for match in SRT_BLOCK_RE.finditer(text.strip()):
        number = int(match.group(1))
        start = match.group(2)
        end = match.group(3)
        body = match.group(4).strip()
        blocks.append(
            SubtitleBlock(
                number=number,
                start=start,
                end=end,
                start_seconds=tc_to_seconds(start),
                end_seconds=tc_to_seconds(end),
                text=body,
            )
        )
    if not blocks:
        raise ValueError(f"SRT를 파싱하지 못했습니다: {path}")
    return blocks


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return KOREAN_WORD_RE.findall(text.lower())


def jaccard_tokens(left: str, right: str) -> float:
    a, b = set(tokenize(left)), set(tokenize(right))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def formality_score(text: str) -> float:
    formal = sum(text.count(token) for token in FORMAL_ENDINGS)
    casual = sum(text.count(token) for token in CASUAL_ENDINGS)
    total = formal + casual
    if total == 0:
        return 0.5
    return formal / total


def repetition_score(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    return max(counts.values()) / len(tokens)


def utterance_style(text: str) -> str:
    compact = normalize_text(text)
    if any(compact.endswith(ending) for ending in NARRATION_ENDINGS):
        return "내레이션형"
    if len(tokenize(compact)) <= 5 and any(token in compact for token in REACTION_TOKENS):
        return "반응형"
    score = formality_score(compact)
    if score >= 0.68:
        return "존댓말형"
    if score <= 0.32:
        return "반말형"
    return "중립·혼합형"


def numeric_features(blocks: list[SubtitleBlock]) -> tuple[np.ndarray, list[str]]:
    frequencies = Counter(normalize_text(block.text) for block in blocks)
    names = [
        "duration", "chars", "cps", "line_count", "max_line_length", "question_count",
        "exclamation_count", "ellipsis_count", "number_flag", "japanese_flag", "placeholder_flag",
        "formal_score", "repetition_score", "duplicate_frequency", "prev_similarity", "next_similarity",
        "formality_shift", "gap_before", "gap_after", "specificity_count", "dialogue_dash_count",
        "abnormal_short_long", "abnormal_long_short",
    ]
    rows = []
    for i, block in enumerate(blocks):
        text = block.text
        compact = normalize_text(text)
        chars = len(compact.replace(" ", ""))
        line_lengths = [len(line) for line in block.lines]
        prev_text = blocks[i - 1].text if i else ""
        next_text = blocks[i + 1].text if i + 1 < len(blocks) else ""
        prev_formality = formality_score(prev_text) if prev_text else formality_score(text)
        next_formality = formality_score(next_text) if next_text else formality_score(text)
        current_formality = formality_score(text)
        gap_before = max(0.0, block.start_seconds - blocks[i - 1].end_seconds) if i else 0.0
        gap_after = max(0.0, blocks[i + 1].start_seconds - block.end_seconds) if i + 1 < len(blocks) else 0.0
        rows.append([
            block.duration,
            chars,
            chars / block.duration,
            len(block.lines),
            max(line_lengths) if line_lengths else 0,
            text.count("?"),
            text.count("!"),
            len(ELLIPSIS_RE.findall(text)),
            float(bool(NUMBER_RE.search(text))),
            float(bool(JAPANESE_RE.search(text))),
            float("[불명]" in text or "번역 불가" in text),
            current_formality,
            repetition_score(text),
            frequencies[compact],
            jaccard_tokens(text, prev_text),
            jaccard_tokens(text, next_text),
            abs(current_formality - (prev_formality + next_formality) / 2),
            min(gap_before, 120.0),
            min(gap_after, 120.0),
            sum(text.count(token) for token in SPECIFIC_ACTION_TOKENS),
            sum(1 for line in block.lines if line.lstrip().startswith("-")),
            float(block.duration < 0.5 and chars > 15),
            float(block.duration > 25 and chars < 5),
        ])
    return np.asarray(rows, dtype=float), names


def combined_context_texts(blocks: list[SubtitleBlock]) -> list[str]:
    texts = []
    for i, block in enumerate(blocks):
        prev_text = blocks[i - 1].text if i else "<BOS>"
        next_text = blocks[i + 1].text if i + 1 < len(blocks) else "<EOS>"
        texts.append(f"현재 {normalize_text(block.text)} 이전 {normalize_text(prev_text)} 다음 {normalize_text(next_text)}")
    return texts


def train_model(baseline_srt: Path, changes_csv: Path, output_dir: Path) -> ModelBundle:
    blocks = parse_srt(baseline_srt)
    changes = pd.read_csv(changes_csv, encoding="utf-8-sig")
    changed = set(int(value) for value in changes["block"].tolist())
    labels = np.asarray([1 if block.number in changed else 0 for block in blocks], dtype=int)

    contexts = combined_context_texts(blocks)
    numeric, numeric_names = numeric_features(blocks)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=9000,
        sublinear_tf=True,
    )
    X_text = vectorizer.fit_transform(contexts)
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(numeric)
    X = hstack([X_text, csr_matrix(X_numeric)], format="csr")

    groups = np.asarray([(block.number - 1) // 50 for block in blocks], dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260724)
    oof = np.zeros(len(blocks), dtype=float)

    for train_idx, valid_idx in splitter.split(X, labels, groups=groups):
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=20260724,
        )
        model.fit(X[train_idx], labels[train_idx])
        oof[valid_idx] = model.predict_proba(X[valid_idx])[:, 1]

    thresholds = np.linspace(0.10, 0.90, 161)
    f2_scores = [fbeta_score(labels, oof >= threshold, beta=2) for threshold in thresholds]
    threshold = float(thresholds[int(np.argmax(f2_scores))])
    predictions = (oof >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    matrix = confusion_matrix(labels, predictions).tolist()
    metrics = {
        "system_version": SYSTEM_VERSION,
        "training_title": "ABF-303",
        "blocks": len(blocks),
        "positive_changed_blocks": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "validation": "5-fold StratifiedGroupKFold, contiguous 50-block groups",
        "roc_auc": float(roc_auc_score(labels, oof)),
        "average_precision": float(average_precision_score(labels, oof)),
        "threshold_max_f2": threshold,
        "f2": float(max(f2_scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": matrix,
        "warning": "동일 작품 내부의 교차검증 결과이며 다른 작품에 대한 일반화 성능을 보장하지 않습니다.",
    }

    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=20260724,
    )
    classifier.fit(X, labels)

    bundle = ModelBundle(
        vectorizer=vectorizer,
        scaler=scaler,
        classifier=classifier,
        numeric_feature_names=numeric_names,
        threshold=threshold,
        metrics=metrics,
    )
    joblib.dump({
        "vectorizer": vectorizer,
        "scaler": scaler,
        "classifier": classifier,
        "numeric_feature_names": numeric_names,
        "threshold": threshold,
        "metrics": metrics,
        "system_version": SYSTEM_VERSION,
    }, output_dir / "risk_model.joblib")
    (output_dir / "model_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    calibration = pd.DataFrame({
        "block": [block.number for block in blocks],
        "changed_label": labels,
        "oof_probability": oof,
        "predicted_at_threshold": predictions,
    })
    calibration.to_csv(output_dir / "ABF-303.model-calibration.csv", index=False, encoding="utf-8-sig")
    return bundle


def predict_probabilities(bundle: ModelBundle, blocks: list[SubtitleBlock]) -> np.ndarray:
    X_text = bundle.vectorizer.transform(combined_context_texts(blocks))
    numeric, _ = numeric_features(blocks)
    X_numeric = bundle.scaler.transform(numeric)
    X = hstack([X_text, csr_matrix(X_numeric)], format="csr")
    return bundle.classifier.predict_proba(X)[:, 1]


def read_optional_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def title_from_name(name: str) -> str | None:
    match = re.search(r"\b(?:ABF|ADN|SSIS|SONE)-\d{3}\b", name.upper())
    return match.group(0) if match else None


def stage_from_path(path: Path) -> str:
    lower = str(path).lower()
    if "final-photo-verified" in lower or "source-verified" in lower:
        return "원문·사진 검증본"
    if "photo-english-verified" in lower:
        return "화면 영문·사진 검증본"
    if "final-photo-guided-retranslated" in lower:
        return "사진 대조 재번역본"
    if "photo-guided-pass1" in lower:
        return "사진 대조 1차 교정본"
    return "기준본"


def srt_priority(path: Path) -> int:
    lower = str(path).lower()
    if "final-photo-verified" in lower:
        return 100
    if "photo-english-verified" in lower:
        return 95
    if "final-photo-guided-retranslated" in lower:
        return 90
    if "photo-guided-pass1" in lower:
        return 80
    return 10


def discover_assets(root: Path) -> list[AssetRecord]:
    candidates: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*.srt"):
        if any(part.startswith("subtitle_forensics") for part in path.parts):
            continue
        title = title_from_name(path.name)
        if title and ("photo" in str(path).lower() or "verified" in str(path).lower()):
            candidates[title].append(path)

    assets = []
    for title, paths in sorted(candidates.items()):
        srt_path = max(paths, key=lambda p: (srt_priority(p), p.stat().st_mtime))
        directory = srt_path.parent
        manifests = sorted(directory.glob(f"{title}*manifest.csv"))
        changes = sorted(directory.glob(f"{title}*changes.csv"))
        manifest = manifests[0] if manifests else None
        change = changes[0] if changes else None
        assets.append(
            AssetRecord(
                title=title,
                srt_path=srt_path,
                manifest_path=manifest,
                changes_path=change,
                stage=stage_from_path(srt_path),
            )
        )
    return assets


def load_error_memory(root: Path) -> pd.DataFrame:
    records = []
    seen = set()
    for path in root.rglob("*changes.csv"):
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        if not {"before", "after"}.issubset(frame.columns):
            continue
        for _, row in frame.iterrows():
            before = normalize_text(str(row.get("before", "")))
            after = normalize_text(str(row.get("after", "")))
            if not before or before == "nan" or before == after or len(before) < 3:
                continue
            key = (before, after)
            if key in seen:
                continue
            seen.add(key)
            records.append({
                "before": before,
                "after": after,
                "basis": str(row.get("basis", "")),
                "confidence": str(row.get("confidence", "")),
                "source_file": str(path),
            })
    return pd.DataFrame(records)


def build_error_memory_index(memory: pd.DataFrame, texts: list[str]):
    if memory.empty:
        return None, None, None
    corpus = memory["before"].tolist() + texts
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(corpus)
    memory_matrix = matrix[: len(memory)]
    text_matrix = matrix[len(memory) :]
    similarities = text_matrix @ memory_matrix.T
    return vectorizer, similarities, memory


def rule_flags(text: str) -> list[str]:
    flags = []
    for name, patterns in KNOWN_REGRESSION_PATTERNS.items():
        if any(re.search(pattern, text) for pattern in patterns):
            flags.append(name)
    if JAPANESE_RE.search(text):
        flags.append("japanese_residue")
    if len(text.splitlines()) > 2:
        flags.append("over_2_lines")
    if any(len(line) > 42 for line in text.splitlines()):
        flags.append("over_42_chars")
    if repetition_score(text) >= 0.6 and len(tokenize(text)) >= 5:
        flags.append("token_repetition")
    if re.search(r"(?:\b\d+\b[ ,]*){3,}", text):
        flags.append("number_sequence")
    return flags


def timing_risk(block: SubtitleBlock) -> tuple[float, list[str]]:
    compact_chars = len(normalize_text(block.text).replace(" ", ""))
    cps = compact_chars / block.duration
    risk = 0.0
    flags = []
    if cps > 20:
        risk += min(70, 30 + (cps - 20) * 5)
        flags.append("high_cps")
    elif cps > 16:
        risk += 25
        flags.append("borderline_cps")
    if block.duration < 0.5 and compact_chars > 15:
        risk += 40
        flags.append("short_duration_long_text")
    if block.duration > 25 and compact_chars < 5:
        risk += 25
        flags.append("long_duration_short_text")
    return min(100.0, risk), flags


def continuity_risk(blocks: list[SubtitleBlock], index: int) -> tuple[float, list[str]]:
    block = blocks[index]
    flags = []
    risk = 0.0
    current_formality = formality_score(block.text)
    neighbor_scores = []
    if index:
        neighbor_scores.append(formality_score(blocks[index - 1].text))
    if index + 1 < len(blocks):
        neighbor_scores.append(formality_score(blocks[index + 1].text))
    if neighbor_scores:
        shift = abs(current_formality - statistics.mean(neighbor_scores))
        if shift > 0.75:
            risk += 35
            flags.append("formality_jump")
        elif shift > 0.5:
            risk += 18
            flags.append("formality_shift")

    compact = normalize_text(block.text)
    if any(word in compact for word in QUESTION_WORDS) and "?" in compact and index + 1 < len(blocks):
        if ("몇 명" in compact or "몇 번" in compact or "얼마" in compact) and not any(
            word in normalize_text(blocks[index + 1].text) for word in NUMBER_WORDS
        ) and not NUMBER_RE.search(blocks[index + 1].text):
            risk += 25
            flags.append("question_answer_checksum")

    if index and normalize_text(blocks[index - 1].text) == compact == normalize_text(blocks[index + 1].text if index + 1 < len(blocks) else ""):
        risk += 30
        flags.append("triple_identical_run")
    return min(100.0, risk), flags


def provenance_dimensions(stage: str, change_row: pd.Series | None, frame_row: pd.Series | None, text: str) -> dict[str, float | str]:
    source = 25.0
    scene = 45.0 if frame_row is not None else 30.0
    speaker = 50.0
    naturalness = 78.0
    basis = ""
    confidence = ""

    if "원문·사진" in stage:
        source, scene = 88.0, 90.0
    elif "영문·사진" in stage:
        source, scene = 72.0, 88.0
    elif "사진 대조 재번역" in stage:
        source, scene = 45.0, 80.0
    elif "1차" in stage:
        source, scene = 25.0, 68.0

    if change_row is not None:
        basis = str(change_row.get("basis", ""))
        confidence = str(change_row.get("confidence", ""))
        if "일본어 원문" in basis:
            source = 96.0 if confidence == "높음" else 75.0
        elif "영문" in basis:
            source = 82.0 if confidence == "높음" else 62.0
        elif "사진" in basis or "장면" in basis:
            source = 58.0 if confidence == "높음" else 38.0
        if "사진" in basis or "장면" in basis:
            scene = 96.0 if confidence == "높음" else 78.0
        if "화자" in basis:
            speaker = 86.0
        if "자연화" in basis or "한국어" in basis or "어순" in basis:
            naturalness = 92.0 if confidence == "높음" else 84.0

    if frame_row is not None:
        try:
            delta = float(frame_row.get("start_to_frame_difference_seconds", frame_row.get("time_difference_seconds", 999)))
            if delta <= 0.2:
                scene = max(scene, 90.0)
            elif delta <= 1.0:
                scene = max(scene, 78.0)
            elif delta > 5:
                scene = min(scene, 45.0)
        except Exception:
            pass

    if any(line.lstrip().startswith("-") for line in text.splitlines()):
        speaker = max(speaker, 72.0)

    naturalness -= max(0, len(text.splitlines()) - 2) * 20
    naturalness -= sum(max(0, len(line) - 42) * 1.5 for line in text.splitlines())
    naturalness = max(0.0, min(100.0, naturalness))

    specificity = sum(text.count(token) for token in SPECIFIC_ACTION_TOKENS)
    hallucination = max(0.0, 100.0 - source)
    if specificity:
        hallucination += min(20.0, specificity * 4.0)
    hallucination = min(100.0, hallucination)

    return {
        "source_meaning_confidence": round(source, 1),
        "scene_fit_confidence": round(scene, 1),
        "speaker_confidence": round(speaker, 1),
        "korean_naturalness": round(naturalness, 1),
        "hallucination_risk": round(hallucination, 1),
        "basis": basis,
        "evidence_confidence": confidence,
    }


def stage_provenance_risk(stage: str, change_row: pd.Series | None) -> float:
    if "원문·사진" in stage:
        base = 14.0
    elif "영문·사진" in stage:
        base = 24.0
    elif "사진 대조 재번역" in stage:
        base = 38.0
    elif "1차" in stage:
        base = 66.0
    else:
        base = 58.0
    if change_row is not None:
        confidence = str(change_row.get("confidence", ""))
        basis = str(change_row.get("basis", ""))
        if "일본어 원문" in basis and confidence == "높음":
            return 5.0
        if confidence == "높음":
            return min(base, 22.0)
        if confidence == "중간":
            return min(base, 48.0)
    return base


def classify_tier(score: float) -> tuple[str, str]:
    if score >= 80:
        return "T1", "원문·원음·사진으로 전면 재복원"
    if score >= 60:
        return "T2", "후보 3개 생성 후 증거 기반 판정"
    if score >= 35:
        return "T3", "화자·문맥·자연스러움 집중 검토"
    return "T4", "구조검사 통과 시 유지"


def scene_boundaries(blocks: list[SubtitleBlock]) -> list[tuple[int, int]]:
    boundaries = [0]
    cues = ("그날", "다음 날", "다음날", "잠시 후", "그 후", "이번에는", "마지막으로", "한편")
    for i in range(1, len(blocks)):
        gap = blocks[i].start_seconds - blocks[i - 1].end_seconds
        cue = normalize_text(blocks[i].text).startswith(cues)
        if gap >= 12 or (gap >= 4 and cue):
            boundaries.append(i)
    boundaries.append(len(blocks))
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def scene_profile(scene_blocks: list[SubtitleBlock]) -> dict[str, Any]:
    styles = Counter(utterance_style(block.text) for block in scene_blocks)
    questions = sum("?" in block.text for block in scene_blocks)
    reactions = sum(utterance_style(block.text) == "반응형" for block in scene_blocks)
    narrations = sum(utterance_style(block.text) == "내레이션형" for block in scene_blocks)
    if narrations / len(scene_blocks) >= 0.4:
        kind = "내레이션 중심"
    elif reactions / len(scene_blocks) >= 0.45:
        kind = "짧은 반응 중심"
    elif questions / len(scene_blocks) >= 0.3:
        kind = "질문·대화 중심"
    else:
        kind = "혼합 대화"
    words = [word for block in scene_blocks for word in tokenize(block.text) if len(word) >= 2]
    stop = {"정말", "조금", "그냥", "지금", "이제", "여기", "그런", "이렇게", "저기", "우리", "나는", "제가"}
    top_words = [word for word, _ in Counter(word for word in words if word not in stop).most_common(8)]
    return {"scene_type": kind, "style_counts": styles, "top_words": top_words}


def validate_srt(blocks: list[SubtitleBlock]) -> dict[str, Any]:
    sequential = [block.number for block in blocks] == list(range(1, len(blocks) + 1))
    monotonic = all(blocks[i].start_seconds >= blocks[i - 1].start_seconds for i in range(1, len(blocks)))
    blank = [block.number for block in blocks if not block.text.strip()]
    japanese = [block.number for block in blocks if JAPANESE_RE.search(block.text)]
    placeholder = [block.number for block in blocks if "[불명]" in block.text or "번역 불가" in block.text]
    over_lines = [block.number for block in blocks if len(block.lines) > 2]
    over_length = [block.number for block in blocks if any(len(line) > 42 for line in block.lines)]
    return {
        "block_count": len(blocks),
        "sequential_numbering": sequential,
        "monotonic_start_times": monotonic,
        "blank_blocks": blank,
        "japanese_residue_blocks": japanese,
        "placeholder_blocks": placeholder,
        "over_2_line_blocks": over_lines,
        "over_42_char_blocks": over_length,
        "hard_validation_pass": sequential and monotonic and not blank and not japanese and not placeholder and not over_lines and not over_length,
    }


def analyze_asset(
    asset: AssetRecord,
    bundle: ModelBundle,
    memory: pd.DataFrame,
    title_output: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    title_output.mkdir(parents=True, exist_ok=True)
    blocks = parse_srt(asset.srt_path)
    probabilities = predict_probabilities(bundle, blocks)
    manifest = read_optional_csv(asset.manifest_path)
    changes = read_optional_csv(asset.changes_path)
    manifest_by_block = {int(row["block"]): row for _, row in manifest.iterrows()} if not manifest.empty and "block" in manifest else {}
    changes_by_block = {int(row["block"]): row for _, row in changes.iterrows()} if not changes.empty and "block" in changes else {}

    all_texts = [normalize_text(block.text) for block in blocks]
    _, memory_similarities, memory_frame = build_error_memory_index(memory, all_texts)

    records = []
    for i, block in enumerate(blocks):
        frame_row = manifest_by_block.get(block.number)
        change_row = changes_by_block.get(block.number)
        rules = rule_flags(block.text)
        timing, timing_flags = timing_risk(block)
        continuity, continuity_flags = continuity_risk(blocks, i)
        dims = provenance_dimensions(asset.stage, change_row, frame_row, block.text)
        provenance = stage_provenance_risk(asset.stage, change_row)

        nearest_bad = ""
        nearest_bad_after = ""
        nearest_bad_basis = ""
        error_similarity = 0.0
        if memory_similarities is not None and memory_frame is not None and len(memory_frame):
            row = memory_similarities.getrow(i)
            if row.nnz:
                local_idx = row.indices[int(np.argmax(row.data))]
                error_similarity = float(row.data.max())
                nearest = memory_frame.iloc[local_idx]
                nearest_bad = str(nearest["before"])
                nearest_bad_after = str(nearest["after"])
                nearest_bad_basis = str(nearest.get("basis", ""))

        rule_score = min(100.0, len(rules) * 24.0 + max(0.0, error_similarity - 0.65) * 120.0)
        model_score = float(probabilities[i] * 100.0)
        final_risk = (
            0.42 * model_score
            + 0.18 * rule_score
            + 0.16 * provenance
            + 0.10 * timing
            + 0.08 * continuity
            + 0.06 * float(dims["hallucination_risk"])
        )
        if change_row is not None and str(change_row.get("confidence", "")) == "높음":
            final_risk *= 0.72
        final_risk = round(min(100.0, final_risk), 2)
        tier, action = classify_tier(final_risk)
        flags = sorted(set(rules + timing_flags + continuity_flags))

        prev_text = blocks[i - 1].text if i else ""
        next_text = blocks[i + 1].text if i + 1 < len(blocks) else ""
        records.append({
            "title": asset.title,
            "stage": asset.stage,
            "block": block.number,
            "timecode": f"{block.start} --> {block.end}",
            "duration_seconds": round(block.duration, 3),
            "text": block.text,
            "prev_text": prev_text,
            "next_text": next_text,
            "style_profile": utterance_style(block.text),
            "frame_file": "" if frame_row is None else str(frame_row.get("frame_file", frame_row.get("reference_frame", ""))),
            "frame_difference_seconds": "" if frame_row is None else frame_row.get("start_to_frame_difference_seconds", frame_row.get("time_difference_seconds", "")),
            "basis": dims["basis"],
            "evidence_confidence": dims["evidence_confidence"],
            "source_meaning_confidence": dims["source_meaning_confidence"],
            "scene_fit_confidence": dims["scene_fit_confidence"],
            "speaker_confidence": dims["speaker_confidence"],
            "korean_naturalness": dims["korean_naturalness"],
            "hallucination_risk": dims["hallucination_risk"],
            "model_change_probability": round(float(probabilities[i]), 5),
            "rule_risk": round(rule_score, 2),
            "provenance_risk": round(provenance, 2),
            "timing_risk": round(timing, 2),
            "continuity_risk": round(continuity, 2),
            "error_memory_similarity": round(error_similarity, 4),
            "nearest_prior_bad_text": nearest_bad,
            "nearest_prior_correction": nearest_bad_after,
            "nearest_prior_basis": nearest_bad_basis,
            "final_risk": final_risk,
            "tier": tier,
            "recommended_action": action,
            "flags": ";".join(flags),
        })

    ledger = pd.DataFrame(records)
    # 절대 위험도와 별개로 작품 내부 상대 이상치를 반영합니다.
    ledger["relative_risk_percentile"] = ledger["final_risk"].rank(method="average", pct=True) * 100.0
    ledger["priority_score"] = (
        0.58 * ledger["final_risk"] + 0.42 * ledger["relative_risk_percentile"]
    ).clip(0, 100).round(2)
    ledger["review_band"] = pd.cut(
        ledger["priority_score"],
        bins=[-0.01, 40, 58, 72, 100.01],
        labels=["P4", "P3", "P2", "P1"],
        include_lowest=True,
    ).astype(str)
    review_actions = {
        "P1": "장면 패킷을 열고 후보 3개를 생성해 반증 검토",
        "P2": "앞뒤 5블록·사진·오류 기억을 함께 검토",
        "P3": "말투·질문응답·가독성 위주 검토",
        "P4": "회귀검사 통과 시 유지",
    }
    ledger["priority_action"] = ledger["review_band"].map(review_actions)

    ledger.to_csv(title_output / f"{asset.title}.evidence-ledger.csv", index=False, encoding="utf-8-sig")
    uncertainty = ledger.sort_values(["priority_score", "final_risk", "block"], ascending=[False, False, True])
    uncertainty.to_csv(title_output / f"{asset.title}.uncertainty-map.csv", index=False, encoding="utf-8-sig")
    uncertainty.head(min(200, len(uncertainty))).to_csv(
        title_output / f"{asset.title}.top-review-queue.csv", index=False, encoding="utf-8-sig"
    )

    packet_columns = [
        "title", "block", "timecode", "text", "prev_text", "next_text", "frame_file",
        "basis", "evidence_confidence", "nearest_prior_bad_text", "nearest_prior_correction",
        "nearest_prior_basis", "final_risk", "relative_risk_percentile", "priority_score",
        "review_band", "priority_action", "flags",
    ]
    with (title_output / f"{asset.title}.review-packets.jsonl").open("w", encoding="utf-8") as handle:
        for record in uncertainty.head(min(100, len(uncertainty)))[packet_columns].to_dict("records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    scenes = []
    for scene_id, (start_idx, end_idx) in enumerate(scene_boundaries(blocks), start=1):
        scene_blocks = blocks[start_idx:end_idx]
        profile = scene_profile(scene_blocks)
        scene_ledger = ledger.iloc[start_idx:end_idx]
        scenes.append({
            "scene_id": scene_id,
            "start_block": scene_blocks[0].number,
            "end_block": scene_blocks[-1].number,
            "start_time": scene_blocks[0].start,
            "end_time": scene_blocks[-1].end,
            "block_count": len(scene_blocks),
            "scene_type": profile["scene_type"],
            "style_counts": dict(profile["style_counts"]),
            "top_words": profile["top_words"],
            "mean_risk": round(float(scene_ledger["final_risk"].mean()), 2),
            "max_risk": round(float(scene_ledger["final_risk"].max()), 2),
            "t1_blocks": int((scene_ledger["tier"] == "T1").sum()),
            "t2_blocks": int((scene_ledger["tier"] == "T2").sum()),
        })
    (title_output / f"{asset.title}.scene-map.json").write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")

    style_groups = []
    for style, group in ledger.groupby("style_profile"):
        style_groups.append({
            "style": style,
            "block_count": int(len(group)),
            "share": round(float(len(group) / len(ledger)), 4),
            "sample_blocks": group.sort_values("final_risk", ascending=False)[["block", "text"]].head(5).to_dict("records"),
            "note": "화자 신원이 아니라 문장 표면형을 분류한 후보 프로필입니다.",
        })
    (title_output / f"{asset.title}.style-profiles.json").write_text(json.dumps(style_groups, ensure_ascii=False, indent=2), encoding="utf-8")

    validation = validate_srt(blocks)
    tier_counts = ledger["tier"].value_counts().to_dict()
    report = {
        "system_version": SYSTEM_VERSION,
        "title": asset.title,
        "stage": asset.stage,
        "source_srt": str(asset.srt_path),
        "manifest": "" if asset.manifest_path is None else str(asset.manifest_path),
        "changes": "" if asset.changes_path is None else str(asset.changes_path),
        "validation": validation,
        "risk_summary": {
            "mean_risk": round(float(ledger["final_risk"].mean()), 2),
            "median_risk": round(float(ledger["final_risk"].median()), 2),
            "max_risk": round(float(ledger["final_risk"].max()), 2),
            "tier_counts": tier_counts,
            "absolute_priority_blocks_T1_T2": int((ledger["tier"].isin(["T1", "T2"])).sum()),
            "review_band_counts": ledger["review_band"].value_counts().to_dict(),
            "relative_priority_blocks_P1_P2": int((ledger["review_band"].isin(["P1", "P2"])).sum()),
        },
        "scene_count": len(scenes),
        "known_limitations": [
            "정지 사진의 의미를 자동으로 새로 판독하지 않았습니다. frame_file은 근거 추적용입니다.",
            "위험 모델은 ABF-303 한 작품의 검증 변경 이력으로 보정돼 작품 간 일반화 성능은 미확정입니다.",
            "style-profiles는 실제 화자 식별이 아니라 문장 형식 분류입니다.",
            "위험 점수는 번역 오류의 확정 판정이 아니라 검수 우선순위입니다.",
        ],
    }
    (title_output / f"{asset.title}.qa-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    readable = f"""# {asset.title} 자막 포렌식 QA 보고서

- 단계: {asset.stage}
- 블록: {len(blocks):,}개
- 장면 묶음: {len(scenes):,}개
- 평균 위험 점수: {report['risk_summary']['mean_risk']}
- 절대등급 T1 전면 재복원: {tier_counts.get('T1', 0):,}개
- 절대등급 T2 후보 경쟁 검토: {tier_counts.get('T2', 0):,}개
- 상대우선 P1: {int((ledger['review_band'] == 'P1').sum()):,}개
- 상대우선 P2: {int((ledger['review_band'] == 'P2').sum()):,}개
- T3 문맥·화자 검토: {tier_counts.get('T3', 0):,}개
- T4 유지 후보: {tier_counts.get('T4', 0):,}개
- 구조검사: {'통과' if validation['hard_validation_pass'] else '경고'}

## 해석

위험 점수는 오류 확률의 확정치가 아니라 검수 우선순위입니다. 특히 T1·T2는 기존 문장을 다듬는 대신 원문·원음·사진에서 다시 복원할 대상으로 봅니다.

## 한계

- 사진 파일은 근거 추적용으로 연결했으며, 사진 내용을 자동으로 단정하지 않았습니다.
- 화자 신원이 없는 자료에서 실제 인물별 말투를 자동 확정하지 않았습니다.
- ABF-303 한 작품에서 보정된 모델이므로 작품 간 검증이 추가로 필요합니다.
"""
    (title_output / f"{asset.title}.qa-report.md").write_text(readable, encoding="utf-8")
    return ledger, report


def write_readme(output_dir: Path, metrics: dict[str, Any], assets: list[AssetRecord]) -> None:
    asset_lines = "\n".join(f"- {asset.title}: {asset.stage} — `{asset.srt_path.name}`" for asset in assets)
    readme = f"""# Subtitle Forensics v{SYSTEM_VERSION}

불완전한 자막·사진·수정 이력을 결합해 **오류를 자동 확정하는 것이 아니라, 깊은 검수가 필요한 블록을 증거 기반으로 순위화**하는 구현체입니다.

## 구현된 핵심

1. **잠재 대본 위험 모델**: ABF-303의 검증 변경 522개를 학습 라벨로 사용합니다.
2. **시간·문맥 특징**: 자막 길이, 초당 글자 수, 앞뒤 문장, 말투 변화, 반복과 질문–응답 신호를 봅니다.
3. **오류 기억**: 지금까지의 `before → after` 변경 이력을 모아 유사한 과거 오류를 탐색합니다.
4. **다차원 근거표**: 원문 의미·장면 적합·화자·한국어 자연도·환각 위험을 분리합니다.
5. **장면 단위 분석**: 긴 공백과 전환 신호를 이용해 블록을 장면 묶음으로 나눕니다.
6. **회귀 검사**: `[불명]`, 일본어 잔존, 3줄, 42자 초과, 숫자 환각, 알려진 오역 패턴을 검사합니다.
7. **이중 검수 큐**: 절대 위험 T1~T4와 작품 내부 상대 우선순위 P1~P4를 함께 제공합니다.
8. **4단계 절대 위험 등급**:
   - T1: 원문·원음·사진으로 전면 재복원
   - T2: 후보 3개 생성 후 증거 기반 판정
   - T3: 화자·문맥·자연스러움 검토
   - T4: 구조검사 통과 시 유지

## 모델 보정 결과

- 검증 방식: {metrics['validation']}
- ROC-AUC: {metrics['roc_auc']:.3f}
- Average Precision: {metrics['average_precision']:.3f}
- 재현율 중심 F2: {metrics['f2']:.3f}
- Precision / Recall: {metrics['precision']:.3f} / {metrics['recall']:.3f}
- 임계값: {metrics['threshold_max_f2']:.3f}

> 주의: 같은 작품 내부의 그룹 교차검증입니다. 다른 작품에서의 정확도를 뜻하지 않습니다.

## 분석 대상

{asset_lines}

## 주요 출력

각 작품 폴더:

- `*.evidence-ledger.csv`: 모든 블록의 증거와 위험 구성요소
- `*.uncertainty-map.csv`: 위험 점수 순 정렬
- `*.top-review-queue.csv`: 상대·절대 위험을 결합한 상위 200개 검수 큐
- `*.review-packets.jsonl`: 앞뒤 문맥·사진·오류 기억을 묶은 상위 100개 장면 패킷
- `*.scene-map.json`: 시간축 장면 묶음
- `*.style-profiles.json`: 실제 화자가 아닌 말투 유형 후보
- `*.qa-report.md/json`: 구조검사와 위험 요약

전체 폴더:

- `portfolio-priority-queue.csv`: 모든 작품을 합친 우선 검수 큐
- `portfolio-summary.csv`: 작품별 위험도 요약
- `risk_model.joblib`: 재사용 가능한 모델
- `model_metrics.json`: 모델 보정 결과
- `regression-memory.csv`: 과거 오류 기억

## 실행

```bash
python subtitle_forensics.py \\
  --root /mnt/data \\
  --output /mnt/data/subtitle_forensics_v1
```

## 중요한 한계

이 버전은 **사진 파일을 블록에 연결하지만 사진 의미를 자동 생성하지 않습니다.** 영상 원음·일본어 N-best ASR·시각 캡션을 추가하면 HLGR 전체 구조로 확장할 수 있습니다. 현재 결과는 오류 확정표가 아니라 가장 효율적으로 깊게 검수할 순서를 제시합니다.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def run(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "model"
    model_dir.mkdir(exist_ok=True)

    baseline = root / "_v6_srt_extract" / "ABF-303.final-deep-natural-ko.v6.srt"
    if not baseline.exists():
        archive = root / "20_SRT_final_deep_natural_ko_v6.zip"
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extract("ABF-303.final-deep-natural-ko.v6.srt", root / "_v6_srt_extract")
    changes = root / "ABF-303_photo_verified_retranslation" / "ABF-303.final-photo-verified-retranslated-ko.v1.changes.csv"
    if not changes.exists():
        raise FileNotFoundError(changes)

    bundle = train_model(baseline, changes, model_dir)
    memory = load_error_memory(root)
    memory.to_csv(output / "regression-memory.csv", index=False, encoding="utf-8-sig")

    assets = discover_assets(root)
    if not assets:
        raise RuntimeError("분석할 사진 대조 자막을 찾지 못했습니다.")

    ledgers = []
    reports = []
    asset_index = []
    for asset in assets:
        title_dir = output / "titles" / asset.title
        ledger, report = analyze_asset(asset, bundle, memory, title_dir)
        ledgers.append(ledger)
        reports.append(report)
        asset_index.append({
            "title": asset.title,
            "stage": asset.stage,
            "srt_path": str(asset.srt_path),
            "manifest_path": "" if asset.manifest_path is None else str(asset.manifest_path),
            "changes_path": "" if asset.changes_path is None else str(asset.changes_path),
        })

    portfolio = pd.concat(ledgers, ignore_index=True)
    portfolio.sort_values(["priority_score", "final_risk", "title", "block"], ascending=[False, False, True, True]).to_csv(
        output / "portfolio-priority-queue.csv", index=False, encoding="utf-8-sig"
    )

    summary_rows = []
    for report in reports:
        tiers = report["risk_summary"]["tier_counts"]
        summary_rows.append({
            "title": report["title"],
            "stage": report["stage"],
            "blocks": report["validation"]["block_count"],
            "mean_risk": report["risk_summary"]["mean_risk"],
            "median_risk": report["risk_summary"]["median_risk"],
            "max_risk": report["risk_summary"]["max_risk"],
            "T1": tiers.get("T1", 0),
            "T2": tiers.get("T2", 0),
            "T3": tiers.get("T3", 0),
            "T4": tiers.get("T4", 0),
            "absolute_priority_blocks_T1_T2": report["risk_summary"]["absolute_priority_blocks_T1_T2"],
            "relative_priority_blocks_P1_P2": report["risk_summary"]["relative_priority_blocks_P1_P2"],
            "P1": report["risk_summary"]["review_band_counts"].get("P1", 0),
            "P2": report["risk_summary"]["review_band_counts"].get("P2", 0),
            "P3": report["risk_summary"]["review_band_counts"].get("P3", 0),
            "P4": report["risk_summary"]["review_band_counts"].get("P4", 0),
            "hard_validation_pass": report["validation"]["hard_validation_pass"],
            "scene_count": report["scene_count"],
        })
    pd.DataFrame(summary_rows).sort_values("mean_risk", ascending=False).to_csv(
        output / "portfolio-summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(asset_index).to_csv(output / "asset-index.csv", index=False, encoding="utf-8-sig")
    write_readme(output, bundle.metrics, assets)

    implementation = {
        "system_version": SYSTEM_VERSION,
        "created_files": sum(1 for _ in output.rglob("*" ) if _.is_file()),
        "titles_analyzed": len(assets),
        "blocks_analyzed": int(len(portfolio)),
        "absolute_priority_blocks_T1_T2": int(portfolio["tier"].isin(["T1", "T2"]).sum()),
        "relative_priority_blocks_P1_P2": int(portfolio["review_band"].isin(["P1", "P2"]).sum()),
        "regression_memory_entries": int(len(memory)),
        "model_metrics": bundle.metrics,
        "sha256_portfolio_queue": hashlib.sha256((output / "portfolio-priority-queue.csv").read_bytes()).hexdigest(),
    }
    (output / "implementation-summary.json").write_text(json.dumps(implementation, ensure_ascii=False, indent=2), encoding="utf-8")

    # Copy the executable as the reusable CLI artifact.
    source = Path(__file__).resolve()
    shutil.copy2(source, output / "subtitle_forensics.py")
    (output / "run_all.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\npython \"$(dirname \"$0\")/subtitle_forensics.py\" --root /mnt/data --output /mnt/data/subtitle_forensics_v1\n",
        encoding="utf-8",
    )
    (output / "run_all.sh").chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-based subtitle forensic prioritization")
    parser.add_argument("--root", type=Path, default=Path("/mnt/data"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/data/subtitle_forensics_v1"))
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == "__main__":
    main()
