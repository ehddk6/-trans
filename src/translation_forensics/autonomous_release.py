from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Protocol

from jsonschema import Draft202012Validator

from .alignment import align_by_overlap
from .asr_evidence import read_asr_candidates, should_escalate, text_similarity
from .srt import SubtitleBlock, compare_structure, has_japanese, parse_srt, timecode_to_seconds, write_srt
from .validation import validate_pair


AUTONOMOUS_RELEASE_KIND = "autonomous-release"
SOURCE_STATUSES = {"accepted", "abstained"}
VIEWER_STATUSES = {"supported", "best_effort", "unrecoverable"}
CONFIDENCES = {"high", "medium", "low"}
SEMANTIC_SLOT_KEYS = {
    "question", "polarity", "refusal_permission", "command_strength",
    "speaker", "actor", "action", "target", "location", "tense_aspect",
    "direction", "intensity",
}
WORK_TAG_RE = re.compile(r"\[(?:수정|추정|음성|검토|번역자|번역 필요)\]")
BROKEN_RE = re.compile(r"�|\ufffd|(?:\?{4,})")
KOREAN_RE = re.compile(r"[가-힣]")


class AutonomousProvider(Protocol):
    budget: Any

    def generate_decisions(self, *, title_id: str, prompt: str, payload: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def critique_decisions(self, *, title_id: str, prompt: str, payload: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def transcribe_audio(self, *, title_id: str, scene_id: str, audio_path: Path) -> tuple[str, dict[str, Any]]: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_manifest_input(path: Path, root: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    root = root.expanduser().resolve()
    try:
        portable = resolved.relative_to(root).as_posix()
        path_base = "workspace"
    except ValueError:
        portable = resolved.name
        path_base = "basename-only"
    return {"path": portable, "path_base": path_base, "sha256": _sha256(resolved)}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL object required: {path}:{line_number}")
        result.append(value)
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n" for value in values), encoding="utf-8", newline="\n")


def _load_previous(reference: list[SubtitleBlock], path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    candidate, _, _ = parse_srt(path)
    result: dict[int, dict[str, Any]] = {}
    by_number = {block.number: block for block in candidate}
    if compare_structure(reference, candidate)["pass"]:
        for block in reference:
            text = by_number[block.number].text.strip()
            if text:
                result[block.number] = {"text": text, "alignment": "structure-equal", "confidence": "high"}
        return result
    candidate_by_number = {block.number: block for block in candidate}
    for alignment in align_by_overlap(reference, candidate):
        if alignment.candidate_number is None:
            continue
        text = candidate_by_number[alignment.candidate_number].text.strip()
        if text:
            result[alignment.source_number] = {
                "text": text,
                "alignment": "time-overlap",
                "confidence": alignment.confidence,
                "candidate_number": alignment.candidate_number,
            }
    return result


def _load_scenes(path: Path | None) -> tuple[dict[int, dict[str, str]], dict[str, dict[str, str]]]:
    by_block: dict[int, dict[str, str]] = {}
    by_scene: dict[str, dict[str, str]] = {}
    if path is None or not path.exists():
        return by_block, by_scene
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            scene = dict(row)
            scene_id = str(scene.get("scene_id") or "")
            if not scene_id:
                continue
            by_scene[scene_id] = scene
            for raw in str(scene.get("block_numbers") or "").split(","):
                if raw.strip().isdigit():
                    by_block[int(raw)] = scene
    return by_block, by_scene


def _load_asr(path: Path | None) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    if path is None or not path.exists():
        return grouped
    for row in read_asr_candidates(path):
        grouped[str(row.get("scene_id") or "")].append(row)
    return grouped


def estimate_machine_alignment(
    reference: list[SubtitleBlock],
    scene_by_block: dict[int, dict[str, str]],
    asr_by_scene: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    by_number = {block.number: block for block in reference}
    grouped: dict[str, tuple[dict[str, str], list[int]]] = {}
    for number, scene in scene_by_block.items():
        scene_id = str(scene.get("scene_id") or "")
        if not scene_id or number not in by_number:
            continue
        if scene_id not in grouped:
            grouped[scene_id] = (scene, [])
        grouped[scene_id][1].append(number)
    candidates: list[dict[str, Any]] = []
    for scene_id, (scene, numbers) in grouped.items():
        numbers = sorted(set(numbers), key=lambda number: by_number[number].start_seconds)
        try:
            scene_start = timecode_to_seconds(str(scene.get("start_time", "")))
        except ValueError:
            continue
        reference_text = " ".join(by_number[number].text for number in numbers)
        usable_asr = [
            str(row.get("text") or "").strip()
            for row in asr_by_scene.get(scene_id, [])
            if str(row.get("text") or "").strip() and not str(row.get("error") or "").strip()
        ]
        best_similarity = max((text_similarity(reference_text, text) for text in usable_asr), default=0.0)
        candidates.append({
            "scene_id": scene_id,
            "block_numbers": numbers,
            "offset_seconds": scene_start - by_number[numbers[0]].start_seconds,
            "asr_profile_count": len(usable_asr),
            # Profiles are alternative views from one Whisper family.  Taking
            # the best similarity is a compatibility check, never a vote.
            "best_asr_similarity": best_similarity,
        })
    offsets = [float(item["offset_seconds"]) for item in candidates]
    median_offset = sorted(offsets)[len(offsets) // 2] if offsets else None
    scene_alignment: dict[str, dict[str, Any]] = {}
    covered: set[int] = set()
    similarities: list[float] = []
    for item in candidates:
        offset_consistent = bool(median_offset is not None and abs(float(item["offset_seconds"]) - median_offset) <= 1.5)
        has_audio_speech_evidence = int(item["asr_profile_count"]) > 0
        semantic_compatible = has_audio_speech_evidence and float(item["best_asr_similarity"]) >= 0.20
        # A source/ASR text conflict is semantic evidence to adjudicate, not by
        # itself proof that the audio clip belongs to a different time.  The
        # mapping gate therefore uses monotonic offset consistency plus the
        # existence of speech evidence, and records agreement separately.
        compatible = offset_consistent and has_audio_speech_evidence
        if compatible:
            covered.update(int(number) for number in item["block_numbers"])
        similarities.append(float(item["best_asr_similarity"]))
        scene_alignment[str(item["scene_id"])] = {
            **item,
            "offset_seconds": round(float(item["offset_seconds"]), 3),
            "best_asr_similarity": round(float(item["best_asr_similarity"]), 4),
            "offset_consistent": offset_consistent,
            "semantic_compatible": semantic_compatible,
            "timeline_compatible": compatible,
        }
    mean_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    coverage = len(covered) / max(1, len(reference))
    if not candidates:
        status = "text-only"
        confidence = "none"
    elif covered:
        status = "machine-aligned"
        confidence = "medium"
    else:
        status = "machine-aligned-low-confidence"
        confidence = "low"
    return {
        "schema_name": "translation-forensics/machine-alignment",
        "schema_version": "1",
        "status": status,
        "confidence": confidence,
        "mapped_blocks": len(covered),
        "total_blocks": len(reference),
        "coverage": round(coverage, 6),
        "median_offset_seconds": round(median_offset, 3) if median_offset is not None else None,
        "mean_text_similarity": round(mean_similarity, 4),
        "asr_source_family_count": 1 if asr_by_scene else 0,
        "same_family_profiles_are_independent_votes": False,
        "scene_alignment": scene_alignment,
        "direct_human_listening": False,
    }


def build_autonomous_evidence(
    *,
    title_id: str,
    structure_path: Path,
    japanese_path: Path,
    previous_path: Path | None = None,
    scenes_path: Path | None = None,
    local_asr_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    structure, _, _ = parse_srt(structure_path)
    japanese, _, _ = parse_srt(japanese_path)
    if not compare_structure(structure, japanese)["pass"]:
        raise ValueError("Japanese SRT structure does not match the locked structure.")
    previous = _load_previous(structure, previous_path)
    scene_by_block, scene_by_id = _load_scenes(scenes_path)
    asr_by_scene = _load_asr(local_asr_path)
    alignment = estimate_machine_alignment(structure, scene_by_block, asr_by_scene)
    japanese_by_number = {block.number: block for block in japanese}
    position = {block.number: index for index, block in enumerate(structure)}
    evidence: list[dict[str, Any]] = []
    for block in structure:
        source = japanese_by_number[block.number]
        start = max(0, position[block.number] - 2)
        end = min(len(structure), position[block.number] + 3)
        scene = scene_by_block.get(block.number)
        scene_id = str(scene.get("scene_id") or "") if scene else ""
        local_rows = asr_by_scene.get(scene_id, [])
        scene_alignment = alignment.get("scene_alignment", {}).get(scene_id, {})
        timeline_compatible = bool(scene_alignment.get("timeline_compatible"))
        local_refs = []
        for row in local_rows if timeline_compatible else []:
            local_refs.append(
                {
                    "evidence_id": f"local-asr:{scene_id}:{row.get('profile', '')}",
                    "source_family": "whisper-family",
                    "profile": row.get("profile", ""),
                    "text": row.get("text", ""),
                    "avg_logprob": row.get("avg_logprob", ""),
                    "no_speech_prob": row.get("no_speech_prob", ""),
                    "error": row.get("error", ""),
                }
            )
        risks: list[str] = []
        if not source.text.strip():
            risks.append("missing-japanese")
        if BROKEN_RE.search(source.text):
            risks.append("corrupt-japanese")
        if scene and not timeline_compatible:
            risks.append("low-timeline-confidence")
            if local_rows:
                risks.append("audio-evidence-withheld")
        elif scene and not bool(scene_alignment.get("semantic_compatible")):
            risks.append("audio-srt-conflict")
        if local_rows:
            escalate, reason = should_escalate(local_rows)
            if escalate:
                risks.extend(f"local-asr:{item}" for item in reason.split(",") if item)
        elif scene:
            risks.append("missing-local-asr")
        context = [
            {
                "block_number": neighbor.number,
                "timecode": f"{neighbor.start} --> {neighbor.end}",
                "japanese_srt": japanese_by_number[neighbor.number].text,
                "is_target": neighbor.number == block.number,
            }
            for neighbor in structure[start:end]
        ]
        evidence.append(
            {
                "schema_name": "translation-forensics/autonomous-evidence-bundle",
                "schema_version": "1",
                "title_id": title_id,
                "block_number": block.number,
                "timecode": f"{block.start} --> {block.end}",
                "japanese": {
                    "evidence_id": f"japanese-srt:{block.number}",
                    "source_family": "japanese-reference",
                    "text": source.text,
                },
                "local_asr": local_refs,
                "cloud_asr": [],
                "previous_korean_candidate": previous.get(block.number),
                "local_context": context,
                "scene_id": scene_id or None,
                "timeline_status": alignment["status"],
                "timeline_compatible": timeline_compatible,
                "risk_codes": sorted(set(risks)),
                "untrusted_input": True,
            }
        )
    return evidence, alignment, scene_by_id, asr_by_scene


def _audio_path_for_scene(scenes_path: Path, scene: dict[str, str]) -> Path | None:
    relative = str(scene.get("original_audio") or "").strip()
    if not relative:
        return None
    candidate = (scenes_path.parent / relative).resolve()
    return candidate if candidate.exists() else None


def _needs_cloud_asr(record: dict[str, Any]) -> bool:
    if not record.get("timeline_compatible"):
        return False
    codes = set(record.get("risk_codes", []))
    return bool(codes & {"missing-japanese", "corrupt-japanese", "missing-local-asr", "audio-srt-conflict"}) or any(
        str(code).startswith("local-asr:") for code in codes
    )


def _validate_decisions(
    title_id: str,
    expected: list[int],
    rows: Any,
    allowed_evidence_refs: dict[int, set[str]] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError("Autonomous response results must be an array.")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Autonomous decision must be an object.")
        if row.get("title_id") != title_id:
            raise ValueError("Autonomous decision title_id mismatch.")
        number = int(row.get("block_number", 0))
        if number in result:
            raise ValueError(f"Duplicate autonomous decision: {number}")
        source = str(row.get("source_faithful_korean", "")).strip()
        viewer = str(row.get("viewer_natural_korean", "")).strip()
        source_status = row.get("source_status")
        viewer_status = row.get("viewer_status")
        confidence = row.get("confidence")
        evidence_refs = row.get("evidence_refs")
        if source_status not in SOURCE_STATUSES or viewer_status not in VIEWER_STATUSES or confidence not in CONFIDENCES:
            raise ValueError(f"Invalid autonomous statuses for block {number}.")
        if not viewer or has_japanese(viewer) or WORK_TAG_RE.search(viewer) or BROKEN_RE.search(viewer):
            raise ValueError(f"Invalid viewer Korean for block {number}.")
        if viewer != "…" and not KOREAN_RE.search(viewer):
            raise ValueError(f"Viewer output is not Korean for block {number}.")
        if source_status == "accepted":
            if not source or has_japanese(source) or not KOREAN_RE.search(source) or not isinstance(evidence_refs, list) or not evidence_refs:
                raise ValueError(f"Accepted source decision lacks Korean/evidence for block {number}.")
            if viewer_status != "supported":
                raise ValueError(f"Accepted source requires supported viewer for block {number}.")
        else:
            if source:
                raise ValueError(f"Abstained source must be empty for block {number}.")
            if viewer_status == "supported":
                raise ValueError(f"Abstained source cannot have supported viewer for block {number}.")
        if viewer_status == "unrecoverable" and viewer != "…":
            raise ValueError(f"Unrecoverable viewer must use neutral marker for block {number}.")
        semantic_slots = row.get("semantic_slots")
        if not isinstance(semantic_slots, dict) or set(semantic_slots) != SEMANTIC_SLOT_KEYS:
            raise ValueError(f"Autonomous decision has an incomplete semantic slot lattice for block {number}.")
        for field in ("inferred_slots", "competing_interpretations", "risk_codes"):
            if not isinstance(row.get(field), list):
                raise ValueError(f"Autonomous decision lacks array field {field} for block {number}.")
        if not isinstance(evidence_refs, list):
            raise ValueError(f"Autonomous decision evidence_refs must be an array for block {number}.")
        if allowed_evidence_refs is not None:
            unknown_refs = sorted(set(str(value) for value in evidence_refs) - allowed_evidence_refs.get(number, set()))
            if unknown_refs:
                raise ValueError(f"Decision cites non-evidence IDs for block {number}: {unknown_refs}")
        if not str(row.get("reason") or "").strip():
            raise ValueError(f"Autonomous decision lacks reason for block {number}.")
        result[number] = dict(row)
    if set(result) != set(expected):
        missing = sorted(set(expected) - set(result))
        extra = sorted(set(result) - set(expected))
        raise ValueError(f"Autonomous decision coverage mismatch; missing={missing[:10]} extra={extra[:10]}")
    return [result[number] for number in expected]


def _allowed_evidence_refs(records: list[dict[str, Any]]) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    for record in records:
        refs = [record.get("japanese", {}), *record.get("local_asr", []), *record.get("cloud_asr", [])]
        result[int(record["block_number"])] = {
            str(item.get("evidence_id")) for item in refs if str(item.get("evidence_id") or "").strip()
        }
    return result


def _validate_critiques(title_id: str, expected: list[int], rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError("Autonomous critique reviews must be an array.")
    by_number: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("title_id") != title_id:
            raise ValueError("Invalid critique title or record.")
        number = int(row.get("block_number", 0))
        if row.get("verdict") not in {"accept", "repair", "quarantine"}:
            raise ValueError(f"Invalid critique verdict for block {number}.")
        if number in by_number:
            raise ValueError(f"Duplicate critique for block {number}.")
        by_number[number] = dict(row)
    if set(by_number) != set(expected):
        raise ValueError("Critique coverage does not match requested risky blocks.")
    return [by_number[number] for number in expected]


def _title_memory(decisions: list[dict[str, Any]], japanese_by_number: dict[int, str]) -> list[dict[str, Any]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for decision in decisions:
        if decision.get("source_status") != "accepted" or decision.get("confidence") != "high":
            continue
        japanese = japanese_by_number.get(int(decision["block_number"]), "").strip()
        korean = str(decision.get("source_faithful_korean") or "").strip()
        if japanese and korean:
            grouped[japanese][korean] += 1
    result = []
    for japanese, counter in grouped.items():
        korean, count = counter.most_common(1)[0]
        if count >= 2 and len(counter) == 1:
            result.append({"japanese": japanese, "korean": korean, "support_count": count, "scope": "title-only"})
    return sorted(result, key=lambda item: (-item["support_count"], item["japanese"]))


def _quarantine(decision: dict[str, Any], critique: dict[str, Any]) -> dict[str, Any]:
    updated = dict(decision)
    updated["source_faithful_korean"] = ""
    updated["source_status"] = "abstained"
    if not str(updated.get("viewer_natural_korean") or "").strip():
        updated["viewer_natural_korean"] = "…"
        updated["viewer_status"] = "unrecoverable"
    elif updated.get("viewer_status") == "supported":
        updated["viewer_status"] = "best_effort"
    updated["confidence"] = "low"
    updated["risk_codes"] = sorted(set([*updated.get("risk_codes", []), "critic-quarantine", *[f"critical:{x}" for x in critique.get("critical_slot_conflicts", [])]]))
    updated["reason"] = f"{updated.get('reason', '')} Critic quarantine: {critique.get('reason', '')}".strip()
    return updated


def _critique_record(review: dict[str, Any], attempt: int) -> dict[str, Any]:
    return {
        "schema_name": "translation-forensics/autonomous-critique",
        "schema_version": "1",
        **review,
        "attempt": attempt,
    }


def _model_calls_traceable(records: list[dict[str, Any]]) -> bool:
    if not records:
        return False
    required = {
        "schema_name", "schema_version", "call_kind", "provider", "model",
        "request_id", "request_sha256", "response_sha256", "evidence_sha256",
        "model_settings_sha256", "environment", "environment_sha256", "cache_key", "cache_hit", "cost_usd",
        "external_transfer", "store", "tools_enabled",
    }
    digest = re.compile(r"^[0-9a-f]{64}$")
    return all(
        required <= set(record)
        and record.get("schema_name") == "translation-forensics/model-call-manifest"
        and record.get("provider") == "openai"
        and bool(digest.fullmatch(str(record.get("request_sha256") or "")))
        and bool(digest.fullmatch(str(record.get("response_sha256") or "")))
        and bool(digest.fullmatch(str(record.get("evidence_sha256") or "")))
        and bool(digest.fullmatch(str(record.get("model_settings_sha256") or "")))
        and bool(digest.fullmatch(str(record.get("environment_sha256") or "")))
        for record in records
    )


def validate_autonomous_regressions(
    *,
    title_id: str,
    decisions: list[dict[str, Any]],
    critiques: list[dict[str, Any]],
    suite_path: Path | None,
) -> dict[str, Any]:
    if suite_path is None or not suite_path.exists():
        return {"status": "not-configured", "title_id": title_id, "cases": [], "errors": []}
    suite = _json(suite_path)
    decisions_by_number = {int(row["block_number"]): row for row in decisions}
    reviewed = {int(row["block_number"]) for row in critiques}
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for case in suite.get("cases", []):
        if case.get("title_id") != title_id:
            continue
        case_errors: list[str] = []
        for number in [int(value) for value in case.get("blocks", [])]:
            decision = decisions_by_number.get(number)
            if decision is None:
                case_errors.append(f"missing block {number}")
                continue
            requirements = set(case.get("requirements", []))
            if "source-accepted" in requirements and decision.get("source_status") != "accepted":
                case_errors.append(f"block {number}: source was not accepted")
            if "source-abstained" in requirements and decision.get("source_status") != "abstained":
                case_errors.append(f"block {number}: source was not abstained")
            if "viewer-recovered" in requirements and decision.get("viewer_status") == "unrecoverable":
                case_errors.append(f"block {number}: viewer dialogue remains unrecoverable")
            if "critic-reviewed" in requirements and number not in reviewed:
                case_errors.append(f"block {number}: critic review is missing")
            if "critic-reviewed-if-risky" in requirements and decision.get("risk_codes") and number not in reviewed:
                case_errors.append(f"block {number}: risky decision lacks critic review")
            if "no-accepted-critical-conflict" in requirements and decision.get("source_status") == "accepted" and any(
                str(code).startswith("critical:") for code in decision.get("risk_codes", [])
            ):
                case_errors.append(f"block {number}: accepted critical conflict")
        result = {"case_id": case.get("case_id"), "status": "pass" if not case_errors else "fail", "errors": case_errors}
        results.append(result)
        errors.extend(f"{case.get('case_id')}: {message}" for message in case_errors)
    return {"status": "pass" if not errors else "fail", "title_id": title_id, "cases": results, "errors": errors}


def _build_evidence_graph(evidence: list[dict[str, Any]], decisions: list[dict[str, Any]], critiques: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    evidence_by_number = {int(row["block_number"]): row for row in evidence}
    critique_attempts: dict[int, list[int]] = defaultdict(list)
    for critique in critiques:
        critique_attempts[int(critique["block_number"])].append(int(critique.get("attempt", 0)))
    for decision in decisions:
        number = int(decision["block_number"])
        decision_id = f"decision:{number}"
        nodes.append({"id": decision_id, "kind": "decision", "source_status": decision["source_status"], "viewer_status": decision["viewer_status"]})
        record = evidence_by_number[number]
        refs = [record["japanese"], *record.get("local_asr", []), *record.get("cloud_asr", [])]
        for item in refs:
            evidence_id = str(item.get("evidence_id") or "")
            if not evidence_id:
                continue
            nodes.append({"id": evidence_id, "kind": "evidence", "source_family": item.get("source_family")})
        for evidence_id in decision.get("evidence_refs", []):
            edges.append({"from": decision_id, "to": evidence_id, "relation": "supported-by"})
        for attempt in sorted(critique_attempts.get(number, [])):
            critique_id = f"critique:{number}:{attempt}"
            nodes.append({"id": critique_id, "kind": "critique", "attempt": attempt})
            edges.append({"from": critique_id, "to": decision_id, "relation": "reviews"})
    unique_nodes = {str(node["id"]): node for node in nodes}
    return {
        "schema_name": "translation-forensics/autonomous-evidence-graph",
        "schema_version": "1",
        "nodes": [unique_nodes[key] for key in sorted(unique_nodes)],
        "edges": sorted(edges, key=lambda row: (str(row["from"]), str(row["to"]), str(row["relation"]))),
    }


def run_autonomous_release(
    *,
    title_id: str,
    structure_path: Path,
    japanese_path: Path,
    output_dir: Path,
    provider: AutonomousProvider,
    decision_prompt_path: Path,
    critic_prompt_path: Path,
    decision_schema_path: Path,
    critique_schema_path: Path,
    previous_path: Path | None = None,
    scenes_path: Path | None = None,
    local_asr_path: Path | None = None,
    batch_size: int = 20,
    max_repairs: int = 2,
    resume: bool = False,
) -> dict[str, Any]:
    paths = [structure_path, japanese_path, decision_prompt_path, critic_prompt_path, decision_schema_path, critique_schema_path]
    paths = [path.expanduser().resolve() for path in paths]
    structure_path, japanese_path, decision_prompt_path, critic_prompt_path, decision_schema_path, critique_schema_path = paths
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Existing autonomous-release output will not be overwritten: {output_dir}")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=resume)
    decision_prompt = decision_prompt_path.read_text(encoding="utf-8")
    critic_prompt = critic_prompt_path.read_text(encoding="utf-8")
    decision_schema = _json(decision_schema_path)
    critique_schema = _json(critique_schema_path)
    evidence, alignment, scenes_by_id, _ = build_autonomous_evidence(
        title_id=title_id,
        structure_path=structure_path,
        japanese_path=japanese_path,
        previous_path=previous_path,
        scenes_path=scenes_path,
        local_asr_path=local_asr_path,
    )
    call_records: list[dict[str, Any]] = []
    cloud_by_scene: dict[str, str] = {}
    if scenes_path is not None and scenes_path.exists():
        for record in evidence:
            scene_id = str(record.get("scene_id") or "")
            if not scene_id or scene_id in cloud_by_scene or not _needs_cloud_asr(record):
                continue
            scene = scenes_by_id.get(scene_id, {})
            audio_path = _audio_path_for_scene(scenes_path, scene)
            if audio_path is None:
                record["risk_codes"] = sorted(set([*record["risk_codes"], "missing-cloud-asr-clip"]))
                continue
            transcript, call = provider.transcribe_audio(title_id=title_id, scene_id=scene_id, audio_path=audio_path)
            cloud_by_scene[scene_id] = transcript
            call_records.append(call)
        for record in evidence:
            scene_id = str(record.get("scene_id") or "")
            if scene_id in cloud_by_scene:
                record["cloud_asr"] = [{
                    "evidence_id": f"cloud-asr:{scene_id}",
                    "source_family": "openai-transcribe-family",
                    "text": cloud_by_scene[scene_id],
                }]

    reference, _, _ = parse_srt(structure_path)
    japanese, _, _ = parse_srt(japanese_path)
    japanese_by_number = {block.number: block.text for block in japanese}
    allowed_evidence_refs = _allowed_evidence_refs(evidence)
    decisions: list[dict[str, Any]] = []
    for offset in range(0, len(evidence), max(1, batch_size)):
        chunk = evidence[offset: offset + max(1, batch_size)]
        checkpoint = checkpoint_dir / f"decisions-{offset:06d}.json"
        if resume and checkpoint.exists():
            saved = _json(checkpoint)
            if not isinstance(saved.get("response"), dict) or not isinstance(saved.get("call_record"), dict):
                raise ValueError(
                    f"Untraceable legacy checkpoint cannot be resumed: {checkpoint}. "
                    "Delete the checkpoint or restart without --resume."
                )
            response = saved["response"]
            saved_call = saved["call_record"]
            restored_call = {**saved_call, "checkpoint_replay": True}
            call_records.append(restored_call)
            restore_spent = getattr(getattr(provider, "budget", None), "restore_spent", None)
            if callable(restore_spent):
                restore_spent(float(saved_call.get("cost_usd", 0.0) or 0.0))
        else:
            payload = {
                "schema_name": "translation-forensics/autonomous-generation-request",
                "schema_version": "1",
                "title_id": title_id,
                "requested_block_numbers": [item["block_number"] for item in chunk],
                "title_memory": _title_memory(decisions, japanese_by_number),
                "evidence": chunk,
            }
            response, call = provider.generate_decisions(title_id=title_id, prompt=decision_prompt, payload=payload, schema=decision_schema)
            call_records.append(call)
            _write_json(checkpoint, {
                "schema_name": "translation-forensics/autonomous-checkpoint",
                "schema_version": "1",
                "response": response,
                "call_record": call,
            })
        rows = _validate_decisions(
            title_id,
            [item["block_number"] for item in chunk],
            response.get("results"),
            allowed_evidence_refs,
        )
        decisions.extend(rows)

    evidence_by_number = {int(item["block_number"]): item for item in evidence}
    decision_by_number = {int(item["block_number"]): item for item in decisions}
    risky = [
        number for number, decision in decision_by_number.items()
        if decision.get("confidence") != "high"
        or decision.get("source_status") == "abstained"
        or decision.get("viewer_status") != "supported"
        or decision.get("risk_codes")
        or decision.get("inferred_slots")
        or decision.get("competing_interpretations")
        or evidence_by_number[number].get("risk_codes")
    ]
    critiques: list[dict[str, Any]] = []
    for offset in range(0, len(risky), max(1, batch_size)):
        numbers = risky[offset: offset + max(1, batch_size)]
        payload = {
            "schema_name": "translation-forensics/autonomous-critique-request",
            "schema_version": "1",
            "title_id": title_id,
            "requested_block_numbers": numbers,
            "items": [{"evidence": evidence_by_number[number], "candidate": decision_by_number[number]} for number in numbers],
        }
        response, call = provider.critique_decisions(title_id=title_id, prompt=critic_prompt, payload=payload, schema=critique_schema)
        call_records.append(call)
        reviews = _validate_critiques(title_id, numbers, response.get("reviews"))
        critiques.extend([_critique_record(review, 0) for review in reviews])
        for review in reviews:
            number = int(review["block_number"])
            if review["verdict"] == "accept":
                continue
            if review["verdict"] == "quarantine":
                decision_by_number[number] = _quarantine(decision_by_number[number], review)
                continue
            current_review = review
            repaired = False
            last_candidate = decision_by_number[number]
            for attempt in range(1, max_repairs + 1):
                repair_payload = {
                    "schema_name": "translation-forensics/autonomous-repair-request",
                    "schema_version": "1",
                    "title_id": title_id,
                    "requested_block_numbers": [number],
                    "title_memory": _title_memory(list(decision_by_number.values()), japanese_by_number),
                    "evidence": [evidence_by_number[number]],
                    "repair_request": {"candidate": decision_by_number[number], "critique": current_review},
                }
                repaired_response, repair_call = provider.generate_decisions(title_id=title_id, prompt=decision_prompt, payload=repair_payload, schema=decision_schema)
                call_records.append(repair_call)
                candidate = _validate_decisions(title_id, [number], repaired_response.get("results"), allowed_evidence_refs)[0]
                last_candidate = candidate
                critic_payload = {
                    "schema_name": "translation-forensics/autonomous-critique-request",
                    "schema_version": "1",
                    "title_id": title_id,
                    "requested_block_numbers": [number],
                    "items": [{"evidence": evidence_by_number[number], "candidate": candidate}],
                }
                critic_response, critic_call = provider.critique_decisions(title_id=title_id, prompt=critic_prompt, payload=critic_payload, schema=critique_schema)
                call_records.append(critic_call)
                current_review = _validate_critiques(title_id, [number], critic_response.get("reviews"))[0]
                critiques.append(_critique_record(current_review, attempt))
                if current_review["verdict"] == "accept":
                    decision_by_number[number] = candidate
                    repaired = True
                    break
                if current_review["verdict"] == "quarantine":
                    break
            if not repaired:
                decision_by_number[number] = _quarantine(last_candidate, current_review)

    final_decisions = [decision_by_number[block.number] for block in reference]
    _validate_decisions(title_id, [block.number for block in reference], final_decisions, allowed_evidence_refs)
    if not _model_calls_traceable(call_records):
        raise ValueError("A model call is missing a request/response provenance hash.")
    regression_path = decision_prompt_path.parent.parent / "regressions" / "autonomous-release-v1.json"
    regression = validate_autonomous_regressions(
        title_id=title_id,
        decisions=final_decisions,
        critiques=critiques,
        suite_path=regression_path,
    )
    if regression["status"] == "fail":
        raise ValueError(f"Autonomous regression failure: {regression['errors']}")
    source_blocks: list[SubtitleBlock] = []
    viewer_blocks: list[SubtitleBlock] = []
    ledger: list[dict[str, Any]] = []
    for block, decision in zip(reference, final_decisions):
        source_text = str(decision.get("source_faithful_korean") or "").strip() or "…"
        viewer_text = str(decision["viewer_natural_korean"]).strip()
        source_blocks.append(SubtitleBlock(block.number, block.start, block.end, source_text, block.start_seconds, block.end_seconds))
        viewer_blocks.append(SubtitleBlock(block.number, block.start, block.end, viewer_text, block.start_seconds, block.end_seconds))
        ledger.append({
            "schema_name": "translation-forensics/autonomous-decision",
            "schema_version": "1",
            **decision,
            "source_srt_text": source_text,
            "human_reviewed": False,
            "human_final_allowed": False,
            "final_promotion_allowed": False,
        })

    source_output = output_dir / f"{title_id}.source-faithful-ko.autonomous-release-v1.srt"
    viewer_output = output_dir / f"{title_id}.viewer-complete-ko.autonomous-release-v1.srt"
    structure_copy = output_dir / "structure.srt"
    decisions_path = output_dir / "autonomous-decisions.jsonl"
    critiques_path = output_dir / "autonomous-critiques.jsonl"
    evidence_path = output_dir / "autonomous-evidence.jsonl"
    uncertainty_path = output_dir / "uncertainty-map.jsonl"
    memory_path = output_dir / "title-memory.json"
    alignment_path = output_dir / "machine-alignment.json"
    calls_path = output_dir / "model-call-manifest.jsonl"
    graph_path = output_dir / "evidence-graph.json"
    qa_path = output_dir / "qa-report.json"
    report_path = output_dir / "autonomous-release-report.json"
    proof_path = output_dir / "autonomous-proof.json"
    manifest_path = output_dir / "autonomous-release-manifest.json"
    packaged_prompts = output_dir / "prompts"
    packaged_schemas = output_dir / "schemas"
    packaged_regressions = output_dir / "regressions"
    packaged_prompts.mkdir(exist_ok=True)
    packaged_schemas.mkdir(exist_ok=True)
    packaged_regressions.mkdir(exist_ok=True)
    prompt_sources = [decision_prompt_path, critic_prompt_path]
    for name in ("autonomous-subtitle-decision-v1.manifest.json", "autonomous-subtitle-decision-v1.cases.jsonl"):
        candidate = decision_prompt_path.parent / name
        if candidate.exists():
            prompt_sources.append(candidate)
    prompt_copies = [packaged_prompts / source.name for source in prompt_sources]
    for source, target in zip(prompt_sources, prompt_copies):
        shutil.copy2(source, target)
    schema_sources = [decision_schema_path, critique_schema_path]
    for name in (
        "autonomous-evidence-bundle.schema.json",
        "autonomous-decision.schema.json",
        "autonomous-critique.schema.json",
        "model-call-manifest.schema.json",
        "autonomous-proof.schema.json",
    ):
        candidate = decision_schema_path.parent / name
        if candidate.exists():
            schema_sources.append(candidate)
    schema_sources = list(dict.fromkeys(schema_sources))
    schema_copies = [packaged_schemas / path.name for path in schema_sources]
    for source, target in zip(schema_sources, schema_copies):
        shutil.copy2(source, target)
    regression_copies: list[Path] = []
    if regression_path.exists():
        target = packaged_regressions / regression_path.name
        shutil.copy2(regression_path, target)
        regression_copies.append(target)
    shutil.copy2(structure_path, structure_copy)
    write_srt(source_output, source_blocks)
    write_srt(viewer_output, viewer_blocks)
    _write_jsonl(decisions_path, ledger)
    _write_jsonl(critiques_path, critiques)
    _write_jsonl(evidence_path, evidence)
    _write_jsonl(uncertainty_path, [row for row in ledger if row["source_status"] == "abstained"])
    _write_json(memory_path, {"schema_name": "translation-forensics/title-memory", "schema_version": "1", "title_id": title_id, "records": _title_memory(final_decisions, japanese_by_number)})
    _write_json(alignment_path, alignment)
    _write_jsonl(calls_path, call_records)
    _write_json(graph_path, _build_evidence_graph(evidence, ledger, critiques))
    srt_qa = validate_pair(structure_copy, source_output, viewer_output, project_root=structure_path.parents[1] if len(structure_path.parents) > 1 else None)
    qa = {
        "schema_name": "translation-forensics/autonomous-qa-report",
        "schema_version": "1",
        "status": "fail" if srt_qa.get("status") == "fail" or regression.get("status") == "fail" else srt_qa.get("status"),
        "structure_same": bool(srt_qa.get("structure_same")),
        "srt_validation": srt_qa,
        "semantic_slot_coverage": sum(1 for row in ledger if row.get("semantic_slots")) / max(1, len(ledger)),
        "accepted_critical_conflicts": sum(1 for row in ledger if row["source_status"] == "accepted" and any(str(code).startswith("critical:") for code in row.get("risk_codes", []))),
        "regression": regression,
        "readability": {
            "source_status": (srt_qa.get("source_faithful") or {}).get("status") if isinstance(srt_qa.get("source_faithful"), dict) else None,
            "viewer_status": (srt_qa.get("viewer_natural") or {}).get("status") if isinstance(srt_qa.get("viewer_natural"), dict) else None,
        },
    }
    _write_json(qa_path, qa)
    counts = Counter(row["source_status"] for row in ledger)
    viewer_counts = Counter(row["viewer_status"] for row in ledger)
    critical_conflicts = sum(1 for row in ledger if row["source_status"] == "accepted" and any(str(code).startswith("critical:") for code in row.get("risk_codes", [])))
    report = {
        "schema_name": "translation-forensics/autonomous-release-report",
        "schema_version": "1",
        "title_id": title_id,
        "release_kind": AUTONOMOUS_RELEASE_KIND,
        "status": "autonomous-release-packaged" if qa.get("status") != "fail" and critical_conflicts == 0 else "autonomous-release-invalid",
        "blocks": len(reference),
        "source_accepted": counts["accepted"],
        "source_abstained": counts["abstained"],
        "viewer_status_counts": dict(sorted(viewer_counts.items())),
        "viewer_empty_blocks": 0,
        "critical_conflicts_in_accepted": critical_conflicts,
        "model_calls": len(call_records),
        "estimated_cost_usd": round(sum(float(row.get("cost_usd", 0.0) or 0.0) for row in call_records), 6),
        "external_transfer_calls": sum(1 for row in call_records if row.get("external_transfer")),
        "cache_replay_calls": sum(1 for row in call_records if row.get("cache_hit")),
        "all_model_calls_traceable": True,
        "alignment_status": alignment["status"],
        "human_reviewed": False,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
        "human_reference_equality": "unidentifiable",
        "100_percent_equal": False,
    }
    _write_json(report_path, report)
    proof = {
        "schema_name": "translation-forensics/autonomous-proof",
        "schema_version": "1",
        "title_id": title_id,
        "release_kind": AUTONOMOUS_RELEASE_KIND,
        "structure_preserved": bool(qa.get("structure_same")),
        "viewer_block_coverage": 1.0,
        "source_decision_coverage": 1.0,
        "source_acceptance_rate": round(counts["accepted"] / max(1, len(reference)), 6),
        "source_abstention_rate": round(counts["abstained"] / max(1, len(reference)), 6),
        "critical_conflicts_in_accepted": critical_conflicts,
        "all_model_calls_traceable": True,
        "external_transfer_calls": report["external_transfer_calls"],
        "cache_replay_byte_deterministic": True,
        "live_model_byte_deterministic": False,
        "human_reference_equality": "unidentifiable",
        "100_percent_equal": False,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
    }
    _write_json(proof_path, proof)
    artifact_paths = [structure_copy, source_output, viewer_output, decisions_path, critiques_path, evidence_path, uncertainty_path, memory_path, alignment_path, calls_path, graph_path, qa_path, report_path, proof_path, *prompt_copies, *schema_copies, *regression_copies]
    manifest = {
        "schema_name": "translation-forensics/autonomous-release-manifest",
        "schema_version": "1",
        "title_id": title_id,
        "release_kind": AUTONOMOUS_RELEASE_KIND,
        "inputs": [_portable_manifest_input(path, structure_path.parent.parent) for path in [structure_path, japanese_path, *[path for path in (previous_path, scenes_path, local_asr_path) if path and path.exists()]]],
        "outputs": [{"path": path.relative_to(output_dir).as_posix(), "sha256": _sha256(path)} for path in artifact_paths],
        "prompt_hashes": {"decision": _sha256(decision_prompt_path), "critic": _sha256(critic_prompt_path)},
        "network_enabled": True,
        "external_transfer_calls": report["external_transfer_calls"],
        "cache_replay_calls": report["cache_replay_calls"],
        "human_reviewed": False,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
    }
    _write_json(manifest_path, manifest)
    if report["status"] != "autonomous-release-packaged":
        raise ValueError("Autonomous release validation failed; checkpoint and invalid package were retained.")
    return {
        "status": report["status"],
        "title_id": title_id,
        "output": str(output_dir),
        "blocks": len(reference),
        "source_accepted": counts["accepted"],
        "source_abstained": counts["abstained"],
        "viewer_complete": True,
        "estimated_cost_usd": report["estimated_cost_usd"],
        "human_final_allowed": False,
        "final_promotion_allowed": False,
    }


def validate_autonomous_release(package_dir: Path) -> dict[str, Any]:
    package_dir = package_dir.expanduser().resolve()
    manifest_path = package_dir / "autonomous-release-manifest.json"
    report_path = package_dir / "autonomous-release-report.json"
    proof_path = package_dir / "autonomous-proof.json"
    decisions_path = package_dir / "autonomous-decisions.jsonl"
    critiques_path = package_dir / "autonomous-critiques.jsonl"
    evidence_path = package_dir / "autonomous-evidence.jsonl"
    calls_path = package_dir / "model-call-manifest.jsonl"
    structure_path = package_dir / "structure.srt"
    qa_path = package_dir / "qa-report.json"
    required = {
        "autonomous-release-manifest.json": manifest_path,
        "autonomous-release-report.json": report_path,
        "autonomous-proof.json": proof_path,
        "autonomous-decisions.jsonl": decisions_path,
        "autonomous-critiques.jsonl": critiques_path,
        "autonomous-evidence.jsonl": evidence_path,
        "model-call-manifest.jsonl": calls_path,
        "structure.srt": structure_path,
        "uncertainty-map.jsonl": package_dir / "uncertainty-map.jsonl",
        "title-memory.json": package_dir / "title-memory.json",
        "machine-alignment.json": package_dir / "machine-alignment.json",
        "evidence-graph.json": package_dir / "evidence-graph.json",
        "qa-report.json": qa_path,
    }
    errors = [f"missing artifact: {name}" for name, path in required.items() if not path.exists()]
    if errors:
        return {"status": "fail", "package": str(package_dir), "errors": errors}

    try:
        manifest = _json(manifest_path)
        report = _json(report_path)
        proof = _json(proof_path)
        qa_report = _json(qa_path)
        reference, _, _ = parse_srt(structure_path)
        decisions = _jsonl(decisions_path)
        critiques = _jsonl(critiques_path)
        evidence = _jsonl(evidence_path)
        calls = _jsonl(calls_path)
    except Exception as exc:
        return {"status": "fail", "package": str(package_dir), "errors": [f"package parse failure: {exc}"]}

    outputs = manifest.get("outputs")
    output_paths: set[str] = set()
    if not isinstance(outputs, list):
        errors.append("manifest outputs must be an array")
        outputs = []
    for item in outputs:
        if not isinstance(item, dict):
            errors.append("manifest output record must be an object")
            continue
        raw = str(item.get("path") or "")
        raw_path = Path(raw)
        if not raw or raw_path.is_absolute() or ".." in raw_path.parts:
            errors.append(f"unsafe manifest output path: {raw!r}")
            continue
        resolved = (package_dir / raw_path).resolve()
        try:
            resolved.relative_to(package_dir)
        except ValueError:
            errors.append(f"manifest output escapes package: {raw}")
            continue
        normalized = resolved.relative_to(package_dir).as_posix()
        if normalized in output_paths:
            errors.append(f"duplicate manifest output: {normalized}")
            continue
        output_paths.add(normalized)
        if not resolved.is_file():
            errors.append(f"manifest output is missing: {normalized}")
        elif _sha256(resolved) != item.get("sha256"):
            errors.append(f"output hash mismatch: {normalized}")

    required_manifest_outputs = {
        "structure.srt",
        "autonomous-decisions.jsonl",
        "autonomous-critiques.jsonl",
        "autonomous-evidence.jsonl",
        "model-call-manifest.jsonl",
        "uncertainty-map.jsonl",
        "title-memory.json",
        "machine-alignment.json",
        "evidence-graph.json",
        "qa-report.json",
        "autonomous-release-report.json",
        "autonomous-proof.json",
    }
    for missing in sorted(required_manifest_outputs - output_paths):
        errors.append(f"required output is not recorded in manifest: {missing}")

    source_candidates = list(package_dir.glob("*.source-faithful-ko.autonomous-release-v*.srt"))
    viewer_candidates = list(package_dir.glob("*.viewer-complete-ko.autonomous-release-v*.srt"))
    if len(source_candidates) != 1 or len(viewer_candidates) != 1:
        errors.append("exactly one source and viewer autonomous SRT are required")
    else:
        for candidate in (*source_candidates, *viewer_candidates):
            if candidate.relative_to(package_dir).as_posix() not in output_paths:
                errors.append(f"SRT is not recorded in manifest: {candidate.name}")
        srt_qa = validate_pair(structure_path, source_candidates[0], viewer_candidates[0])
        if srt_qa.get("status") == "fail" or not srt_qa.get("structure_same"):
            errors.append("autonomous SRT structural/text validation failed")

    expected = [block.number for block in reference]
    evidence_numbers: list[int] = []
    allowed_refs: dict[int, set[str]] = {}
    try:
        evidence_numbers = [int(row.get("block_number", 0)) for row in evidence]
        allowed_refs = _allowed_evidence_refs(evidence)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid evidence record: {exc}")
    if len(evidence_numbers) != len(set(evidence_numbers)) or set(evidence_numbers) != set(expected):
        errors.append("evidence coverage mismatch")
    title_id = str(report.get("title_id") or "")
    if not title_id:
        errors.append("release report title_id is missing")
    if any(row.get("title_id") != title_id for row in evidence):
        errors.append("evidence title_id mismatch")
    try:
        validated_decisions = _validate_decisions(title_id, expected, decisions, allowed_refs)
    except Exception as exc:
        errors.append(f"decision validation failed: {exc}")
        validated_decisions = decisions

    expected_set = set(expected)
    reviewed: set[int] = set()
    for row in critiques:
        try:
            number = int(row.get("block_number", 0))
            attempt = int(row.get("attempt", 0))
        except (TypeError, ValueError):
            errors.append("critique block_number/attempt is invalid")
            continue
        if row.get("title_id") != title_id or number not in expected_set:
            errors.append(f"critique title/block mismatch: {number}")
        if row.get("verdict") not in {"accept", "repair", "quarantine"} or attempt < 0:
            errors.append(f"invalid critique record: {number}")
        reviewed.add(number)

    evidence_by_number: dict[int, dict[str, Any]] = {}
    for row in evidence:
        try:
            evidence_by_number[int(row.get("block_number", 0))] = row
        except (TypeError, ValueError):
            continue
    risky: set[int] = set()
    for row in validated_decisions:
        try:
            number = int(row.get("block_number", 0))
        except (TypeError, ValueError):
            continue
        evidence_row = evidence_by_number.get(number, {})
        if (
            row.get("confidence") != "high"
            or row.get("source_status") == "abstained"
            or row.get("viewer_status") != "supported"
            or row.get("risk_codes")
            or row.get("inferred_slots")
            or row.get("competing_interpretations")
            or evidence_row.get("risk_codes")
        ):
            risky.add(number)
    for number in sorted(risky - reviewed):
        errors.append(f"risky decision lacks critic review: {number}")

    calls_traceable = _model_calls_traceable(calls)
    if not calls_traceable:
        errors.append("model-call provenance is missing, empty, or invalid")
    try:
        expected_cost = round(sum(float(row.get("cost_usd", 0.0) or 0.0) for row in calls), 6)
    except (TypeError, ValueError):
        expected_cost = -1.0
        errors.append("model-call cost field is invalid")
    try:
        reported_calls = int(report.get("model_calls", -1))
    except (TypeError, ValueError):
        reported_calls = -1
        errors.append("release report model_calls is invalid")
    if reported_calls != len(calls):
        errors.append("release report model_calls does not match call manifest")
    try:
        reported_cost = float(report.get("estimated_cost_usd", -1.0))
    except (TypeError, ValueError):
        reported_cost = -1.0
        errors.append("release report estimated cost is invalid")
    if abs(reported_cost - expected_cost) > 1e-6:
        errors.append("release report estimated cost does not match call manifest")
    if report.get("all_model_calls_traceable") is not calls_traceable:
        errors.append("release report model-call provenance flag is inconsistent")
    if proof.get("all_model_calls_traceable") is not calls_traceable:
        errors.append("proof model-call provenance flag is inconsistent")

    schema_checks = (
        ("autonomous-evidence-bundle.schema.json", evidence),
        ("autonomous-decision.schema.json", decisions),
        ("autonomous-critique.schema.json", critiques),
        ("model-call-manifest.schema.json", calls),
        ("autonomous-proof.schema.json", [proof]),
    )
    for schema_name, records in schema_checks:
        schema_path = package_dir / "schemas" / schema_name
        if not schema_path.exists():
            errors.append(f"missing record schema: {schema_name}")
            continue
        validator = Draft202012Validator(_json(schema_path))
        for index, record in enumerate(records, 1):
            first_error = next(iter(validator.iter_errors(record)), None)
            if first_error is not None:
                errors.append(f"{schema_name} record {index}: {first_error.message}")

    if report.get("release_kind") != AUTONOMOUS_RELEASE_KIND:
        errors.append("release kind is invalid")
    if report.get("status") != "autonomous-release-packaged":
        errors.append("release report is not in packaged state")
    if report.get("human_final_allowed") is not False or report.get("final_promotion_allowed") is not False:
        errors.append("release report human-final boundary is invalid")
    if qa_report.get("status") == "fail":
        errors.append("packaged QA report is failing")
    if proof.get("human_reference_equality") != "unidentifiable" or proof.get("100_percent_equal") is not False:
        errors.append("human-reference claim boundary is invalid")
    if proof.get("human_final_allowed") is not False or proof.get("final_promotion_allowed") is not False:
        errors.append("proof human-final boundary is invalid")
    if manifest.get("title_id") != title_id:
        errors.append("manifest title_id mismatch")
    if manifest.get("final_promotion_allowed") is not False:
        errors.append("autonomous package must not allow final promotion")

    return {
        "status": "pass" if not errors else "fail",
        "package": str(package_dir),
        "title_id": title_id,
        "blocks": len(reference),
        "source_accepted": report.get("source_accepted"),
        "source_abstained": report.get("source_abstained"),
        "model_calls": len(calls),
        "errors": errors,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
    }


def prove_autonomous_claim(package_dir: Path, output_path: Path | None = None) -> dict[str, Any]:
    validation = validate_autonomous_release(package_dir)
    package_dir = package_dir.expanduser().resolve()
    report = _json(package_dir / "autonomous-release-report.json") if (package_dir / "autonomous-release-report.json").exists() else {}
    proof = {
        "schema_name": "translation-forensics/autonomous-claim-proof",
        "schema_version": "1",
        "status": "pass" if validation["status"] == "pass" else "fail",
        "title_id": report.get("title_id"),
        "release_kind": AUTONOMOUS_RELEASE_KIND,
        "structure_preserved": validation["status"] == "pass",
        "viewer_block_coverage": 1.0 if validation["status"] == "pass" else None,
        "source_accepted": report.get("source_accepted"),
        "source_abstained": report.get("source_abstained"),
        "model_calls": report.get("model_calls"),
        "estimated_cost_usd": report.get("estimated_cost_usd"),
        "human_reference_equality": "unidentifiable",
        "100_percent_equal": False,
        "human_reviewed": False,
        "human_final_allowed": False,
        "final_promotion_allowed": False,
        "validation_errors": validation.get("errors", []),
    }
    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        if output_path.exists():
            raise FileExistsError(f"Existing proof will not be overwritten: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output_path, proof)
    return proof
