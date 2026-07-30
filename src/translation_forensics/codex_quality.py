from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .codex_exec_provider import CodexExecProvider, canonical_json, sha256_json, validate_call_receipt
from .local_asr import FasterWhisperBackend, LocalASRError, ReazonSpeechBackend, run_conflict_asr_rerun
from .srt import SubtitleBlock, compare_structure, parse_srt, write_srt


CRITICAL_SLOTS = (
    "speech_act",
    "question",
    "polarity",
    "refusal_permission",
    "command_strength",
    "speaker",
    "actor",
    "action",
    "target",
    "location",
    "tense_aspect",
    "direction",
    "intensity",
)
TITLE_GATES = {
    "ADN-622": {"minimum_accepted_rate": 0.95, "maximum_ellipsis_rate": 0.02},
    "JUQ-439": {"minimum_accepted_rate": 0.90, "maximum_ellipsis_rate": 0.05},
    "SSIS-908": {"minimum_accepted_rate": 0.85, "maximum_ellipsis_rate": 0.08},
}
KOREAN_SHORT_RE = re.compile(r"^(?:아+|어+|응+|네+|싫어|안 돼|좋아|더|잠깐|…)[.!?…~]*$")


class CodexQualityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
        newline="\n",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        values.append(value)
    return values


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    try:
        portable = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        portable = str(path.resolve())
    return {"path": portable, "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _load_by_block(path: Path, expected: list[int], label: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        try:
            number = int(row.get("block_number"))
        except (TypeError, ValueError) as exc:
            raise CodexQualityError(f"{label} record lacks integer block_number") from exc
        if number in result:
            raise CodexQualityError(f"Duplicate {label} block {number}")
        result[number] = row
    if sorted(result) != expected:
        raise CodexQualityError(f"{label} coverage differs from locked SRT")
    return result


def build_scene_batches(
    blocks: list[SubtitleBlock],
    *,
    max_blocks: int = 20,
    maximum_gap_seconds: float = 15.0,
) -> list[list[SubtitleBlock]]:
    if max_blocks <= 0:
        raise ValueError("max_blocks must be positive")
    scenes: list[list[SubtitleBlock]] = []
    current: list[SubtitleBlock] = []
    for block in blocks:
        gap = block.start_seconds - current[-1].end_seconds if current else 0.0
        if current and (len(current) >= max_blocks or gap > maximum_gap_seconds):
            scenes.append(current)
            current = []
        current.append(block)
    if current:
        scenes.append(current)
    return scenes


def _normalized_slot(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    return re.sub(r"\s+", " ", str(value).strip().casefold()) or None

CRITICAL_FLIPPING_SLOTS = {"polarity", "refusal_permission", "command_strength"}

def _coherence_check(consensus_frame):
    issues = []
    sa = str(consensus_frame.get("speech_act") or "")
    q = consensus_frame.get("question")
    cs = str(consensus_frame.get("command_strength") or "")
    pol = consensus_frame.get("polarity")
    rp = consensus_frame.get("refusal_permission")
    if q is True and sa and sa not in ("question",):
        issues.append("question_true_but_speech_act_not_question")
    if q is False and sa == "question":
        issues.append("question_false_but_speech_act_question")
    if cs and sa and sa not in ("command", "request", "prohibition", "permission"):
        issues.append("command_strength_without_command_speech_act")
    if pol == "positive" and rp:
        issues.append("positive_polarity_with_refusal_permission")
    if pol == "negative" and rp == "granted":
        issues.append("negative_polarity_with_granted_permission")
    if consensus_frame.get("action") and not consensus_frame.get("actor"):
        issues.append("action_without_actor")
    return issues


def _graduated_source_status(conflicts, coverage_gaps, source_quality_status, num_asr_families, consensus_frame):
    flipping_conflicts = [s for s in conflicts if s in CRITICAL_FLIPPING_SLOTS]
    descriptive_conflicts = [s for s in conflicts if s not in CRITICAL_FLIPPING_SLOTS]
    coherence_issues = _coherence_check(consensus_frame)
    if source_quality_status in ("suspect", "unusable") and num_asr_families < 2:
        return "abstained", "unrecoverable", ["damaged_source_without_dual_acoustic_evidence"]
    has_agreed_speech_act = bool(consensus_frame.get("speech_act"))
    flipping_dangerous = bool(flipping_conflicts) or any(
        issue in coherence_issues for issue in [
            "positive_polarity_with_refusal_permission",
            "negative_polarity_with_granted_permission",
            "question_false_but_speech_act_question",
        ]
    )
    if not has_agreed_speech_act and flipping_dangerous:
        return "abstained", "unrecoverable", flipping_conflicts + coherence_issues + ["no_agreed_semantic_core"]
    if flipping_dangerous:
        return "accepted", "best_effort", flipping_conflicts + coherence_issues + ["meaning_flipping_conflict"]
    if descriptive_conflicts:
        return "accepted", "best_effort", descriptive_conflicts + ["descriptive_coverage_gap"]
    if coherence_issues:
        return "accepted", "best_effort", coherence_issues
    return "accepted", "supported", []


def _semantic_slot_value(slot: str, value: Any) -> Any:
    normalized = _normalized_slot(value)
    if normalized in {None, "unknown"}:
        return None
    if slot in {"refusal_permission", "command_strength"} and normalized == "none":
        return None
    if slot == "polarity" and normalized == "neutral":
        return None
    return normalized


def compare_independent_frames(
    terra_frames: list[dict[str, Any]],
    sol_frames: list[dict[str, Any]],
    expected_blocks: list[int],
) -> list[dict[str, Any]]:
    def keyed(rows: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for row in rows:
            number = int(row.get("block_number", 0) or 0)
            if number in result:
                raise CodexQualityError(f"Duplicate {label} frame for block {number}")
            result[number] = row
        if sorted(result) != expected_blocks:
            raise CodexQualityError(f"{label} frame coverage differs from scene blocks")
        return result

    terra = keyed(terra_frames, "Terra")
    sol = keyed(sol_frames, "Sol")
    agreements: list[dict[str, Any]] = []
    for number in expected_blocks:
        left, right = terra[number], sol[number]
        left_values = {slot: _semantic_slot_value(slot, left.get(slot)) for slot in CRITICAL_SLOTS}
        right_values = {slot: _semantic_slot_value(slot, right.get(slot)) for slot in CRITICAL_SLOTS}
        conflicts = [
            slot
            for slot in CRITICAL_SLOTS
            if left_values[slot] is not None
            and right_values[slot] is not None
            and left_values[slot] != right_values[slot]
        ]
        coverage_gaps = [
            slot
            for slot in CRITICAL_SLOTS
            if (left_values[slot] is None) != (right_values[slot] is None)
        ]
        consensus = {
            slot: left_values[slot] if left_values[slot] == right_values[slot] else None
            for slot in CRITICAL_SLOTS
        }
        agreements.append(
            {
                "schema_name": "translation-forensics/codex-frame-agreement",
                "schema_version": "1",
                "block_number": number,
                "terra_frame_sha256": sha256_json(left),
                "sol_frame_sha256": sha256_json(right),
                "critical_slot_conflicts": conflicts,
                "slot_coverage_gaps": coverage_gaps,
                "consensus_frame": consensus,
                "agreed": not conflicts,
                "fully_agreed": not conflicts and not coverage_gaps,
            }
        )
    return agreements


def _response_rows(response: dict[str, Any], field: str, expected: list[int], scene_id: str) -> list[dict[str, Any]]:
    if response.get("scene_id") != scene_id:
        raise CodexQualityError(f"Response scene_id mismatch: {response.get('scene_id')!r} != {scene_id!r}")
    rows = response.get(field)
    if not isinstance(rows, list):
        raise CodexQualityError(f"Response {field} must be an array")
    numbers = [int(row.get("block_number", 0) or 0) for row in rows if isinstance(row, dict)]
    if numbers != expected:
        raise CodexQualityError(f"Response {field} must cover scene blocks once and in order")
    return rows


def _scene_payload(
    scene_id: str,
    scene: list[SubtitleBlock],
    source_quality: dict[int, dict[str, Any]],
    acoustic: dict[int, dict[str, Any]],
    *,
    context_before: list[SubtitleBlock] | None = None,
    context_after: list[SubtitleBlock] | None = None,
) -> dict[str, Any]:
    def context_rows(blocks: list[SubtitleBlock]) -> list[dict[str, Any]]:
        return [
            {"block_number": block.number, "start": block.start, "end": block.end, "japanese": block.text}
            for block in blocks
        ]

    acoustic_windows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for block in scene:
        for transcript in acoustic[block.number].get("transcripts", []):
            key = (
                str(transcript.get("window_id") or ""),
                str(transcript.get("source_family") or ""),
                str(transcript.get("audio_sha256") or ""),
            )
            acoustic_windows.setdefault(key, transcript)
    return {
        "scene_id": scene_id,
        "locked_blocks": [
            {
                "block_number": block.number,
                "start": block.start,
                "end": block.end,
                "japanese_srt": block.text,
                "source_quality_status": source_quality[block.number]["source_quality_status"],
                "source_quality_reasons": source_quality[block.number].get("reason_codes", []),
                "evidence_refs": acoustic[block.number].get("evidence_refs", []),
                "independent_source_families": acoustic[block.number].get("independent_source_families", []),
            }
            for block in scene
        ],
        "acoustic_windows": list(acoustic_windows.values()),
        "neighbor_context": {
            "before": context_rows(context_before or []),
            "after": context_rows(context_after or []),
        },
        "screen_context_policy": "not-provided; never infer spoken meaning from pixels",
    }


def _persist_call(
    *,
    provider: CodexExecProvider,
    role: str,
    title_id: str,
    call_id: str,
    prompt: str,
    payload: dict[str, Any],
    schema: dict[str, Any],
    scene_dir: Path,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response, receipt = provider.run_structured(
        role=role,
        title_id=title_id,
        call_id=call_id,
        prompt=prompt,
        payload=payload,
        schema=schema,
        resume=resume,
    )
    _write_json(scene_dir / f"{call_id}.response.json", response)
    _write_json(scene_dir / f"{call_id}.receipt.json", receipt)
    return response, receipt


def _enforce_evidence_gate(
    rows: list[dict[str, Any]],
    agreements: dict[int, dict[str, Any]],
    source_quality: dict[int, dict[str, Any]],
    acoustic: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Graduated gate using _graduated_source_status per block."""
    enforced: list[dict[str, Any]] = []
    for row in rows:
        number = int(row["block_number"])
        quality_status = str(source_quality[number]["source_quality_status"])
        families = sorted(set(acoustic[number].get("independent_source_families", [])))
        agreement = agreements.get(number, {})
        conflicts = list(agreement.get("critical_slot_conflicts", []))
        gaps = agreement.get("slot_coverage_gaps", [])
        consensus_frame = agreement.get("consensus_frame", {})
        output = dict(row)
        grad_status, grad_quality, grad_issues = _graduated_source_status(
            conflicts, gaps, quality_status, len(families), consensus_frame
        )
        output["source_status"] = grad_status
        output["viewer_status"] = grad_quality
        if grad_issues:
            output["reason"] = "; ".join(grad_issues)
        if grad_status in {"abstained", "unresolved"}:
            output["source_faithful_korean"] = str(output.get("source_faithful_korean") or "…").strip() or "…"
            output["viewer_natural_korean"] = str(output.get("viewer_natural_korean") or "…").strip() or "…"
        enforced.append(output)
    return enforced


def _quarantine_reviews(rows: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Graduated quarantine: only block if unsupported critical meaning, incoherence, or empty core."""
    by_number = {int(row["block_number"]): row for row in reviews}
    result: list[dict[str, Any]] = []
    for row in rows:
        number = int(row["block_number"])
        review = by_number[number]
        verdict = review.get("verdict", "")
        has_unsupported = bool(review.get("unsupported_additions"))
        has_conflicts = bool(review.get("critical_slot_conflicts"))
        if verdict == "quarantine" and (has_unsupported or has_conflicts):
            result.append(
                {
                    **row,
                    "source_faithful_korean": "…",
                    "viewer_natural_korean": "…",
                    "source_status": "abstained",
                    "viewer_status": "unrecoverable",
                    "reason": f"Sol quarantine after repair budget: {review.get('reason')}",
                }
            )
        else:
            result.append(row)
    return result


def evaluate_evidence_ceiling(
    *,
    title_id: str,
    expected_blocks: list[int],
    source_quality: dict[int, dict[str, Any]],
    acoustic: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    eligible = [
        number
        for number in expected_blocks
        if source_quality[number].get("source_quality_status") == "trusted"
        or len(set(acoustic[number].get("independent_source_families", []))) >= 2
    ]
    maximum_rate = len(eligible) / max(1, len(expected_blocks))
    gate = TITLE_GATES.get(title_id)
    minimum_rate = float(gate["minimum_accepted_rate"]) if gate else 0.0
    return {
        "schema_name": "translation-forensics/codex-quality-evidence-ceiling",
        "schema_version": "1",
        "title_id": title_id,
        "status": "pass" if maximum_rate >= minimum_rate else "fail",
        "block_count": len(expected_blocks),
        "eligible_block_count": len(eligible),
        "eligible_block_numbers": eligible,
        "maximum_possible_accepted_rate": round(maximum_rate, 6),
        "minimum_required_accepted_rate": minimum_rate,
        "policy": "trusted Japanese or two block-local independent ASR families",
        "model_calls_allowed": maximum_rate >= minimum_rate,
        "human_equal": False,
        "human_final": False,
        "final_promotion_allowed": False,
    }


def run_codex_quality_title(
    *,
    title_id: str,
    structure_path: Path,
    source_quality_map_path: Path,
    acoustic_evidence_path: Path,
    audio_path: Path | None,
    output_dir: Path,
    provider: CodexExecProvider,
    prompt_dir: Path,
    schema_dir: Path,
    max_scene_blocks: int = 20,
    maximum_scene_gap_seconds: float = 60.0,
    max_repairs: int = 2,
    force_cpu: bool = False,
    allow_model_download: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file() and resume:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("inputs", {}).get("structure", {}).get("sha256") == _sha256(structure_path)
            and existing.get("inputs", {}).get("source_quality_map", {}).get("sha256") == _sha256(source_quality_map_path)
            and existing.get("inputs", {}).get("acoustic_evidence", {}).get("sha256") == _sha256(acoustic_evidence_path)
        ):
            return {**existing, "cache_hit": True, "output": str(output_dir)}
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Codex quality package already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    structure, _, _ = parse_srt(structure_path)
    expected = [block.number for block in structure]
    source_quality = _load_by_block(source_quality_map_path, expected, "source quality")
    acoustic = _load_by_block(acoustic_evidence_path, expected, "acoustic evidence")
    ceiling = evaluate_evidence_ceiling(
        title_id=title_id,
        expected_blocks=expected,
        source_quality=source_quality,
        acoustic=acoustic,
    )
    if ceiling["status"] != "pass":
        ceiling_path = output_dir / "evidence-feasibility.json"
        _write_json(ceiling_path, ceiling)
        manifest = {
            "schema_name": "translation-forensics/codex-quality-manifest",
            "schema_version": "1",
            "title_id": title_id,
            "release_kind": "autonomous-quality-candidate",
            "status": "evidence-ceiling-failed",
            "inputs": {
                "structure": _artifact(structure_path, output_dir),
                "source_quality_map": _artifact(source_quality_map_path, output_dir),
                "acoustic_evidence": _artifact(acoustic_evidence_path, output_dir),
            },
            "evidence_feasibility": _artifact(ceiling_path, output_dir),
            "block_count": len(structure),
            "model_call_count": 0,
            "all_model_calls_traceable": True,
            "api_key_used": False,
            "human_equal": False,
            "human_final": False,
            "final_promotion_allowed": False,
        }
        _write_json(manifest_path, manifest)
        return {**manifest, "cache_hit": False, "output": str(output_dir)}
    scenes = build_scene_batches(
        structure,
        max_blocks=max_scene_blocks,
        maximum_gap_seconds=maximum_scene_gap_seconds,
    )
    prompts = {
        "meaning": (prompt_dir / "codex-quality-meaning-frame-v1.md").read_text(encoding="utf-8"),
        "translation": (prompt_dir / "codex-quality-translation-v1.md").read_text(encoding="utf-8"),
        "critic": (prompt_dir / "codex-quality-critic-v1.md").read_text(encoding="utf-8"),
        "repair": (prompt_dir / "codex-quality-repair-v1.md").read_text(encoding="utf-8"),
    }
    schemas = {
        "frame": json.loads((schema_dir / "codex-quality-frame-response.schema.json").read_text(encoding="utf-8")),
        "translation": json.loads((schema_dir / "codex-quality-translation-response.schema.json").read_text(encoding="utf-8")),
        "critique": json.loads((schema_dir / "codex-quality-critique-response.schema.json").read_text(encoding="utf-8")),
    }
    all_decisions: list[dict[str, Any]] = []
    all_agreements: list[dict[str, Any]] = []
    all_receipts: list[dict[str, Any]] = []
    all_critiques: list[dict[str, Any]] = []
    conflict_backends: list[Any] | None = None

    for scene_index, scene in enumerate(scenes):
        scene_id = f"scene-{scene_index + 1:04d}"
        numbers = [block.number for block in scene]
        before = scenes[scene_index - 1][-2:] if scene_index else []
        after = scenes[scene_index + 1][:2] if scene_index + 1 < len(scenes) else []
        payload = _scene_payload(scene_id, scene, source_quality, acoustic, context_before=before, context_after=after)
        scene_dir = output_dir / "scenes" / scene_id
        with ThreadPoolExecutor(max_workers=2) as pool:
            terra_future = pool.submit(
                _persist_call,
                provider=provider,
                role="meaning-frame-terra",
                title_id=title_id,
                call_id=f"{scene_id}.meaning.terra",
                prompt=prompts["meaning"],
                payload=payload,
                schema=schemas["frame"],
                scene_dir=scene_dir,
                resume=resume,
            )
            sol_future = pool.submit(
                _persist_call,
                provider=provider,
                role="meaning-frame-sol",
                title_id=title_id,
                call_id=f"{scene_id}.meaning.sol",
                prompt=prompts["meaning"],
                payload=payload,
                schema=schemas["frame"],
                scene_dir=scene_dir,
                resume=resume,
            )
            terra_response, terra_receipt = terra_future.result()
            sol_response, sol_receipt = sol_future.result()
        terra_frames = _response_rows(terra_response, "frames", numbers, scene_id)
        sol_frames = _response_rows(sol_response, "frames", numbers, scene_id)
        agreements = compare_independent_frames(terra_frames, sol_frames, numbers)
        rerun_performed = False
        rerun_error = ""
        conflict_numbers = [
            int(agreement["block_number"])
            for agreement in agreements
            if agreement["critical_slot_conflicts"]
        ]
        if conflict_numbers and audio_path is not None:
            try:
                if conflict_backends is None:
                    conflict_backends = [
                        FasterWhisperBackend(
                            force_cpu=force_cpu,
                            local_files_only=not allow_model_download,
                        ),
                        ReazonSpeechBackend(),
                    ]
                by_block = {block.number: block for block in scene}
                rerun_evidence = run_conflict_asr_rerun(
                    title_id=title_id,
                    audio_path=audio_path,
                    blocks=[by_block[number] for number in conflict_numbers],
                    output_dir=scene_dir / "conflict-asr-rerun-v2",
                    force_cpu=force_cpu,
                    allow_model_download=allow_model_download,
                    resume=resume,
                    backends=conflict_backends,
                )
                for number, evidence in rerun_evidence.items():
                    acoustic[number] = {
                        **evidence,
                        "transcripts": acoustic[number].get("transcripts", []) + evidence.get("transcripts", []),
                        "evidence_refs": sorted(
                            set(acoustic[number].get("evidence_refs", [])) | set(evidence.get("evidence_refs", []))
                        ),
                        "independent_source_families": sorted(
                            set(acoustic[number].get("independent_source_families", []))
                            | set(evidence.get("independent_source_families", []))
                        ),
                    }
                retry_scene_id = f"{scene_id}-conflict-rerun"
                retry_scene = [by_block[number] for number in conflict_numbers]
                rerun_payload = _scene_payload(
                    retry_scene_id,
                    retry_scene,
                    source_quality,
                    acoustic,
                    context_before=before,
                    context_after=after,
                )
                rerun_payload["original_scene_context"] = [
                    {
                        "block_number": block.number,
                        "start": block.start,
                        "end": block.end,
                        "japanese_srt": block.text,
                    }
                    for block in scene
                ]
                rerun_payload["boundary_expansion_asr_rerun"] = {
                    "performed": True,
                    "block_numbers": conflict_numbers,
                    "maximum_clip_seconds": 28.0,
                }
                with ThreadPoolExecutor(max_workers=2) as pool:
                    terra_retry_future = pool.submit(
                        _persist_call,
                        provider=provider,
                        role="meaning-frame-terra",
                        title_id=title_id,
                        call_id=f"{scene_id}.meaning-rerun.terra",
                        prompt=prompts["meaning"],
                        payload=rerun_payload,
                        schema=schemas["frame"],
                        scene_dir=scene_dir,
                        resume=resume,
                    )
                    sol_retry_future = pool.submit(
                        _persist_call,
                        provider=provider,
                        role="meaning-frame-sol",
                        title_id=title_id,
                        call_id=f"{scene_id}.meaning-rerun.sol",
                        prompt=prompts["meaning"],
                        payload=rerun_payload,
                        schema=schemas["frame"],
                        scene_dir=scene_dir,
                        resume=resume,
                    )
                    terra_retry, terra_retry_receipt = terra_retry_future.result()
                    sol_retry, sol_retry_receipt = sol_retry_future.result()
                retry_agreements = {
                    int(row["block_number"]): row
                    for row in compare_independent_frames(
                        _response_rows(terra_retry, "frames", conflict_numbers, retry_scene_id),
                        _response_rows(sol_retry, "frames", conflict_numbers, retry_scene_id),
                        conflict_numbers,
                    )
                }
                agreements = [
                    retry_agreements.get(int(agreement["block_number"]), agreement)
                    for agreement in agreements
                ]
                all_receipts.extend((terra_retry_receipt, sol_retry_receipt))
                rerun_performed = True
            except (FileExistsError, LocalASRError, OSError, RuntimeError, ValueError) as exc:
                rerun_error = str(exc)
        agreement_by_number = {int(row["block_number"]): row for row in agreements}
        all_agreements.extend(
            {
                **row,
                "scene_id": scene_id,
                "boundary_expansion_asr_rerun": rerun_performed,
                "boundary_expansion_asr_rerun_error": rerun_error,
            }
            for row in agreements
        )
        all_receipts.extend((terra_receipt, sol_receipt))

        translation_payload = {
            **payload,
            "consensus": [
                {
                    **agreement,
                    "forced_source_status": "accepted",
                    "render_blocking_conflicts": agreement.get("critical_slot_conflicts", []),
                }
                for agreement in agreements
            ],
        }
        translation_response, translation_receipt = _persist_call(
            provider=provider,
            role="translation-terra",
            title_id=title_id,
            call_id=f"{scene_id}.translation.terra",
            prompt=prompts["translation"],
            payload=translation_payload,
            schema=schemas["translation"],
            scene_dir=scene_dir,
            resume=resume,
        )
        candidate_rows = _response_rows(translation_response, "blocks", numbers, scene_id)
        candidate_rows = _enforce_evidence_gate(candidate_rows, agreement_by_number, source_quality, acoustic)
        all_receipts.append(translation_receipt)
        candidate_by_number = {int(row["block_number"]): row for row in candidate_rows}
        final_reviews_by_number: dict[int, dict[str, Any]] = {}
        active_numbers = list(numbers)
        by_block = {block.number: block for block in scene}

        for repair_attempt in range(max_repairs + 1):
            if repair_attempt == 0:
                critique_scene_id = scene_id
                critique_payload = {
                    **translation_payload,
                    "candidate_blocks": [candidate_by_number[number] for number in active_numbers],
                    "repair_attempt": repair_attempt,
                }
            else:
                critique_scene_id = f"{scene_id}-repair-review-{repair_attempt}"
                critique_payload = _scene_payload(
                    critique_scene_id,
                    [by_block[number] for number in active_numbers],
                    source_quality,
                    acoustic,
                    context_before=before,
                    context_after=after,
                )
                critique_payload.update(
                    {
                        "consensus": [agreement_by_number[number] for number in active_numbers],
                        "candidate_blocks": [candidate_by_number[number] for number in active_numbers],
                        "repair_attempt": repair_attempt,
                        "original_scene_id": scene_id,
                    }
                )
            critique_response, critique_receipt = _persist_call(
                provider=provider,
                role="critique-sol",
                title_id=title_id,
                call_id=f"{scene_id}.critique.sol.{repair_attempt}",
                prompt=prompts["critic"],
                payload=critique_payload,
                schema=schemas["critique"],
                scene_dir=scene_dir,
                resume=resume,
            )
            reviews = _response_rows(critique_response, "reviews", active_numbers, critique_scene_id)
            all_receipts.append(critique_receipt)
            all_critiques.extend(
                {
                    **review,
                    "scene_id": scene_id,
                    "evaluation_scene_id": critique_scene_id,
                    "repair_attempt": repair_attempt,
                }
                for review in reviews
            )
            final_reviews_by_number.update({int(review["block_number"]): review for review in reviews})
            repair_numbers = [
                int(review["block_number"])
                for review in reviews
                if review.get("verdict") == "repair"
                and not review.get("critical_slot_conflicts")
                and not review.get("unsupported_additions")
            ]
            blocking = [review for review in reviews if review.get("verdict") == "quarantine" or review.get("critical_slot_conflicts") or review.get("unsupported_additions")]
            if not repair_numbers and not blocking:
                break
            if repair_attempt >= max_repairs or not repair_numbers:
                break
            repair_scene_id = f"{scene_id}-repair-{repair_attempt + 1}"
            repair_payload = _scene_payload(
                repair_scene_id,
                [by_block[number] for number in repair_numbers],
                source_quality,
                acoustic,
                context_before=before,
                context_after=after,
            )
            repair_payload.update(
                {
                    "consensus": [agreement_by_number[number] for number in repair_numbers],
                    "candidate_blocks": [candidate_by_number[number] for number in repair_numbers],
                    "sol_reviews": [final_reviews_by_number[number] for number in repair_numbers],
                    "repair_only_block_numbers": repair_numbers,
                    "repair_attempt": repair_attempt + 1,
                    "original_scene_id": scene_id,
                }
            )
            repair_response, repair_receipt = _persist_call(
                provider=provider,
                role="repair-terra",
                title_id=title_id,
                call_id=f"{scene_id}.repair.terra.{repair_attempt + 1}",
                prompt=prompts["repair"],
                payload=repair_payload,
                schema=schemas["translation"],
                scene_dir=scene_dir,
                resume=resume,
            )
            repaired_rows = _response_rows(repair_response, "blocks", repair_numbers, repair_scene_id)
            repaired_rows = _enforce_evidence_gate(repaired_rows, agreement_by_number, source_quality, acoustic)
            candidate_by_number.update({int(row["block_number"]): row for row in repaired_rows})
            active_numbers = repair_numbers
            all_receipts.append(repair_receipt)

        candidate_rows = [candidate_by_number[number] for number in numbers]
        if sorted(final_reviews_by_number) != numbers:
            raise CodexQualityError(f"Sol final review coverage is incomplete for {scene_id}")
        final_reviews = [final_reviews_by_number[number] for number in numbers]
        candidate_rows = _quarantine_reviews(candidate_rows, final_reviews)
        reviews_by_number = final_reviews_by_number
        for row in candidate_rows:
            number = int(row["block_number"])
            all_decisions.append(
                {
                    "schema_name": "translation-forensics/codex-quality-decision",
                    "schema_version": "1",
                    "title_id": title_id,
                    "scene_id": scene_id,
                    "utterance_id": f"{scene_id}:u-{number:05d}",
                    **row,
                    "source_quality_status": source_quality[number]["source_quality_status"],
                    "source_quality_reason_codes": source_quality[number].get("reason_codes", []),
                    "independent_source_families": acoustic[number].get("independent_source_families", []),
                    "acoustic_evidence_refs": acoustic[number].get("evidence_refs", []),
                    "acoustic_evidence_sha256": sha256_json(acoustic[number]),
                    "consensus_frame": agreement_by_number[number]["consensus_frame"],
                    "critical_slot_conflicts": agreement_by_number[number]["critical_slot_conflicts"],
                    "slot_coverage_gaps": agreement_by_number[number].get("slot_coverage_gaps", []),
                    "terra_frame_sha256": agreement_by_number[number]["terra_frame_sha256"],
                    "sol_frame_sha256": agreement_by_number[number]["sol_frame_sha256"],
                    "sol_final_verdict": reviews_by_number[number]["verdict"],
                    "repair_history": [
                        review
                        for review in all_critiques
                        if review.get("scene_id") == scene_id and int(review.get("block_number", 0)) == number
                    ],
                    "human_equal": False,
                    "human_final": False,
                    "final_promotion_allowed": False,
                }
            )

    decisions_path = output_dir / "decisions.jsonl"
    agreements_path = output_dir / "frame-agreements.jsonl"
    critiques_path = output_dir / "critiques.jsonl"
    receipts_path = output_dir / "model-call-receipts.jsonl"
    decision_schema = json.loads((schema_dir / "codex-quality-decision.schema.json").read_text(encoding="utf-8"))
    decision_validator = Draft202012Validator(decision_schema)
    for row in all_decisions:
        schema_error = next(iter(decision_validator.iter_errors(row)), None)
        if schema_error is not None:
            raise CodexQualityError(
                f"Decision schema error for block {row.get('block_number')}: {schema_error.message}"
            )
    _write_jsonl(decisions_path, all_decisions)
    _write_jsonl(agreements_path, all_agreements)
    _write_jsonl(critiques_path, all_critiques)
    _write_jsonl(receipts_path, all_receipts)

    by_number = {int(row["block_number"]): row for row in all_decisions}
    source_blocks = [
        SubtitleBlock(block.number, block.start, block.end, str(by_number[block.number]["source_faithful_korean"]), block.start_seconds, block.end_seconds)
        for block in structure
    ]
    viewer_blocks = [
        SubtitleBlock(block.number, block.start, block.end, str(by_number[block.number]["viewer_natural_korean"]), block.start_seconds, block.end_seconds)
        for block in structure
    ]
    source_path = output_dir / f"{title_id}.source-faithful-ko.autonomous-quality-v1.srt"
    viewer_path = output_dir / f"{title_id}.viewer-natural-ko.autonomous-quality-v1.srt"
    write_srt(source_path, source_blocks)
    write_srt(viewer_path, viewer_blocks)

    manifest = {
        "schema_name": "translation-forensics/codex-quality-manifest",
        "schema_version": "1",
        "title_id": title_id,
        "release_kind": "autonomous-quality-candidate",
        "status": "candidate-generated",
        "inputs": {
            "structure": _artifact(structure_path, output_dir),
            "source_quality_map": _artifact(source_quality_map_path, output_dir),
            "acoustic_evidence": _artifact(acoustic_evidence_path, output_dir),
        },
        "outputs": {
            "source_faithful": _artifact(source_path, output_dir),
            "viewer_natural": _artifact(viewer_path, output_dir),
            "decisions": _artifact(decisions_path, output_dir),
            "frame_agreements": _artifact(agreements_path, output_dir),
            "critiques": _artifact(critiques_path, output_dir),
            "model_call_receipts": _artifact(receipts_path, output_dir),
        },
        "block_count": len(structure),
        "scene_count": len(scenes),
        "model_call_count": len(all_receipts),
        "all_model_calls_traceable": all(not validate_call_receipt(receipt) for receipt in all_receipts),
        "models": {"generation": "gpt-5.6-terra", "independent_critic": "gpt-5.6-sol"},
        "api_key_used": False,
        "human_equal": False,
        "human_final": False,
        "final_promotion_allowed": False,
    }
    _write_json(manifest_path, manifest)
    validation = validate_codex_quality(output_dir)
    _write_json(output_dir / "qa-report.json", validation)
    manifest["status"] = "quality-gates-passed" if validation["status"] == "pass" else "quality-gates-failed"
    manifest["qa_report"] = _artifact(output_dir / "qa-report.json", output_dir)
    _write_json(manifest_path, manifest)
    return {**manifest, "cache_hit": False, "output": str(output_dir)}


def _resolve_ref(package_dir: Path, record: dict[str, Any]) -> Path:
    path = Path(str(record.get("path") or ""))
    if not path.is_absolute():
        path = package_dir / path
    return path.resolve()


def validate_codex_quality(package_dir: Path) -> dict[str, Any]:
    package_dir = package_dir.expanduser().resolve()
    errors: list[str] = []
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.is_file():
        return {"status": "fail", "errors": ["manifest.json is missing"], "final_promotion_allowed": False}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    title_id = str(manifest.get("title_id") or "")
    if manifest.get("release_kind") != "autonomous-quality-candidate":
        errors.append("release_kind must be autonomous-quality-candidate")
    if any(manifest.get(field) is not False for field in ("human_equal", "human_final", "final_promotion_allowed")):
        errors.append("human/final claim boundary is invalid")
    if manifest.get("status") == "evidence-ceiling-failed":
        try:
            feasibility_path = _resolve_ref(package_dir, manifest["evidence_feasibility"])
            feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
            if manifest["evidence_feasibility"].get("sha256") != _sha256(feasibility_path):
                errors.append("evidence feasibility hash mismatch")
            if feasibility.get("status") != "fail" or feasibility.get("model_calls_allowed") is not False:
                errors.append("evidence ceiling failure contract is invalid")
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
            feasibility = {}
        return {
            "schema_name": "translation-forensics/codex-quality-validation",
            "schema_version": "1",
            "status": "fail",
            "phase": "evidence-feasibility",
            "title_id": title_id,
            "block_count": int(feasibility.get("block_count", 0) or 0),
            "maximum_possible_accepted_rate": feasibility.get("maximum_possible_accepted_rate"),
            "minimum_required_accepted_rate": feasibility.get("minimum_required_accepted_rate"),
            "model_call_count": 0,
            "errors": errors or ["Evidence ceiling is below the title acceptance threshold"],
            "human_equal": False,
            "human_final": False,
            "final_promotion_allowed": False,
        }
    try:
        structure_path = _resolve_ref(package_dir, manifest["inputs"]["structure"])
        decisions_path = _resolve_ref(package_dir, manifest["outputs"]["decisions"])
        source_path = _resolve_ref(package_dir, manifest["outputs"]["source_faithful"])
        viewer_path = _resolve_ref(package_dir, manifest["outputs"]["viewer_natural"])
        receipts_path = _resolve_ref(package_dir, manifest["outputs"]["model_call_receipts"])
        structure, _, _ = parse_srt(structure_path)
        source, _, _ = parse_srt(source_path)
        viewer, _, _ = parse_srt(viewer_path)
        if not compare_structure(structure, source)["pass"] or not compare_structure(structure, viewer)["pass"]:
            errors.append("locked SRT structure changed")
        decisions = _read_jsonl(decisions_path)
        by_number = {int(row.get("block_number", 0)): row for row in decisions}
        expected = [block.number for block in structure]
        if sorted(by_number) != expected or len(by_number) != len(decisions):
            errors.append("decision coverage differs from locked SRT")
        if [block.text for block in source] != [str(by_number[number].get("source_faithful_korean")) for number in expected]:
            errors.append("source-faithful SRT differs from decisions")
        if [block.text for block in viewer] != [str(by_number[number].get("viewer_natural_korean")) for number in expected]:
            errors.append("viewer-natural SRT differs from decisions")

        critical_conflicts = 0
        damaged_without_dual = 0
        accepted = 0
        ellipsis = 0
        reused: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for number in expected:
            row = by_number[number]
            if any(row.get(field) is not False for field in ("human_equal", "human_final", "final_promotion_allowed")):
                errors.append(f"block {number}: claim boundary is invalid")
            if row.get("source_status") == "accepted":
                accepted += 1
                if row.get("critical_slot_conflicts"):
                    critical_conflicts += 1
                if row.get("source_quality_status") in {"suspect", "unusable"} and len(set(row.get("independent_source_families", []))) < 2:
                    damaged_without_dual += 1
            viewer_text = str(row.get("viewer_natural_korean") or "").strip()
            if viewer_text == "…":
                ellipsis += 1
            normalized = re.sub(r"\s+", "", viewer_text)
            if normalized and normalized != "…" and not KOREAN_SHORT_RE.fullmatch(normalized):
                reused[normalized].append(row)
        if critical_conflicts:
            errors.append(f"accepted blocks retain {critical_conflicts} critical frame conflicts")
        if damaged_without_dual:
            errors.append(f"{damaged_without_dual} damaged-source blocks were accepted without two ASR families")
        mass_copy_groups = 0
        for rows in reused.values():
            signatures = {str(row.get("acoustic_evidence_sha256") or "") for row in rows}
            if len(rows) >= 5 and len(signatures) >= 3:
                mass_copy_groups += 1
        if mass_copy_groups:
            errors.append(f"detected {mass_copy_groups} mass-copy groups across different acoustic evidence")

        receipts = _read_jsonl(receipts_path)
        receipt_errors = [error for receipt in receipts for error in validate_call_receipt(receipt)]
        if receipt_errors:
            errors.extend(f"receipt: {error}" for error in receipt_errors[:20])
        if len(receipts) != int(manifest.get("model_call_count", -1)):
            errors.append("model call receipt count mismatch")
        for area in (manifest.get("inputs", {}), manifest.get("outputs", {})):
            for label, record in area.items():
                path = _resolve_ref(package_dir, record)
                if not path.is_file() or record.get("sha256") != _sha256(path):
                    errors.append(f"artifact hash mismatch: {label}")
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
        accepted = ellipsis = critical_conflicts = damaged_without_dual = mass_copy_groups = 0
        structure = []
        receipts = []

    block_count = len(structure)
    accepted_rate = accepted / max(1, block_count)
    ellipsis_rate = ellipsis / max(1, block_count)
    gate = TITLE_GATES.get(title_id)
    metric_gate_passed = True
    if gate:
        if accepted_rate < gate["minimum_accepted_rate"]:
            errors.append(
                f"accepted rate {accepted_rate:.4f} is below {gate['minimum_accepted_rate']:.2f} for {title_id}"
            )
            metric_gate_passed = False
        if ellipsis_rate > gate["maximum_ellipsis_rate"]:
            errors.append(
                f"ellipsis rate {ellipsis_rate:.4f} exceeds {gate['maximum_ellipsis_rate']:.2f} for {title_id}"
            )
            metric_gate_passed = False
    return {
        "schema_name": "translation-forensics/codex-quality-validation",
        "schema_version": "1",
        "status": "pass" if not errors else "fail",
        "title_id": title_id,
        "block_count": block_count,
        "accepted_count": accepted,
        "accepted_rate": round(accepted_rate, 6),
        "ellipsis_count": ellipsis,
        "ellipsis_rate": round(ellipsis_rate, 6),
        "critical_conflicts_in_accepted": critical_conflicts,
        "damaged_source_accepted_without_dual_asr": damaged_without_dual,
        "mass_copy_groups": mass_copy_groups,
        "model_call_receipts": len(receipts),
        "metric_gate_passed": metric_gate_passed,
        "errors": errors,
        "human_equal": False,
        "human_final": False,
        "final_promotion_allowed": False,
    }


def _discover_baseline_srt(baseline: Path, title_id: str) -> Path:
    baseline = baseline.expanduser().resolve()
    if baseline.is_file():
        return baseline
    if not baseline.is_dir():
        raise FileNotFoundError(f"Baseline does not exist: {baseline}")
    preferred = sorted(baseline.rglob(f"{title_id}.gpt56-direct-v1.viewer-natural.srt"))
    candidates = preferred or sorted(baseline.rglob("*viewer*.srt"))
    if not candidates:
        raise FileNotFoundError(f"No viewer baseline SRT found under {baseline}")
    if len(candidates) > 1 and not preferred:
        raise CodexQualityError(
            "Baseline directory is ambiguous; pass one SRT file: " + ", ".join(str(path) for path in candidates[:5])
        )
    return candidates[0]


def stratified_proxy_sample(decisions: list[dict[str, Any]], *, title_id: str, sample_size: int = 120) -> list[int]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    total = len(decisions)
    if total <= sample_size:
        return sorted(int(row["block_number"]) for row in decisions)
    strata: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in decisions:
        number = int(row["block_number"])
        timeline_band = min(2, int((number - 1) * 3 / max(1, total)))
        strata[(str(row.get("source_quality_status") or "unknown"), timeline_band)].append(number)
    for key, numbers in strata.items():
        numbers.sort(
            key=lambda number: hashlib.sha256(f"codex-quality-v1:{title_id}:{key}:{number}".encode()).hexdigest()
        )
    selected: list[int] = []
    keys = sorted(strata)
    while len(selected) < sample_size:
        progressed = False
        for key in keys:
            if strata[key] and len(selected) < sample_size:
                selected.append(strata[key].pop(0))
                progressed = True
        if not progressed:
            break
    return sorted(selected)


def _blind_side(title_id: str, block_number: int) -> str:
    digest = hashlib.sha256(f"codex-quality-blind-v1:{title_id}:{block_number}".encode()).digest()
    return "A" if digest[0] % 2 == 0 else "B"


def evaluate_codex_quality(
    *,
    package_dir: Path,
    baseline: Path,
    provider: CodexExecProvider,
    prompt_dir: Path,
    schema_dir: Path,
    sample_size: int = 120,
    batch_size: int = 20,
    resume: bool = True,
) -> dict[str, Any]:
    package_dir = package_dir.expanduser().resolve()
    validation = validate_codex_quality(package_dir)
    if validation["block_count"] <= 0:
        raise CodexQualityError("Candidate package is not structurally evaluable")
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    title_id = str(manifest["title_id"])
    structure_path = _resolve_ref(package_dir, manifest["inputs"]["structure"])
    acoustic_path = _resolve_ref(package_dir, manifest["inputs"]["acoustic_evidence"])
    decisions_path = _resolve_ref(package_dir, manifest["outputs"]["decisions"])
    candidate_path = _resolve_ref(package_dir, manifest["outputs"]["viewer_natural"])
    baseline_path = _discover_baseline_srt(baseline, title_id)
    structure, _, _ = parse_srt(structure_path)
    candidate, _, _ = parse_srt(candidate_path)
    baseline_blocks, _, _ = parse_srt(baseline_path)
    if not compare_structure(structure, candidate)["pass"] or not compare_structure(structure, baseline_blocks)["pass"]:
        raise CodexQualityError("Candidate and baseline must share the locked SRT structure")
    decisions = _read_jsonl(decisions_path)
    acoustic = _load_by_block(acoustic_path, [block.number for block in structure], "acoustic evidence")
    decision_by_number = {int(row["block_number"]): row for row in decisions}
    candidate_by_number = {block.number: block.text for block in candidate}
    baseline_by_number = {block.number: block.text for block in baseline_blocks}
    sample = stratified_proxy_sample(decisions, title_id=title_id, sample_size=sample_size)
    evaluation_dir = package_dir / "evaluation"
    result_path = evaluation_dir / "proxy-evaluation.json"
    if result_path.is_file() and resume:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("candidate_sha256") == _sha256(candidate_path)
            and existing.get("baseline_sha256") == _sha256(baseline_path)
        ):
            return {**existing, "cache_hit": True, "output": str(evaluation_dir)}
    if result_path.exists() and not resume:
        raise FileExistsError(f"Proxy evaluation already exists: {result_path}")
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    prompt = (prompt_dir / "codex-quality-proxy-evaluator-v1.md").read_text(encoding="utf-8")
    schema = json.loads((schema_dir / "codex-quality-proxy-response.schema.json").read_text(encoding="utf-8"))
    key_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    for number in sample:
        candidate_side = _blind_side(title_id, number)
        baseline_side = "B" if candidate_side == "A" else "A"
        key_rows.append({"block_number": number, "candidate_side": candidate_side, "baseline_side": baseline_side})
        alternatives = {
            candidate_side: candidate_by_number[number],
            baseline_side: baseline_by_number[number],
        }
        blind_rows.append(
            {
                "block_number": number,
                "source_quality_status": decision_by_number[number].get("source_quality_status"),
                "consensus_frame": decision_by_number[number].get("consensus_frame"),
                "acoustic_transcripts": acoustic[number].get("transcripts", []),
                "candidate_A": alternatives["A"],
                "candidate_B": alternatives["B"],
            }
        )
    _write_jsonl(evaluation_dir / "sample.jsonl", blind_rows)

    evaluator_judgments: dict[str, list[dict[str, Any]]] = {"terra": [], "sol": []}
    receipts: list[dict[str, Any]] = []
    roles = (("terra", "proxy-evaluator-terra"), ("sol", "proxy-evaluator-sol"))
    for offset in range(0, len(blind_rows), batch_size):
        batch = blind_rows[offset : offset + batch_size]
        batch_id = f"proxy-{offset // batch_size + 1:03d}"
        expected = [int(row["block_number"]) for row in batch]
        payload = {"batch_id": batch_id, "title_id": title_id, "blocks": batch}
        for label, role in roles:
            response, receipt = provider.run_structured(
                role=role,
                title_id=title_id,
                call_id=f"{batch_id}.{label}",
                prompt=prompt,
                payload=payload,
                schema=schema,
                resume=resume,
            )
            rows = _response_rows(response, "judgments", expected, batch_id)
            evaluator_judgments[label].extend(rows)
            receipts.append(receipt)
            _write_json(evaluation_dir / f"{batch_id}.{label}.response.json", response)
            _write_json(evaluation_dir / f"{batch_id}.{label}.receipt.json", receipt)
    _write_jsonl(evaluation_dir / "model-call-receipts.jsonl", receipts)

    key = {int(row["block_number"]): row for row in key_rows}
    evaluator_results: dict[str, dict[str, Any]] = {}
    all_passed = True
    for label, rows in evaluator_judgments.items():
        wins = ties = losses = critical_losses = 0
        resolved: list[dict[str, Any]] = []
        for row in rows:
            number = int(row["block_number"])
            candidate_side = key[number]["candidate_side"]
            baseline_side = key[number]["baseline_side"]
            winner = row["winner"]
            outcome = "tie" if winner == "tie" else "win" if winner == candidate_side else "loss"
            wins += outcome == "win"
            ties += outcome == "tie"
            losses += outcome == "loss"
            critical_side = row["critical_error_side"]
            candidate_critical = critical_side in {candidate_side, "both"} and critical_side not in {baseline_side}
            critical_losses += bool(candidate_critical)
            resolved.append({**row, "candidate_outcome": outcome, "candidate_critical_loss": candidate_critical})
        non_loss_rate = (wins + ties) / max(1, len(rows))
        passed = non_loss_rate >= 0.90 and critical_losses == 0
        all_passed = all_passed and passed
        evaluator_results[label] = {
            "sample_size": len(rows),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_or_tie_rate": round(non_loss_rate, 6),
            "critical_losses": critical_losses,
            "passed": passed,
            "judgments": resolved,
        }
    result = {
        "schema_name": "translation-forensics/codex-quality-proxy-evaluation",
        "schema_version": "1",
        "title_id": title_id,
        "status": "pass" if all_passed else "fail",
        "engineering_proxy_only": True,
        "human_equivalence_certified": False,
        "sample_seed": f"codex-quality-v1:{title_id}",
        "stratified_sample_size": len(sample),
        "candidate_sha256": _sha256(candidate_path),
        "baseline_sha256": _sha256(baseline_path),
        "evaluators": evaluator_results,
        "model_call_receipts": len(receipts),
        "all_model_calls_traceable": all(not validate_call_receipt(receipt) for receipt in receipts),
        "human_equal": False,
        "human_final": False,
        "final_promotion_allowed": False,
    }
    _write_json(result_path, result)
    return {**result, "cache_hit": False, "output": str(evaluation_dir)}
