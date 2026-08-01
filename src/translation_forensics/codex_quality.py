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
from .asr_fusion import add_asr_fusion
from .graduated_recovery import (
    build_frame_agreement,
    derive_slot_corroboration,
    apply_graduated_evidence_gate,
    apply_review_outcomes,
    coherence_check,
    compatibility_status,
    graduated_source_status,
    has_usable_dual_acoustic,
    needs_context_rerun,
)
from .utterance_routing import apply_utterance_route

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
    "ADN-622": {"minimum_accepted_rate": 0.95, "minimum_safe_usable_rate": 0.95, "maximum_ellipsis_rate": 0.02},
    "JUQ-439": {"minimum_accepted_rate": 0.90, "minimum_safe_usable_rate": 0.90, "maximum_ellipsis_rate": 0.05},
    "SSIS-908": {"minimum_accepted_rate": 0.85, "minimum_safe_usable_rate": 0.85, "maximum_ellipsis_rate": 0.08},
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


def _review_repair_eligible(review: dict[str, Any]) -> bool:
    """Allow Terra repair only for claim-level, non-blocking findings."""
    if str(review.get("verdict") or "") != "repair":
        return False
    findings = [finding for finding in (review.get("claim_findings") or []) if isinstance(finding, dict)]
    if any(finding.get("disposition") == "blocking" for finding in findings):
        return False
    unsupported_additions = [
        str(value).strip()
        for value in (review.get("unsupported_additions") or [])
        if str(value).strip()
    ]
    unsupported = bool(unsupported_additions)
    if unsupported and not findings:
        return False
    if unsupported:
        repairable = {"omit", "repair"}
        parsed_slots = {
            addition.split(":", 1)[0].strip().casefold()
            for addition in unsupported_additions
        }
        known_slots = set(CRITICAL_SLOTS) | {"translation"}
        if parsed_slots <= known_slots:
            for slot in parsed_slots:
                if not any(
                    str(finding.get("slot") or "").casefold() == slot
                    and finding.get("disposition") in repairable
                    for finding in findings
                ):
                    return False
        elif not findings or not all(
            finding.get("disposition") in repairable for finding in findings
        ):
            return False
    # A legacy review without claim findings keeps the old conservative rule.
    if not findings and review.get("critical_slot_conflicts"):
        return False
    return True




def compare_independent_frames(
    terra_frames: list[dict[str, Any]],
    sol_frames: list[dict[str, Any]],
    expected_blocks: list[int],
    *,
    acoustic: dict[int, dict[str, Any]] | None = None,
    source_quality: dict[int, dict[str, Any]] | None = None,
    context_retried: bool = False,
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
        if acoustic is not None and source_quality is not None:
            corrob = derive_slot_corroboration(
                left, right,
                acoustic_record=acoustic[number],
                source_quality_record=source_quality[number],
                block_number=number,
            )
        else:
            corrob = None
        agreement = build_frame_agreement(
            left,
            right,
            corroboration=corrob,
            context_retried=context_retried,
        )
        agreement["block_number"] = number
        agreement["terra_frame_sha256"] = sha256_json(left)
        agreement["sol_frame_sha256"] = sha256_json(right)
        agreements.append(agreement)
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
                "asr_fusion": acoustic[block.number].get("asr_fusion", {}),
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
        or has_usable_dual_acoustic(acoustic[number])
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
        "policy": "trusted Japanese or downstream-compatible two block-local independent ASR families",
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
    evidence_repair_path: Path | None = None,
    max_scene_blocks: int = 20,
    maximum_scene_gap_seconds: float = 60.0,
    max_repairs: int = 2,
    force_cpu: bool = False,
    allow_model_download: bool = True,
    resume: bool = True,
    local_asr_call_count: int = 0,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    repair_cache_identity: str | None = None
    if evidence_repair_path is not None and evidence_repair_path.is_file():
        try:
            repair_cache_identity = str(
                json.loads(evidence_repair_path.read_text(encoding="utf-8")).get("cache_identity") or ""
            ) or None
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            repair_cache_identity = None
    if manifest_path.is_file() and resume:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_repair = existing.get("inputs", {}).get("evidence_repair")
        repair_matches = (
            evidence_repair_path is None
            and existing_repair is None
        ) or (
            evidence_repair_path is not None
            and isinstance(existing_repair, dict)
            and existing_repair.get("sha256") == _sha256(evidence_repair_path)
        )
        repair_mode_matches = (
            (evidence_repair_path is None and existing.get("status") != "evidence-ceiling-failed-after-repair")
            or (evidence_repair_path is not None and existing.get("status") == "evidence-ceiling-failed-after-repair")
        )
        repair_identity_matches = (
            evidence_repair_path is None
            or (repair_cache_identity is not None and existing.get("repair_cache_identity") == repair_cache_identity)
        )
        if (
            existing.get("inputs", {}).get("structure", {}).get("sha256") == _sha256(structure_path)
            and existing.get("inputs", {}).get("source_quality_map", {}).get("sha256") == _sha256(source_quality_map_path)
            and existing.get("inputs", {}).get("acoustic_evidence", {}).get("sha256") == _sha256(acoustic_evidence_path)
            and repair_matches
            and repair_mode_matches
            and repair_identity_matches
        ):
            return {**existing, "cache_hit": True, "output": str(output_dir)}
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Codex quality package already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    structure, _, _ = parse_srt(structure_path)
    expected = [block.number for block in structure]
    source_text_by_number = {block.number: block.text for block in structure}
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
        failed_status = (
            "evidence-ceiling-failed-after-repair"
            if evidence_repair_path is not None
            else "evidence-ceiling-failed"
        )
        manifest = {
            "schema_name": "translation-forensics/codex-quality-manifest",
            "schema_version": "1",
            "title_id": title_id,
            "release_kind": "autonomous-quality-candidate",
            "status": failed_status,
            "inputs": {
                "structure": _artifact(structure_path, output_dir),
                "source_quality_map": _artifact(source_quality_map_path, output_dir),
                "acoustic_evidence": _artifact(acoustic_evidence_path, output_dir),
            },
            "evidence_feasibility": _artifact(ceiling_path, output_dir),
            "block_count": len(structure),
            "model_call_count": 0,
            "semantic_model_call_count": 0,
            "local_asr_call_count": int(local_asr_call_count),
            "repair_cache_identity": repair_cache_identity,
            "all_model_calls_traceable": True,
            "api_key_used": False,
            "human_equal": False,
            "human_final": False,
            "final_promotion_allowed": False,
        }
        if evidence_repair_path is not None:
            manifest["inputs"]["evidence_repair"] = _artifact(evidence_repair_path, output_dir)
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
        agreements = compare_independent_frames(terra_frames, sol_frames, numbers, acoustic=acoustic, source_quality=source_quality)
        rerun_performed = False
        rerun_numbers: set[int] = set()
        rerun_error = ""
        conflict_numbers = [
            int(agreement["block_number"])
            for agreement in agreements
            if needs_context_rerun(agreement)
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
                    acoustic[number] = add_asr_fusion(
                        {
                            **acoustic[number],
                            "transcripts": acoustic[number].get("transcripts", [])
                            + evidence.get("transcripts", []),
                            "evidence_refs": sorted(
                                set(acoustic[number].get("evidence_refs", []))
                                | set(evidence.get("evidence_refs", []))
                            ),
                            "independent_source_families": sorted(
                                set(acoustic[number].get("independent_source_families", []))
                                | set(evidence.get("independent_source_families", []))
                            ),
                        }
                    )
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
                        acoustic=acoustic,
                        source_quality=source_quality,
                        context_retried=True,
                    )
                }
                agreements = [
                    retry_agreements.get(int(agreement["block_number"]), agreement)
                    for agreement in agreements
                ]
                all_receipts.extend((terra_retry_receipt, sol_retry_receipt))
                rerun_performed = True
                rerun_numbers.update(conflict_numbers)
            except (FileExistsError, LocalASRError, OSError, RuntimeError, ValueError) as exc:
                rerun_error = str(exc)
        by_block = {block.number: block for block in scene}
        agreements = [
            apply_utterance_route(
                agreement,
                source_text=by_block[int(agreement["block_number"])].text,
                acoustic_record=acoustic[int(agreement["block_number"])],
                source_quality_record=source_quality[int(agreement["block_number"])],
            )
            for agreement in agreements
        ]
        agreement_by_number = {int(row["block_number"]): row for row in agreements}
        # Rerun evidence mutates the acoustic ledger. Rebuild the ordinary scene
        # payload so translation and the first Sol critique see the same fused
        # evidence that produced the final agreement records.
        payload = _scene_payload(
            scene_id,
            scene,
            source_quality,
            acoustic,
            context_before=before,
            context_after=after,
        )
        all_agreements.extend(
            {
                **row,
                "scene_id": scene_id,
                "boundary_expansion_asr_rerun": int(row["block_number"]) in rerun_numbers,
                "boundary_expansion_asr_rerun_error": rerun_error,
            }
            for row in agreements
        )
        all_receipts.extend((terra_receipt, sol_receipt))

        translation_payload = {
            **payload,
            "consensus": agreements,
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
        candidate_rows = apply_graduated_evidence_gate(candidate_rows, agreement_by_number, source_quality, acoustic)
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
                if _review_repair_eligible(review)
            ]
            blocking = [
                review
                for review in reviews
                if review.get("verdict") == "quarantine"
                or any(
                    finding.get("disposition") == "blocking"
                    for finding in review.get("claim_findings", [])
                    if isinstance(finding, dict)
                )
                or (
                    review.get("verdict") == "repair"
                    and not _review_repair_eligible(review)
                )
            ]
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
            repaired_rows = apply_graduated_evidence_gate(repaired_rows, agreement_by_number, source_quality, acoustic)
            candidate_by_number.update({int(row["block_number"]): row for row in repaired_rows})
            active_numbers = repair_numbers
            all_receipts.append(repair_receipt)

        candidate_rows = [candidate_by_number[number] for number in numbers]
        if sorted(final_reviews_by_number) != numbers:
            raise CodexQualityError(f"Sol final review coverage is incomplete for {scene_id}")
        final_reviews = [final_reviews_by_number[number] for number in numbers]
        candidate_rows = apply_review_outcomes(candidate_rows, final_reviews)
        reviews_by_number = final_reviews_by_number
        for row in candidate_rows:
            number = int(row["block_number"])
            all_decisions.append(
                {
                    "schema_name": "translation-forensics/codex-quality-decision",
                    "schema_version": "2",
                    "title_id": title_id,
                    "scene_id": scene_id,
                    "utterance_id": f"{scene_id}:u-{number:05d}",
                    "boundary_expansion_asr_rerun": number in rerun_numbers,
                    **row,
                    "source_quality_status": source_quality[number]["source_quality_status"],
                    "source_quality_reason_codes": source_quality[number].get("reason_codes", []),
                    "independent_source_families": acoustic[number].get("independent_source_families", []),
                    "acoustic_evidence_refs": acoustic[number].get("evidence_refs", []),
                    "acoustic_evidence_sha256": sha256_json(acoustic[number]),
                    "consensus_frame": agreement_by_number[number]["consensus_frame"],
                    "selected_frame": agreement_by_number[number].get("selected_frame", agreement_by_number[number]["consensus_frame"]),
                    "critical_slot_conflicts": agreement_by_number[number]["critical_slot_conflicts"],
                    "render_blocking_conflicts": agreement_by_number[number].get("render_blocking_conflicts", agreement_by_number[number]["critical_slot_conflicts"]),
                    "slot_coverage_gaps": agreement_by_number[number].get("slot_coverage_gaps", []),
                    "resolved_conflicts": agreement_by_number[number].get("resolved_conflicts", []),
                    "coherence_findings": agreement_by_number[number].get("coherence_findings", []),
                    "slot_provenance": agreement_by_number[number].get("slot_provenance", {}),
                    "omitted_slots": agreement_by_number[number].get("omitted_slots", []),
                    "uncertainty_codes": row.get("uncertainty_codes", []),
                    "recovery_state": row.get("recovery_state", "abstained"),
                    "rendered_slots": row.get("rendered_slots", []),
                    "terra_frame_sha256": agreement_by_number[number]["terra_frame_sha256"],
                    "sol_frame_sha256": agreement_by_number[number]["sol_frame_sha256"],
                    "sol_final_verdict": reviews_by_number[number]["verdict"],
                    "repair_history": [
                        review
                        for review in all_critiques
                        if review.get("scene_id") == scene_id and int(review.get("block_number", 0)) == number
                    ],
                    "utterance_kind": agreement_by_number[number].get("utterance_kind", "unknown"),
                    "utterance_kind_confidence": agreement_by_number[number].get("utterance_kind_confidence", "low"),
                    "utterance_kind_reason_codes": agreement_by_number[number].get("utterance_kind_reason_codes", []),
                    "utterance_kind_evidence_refs": agreement_by_number[number].get("utterance_kind_evidence_refs", []),
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
        "semantic_model_call_count": len(all_receipts),
        "local_asr_call_count": int(local_asr_call_count),
        "repair_cache_identity": repair_cache_identity,
        "all_model_calls_traceable": all(not validate_call_receipt(receipt) for receipt in all_receipts),
        "models": {"generation": "gpt-5.6-terra", "independent_critic": "gpt-5.6-sol"},
        "api_key_used": False,
        "human_equal": False,
        "human_final": False,
        "final_promotion_allowed": False,
    }
    if evidence_repair_path is not None:
        manifest["inputs"]["evidence_repair"] = _artifact(evidence_repair_path, output_dir)
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


def _validate_repair_lineage(
    manifest: dict[str, Any],
    package_dir: Path,
    errors: list[str],
) -> dict[str, Any] | None:
    """Validate repair provenance for both ceiling-failed and success packages."""
    record = manifest.get("inputs", {}).get("evidence_repair")
    if not isinstance(record, dict):
        errors.append("repair-enabled package lacks evidence repair artifact")
        return None
    try:
        repair_path = _resolve_ref(package_dir, record)
        if not repair_path.is_file() or record.get("sha256") != _sha256(repair_path):
            errors.append("evidence repair artifact hash mismatch")
            return None
        report = json.loads(repair_path.read_text(encoding="utf-8"))
        if not report.get("cache_identity"):
            errors.append("repair cache identity is missing")
        elif manifest.get("repair_cache_identity") != report.get("cache_identity"):
            errors.append("repair cache identity mismatch")
        lineage = report.get("lineage") or {}
        structure_record = manifest.get("inputs", {}).get("structure", {})
        source_record = manifest.get("inputs", {}).get("source_quality_map", {})
        acoustic_record = manifest.get("inputs", {}).get("acoustic_evidence", {})
        structure_path = _resolve_ref(package_dir, structure_record)
        source_path = _resolve_ref(package_dir, source_record)
        acoustic_path = _resolve_ref(package_dir, acoustic_record)
        for label, artifact, path in (
            ("structure", structure_record, structure_path),
            ("source_quality_map", source_record, source_path),
            ("acoustic_evidence", acoustic_record, acoustic_path),
        ):
            if not path.is_file() or artifact.get("sha256") != _sha256(path):
                errors.append(f"artifact hash mismatch: {label}")
        if lineage.get("structure_sha256") and lineage.get("structure_sha256") != structure_record.get("sha256"):
            errors.append("repair lineage structure hash mismatch")
        if report.get("merged_acoustic_sha256") and report.get("merged_acoustic_sha256") != acoustic_record.get("sha256"):
            errors.append("repair lineage merged acoustic hash mismatch")
        if report.get("regenerated_source_quality_sha256") and report.get("regenerated_source_quality_sha256") != source_record.get("sha256"):
            errors.append("repair lineage regenerated source-quality hash mismatch")
        if not isinstance(report.get("policy"), dict) or not report["policy"].get("policy_version"):
            errors.append("repair lineage policy is missing")
        audit_input = report.get("source_quality_audit_input")
        if not isinstance(audit_input, dict) or audit_input.get("acoustic_sha256") != acoustic_record.get("sha256"):
            errors.append("source-quality audit input lineage is missing or stale")
        repaired_evidence = repair_path.parent / str(
            report.get("evidence_path") or "repaired-block-acoustic-evidence.jsonl"
        )
        if report.get("evidence_sha256") and repaired_evidence.is_file() and report.get("evidence_sha256") != _sha256(repaired_evidence):
            errors.append("repair evidence ledger hash mismatch")
        structure, _, _ = parse_srt(structure_path)
        expected = [block.number for block in structure]
        source_quality = _load_by_block(source_path, expected, "source quality")
        acoustic = _load_by_block(acoustic_path, expected, "acoustic evidence")
        ceiling = evaluate_evidence_ceiling(
            title_id=str(manifest.get("title_id") or ""),
            expected_blocks=expected,
            source_quality=source_quality,
            acoustic=acoustic,
        )
        final_ceiling = report.get("final_ceiling")
        if isinstance(final_ceiling, dict):
            for field in ("eligible_block_count", "eligible_block_numbers", "maximum_possible_accepted_rate", "status"):
                if final_ceiling.get(field) != ceiling.get(field):
                    errors.append(f"repair final ceiling disagrees with recomputed {field}")
        return report
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"repair lineage validation failed: {exc}")
        return None


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
    if manifest.get("status") in {"evidence-ceiling-failed", "evidence-ceiling-failed-after-repair"}:
        try:
            feasibility_path = _resolve_ref(package_dir, manifest["evidence_feasibility"])
            feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
            if manifest["evidence_feasibility"].get("sha256") != _sha256(feasibility_path):
                errors.append("evidence feasibility hash mismatch")
            if manifest.get("status") == "evidence-ceiling-failed-after-repair":
                repair_record = manifest.get("inputs", {}).get("evidence_repair")
                if not isinstance(repair_record, dict):
                    errors.append("post-repair ceiling failure lacks evidence repair artifact")
                else:
                    repair_path = _resolve_ref(package_dir, repair_record)
                    if not repair_path.is_file() or repair_record.get("sha256") != _sha256(repair_path):
                        errors.append("evidence repair artifact hash mismatch")
                    else:
                        repair_report = json.loads(repair_path.read_text(encoding="utf-8"))
                        if not repair_report.get("cache_identity"):
                            errors.append("repair cache identity is missing")
                        elif manifest.get("repair_cache_identity") != repair_report.get("cache_identity"):
                            errors.append("repair cache identity mismatch")
                        lineage = repair_report.get("lineage") or {}
                        structure_record = manifest.get("inputs", {}).get("structure", {})
                        source_record = manifest.get("inputs", {}).get("source_quality_map", {})
                        acoustic_record = manifest.get("inputs", {}).get("acoustic_evidence", {})
                        structure_path = _resolve_ref(package_dir, structure_record)
                        source_path = _resolve_ref(package_dir, source_record)
                        acoustic_path = _resolve_ref(package_dir, acoustic_record)
                        for label, record, path in (
                            ("structure", structure_record, structure_path),
                            ("source_quality_map", source_record, source_path),
                            ("acoustic_evidence", acoustic_record, acoustic_path),
                        ):
                            if not path.is_file() or record.get("sha256") != _sha256(path):
                                errors.append(f"artifact hash mismatch: {label}")
                        if lineage.get("structure_sha256") and lineage.get("structure_sha256") != structure_record.get("sha256"):
                            errors.append("repair lineage structure hash mismatch")
                        if repair_report.get("merged_acoustic_sha256") and repair_report.get("merged_acoustic_sha256") != acoustic_record.get("sha256"):
                            errors.append("repair lineage merged acoustic hash mismatch")
                        if repair_report.get("regenerated_source_quality_sha256") and repair_report.get("regenerated_source_quality_sha256") != source_record.get("sha256"):
                            errors.append("repair lineage regenerated source-quality hash mismatch")
                        policy = repair_report.get("policy")
                        if not isinstance(policy, dict) or not policy.get("policy_version"):
                            errors.append("repair lineage policy is missing")
                        repaired_evidence = repair_path.parent / str(
                            repair_report.get("evidence_path") or "repaired-block-acoustic-evidence.jsonl"
                        )
                        if repair_report.get("evidence_sha256") and repaired_evidence.is_file() and repair_report.get("evidence_sha256") != _sha256(repaired_evidence):
                            errors.append("repair evidence ledger hash mismatch")
                        # A ceiling failure must never contain semantic Terra/Sol receipts.
                        receipts_candidate = package_dir / "model-call-receipts.jsonl"
                        if receipts_candidate.is_file() and _read_jsonl(receipts_candidate):
                            errors.append("ceiling-failed package contains semantic model receipts")
                        try:
                            structure, _, _ = parse_srt(structure_path)
                            expected = [block.number for block in structure]
                            recomputed_source = _load_by_block(source_path, expected, "source quality")
                            recomputed_acoustic = _load_by_block(acoustic_path, expected, "acoustic evidence")
                            recomputed = evaluate_evidence_ceiling(
                                title_id=title_id,
                                expected_blocks=expected,
                                source_quality=recomputed_source,
                                acoustic=recomputed_acoustic,
                            )
                            for field in ("eligible_block_count", "eligible_block_numbers", "maximum_possible_accepted_rate", "status"):
                                if recomputed.get(field) != feasibility.get(field):
                                    errors.append(f"stored evidence ceiling disagrees with recomputed {field}")
                            final_ceiling = repair_report.get("final_ceiling")
                            if isinstance(final_ceiling, dict) and final_ceiling.get("eligible_block_numbers") != recomputed.get("eligible_block_numbers"):
                                errors.append("repair final ceiling eligible block list mismatch")
                        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                            errors.append(f"repair ceiling recomputation failed: {exc}")
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
        if manifest.get("inputs", {}).get("evidence_repair") is not None:
            _validate_repair_lineage(manifest, package_dir, errors)
        structure_path = _resolve_ref(package_dir, manifest["inputs"]["structure"])
        decisions_path = _resolve_ref(package_dir, manifest["outputs"]["decisions"])
        source_path = _resolve_ref(package_dir, manifest["outputs"]["source_faithful"])
        viewer_path = _resolve_ref(package_dir, manifest["outputs"]["viewer_natural"])
        acoustic_path = _resolve_ref(package_dir, manifest["inputs"]["acoustic_evidence"])
        receipts_path = _resolve_ref(package_dir, manifest["outputs"]["model_call_receipts"])
        structure, _, _ = parse_srt(structure_path)
        source, _, _ = parse_srt(source_path)
        viewer, _, _ = parse_srt(viewer_path)
        acoustic = _load_by_block(acoustic_path, [block.number for block in structure], "acoustic evidence")
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
        safe_usable = 0
        ellipsis = 0
        reused: dict[str, list[dict[str, Any]]] = defaultdict(list)
        recovery_counts: dict[str, int] = defaultdict(int)
        for number in expected:
            row = by_number[number]
            if any(row.get(field) is not False for field in ("human_equal", "human_final", "final_promotion_allowed")):
                errors.append(f"block {number}: claim boundary is invalid")
            if row.get("source_status") == "accepted":
                accepted += 1
                if row.get("render_blocking_conflicts"):
                    critical_conflicts += 1
                if row.get("source_quality_status") in {"suspect", "unusable"} and not has_usable_dual_acoustic(acoustic[number], allow_legacy_without_fusion=False):
                    damaged_without_dual += 1
            recovery_state = str(row.get("recovery_state") or "abstained")
            recovery_counts[recovery_state] = recovery_counts.get(recovery_state, 0) + 1
            viewer_text = str(row.get("viewer_natural_korean") or "").strip()
            if viewer_text == "…":
                ellipsis += 1
            source_text = str(row.get("source_faithful_korean") or "").strip()
            if (
                row.get("source_status") in ("accepted", "unresolved")
                and recovery_state != "abstained"
                and source_text != "…"
                and viewer_text != "…"
            ):
                safe_usable += 1
            normalized = re.sub(r"\s+", "", viewer_text)
            if normalized and normalized != "…" and not KOREAN_SHORT_RE.fullmatch(normalized):
                reused[normalized].append(row)
        if critical_conflicts:
            errors.append(f"accepted blocks retain {critical_conflicts} render-blocking conflicts")
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
        accepted = safe_usable = ellipsis = critical_conflicts = damaged_without_dual = mass_copy_groups = 0
        structure = []
        receipts = []
        recovery_counts = {}

    block_count = len(structure)
    accepted_rate = accepted / max(1, block_count)
    ellipsis_rate = ellipsis / max(1, block_count)
    gate = TITLE_GATES.get(title_id)
    metric_gate_passed = True
    if gate:
        safe_usable_rate = safe_usable / max(1, block_count)
        if safe_usable_rate < gate["minimum_safe_usable_rate"]:
            errors.append(
                f"safe usable-output rate {safe_usable_rate:.4f} is below {gate['minimum_safe_usable_rate']:.2f} for {title_id}"
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
        "minimum_accepted_rate": gate.get("minimum_accepted_rate") if gate else None,
        "minimum_safe_usable_rate": gate.get("minimum_safe_usable_rate") if gate else None,
        "safe_usable_rate": round(safe_usable / max(1, block_count), 6),
        "recovery_state_counts": dict(sorted(recovery_counts.items())),
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
