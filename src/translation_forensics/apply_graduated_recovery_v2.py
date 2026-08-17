from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path


class PatchError(RuntimeError):
    pass


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(f"{label}: expected one exact match, found {count}")
    return text.replace(old, new, 1)


def replace_regex_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise PatchError(f"{label}: expected one regex match, found {count}")
    return updated


def patch_codex_quality(text: str) -> str:
    if "from .graduated_recovery import (" in text:
        raise PatchError("codex_quality.py already imports graduated_recovery")

    text = replace_once(
        text,
        "from .local_asr import FasterWhisperBackend, LocalASRError, ReazonSpeechBackend, run_conflict_asr_rerun\n",
        '''from .local_asr import FasterWhisperBackend, LocalASRError, ReazonSpeechBackend, run_conflict_asr_rerun
from .graduated_recovery import (
    CRITICAL_SLOTS,
    apply_graduated_evidence_gate,
    apply_review_outcomes,
    build_frame_agreement,
    derive_slot_corroboration,
    needs_context_rerun,
)
from .asr_fusion import add_asr_fusion
from .utterance_routing import apply_utterance_route
''',
        "graduated recovery import",
    )
    text = replace_regex_once(
        text,
        r"CRITICAL_SLOTS = \(\n(?:    .+\n)+\)\n",
        "",
        "remove local critical slots",
    )

    text = replace_once(
        text,
        """                "japanese_srt": block.text,
                "source_quality_status": source_quality[block.number]["source_quality_status"],
""",
        """                "japanese_srt": block.text,
                "source_text_evidence_ref": f"source-srt:block-{block.number}",
                "source_quality_status": source_quality[block.number]["source_quality_status"],
""",
        "source text evidence reference",
    )
    text = replace_once(
        text,
        """                "independent_source_families": acoustic[block.number].get("independent_source_families", []),
""",
        """                "independent_source_families": acoustic[block.number].get("independent_source_families", []),
                "asr_fusion": acoustic[block.number].get("asr_fusion", {}),
""",
        "scene payload ASR fusion",
    )

    text = replace_once(
        text,
        '    acoustic = _load_by_block(acoustic_evidence_path, expected, "acoustic evidence")\n',
        """    acoustic = {
        number: add_asr_fusion(record)
        for number, record in _load_by_block(
            acoustic_evidence_path,
            expected,
            "acoustic evidence",
        ).items()
    }
""",
        "load additive ASR fusion",
    )

    compare = '''def compare_independent_frames(
    terra_frames: list[dict[str, Any]],
    sol_frames: list[dict[str, Any]],
    expected_blocks: list[int],
    *,
    context_retried: bool = False,
    acoustic_by_block: dict[int, dict[str, Any]] | None = None,
    source_quality_by_block: dict[int, dict[str, Any]] | None = None,
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
        corroboration = derive_slot_corroboration(
            left,
            right,
            acoustic_record=(acoustic_by_block or {}).get(number, {}),
            source_quality_record=(source_quality_by_block or {}).get(number, {}),
            block_number=number,
        )
        agreement = build_frame_agreement(
            left,
            right,
            context_retried=context_retried,
            corroboration=corroboration,
        )
        agreement.update(
            {
                "block_number": number,
                "terra_frame_sha256": sha256_json(left),
                "sol_frame_sha256": sha256_json(right),
            }
        )
        agreements.append(agreement)
    return agreements

'''
    text = replace_regex_once(
        text,
        r"(?s)def compare_independent_frames\(.*?\n(?=def _response_rows)",
        compare,
        "compare independent frames",
    )

    text = replace_once(
        text,
        '        agreements = compare_independent_frames(terra_frames, sol_frames, numbers)\n',
        '        agreements = compare_independent_frames(\n            terra_frames,\n            sol_frames,\n            numbers,\n            acoustic_by_block=acoustic,\n            source_quality_by_block=source_quality,\n        )\n',
        "initial corroboration inputs",
    )

    gates = '''def _enforce_evidence_gate(
    rows: list[dict[str, Any]],
    agreements: dict[int, dict[str, Any]],
    source_quality: dict[int, dict[str, Any]],
    acoustic: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    return apply_graduated_evidence_gate(rows, agreements, source_quality, acoustic)


def _quarantine_reviews(rows: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return apply_review_outcomes(rows, reviews)

'''
    text = replace_regex_once(
        text,
        r"(?s)def _enforce_evidence_gate\(.*?\n(?=def evaluate_evidence_ceiling)",
        gates,
        "graduated gate and review application",
    )

    text = replace_once(
        text,
        '''        conflict_numbers = [
            int(agreement["block_number"])
            for agreement in agreements
            if agreement["critical_slot_conflicts"]
        ]
''',
        '''        conflict_numbers = [
            int(agreement["block_number"])
            for agreement in agreements
            if needs_context_rerun(agreement)
        ]
''',
        "context rerun selection",
    )

    text = replace_once(
        text,
        '                    acoustic[number] = {\n                        **evidence,\n                        "transcripts": acoustic[number].get("transcripts", []) + evidence.get("transcripts", []),\n                        "evidence_refs": sorted(\n                            set(acoustic[number].get("evidence_refs", [])) | set(evidence.get("evidence_refs", []))\n                        ),\n                        "independent_source_families": sorted(\n                            set(acoustic[number].get("independent_source_families", []))\n                            | set(evidence.get("independent_source_families", []))\n                        ),\n                    }\n',
        '                    acoustic[number] = add_asr_fusion(\n                        {\n                            **evidence,\n                            "transcripts": acoustic[number].get("transcripts", [])\n                            + evidence.get("transcripts", []),\n                            "evidence_refs": sorted(\n                                set(acoustic[number].get("evidence_refs", []))\n                                | set(evidence.get("evidence_refs", []))\n                            ),\n                            "independent_source_families": sorted(\n                                set(acoustic[number].get("independent_source_families", []))\n                                | set(evidence.get("independent_source_families", []))\n                            ),\n                        }\n                    )\n',
        "recompute ASR fusion after context merge",
    )

    text = replace_once(
        text,
        '''                    for row in compare_independent_frames(
                        _response_rows(terra_retry, "frames", conflict_numbers, retry_scene_id),
                        _response_rows(sol_retry, "frames", conflict_numbers, retry_scene_id),
                        conflict_numbers,
                    )
''',
        '''                    for row in compare_independent_frames(
                        _response_rows(terra_retry, "frames", conflict_numbers, retry_scene_id),
                        _response_rows(sol_retry, "frames", conflict_numbers, retry_scene_id),
                        conflict_numbers,
                        context_retried=True,
                        acoustic_by_block=acoustic,
                        source_quality_by_block=source_quality,
                    )
''',
        "context retry recovery state",
    )

    text = replace_once(
        text,
        '        agreement_by_number = {int(row["block_number"]): row for row in agreements}\n',
        '        by_block = {block.number: block for block in scene}\n        agreements = [\n            apply_utterance_route(\n                agreement,\n                source_text=by_block[int(agreement["block_number"])].text,\n                acoustic_record=acoustic[int(agreement["block_number"])],\n                source_quality_record=source_quality[int(agreement["block_number"])],\n            )\n            for agreement in agreements\n        ]\n        agreement_by_number = {int(row["block_number"]): row for row in agreements}\n',
        "attach utterance routing",
    )

    text = replace_regex_once(
        text,
        r'''(?s)        translation_payload = \{\n            \*\*payload,\n            "consensus": \[\n.*?        \}\n        translation_response, translation_receipt = _persist_call\(''',
        '''        translation_payload = {
            **payload,
            # Keep the historical key for prompt/cache plumbing; records are now v2
            # slot-level arbitration objects rather than binary consensus gates.
            "consensus": agreements,
        }
        translation_response, translation_receipt = _persist_call(''',
        "translation payload",
    )

    text = replace_once(
        text,
        '''            repair_numbers = [
                int(review["block_number"])
                for review in reviews
                if review.get("verdict") == "repair"
                and not review.get("critical_slot_conflicts")
                and not review.get("unsupported_additions")
            ]
            blocking = [review for review in reviews if review.get("verdict") == "quarantine" or review.get("critical_slot_conflicts") or review.get("unsupported_additions")]
''',
        '''            repair_numbers = [
                int(review["block_number"])
                for review in reviews
                if review.get("verdict") == "repair"
                and not any(
                    finding.get("disposition") == "blocking"
                    for finding in review.get("claim_findings", [])
                    if isinstance(finding, dict)
                )
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
            ]
''',
        "claim-level repair queue",
    )

    text = replace_once(
        text,
        '''                    "schema_name": "translation-forensics/codex-quality-decision",
                    "schema_version": "1",
''',
        '''                    "schema_name": "translation-forensics/codex-quality-decision",
                    "schema_version": "2",
''',
        "decision schema version",
    )

    text = replace_once(
        text,
        '''                    "consensus_frame": agreement_by_number[number]["consensus_frame"],
                    "critical_slot_conflicts": agreement_by_number[number]["critical_slot_conflicts"],
                    "slot_coverage_gaps": agreement_by_number[number].get("slot_coverage_gaps", []),
''',
        '''                    "consensus_frame": agreement_by_number[number]["consensus_frame"],
                    "selected_frame": agreement_by_number[number]["selected_frame"],
                    "slot_provenance": agreement_by_number[number]["slot_provenance"],
                    "rendered_slots": sorted(set(row.get("rendered_slots", []))),
                    "omitted_slots": sorted(
                        set(agreement_by_number[number]["omitted_slots"])
                        | (
                            set(agreement_by_number[number]["rendered_slots"])
                            - set(row.get("rendered_slots", []))
                        )
                    ),
                    "critical_slot_conflicts": agreement_by_number[number]["critical_slot_conflicts"],
                    "render_blocking_conflicts": agreement_by_number[number]["render_blocking_conflicts"],
                    "resolved_conflicts": agreement_by_number[number]["resolved_conflicts"],
                    "coherence_findings": agreement_by_number[number]["coherence_findings"],
                    "uncertainty_codes": sorted(
                        set(row.get("uncertainty_codes", []))
                        | set(agreement_by_number[number].get("uncertainty_codes", []))
                    ),
                    "slot_coverage_gaps": agreement_by_number[number].get("slot_coverage_gaps", []),
                    "utterance_kind": agreement_by_number[number].get("utterance_kind", "unknown"),
                    "utterance_kind_confidence": agreement_by_number[number].get(
                        "utterance_kind_confidence",
                        "low",
                    ),
                    "utterance_kind_reason_codes": agreement_by_number[number].get(
                        "utterance_kind_reason_codes",
                        [],
                    ),
                    "utterance_kind_evidence_refs": agreement_by_number[number].get(
                        "utterance_kind_evidence_refs",
                        [],
                    ),
''',
        "decision arbitration fields",
    )

    text = replace_once(
        text,
        '''        critical_conflicts = 0
        damaged_without_dual = 0
        accepted = 0
        ellipsis = 0
        reused: dict[str, list[dict[str, Any]]] = defaultdict(list)
''',
        '''        critical_conflicts = 0
        damaged_without_dual = 0
        accepted = 0
        safe_usable = 0
        meaningful_lexical = 0
        minimal_speech_acts = 0
        vocalizations = 0
        ellipsis = 0
        recovery_counts: Counter[str] = Counter()
        reused: dict[str, list[dict[str, Any]]] = defaultdict(list)
''',
        "validation counters",
    )

    text = replace_once(
        text,
        '''            if row.get("source_status") == "accepted":
                accepted += 1
                if row.get("critical_slot_conflicts"):
                    critical_conflicts += 1
                if row.get("source_quality_status") in {"suspect", "unusable"} and len(set(row.get("independent_source_families", []))) < 2:
                    damaged_without_dual += 1
            viewer_text = str(row.get("viewer_natural_korean") or "").strip()
            if viewer_text == "…":
                ellipsis += 1
''',
        '''            recovery_state = str(row.get("recovery_state") or "abstained")
            recovery_counts[recovery_state] += 1
            if row.get("source_status") == "accepted":
                accepted += 1
                if row.get("render_blocking_conflicts"):
                    critical_conflicts += 1
                if row.get("source_quality_status") in {"suspect", "unusable"} and len(set(row.get("independent_source_families", []))) < 2:
                    damaged_without_dual += 1
            viewer_text = str(row.get("viewer_natural_korean") or "").strip()
            if viewer_text == "…":
                ellipsis += 1
            elif recovery_state != "abstained":
                safe_usable += 1
                if recovery_state == "vocalization":
                    vocalizations += 1
                else:
                    meaningful_lexical += 1
                if recovery_state == "minimal_speech_act":
                    minimal_speech_acts += 1
''',
        "validation per-block metrics",
    )

    text = replace_once(
        text,
        '''        accepted = ellipsis = critical_conflicts = damaged_without_dual = mass_copy_groups = 0
        structure = []
        receipts = []
    block_count = len(structure)
    accepted_rate = accepted / max(1, block_count)
    ellipsis_rate = ellipsis / max(1, block_count)
''',
        '''        accepted = safe_usable = meaningful_lexical = minimal_speech_acts = vocalizations = 0
        ellipsis = critical_conflicts = damaged_without_dual = mass_copy_groups = 0
        recovery_counts = Counter()
        structure = []
        receipts = []
    block_count = len(structure)
    accepted_rate = accepted / max(1, block_count)
    safe_usable_rate = safe_usable / max(1, block_count)
    ellipsis_rate = ellipsis / max(1, block_count)
''',
        "validation exception and rates",
    )

    text = replace_once(
        text,
        '''        if accepted_rate < gate["minimum_accepted_rate"]:
            errors.append(
                f"accepted rate {accepted_rate:.4f} is below {gate['minimum_accepted_rate']:.2f} for {title_id}"
            )
            metric_gate_passed = False
''',
        '''        if safe_usable_rate < gate["minimum_accepted_rate"]:
            errors.append(
                f"safe usable-output rate {safe_usable_rate:.4f} is below {gate['minimum_accepted_rate']:.2f} for {title_id}"
            )
            metric_gate_passed = False
''',
        "safe usable title gate",
    )

    text = replace_once(
        text,
        '''        "accepted_count": accepted,
        "accepted_rate": round(accepted_rate, 6),
        "ellipsis_count": ellipsis,
''',
        '''        "accepted_count": accepted,
        "accepted_rate": round(accepted_rate, 6),
        "safe_usable_count": safe_usable,
        "safe_usable_rate": round(safe_usable_rate, 6),
        "meaningful_lexical_count": meaningful_lexical,
        "minimal_speech_act_count": minimal_speech_acts,
        "vocalization_count": vocalizations,
        "recovery_state_counts": dict(sorted(recovery_counts.items())),
        "ellipsis_count": ellipsis,
''',
        "validation output metrics",
    )
    return text


def patch_local_asr(text: str) -> str:
    if "from .asr_fusion import add_asr_fusion" in text:
        raise PatchError("local_asr.py already imports add_asr_fusion")

    text = replace_once(
        text,
        "from .asr_evidence import normalize_japanese, text_similarity\n",
        "from .asr_evidence import normalize_japanese, text_similarity\n"
        "from .asr_fusion import add_asr_fusion\n",
        "local ASR fusion import",
    )
    text = replace_once(
        text,
        "    block_records = map_transcripts_to_blocks(blocks, windows, records)\n",
        "    block_records = [\n"
        "        add_asr_fusion(record)\n"
        "        for record in map_transcripts_to_blocks(blocks, windows, records)\n"
        "    ]\n",
        "full ASR block fusion",
    )
    text = replace_once(
        text,
        "    _write = \"\".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + \"\\n\" for record in records)\n",
        "    records = [add_asr_fusion(record) for record in records]\n"
        "    _write = \"\".join(\n"
        "        json.dumps(record, ensure_ascii=False, sort_keys=True) + \"\\n\"\n"
        "        for record in records\n"
        "    )\n",
        "conflict rerun fusion",
    )
    text = replace_once(
        text,
        "    output_dir.mkdir(parents=True, exist_ok=True)\n"
        "    ledger_path.write_text(\n",
        "    block_records = [add_asr_fusion(record) for record in block_records]\n"
        "    output_dir.mkdir(parents=True, exist_ok=True)\n"
        "    ledger_path.write_text(\n",
        "timestamped block fusion",
    )
    return text


def patch_test_codex_quality(text: str) -> str:
    text = replace_once(
        text,
        '''                        "viewer_natural_korean": f"멈춰 {number}",
                        "source_status": "accepted",
                        "viewer_status": "supported",
                        "evidence_refs": [],
                        "reason": "supported",
''',
        '''                        "viewer_natural_korean": f"멈춰 {number}",
                        "conservative_source_faithful_korean": f"멈춰 {number}",
                        "conservative_viewer_natural_korean": f"멈춰 {number}",
                        "source_status": "accepted",
                        "viewer_status": "supported",
                        "recovery_state": "accepted_consensus",
                        "fallback_recovery_state": "accepted_consensus",
                        "rendered_slots": ["speech_act", "polarity", "action"],
                        "fallback_rendered_slots": ["speech_act", "polarity", "action"],
                        "uncertainty_codes": [],
                        "evidence_refs": [],
                        "reason": "supported",
''',
        "fake translation v2 fields",
    )
    text = replace_once(
        text,
        '''                        "unsupported_additions": [],
                        "naturalness_issues": [],
                        "reason": "supported",
''',
        '''                        "unsupported_additions": [],
                        "naturalness_issues": [],
                        "claim_findings": [],
                        "fallback_blocking": False,
                        "reason": "supported",
''',
        "fake critique v2 fields",
    )
    text = replace_once(
        text,
        '''    assert validation["accepted_rate"] == 1.0
''',
        '''    assert validation["accepted_rate"] == 1.0
    assert validation["safe_usable_rate"] == 1.0
    assert validation["recovery_state_counts"] == {"accepted_consensus": 2}
''',
        "end-to-end v2 assertions",
    )
    return text


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=path.parent,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _payload_files(bundle_root: Path) -> list[Path]:
    relative = [
        Path("src/translation_forensics/graduated_recovery.py"),
        Path("src/translation_forensics/asr_fusion.py"),
        Path("src/translation_forensics/utterance_routing.py"),
        Path("prompts/codex-quality-meaning-frame-v1.md"),
        Path("prompts/codex-quality-translation-v1.md"),
        Path("prompts/codex-quality-critic-v1.md"),
        Path("prompts/codex-quality-repair-v1.md"),
        Path("schemas/codex-quality-frame-response.schema.json"),
        Path("schemas/codex-quality-translation-response.schema.json"),
        Path("schemas/codex-quality-critique-response.schema.json"),
        Path("schemas/codex-quality-decision.schema.json"),
        Path("schemas/block-acoustic-evidence.schema.json"),
        Path("tests/unit/test_graduated_recovery.py"),
        Path("tests/unit/test_asr_fusion.py"),
        Path("tests/unit/test_utterance_routing.py"),
    ]
    missing = [path for path in relative if not (bundle_root / path).is_file()]
    if missing:
        raise PatchError(
            "bundle payload is incomplete: "
            + ", ".join(path.as_posix() for path in missing)
        )
    return relative


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply graduated recovery v2 to the inspected branch."
    )
    parser.add_argument("repo_root", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate target anchors and payload without writing files.",
    )
    args = parser.parse_args()

    root = args.repo_root.expanduser().resolve()
    bundle_root = Path(__file__).resolve().parents[1]
    codex_path = root / "src" / "translation_forensics" / "codex_quality.py"
    local_asr_path = root / "src" / "translation_forensics" / "local_asr.py"
    test_path = root / "tests" / "unit" / "test_codex_quality.py"
    for path in (codex_path, local_asr_path, test_path):
        if not path.is_file():
            raise PatchError(f"required file is missing: {path}")

    payload = _payload_files(bundle_root)
    for relative in payload:
        if relative.suffix == ".json":
            json.loads((bundle_root / relative).read_text(encoding="utf-8"))

    codex_updated = patch_codex_quality(
        codex_path.read_text(encoding="utf-8")
    )
    local_asr_updated = patch_local_asr(
        local_asr_path.read_text(encoding="utf-8")
    )
    test_updated = patch_test_codex_quality(
        test_path.read_text(encoding="utf-8")
    )

    if args.check:
        print("Patch anchors and payload validated; no files written.")
        return 0

    for relative in payload:
        source = bundle_root / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    _atomic_write(codex_path, codex_updated)
    _atomic_write(local_asr_path, local_asr_updated)
    _atomic_write(test_path, test_updated)
    print(f"Updated {codex_path}")
    print(f"Updated {local_asr_path}")
    print(f"Updated {test_path}")
    print(f"Copied {len(payload)} v2 payload files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
