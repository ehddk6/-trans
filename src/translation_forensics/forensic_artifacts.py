from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .forensic_model import frame_conflicts, read_jsonl
from .scenes import scenes_from_csv


def initialize_speaker_state(scenes_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"기존 speaker state를 덮어쓰지 않습니다: {output_path}")
    records = []
    for scene in scenes_from_csv(scenes_path):
        records.append({"scene_id": scene.get("scene_id"), "participants": [], "current_speaker": "unknown", "previous_speaker": "unknown", "addressee": "unknown", "speaker_confidence": "unknown", "relationship": "unknown", "register_by_speaker": {}, "address_terms": [], "current_action_by_participant": {}, "question_owner": "unknown", "expected_responder": "unknown", "review_status": "unreviewed"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"status": "speaker-state-template", "output": str(output_path), "scenes": len(records), "review_status": "not-reviewed"}


def initialize_phonetic_candidates(queue_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"기존 phonetic candidate를 덮어쓰지 않습니다: {output_path}")
    records = [{"block_number": queue.get("block_number"), "title_id": queue.get("title_id", ""), "phonetic_candidate_id": "", "candidate": "", "mora_form": "", "supported_by": [], "contradicted_by": [], "boundary_scope": "", "status": "not-analyzed", "note": "음가 후보는 실제 음향·문법 근거가 있을 때만 사람 검수자가 추가합니다."} for queue in read_jsonl(queue_path)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8", newline="\n")
    return {"status": "phonetic-candidate-template", "output": str(output_path), "blocks": len(records), "analysis_status": "not-analyzed"}


def build_slot_conflicts(hypotheses_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"기존 slot conflict 보고서를 덮어쓰지 않습니다: {output_path}")
    by_block: dict[int, list[dict[str, Any]]] = {}
    for item in read_jsonl(hypotheses_path):
        try:
            by_block.setdefault(int(item["block_number"]), []).append(item)
        except (KeyError, TypeError, ValueError):
            continue
    rows = []
    for number, items in sorted(by_block.items()):
        for left_index, left in enumerate(items):
            for right in items[left_index + 1:]:
                for conflict in frame_conflicts(left.get("semantic_frame", {}), right.get("semantic_frame", {})):
                    rows.append({"block_number": number, "left_hypothesis_id": left.get("hypothesis_id", ""), "right_hypothesis_id": right.get("hypothesis_id", ""), **conflict, "action": "needs-human-review"})
    fields = ["block_number", "left_hypothesis_id", "right_hypothesis_id", "slot", "severity", "left", "right", "action"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return {"status": "slot-conflicts-created", "output": str(output_path), "conflicts": len(rows), "critical_conflicts": sum(row["severity"] == "critical" for row in rows)}
