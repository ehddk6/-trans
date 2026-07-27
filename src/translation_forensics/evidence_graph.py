from __future__ import annotations

import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .forensic_model import read_jsonl


def _node(identifier: str, kind: str, **fields: Any) -> dict[str, Any]:
    return {"record_type": "node", "id": identifier, "kind": kind, **fields}


def _edge(source: str, target: str, relation: str, **fields: Any) -> dict[str, Any]:
    return {"record_type": "edge", "source": source, "target": target, "relation": relation, **fields}


def build_evidence_graph(title: str, queue_path: Path, output_path: Path, *, decisions_path: Path | None = None, frames_path: Path | None = None, hypotheses_path: Path | None = None, evaluation_path: Path | None = None) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"기존 evidence graph를 덮어쓰지 않습니다: {output_path}")
    queue = read_jsonl(queue_path)
    decisions = {int(row["block_number"]): row for row in read_jsonl(decisions_path)} if decisions_path and decisions_path.exists() else {}
    frames = {int(row["block_number"]): row for row in read_jsonl(frames_path)} if frames_path and frames_path.exists() else {}
    hypotheses = read_jsonl(hypotheses_path) if hypotheses_path and hypotheses_path.exists() else []
    generated_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = [_node(f"title:{title}", "title", title_id=title, created_at=generated_at), _node(f"timeline:{title}", "timeline", title_id=title, created_at=generated_at)]
    known_nodes = {f"title:{title}", f"timeline:{title}"}
    orphan_refs: list[dict[str, Any]] = []
    for item in queue:
        number = int(item["block_number"])
        block_id = f"block:{title}:{number}"
        known_nodes.add(block_id)
        rows.append(_node(block_id, "block", block_number=number, timecode=item.get("timecode"), review_band=item.get("review_band")))
        rows.append(_edge(f"title:{title}", block_id, "contains"))
        rows.append(_edge(f"timeline:{title}", block_id, "aligned_to"))
        for ref in item.get("evidence_refs", []):
            evidence_id = f"evidence:{ref}"
            if evidence_id not in known_nodes:
                known_nodes.add(evidence_id); rows.append(_node(evidence_id, "evidence", evidence_ref=ref, source_family="unclassified", review_status="unreviewed"))
            rows.append(_edge(evidence_id, block_id, "supports", scope="candidate-only", created_at=generated_at))
        if number in decisions:
            decision_id = f"decision:{title}:{number}"
            known_nodes.add(decision_id); rows.append(_node(decision_id, "translation_decision", status=decisions[number].get("status"), confidence=decisions[number].get("confidence")))
            rows.append(_edge(decision_id, block_id, "derived_from", review_status=decisions[number].get("status")))
            for ref in decisions[number].get("evidence_refs", []):
                evidence_id = f"evidence:{ref}"
                if evidence_id not in known_nodes:
                    orphan_refs.append({"block_number": number, "reference": ref, "location": "decision"})
                else:
                    rows.append(_edge(evidence_id, decision_id, "supports"))
        if number in frames:
            frame_id = f"frame:{title}:{number}"
            known_nodes.add(frame_id); rows.append(_node(frame_id, "semantic_frame", review_status=frames[number].get("review_status")))
            rows.append(_edge(frame_id, block_id, "derived_from", review_status=frames[number].get("review_status")))
    for index, hypothesis in enumerate(hypotheses, 1):
        number = int(hypothesis.get("block_number", 0))
        hypothesis_id = f"hypothesis:{title}:{number}:{hypothesis.get('hypothesis_id', index)}"
        known_nodes.add(hypothesis_id); rows.append(_node(hypothesis_id, "hypothesis", status=hypothesis.get("status")))
        block_id = f"block:{title}:{number}"
        if block_id in known_nodes:
            rows.append(_edge(hypothesis_id, block_id, "derived_from", review_status=hypothesis.get("status")))
        for ref in hypothesis.get("supported_by", []):
            evidence_id = f"evidence:{ref}"
            if evidence_id in known_nodes:
                rows.append(_edge(evidence_id, hypothesis_id, "supports"))
            else:
                orphan_refs.append({"block_number": number, "reference": ref, "location": "hypothesis.supported_by"})
    if evaluation_path and evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        evaluation_id = f"evaluation:{title}:{evaluation.get('created_at', 'unknown')}"
        rows.append(_node(evaluation_id, "evaluation", evaluation_status=evaluation.get("evaluation_status"), review_complete=evaluation.get("review_complete")))
        rows.append(_edge(evaluation_id, f"title:{title}", "evaluated_by", created_at=generated_at))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
    orphan_path = output_path.with_name(output_path.name.replace(".jsonl", ".orphan-reference-report.csv"))
    with orphan_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=["block_number", "reference", "location"]); writer.writeheader(); writer.writerows(orphan_refs)
    provenance_path = output_path.with_name(output_path.name.replace(".jsonl", ".provenance-report.json"))
    node_count = sum(row["record_type"] == "node" for row in rows); edge_count = sum(row["record_type"] == "edge" for row in rows)
    provenance_path.write_text(json.dumps({"title_id": title, "created_at": generated_at, "graph": str(output_path), "nodes": node_count, "edges": edge_count, "queue_blocks": len(queue), "decision_coverage": len(decisions), "semantic_frame_coverage": len(frames), "hypothesis_count": len(hypotheses), "orphan_references": len(orphan_refs), "status": "pass" if not orphan_refs else "warning"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"status": "graph-created" if not orphan_refs else "graph-created-with-orphans", "output": str(output_path), "orphan_report": str(orphan_path), "provenance_report": str(provenance_path), "nodes": node_count, "edges": edge_count, "orphan_references": len(orphan_refs)}
