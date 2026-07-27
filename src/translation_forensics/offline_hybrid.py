from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .srt import parse_srt, render_srt, SubtitleBlock

def get_block_dict(path: Path) -> dict[int, SubtitleBlock]:
    if not path.exists():
        return {}
    blocks, _, _ = parse_srt(path)
    return {b.number: b for b in blocks}

def build_offline_hybrid(titles_file: Path, workspace_root: Path, output_dir: Path) -> None:
    if not titles_file.exists():
        raise FileNotFoundError(f"Titles file not found: {titles_file}")
    
    titles = [t.strip() for t in titles_file.read_text("utf-8-sig").splitlines() if t.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for title in titles:
        logging.info(f"Processing offline hybrid for {title}")
        title_dir = workspace_root / title
        if not title_dir.exists():
            logging.warning(f"Workspace for {title} not found at {title_dir}, skipping.")
            continue
        
        # Paths
        cw_sf = title_dir / "closed-world" / "closed-world-validated-v1" / "source-faithful.preview.srt"
        cw_vn = title_dir / "closed-world" / "closed-world-validated-v1" / "viewer-natural.preview.srt"
        ir_sf = title_dir / "inferred-recovery-v4" / f"{title}.source-faithful-ko.inferred-recovery-v4.srt"
        ir_vn = title_dir / "inferred-recovery-v4" / f"{title}.viewer-natural-ko.inferred-recovery-v4.srt"
        mf_sf = title_dir / "runs" / "2026-07-26-full-execution-v1" / "machine-final-v1" / f"{title}.source-faithful-ko.machine-final-v1.srt"
        mf_vn = title_dir / "runs" / "2026-07-26-full-execution-v1" / "machine-final-v1" / f"{title}.viewer-natural-ko.machine-final-v1.srt"
        
        if not mf_sf.exists():
            mf_sf = title_dir / "machine-final-v1" / f"{title}.source-faithful-ko.machine-final-v1.srt"
        if not mf_vn.exists():
            mf_vn = title_dir / "machine-final-v1" / f"{title}.viewer-natural-ko.machine-final-v1.srt"
            
        blocks_cw_sf = get_block_dict(cw_sf)
        blocks_cw_vn = get_block_dict(cw_vn)
        blocks_ir_sf = get_block_dict(ir_sf)
        blocks_ir_vn = get_block_dict(ir_vn)
        blocks_mf_sf = get_block_dict(mf_sf)
        blocks_mf_vn = get_block_dict(mf_vn)
        
        all_block_numbers = sorted(set(list(blocks_cw_sf.keys()) + list(blocks_mf_sf.keys()) + list(blocks_ir_sf.keys())))
        if not all_block_numbers:
            logging.warning(f"No blocks found for {title}, skipping.")
            continue
            
        hybrid_sf_blocks = []
        hybrid_vn_blocks = []
        
        for num in all_block_numbers:
            base_block = blocks_mf_sf.get(num) or blocks_cw_sf.get(num) or blocks_ir_sf.get(num)
            
            # Determine source-faithful
            sf_text = ""
            cw_sf_b = blocks_cw_sf.get(num)
            ir_sf_b = blocks_ir_sf.get(num)
            
            if cw_sf_b and "[미확정]" not in cw_sf_b.text and cw_sf_b.text.strip():
                sf_text = cw_sf_b.text
            elif ir_sf_b and ir_sf_b.text.strip():
                sf_text = ir_sf_b.text
                
            # Determine viewer-complete (natural)
            vn_text = "…"
            cw_vn_b = blocks_cw_vn.get(num)
            ir_vn_b = blocks_ir_vn.get(num)
            mf_vn_b = blocks_mf_vn.get(num)
            
            if cw_vn_b and "[미확정]" not in cw_vn_b.text and cw_vn_b.text.strip():
                vn_text = cw_vn_b.text
            elif ir_vn_b and ir_vn_b.text.strip():
                vn_text = ir_vn_b.text
            elif mf_vn_b and mf_vn_b.text.strip():
                vn_text = mf_vn_b.text
                
            if not vn_text.strip():
                vn_text = "…"
                
            hybrid_sf_blocks.append(SubtitleBlock(
                number=num,
                start=base_block.start,
                end=base_block.end,
                text=sf_text,
                start_seconds=base_block.start_seconds,
                end_seconds=base_block.end_seconds,
            ))
            
            hybrid_vn_blocks.append(SubtitleBlock(
                number=num,
                start=base_block.start,
                end=base_block.end,
                text=vn_text,
                start_seconds=base_block.start_seconds,
                end_seconds=base_block.end_seconds,
            ))
            
        title_out_dir = output_dir / title / "offline-hybrid-v1"
        title_out_dir.mkdir(parents=True, exist_ok=True)
        
        out_sf_path = title_out_dir / f"{title}.source-faithful-ko.offline-hybrid-v1.srt"
        out_vn_path = title_out_dir / f"{title}.viewer-complete-ko.offline-hybrid-v1.srt"
        
        out_sf_path.write_text(render_srt(hybrid_sf_blocks), encoding="utf-8-sig")
        out_vn_path.write_text(render_srt(hybrid_vn_blocks), encoding="utf-8-sig")
        logging.info(f"Generated {out_sf_path} and {out_vn_path}")
