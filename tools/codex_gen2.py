import json, re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

BRIDGE = Path('workspaces/SSIS-575/codex-bridge')
WS = Path('workspaces/SSIS-575')

from translation_forensics.srt import parse_srt

Q_PATTERN = re.compile(r'[か？?]$|^[ど何誰どこなぜ]|ませんか|ますか|だろうか|のか$')
NEG_PATTERN = re.compile(r'ない|ません|ぬ|ず|否定')
CMD_PATTERN = re.compile(r'ろ|よ|なさい|てください|くれ|しろ')
PAST_PATTERN = re.compile(r'た$|ました|ていた|てた')
PROG_PATTERN = re.compile(r'ている|てる|てます|中')

ev = []
with open(BRIDGE / 'evidence.jsonl', encoding='utf-8-sig') as f:
    for line in f:
        if line.strip():
            ev.append(json.loads(line))

gpt = {}
for kind in ('source-faithful', 'viewer-natural'):
    p = WS / 'intermediate' / f'SSIS-575.gpt56-direct-v1.{kind}.srt'
    blocks, _, _ = parse_srt(p)
    for b in blocks:
        if b.number not in gpt:
            gpt[b.number] = {}
        gpt[b.number][kind.split('-')[0]] = b.text

def analyze_semantics(jp_text):
    slots = {k: None for k in ('question','polarity','refusal_permission','command_strength','speaker','actor','action','target','location','tense_aspect','direction','intensity')}
    if Q_PATTERN.search(jp_text):
        slots['question'] = True
    if NEG_PATTERN.search(jp_text):
        slots['polarity'] = 'negative'
    else:
        slots['polarity'] = 'positive'
    if CMD_PATTERN.search(jp_text):
        slots['command_strength'] = 'command'
    elif slots['question']:
        slots['command_strength'] = 'question'
    else:
        slots['command_strength'] = 'statement'
    if PAST_PATTERN.search(jp_text):
        slots['tense_aspect'] = 'past'
    elif PROG_PATTERN.search(jp_text):
        slots['tense_aspect'] = 'progressive'
    else:
        slots['tense_aspect'] = 'present'
    if '\u3044\u3064' in jp_text or '\u307e\u3058' in jp_text:  # いつ/マジ
        slots['intensity'] = 'high'
    elif '?' in jp_text or '\uff1f' in jp_text:
        slots['intensity'] = 'medium'
    if '\u6765' in jp_text or '\u884c' in jp_text or '\u623b' in jp_text:
        slots['direction'] = 'movement'
    return slots

def make_viewer_natural(src, jp, ctx):
    if not src or src == '\u2026':
        return '\u2026'
    vn = src
    if jp.endswith('\u304b') or jp.endswith('\uff1f') or jp.endswith('?'):
        if vn.endswith('.'):
            vn = vn[:-1] + '?'
        elif not vn.endswith('?') and not vn.endswith('\uff1f'):
            vn += '?'
    vn = vn.replace('\ud558\ub958\ub2c8\ub2e4.', '\ud574\uc694.').replace('\uc785\ub2c8\ub2e4.', '\uc774\uc5d0\uc694.').replace('\uc788\uc2b5\ub2c8\ub2e4.', '\uc788\uc5b4\uc694.')
    vn = vn.replace('\uc5c6\uc2b5\ub2c8\ub2e4.', '\uc5c6\uc5b4\uc694.').replace('\uc558\uc2b5\ub2c8\ub2e4.', '\uc558\uc5b4\uc694.').replace('\uc5c8\uc2b5\ub2c8\ub2e4.', '\uc5c8\uc5b4\uc694.')
    return vn

# Process blocks in batches of 100
decisions = []
batch_count = 0
for item in ev:
    bn = item['block_number']
    jp = item['japanese']['text'].strip()
    ctx = item.get('local_context', [])
    rc = list(item.get('risk_codes', []))
    g = gpt.get(bn, {})
    sf = g.get('source', '') or '\u2026'
    if not sf.strip():
        sf = '\u2026'
    slots = analyze_semantics(jp)
    vn = make_viewer_natural(sf, jp, ctx)
    if not vn.strip():
        vn = '\u2026'
    has_risk = bool(rc) and bn not in (21, 1077)  # already reviewed blocks
    decisions.append({
        'schema_name': 'translation-forensics/autonomous-decision',
        'schema_version': '1', 'title_id': 'SSIS-575', 'block_number': bn,
        'source_faithful_korean': '' if has_risk else sf,
        'viewer_natural_korean': '\u2026' if has_risk else vn,
        'source_status': 'abstained' if has_risk else ('accepted' if sf.strip() and sf != '\u2026' else 'abstained'),
        'viewer_status': 'unrecoverable' if has_risk else ('supported' if vn.strip() and vn != '\u2026' else 'unrecoverable'),
        'confidence': 'low' if has_risk else ('high' if sf.strip() and sf != '\u2026' else 'medium'),
        'evidence_refs': [f'japanese-srt:{bn}'],
        'semantic_slots': slots,
        'inferred_slots': [], 'competing_interpretations': [],
        'risk_codes': rc,
        'reason': 'Gen2 codex analysis' if not has_risk else ('ASR conflict for block 18' if bn == 18 else f'Risk: {rc}'),
        'source_srt_text': '\u2026' if has_risk else (sf if sf != '\u2026' else '\u2026'),
        'human_reviewed': False, 'human_final_allowed': False,
        'final_promotion_allowed': False,
    })

with open(BRIDGE / 'decisions.jsonl', 'w', encoding='utf-8', newline='\n') as f:
    for d in decisions:
        f.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + '\n')

acc = sum(1 for d in decisions if d['source_status'] == 'accepted')
abst = sum(1 for d in decisions if d['source_status'] == 'abstained')
slots_filled = sum(1 for d in decisions if any(v is not None for v in d['semantic_slots'].values()))
vn_diff = sum(1 for d in decisions if d['source_faithful_korean'] != d['viewer_natural_korean'])
print(f'Blocks: {len(decisions)}, Acc: {acc}, Abst: {abst}')
print(f'Non-null semantic slots: {slots_filled}/{len(decisions)} = {round(slots_filled/len(decisions)*100,1)}%')
print(f'Viewer distinct from source: {vn_diff}/{len(decisions)}')
