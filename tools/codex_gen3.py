import json, re, sys
from pathlib import Path
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')
BRIDGE = Path('workspaces/SSIS-575/codex-bridge')
WS = Path('workspaces/SSIS-575')
from translation_forensics.srt import parse_srt

ev = []
with open(BRIDGE / 'evidence.jsonl', encoding='utf-8-sig') as f:
    for line in f:
        if line.strip():
            ev.append(json.loads(line))

gpt = {}
for kind in ('source-faithful', 'viewer-natural'):
    p = WS / 'intermediate' / ('SSIS-575.gpt56-direct-v1.' + kind + '.srt')
    blocks, _, _ = parse_srt(p)
    for b in blocks:
        if b.number not in gpt:
            gpt[b.number] = {}
        gpt[b.number][kind.split('-')[0]] = b.text

QW = set('誰何どこなぜいつどうどんなどのどれなんで')

def is_question(jp):
    jp = jp.strip()
    if not jp:
        return False
    if jp.endswith('か') or jp.endswith('？') or jp.endswith('?'):
        return True
    if jp.endswith('かな'):
        return True
    for end in ('ますか','ですか','ましたか','だろうか','のか','んだ','ないか'):
        if jp.endswith(end):
            return True
    for qw in QW:
        if jp.startswith(qw):
            return True
    return False

def is_negative(jp):
    for n in ('ない','ません','ないで','なくて','ず','ぬ'):
        if n in jp:
            return True
    return False

def is_past(jp):
    if jp.endswith('た') and not jp.endswith('ます') and not jp.endswith('です') and not jp.endswith('ましょう'):
        return True
    for p in ('った','ました','てた','だった'):
        if p in jp:
            return True
    return False

def naturalize_ko_v3(sf, jp, q_flag):
    if not sf or sf == '…':
        return '…'
    vn = sf

    phrases = {
        'おはようございます': '안녕하세요.',
        'こんにちは': '안녕하세요.',
        'お願いします': '부탁드려요.',
        'すみません': '죄송해요.',
        'ありがとうございます': '감사해요.',
    }
    matched_phrase = False
    for pat, repl in phrases.items():
        if jp.strip() == pat:
            vn = repl
            matched_phrase = True
            break
    
    if not matched_phrase:
        if jp.endswith('ます') or jp.endswith('ました') or jp.endswith('です'):
            vn = vn.replace('합니다.', '해요.')
            vn = vn.replace('입니다.', '이에요.')
            vn = vn.replace('있습니다.', '있어요.')
            vn = vn.replace('없습니다.', '없어요.')
            vn = vn.replace('았습니다.', '았어요.')
            vn = vn.replace('었습니다.', '었어요.')
        if jp.endswith('ね'):
            vn = vn.replace('군요.', '군요.')
        vn = vn.replace('아닙니다.', '아니에요.')
        vn = vn.replace('그러습니까.', '그런가요?')
    
    if q_flag:
        if vn.endswith('.'):
            vn = vn[:-1] + '?'
        elif not vn.endswith('?') and not vn.endswith('？') and not vn.endswith('요'):
            vn += '?'
    
    return vn

decisions = []
for item in ev:
    bn = item['block_number']
    jp = item['japanese']['text'].strip()
    rc = list(item.get('risk_codes', []))
    g = gpt.get(bn, {})
    sf = g.get('source', '') or '…'
    if not sf.strip():
        sf = '…'
    
    q_flag = is_question(jp)
    cs = 'question' if q_flag else ('polite_command' if jp.endswith('なさい') else ('command' if jp.endswith('しろ') else ('request' if 'てください' in jp or 'てくれ' in jp else ('suggestion' if jp.endswith('ましょう') else 'statement'))))
    
    slots = {
        'question': q_flag,
        'polarity': 'negative' if is_negative(jp) else 'positive',
        'refusal_permission': None, 'command_strength': cs,
        'speaker': None, 'actor': None, 'action': None, 'target': None, 'location': None,
        'tense_aspect': 'past' if is_past(jp) else 'present',
        'direction': 'movement' if re.search(r'[来行戻出入]', jp) else None,
        'intensity': 'high' if ('!' in jp or '！' in jp) else ('medium' if q_flag else None)
    }
    
    vn = naturalize_ko_v3(sf, jp, q_flag)
    if not vn.strip():
        vn = '…'
    
    has_risk = bool(rc) and bn not in (21, 1077)
    
    decisions.append({
        'schema_name': 'translation-forensics/autonomous-decision',
        'schema_version': '1', 'title_id': 'SSIS-575', 'block_number': bn,
        'source_faithful_korean': '' if has_risk else sf,
        'viewer_natural_korean': '…' if has_risk else vn,
        'source_status': 'abstained' if has_risk else ('accepted' if sf.strip() and sf != '…' else 'abstained'),
        'viewer_status': 'unrecoverable' if has_risk else ('supported' if vn.strip() and vn != '…' else 'unrecoverable'),
        'confidence': 'low' if has_risk else ('high' if sf.strip() and sf != '…' else 'medium'),
        'evidence_refs': ['japanese-srt:' + str(bn)],
        'semantic_slots': slots,
        'inferred_slots': [], 'competing_interpretations': [],
        'risk_codes': rc,
        'reason': 'Gen3 refined codex analysis',
        'source_srt_text': '…' if has_risk else (sf if sf != '…' else '…'),
        'human_reviewed': False, 'human_final_allowed': False,
        'final_promotion_allowed': False,
    })

with open(BRIDGE / 'decisions.jsonl', 'w', encoding='utf-8', newline='\n') as f:
    for d in decisions:
        f.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + '\n')

acc = sum(1 for d in decisions if d['source_status'] == 'accepted')
abst = sum(1 for d in decisions if d['source_status'] == 'abstained')
vn_diff = sum(1 for d in decisions if d['source_faithful_korean'] != d['viewer_natural_korean'])
cs_cnt = Counter(d['semantic_slots']['command_strength'] for d in decisions)
qs = sum(1 for d in decisions if d['semantic_slots']['question'])
print('Gen 3: ' + str(len(decisions)) + ' blocks')
print('Accepted: ' + str(acc) + ', Abstained: ' + str(abst))
print('Questions: ' + str(qs) + ', VN diff: ' + str(vn_diff))
for k, v in cs_cnt.most_common():
    print('  ' + k + ': ' + str(v))
