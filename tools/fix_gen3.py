import sys
sys.stdout.reconfigure(encoding="utf-8")
with open("tools/codex_gen3.py","r",encoding="utf-8") as f:
    c = f.read()
old = "WS / 'intermediate' / 'SSIS-575.gpt56-direct-v1.' + kind + '.srt'"
new = "WS / 'intermediate' / ('SSIS-575.gpt56-direct-v1.' + kind + '.srt')"
c = c.replace(old, new)
with open("tools/codex_gen3.py","w",encoding="utf-8") as f:
    f.write(c)
print("fixed")
