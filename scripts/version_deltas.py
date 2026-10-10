#!/usr/bin/env python3
"""Version delta: kernel symbol sets + host-side string sets across flash_serve versions."""
import json, os, subprocess, collections, re, sys

ROOT='/home/rangizingo/re-halogen'
VERS=['0.15.0','0.16.4','0.17.0','0.17.2','0.17.3']
def path(v):
    return f'{ROOT}/binaries/flash_serve' if v=='0.17.3' else f'{ROOT}/work/versions/{v}/flash_serve'

def ksyms(v):
    mp = f'{ROOT}/work/fatbin-manifest.json' if v=='0.17.3' else f'{ROOT}/work/versions/{v}/fatbin-manifest.json'
    if not os.path.exists(mp): return None
    m=json.load(open(mp))
    s=set()
    for b in m['blobs']:
        s.update(b.get('symbol_names') or [])
    return s

def strings(v):
    r=subprocess.run(['strings','-a',path(v)],capture_output=True,text=True)
    return set(l for l in r.stdout.splitlines() if len(l)>=12)

out=[]
sets={v:ksyms(v) for v in VERS}
strs={v:strings(v) for v in VERS}
for i in range(1,len(VERS)):
    a,b=VERS[i-1],VERS[i]
    ka,kb=sets[a],sets[b]
    if ka is None or kb is None:
        out.append(f'{a}->{b}: fatbin manifest missing'); continue
    added=sorted(kb-ka); removed=sorted(ka-kb)
    out.append(f'== kernels {a} -> {b}: +{len(added)} -{len(removed)} (total {len(ka)} -> {len(kb)})')
    for k in added[:40]: out.append('  + '+k)
    for k in removed[:40]: out.append('  - '+k)
sa,sb=strs['0.17.2'],strs['0.17.3']
out.append(f'== strings 0.17.2 -> 0.17.3: +{len(sb-sa)} -{len(sa-sb)}')
for s in sorted(sb-sa)[:80]: out.append('  + '+s[:150])
for s in sorted(sa-sb)[:80]: out.append('  - '+s[:150])
open(f'{ROOT}/tmp/deltas_raw.txt','w').write('\n'.join(out))
print('\n'.join(out[:120]))
print('...full file tmp/deltas_raw.txt', len(out), 'lines')
