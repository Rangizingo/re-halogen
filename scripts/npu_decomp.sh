#!/bin/bash
# Decompile targeted halogen-npu procedures with REA's Ghidra provider.
export GHIDRA_INSTALL_DIR=/home/rangizingo/.hermes/tools/ghidra-12.1.2
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export TMPDIR=/home/rangizingo/re-halogen/tmp
export REA_GHIDRA_STARTUP_TIMEOUT_MS=1800000
export PATH=/home/rangizingo/.hermes/tools/node-26.7.0-linux-x64/bin:/usr/bin:/bin
cd /home/rangizingo/re-halogen
mkdir -p tmp/npu
for a in "$@"; do
  rea function binaries/halogen-npu $a --provider ghidra --format json > tmp/npu/f_$a.json 2> tmp/npu/f_$a.err
  python3 - "$a" <<'EOF'
import json,sys
a=sys.argv[1]
try:
    d=json.load(open(f'tmp/npu/f_{a}.json'))
    nr=d['normalized_result']
    open(f'tmp/npu/{a}.c','w').write(nr['pseudocode'])
    open(f'tmp/npu/{a}.meta.json','w').write(json.dumps({'evidence_id':d['evidence_id'],'procedure':nr.get('procedure'),'callers':nr.get('callers'),'callees':nr.get('callees'),'referenced_strings':nr.get('referenced_strings')},indent=1))
    print(a,'ok',d['evidence_id'][:19],len(nr['pseudocode']))
except Exception as e:
    print(a,'FAIL',e)
EOF
done
