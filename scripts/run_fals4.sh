#!/bin/bash
export GHIDRA_INSTALL_DIR=/home/rangizingo/.hermes/tools/ghidra-12.1.2
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export TMPDIR=/home/rangizingo/re-halogen/tmp
export REA_GHIDRA_STARTUP_TIMEOUT_MS=1800000
export PATH=/home/rangizingo/.hermes/tools/node-26.7.0-linux-x64/bin:/usr/bin:/bin
cd /home/rangizingo/re-halogen
for a in 0x19098d0 0x1909e50 0x1851dc0; do
  rea function flash_serve $a --provider ghidra --format json > gf_$a.json 2> gf_$a.err
  echo "$a exit=$?" >> /home/rangizingo/re-halogen/analyze.status
done
