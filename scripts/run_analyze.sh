#!/bin/bash
export GHIDRA_INSTALL_DIR=/home/rangizingo/.hermes/tools/ghidra-12.1.2
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export TMPDIR=/home/rangizingo/re-halogen/tmp
export REA_GHIDRA_STARTUP_TIMEOUT_MS=1800000
export PATH=/home/rangizingo/.hermes/tools/node-26.7.0-linux-x64/bin:/usr/bin:/bin
cd /home/rangizingo/re-halogen
rea analyze flash_serve --provider ghidra --snapshot /home/rangizingo/re-halogen/flash.snap --format json > /home/rangizingo/re-halogen/analyze2.json 2> /home/rangizingo/re-halogen/analyze2.err
echo "analyze exit=$?" >> /home/rangizingo/re-halogen/analyze.status
rea function flash_serve main --provider ghidra --snapshot /home/rangizingo/re-halogen/flash.snap --format json > /home/rangizingo/re-halogen/main_fn.json 2> /home/rangizingo/re-halogen/main_fn.err
echo "function exit=$?" >> /home/rangizingo/re-halogen/analyze.status
