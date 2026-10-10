#!/bin/bash
# Test the AMD Radeon F32 Ghidra processor module on real Radeon firmware
# (navi10_asd.bin = Animation State Data microcode for Navi 10 / RDNA1).
export GHIDRA_INSTALL_DIR=/home/rangizingo/.hermes/tools/ghidra-12.1.2
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export TMPDIR=/home/rangizingo/re-halogen/tmp
export PATH=/home/rangizingo/.hermes/tools/node-26.7.0-linux-x64/bin:/usr/bin:/bin
cd /home/rangizingo/re-halogen
mkdir -p tmp/proj-f32
$GHIDRA_INSTALL_DIR/support/analyzeHeadless tmp/proj-f32 f32test \
  -import tmp/navi10_asd.bin \
  -loader BinaryLoader -loader-baseAddr 0x0 \
  -processor "F32:BE:64:default" \
  -scriptPath /home/rangizingo/re-halogen/scripts \
  -postScript F32Report.java \
  > tmp/f32-headless.log 2>&1
echo "exit=$?" > tmp/f32-headless.status
