#!/bin/bash
# Run compare.py for every k_lq4w instance (448). On Ai-core, from ~/re-halogen:
#   bash rewrite/specs/lq4w_run_all.sh [FILTER_REGEX] >> /tmp/g01_run.log
# FILTER_REGEX is matched against the mangled name (for example 'ILi1ELi16ELi1E' for NC=1,W=16,M=1).
# The rewrite is compiled once per instance with the template arguments passed through
# HIPCC_COMPILE_FLAGS_APPEND (compare.py calls hipcc and inherits the environment).
# Sizes: rows N = 1000 and 2800 (K = 32*ceil(N/20), 50 and 140 groups; 140 > 128 takes the vendor F=1 loop path).
# E=5 splits rows in 512-row blocks, so its sizes are 1024 and 3072.
filter=${1:-.}
tsv=${LQ_TSV:-analysis/function-map-kernel-stubs.tsv}  # not on Ai-core: scp it to /tmp and set LQ_TSV
re='k_lq4wILi([0-9]+)ELi([0-9]+)ELi([0-9]+)ELb([0-9])ELi([0-9])ELi([0-9])E'
grep -P '\tlq4w$' "$tsv" | cut -f2 | grep -E "$filter" | while read -r name; do
  [[ $name =~ $re ]] || continue
  nc=${BASH_REMATCH[1]}; w=${BASH_REMATCH[2]}; m=${BASH_REMATCH[3]}
  b=${BASH_REMATCH[4]}; e=${BASH_REMATCH[5]}; f=${BASH_REMATCH[6]}
  sizes=1000,2800; [ "$e" = 5 ] && sizes=1024,3072
  HIPCC_COMPILE_FLAGS_APPEND="-DLQ_N=$nc -DLQ_W=$w -DLQ_M=$m -DLQ_B=$b -DLQ_E=$e -DLQ_F=$f" \
    python3 -I rewrite/harness/compare.py rewrite/specs/lq4w_c${nc}_b${b}_e${e}.json --kernel "$name" \
      --seeds 1,2,3 --sizes "$sizes" --table "${LQ_TABLE:-$HOME/re-halogen/rewrite/tables/g01.md}" 2>&1 | grep -v '^MATCH' \
      | sed "s|^|[$nc,$w,$m,$b,$e,$f] |"
  echo "done $nc $w $m $b $e $f"
done
