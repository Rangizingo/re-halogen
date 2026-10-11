#!/bin/bash
# Run compare.py for the 32 k_lq4w_grp instances, three spec variants each (A, B, C). On Ai-core, from ~/re-halogen:
#   LQ_TSV=/tmp/g01_stubs.tsv bash rewrite/specs/lq4w_grp_run_all.sh [FILTER_REGEX]
# Template args: (W,M) = (16,1) -> small-R split 16; (32,2) -> small-R split 8. Variant C needs R % 512 == 0.
filter=${1:-.}
tsv=${LQ_TSV:-analysis/function-map-kernel-stubs.tsv}
re='k_lq4w_grpILi([0-9]+)ELi([0-9]+)ELi([0-9]+)ELi([0-9]+)E'
grep -P '\tlq4w$' "$tsv" | cut -f2 | grep -E "$filter" | while read -r name; do
  [[ $name =~ $re ]] || continue
  nc=${BASH_REMATCH[1]}; w=${BASH_REMATCH[2]}; mb=${BASH_REMATCH[3]}; f=${BASH_REMATCH[4]}
  ms=8; [ "$w" = 16 ] && ms=16
  for v in A B C; do
    sizes=1000,2800; [ $v = C ] && sizes=1024,3072
    HIPCC_COMPILE_FLAGS_APPEND="-DLQ_GRP_N=$nc -DLQ_GRP_W=$w -DLQ_GRP_MB=$mb -DLQ_GRP_MS=$ms -DLQ_GRP_F=$f" \
      python3 -I rewrite/harness/compare.py rewrite/specs/lq4w_grp_c${nc}_$v.json --kernel "$name" \
        --seeds 1,2,3 --sizes "$sizes" --table "${LQ_TABLE:-$HOME/re-halogen/rewrite/tables/g01.md}" 2>&1 | grep -v '^MATCH' \
        | sed "s|^|[grp $nc $w $mb $f $v] |"
  done
  echo "done grp $nc $w $mb $f"
done
