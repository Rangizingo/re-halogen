#!/bin/bash
# Run compare.py for the 8 k_lq4w_pair instances. On Ai-core, from ~/re-halogen:
#   LQ_TSV=/tmp/g01_stubs.tsv bash rewrite/specs/lq4w_pair_run_all.sh
# Template args reach the rewrite through HIPCC_COMPILE_FLAGS_APPEND. Sizes: rows 1000 and 2800.
filter=${1:-.}
tsv=${LQ_TSV:-analysis/function-map-kernel-stubs.tsv}  # not on Ai-core: scp it to /tmp and set LQ_TSV
re='k_lq4w_pairILi([0-9]+)ELi16ELi([0-9]+)E'
grep -P '\tlq4w$' "$tsv" | cut -f2 | grep -E "$filter" | while read -r name; do
  [[ $name =~ $re ]] || continue
  nc=${BASH_REMATCH[1]}; f=${BASH_REMATCH[2]}
  HIPCC_COMPILE_FLAGS_APPEND="-DLQ_PAIR_N=$nc -DLQ_PAIR_F=$f" \
    python3 -I rewrite/harness/compare.py rewrite/specs/lq4w_pair_c${nc}.json --kernel "$name" \
      --seeds 1,2,3 --sizes 1000,2800 --table "${LQ_TABLE:-$HOME/re-halogen/rewrite/tables/g01.md}" 2>&1 | grep -v '^MATCH' \
      | sed "s|^|[pair $nc $f] |"
  echo "done pair $nc $f"
done
