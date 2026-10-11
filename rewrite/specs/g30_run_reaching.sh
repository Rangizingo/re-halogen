#!/bin/bash
# g30 reaching runs (integer constant audit, ruling 4c). On Ai-core, from a checkout root (needs rewrite/harness/run_kernel from build.sh):
#   bash rewrite/specs/g30_run_reaching.sh OUTDIR [FILTER_REGEX]
# Reads rewrite/specs/g30_reaching_jobs.tsv: instance<TAB>spec<TAB>compile flags<TAB>sizes<TAB>tag. Flags -DG30=1 is a placeholder (no template flags).
# Four groups, by tag prefix:
#   tail_*  rows N = 1001,1002,1003,1005 (N % 4 and N % 16 != 0): partial last block of the M=2 W=32 final stage (lq4w, lq4w_grp, lq6k, lq8s32, lq8w)
#   thr_*   K = 32*G with G = 1,2,127,128,129,255,256,257,300,511,512,513,600 (24 rows): the vendor F=1 scale-preload thresholds (G <= 128 / 256 / 512)
#   lq4mt_* / lq4m_* / lf32mt_*  existing linear_q4 / linear_f32 specs at K = 1, B-1, B, B+1, UB-1, UB, UB+1, (2U-1)B-1.. 3UB+7 around the unrolled-loop bound (2U-1)B
# Every row must print MATCH; results in rewrite/tables/g30.md.
out=${1:?outdir}; filter=${2:-.}
mkdir -p "$out"
export CMP_TMP=${CMP_TMP:-$HOME/cmp_tmp}
one() {
  IFS=$'\t' read -r name spec flags sizes tag <<< "$1"
  HIPCC_COMPILE_FLAGS_APPEND="$flags" python3 -I rewrite/harness/compare.py "rewrite/specs/$spec" --kernel "$name" \
    --seeds 1 --sizes "$sizes" --table "$out/$tag.md" > "$out/$tag.log" 2>&1
  echo "done $tag rc=$?" >> "$out/_progress.txt"
}
export -f one; export out
grep -E "$filter" rewrite/specs/g30_reaching_jobs.tsv | tr '\n' '\0' | xargs -0 -n1 -P 2 bash -c 'one "$0"'
