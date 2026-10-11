#!/bin/bash
# Orchestrator helper (p7-12). usage: rewrite/collect.sh <group> <agent-worktree>  : copy rewrite files from an agent worktree into the main checkout, tables from Ai-core
set -euo pipefail
g=$1; wt=$2/rewrite; repo=/Users/peterb/Documents/AI/re-halogen
cd "$repo"
for d in kernels specs; do
  for f in "$wt/$d"/*; do b=$(basename "$f"); case $b in scale.hip|k_scale_silu*) continue;; esac; cp -R "$f" rewrite/$d/; done
done
cp "$wt/reports/$g.md" rewrite/reports/
for t in $(ssh ai-core "cd ~/re-halogen/rewrite/tables && ls $g.md $g-*.md 2>/dev/null | grep -v -e scratch -e dev"); do scp -q "ai-core:re-halogen/rewrite/tables/$t" rewrite/tables/; done
# Mac vs Ai-core sha1 of every kernel the group wrote
ks=$(cd "$wt" && ls kernels | grep -v "^scale.hip$" | tr "\n" " ")
(cd rewrite && for k in $ks; do shasum kernels/$k; done) > /tmp/c.sha
ssh ai-core "cd ~/re-halogen/rewrite && for k in $ks; do shasum kernels/\$k; done" | diff - /tmp/c.sha && echo "sha SAME ($(echo $ks | wc -w) kernels)"
echo "MATCH instances: $(awk -F'|' '$5 ~ /^ *MATCH/ {print $2}' rewrite/tables/$g.md | sort -u | wc -l)  MISMATCH rows: $(grep -c MISMATCH rewrite/tables/$g.md || true)  negctl MISMATCH rows: $(grep -c MISMATCH rewrite/tables/$g-negctl.md 2>/dev/null || echo 0)"
# Every spec a MATCH row names must be committed in rewrite/specs, or the row cannot be reproduced.
miss=$(awk -F'|' '$5 ~ /^ *MATCH/ {split($4,a," "); print a[1]}' rewrite/tables/$g.md | sort -u | while read -r s; do [ -f "rewrite/specs/$s" ] || echo "$s"; done)
[ -z "$miss" ] && echo "specs: all present" || echo "specs MISSING from rewrite/specs: $miss"
