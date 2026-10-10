#!/bin/bash
# Run halogen-tools verify + inspect on every crafted .hgn in work/hgn/files,
# record verdicts to tmp/verify_matrix.tsv (phase 3 acceptance matrix).
cd /home/rangizingo/re-halogen
OUT=tmp/verify_matrix.tsv
: > "$OUT"
for f in work/hgn/files/*.hgn; do
  b=$(basename "$f")
  VOUT=$(timeout 60 ./binaries/halogen-tools verify "$f" </dev/null 2>&1)
  VEX=$?
  IOUT=$(timeout 60 ./binaries/halogen-tools inspect "$f" --json --no-hash </dev/null 2>/dev/null | head -c 400)
  IEX=$?
  printf '%s\t%s\t%s\t%s\t%s\n' "$b" "$VEX" "$(echo "$VOUT" | head -1)" "$IEX" "$(echo "$IOUT" | head -c 200)" >> "$OUT"
done
echo "done $(wc -l < "$OUT")"
