#!/usr/bin/env bash
# Fetch the HT decoder test slices by HTTP range reads (about 491 MB, mostly the w4b expert reference).
# usage: bash scripts/fetch_ht_testdata.sh <out_dir>   then: python3 -I scripts/ht_decode.py <out_dir>
# Offsets come from the .hgn tables of qwen38-flash-next-ht43.hgn and qwen38-flash-next-w4b.hgn
# (peonist-ai/halogen-qwen3.8-flash-next, read 2026-10-10). If the vendor republishes, re-read the tables.
set -euo pipefail
out="${1:?usage: fetch_ht_testdata.sh <out_dir>}"
mkdir -p "$out"
base=https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next/resolve/main
ht=qwen38-flash-next-ht43.hgn
w4b=qwen38-flash-next-w4b.hgn

get() { # name offset length file
  curl -sSfL -r "$2-$(($2 + $3 - 1))" -o "$out/$1" "$base/$4"
  local got
  got=$(wc -c < "$out/$1" | tr -d ' ')
  [ "$got" = "$3" ] || { echo "fetch_ht_testdata: $1 is $got bytes, expected $3" >&2; exit 1; }
}

get k3_weight.ht            4050329344    655360 "$ht"
get k3_suh.f16              4050984704      5120 "$ht"
get k3_svh.f16              4050989824      1024 "$ht"
get k3_weight_w4b.q4c      56043991552    737344 "$w4b"
get sh3_down.ht             5033252608    819200 "$ht"
get sh3_down_suh.f16        5034071808      1280 "$ht"
get sh3_down_svh.f16        5034073088      5120 "$ht"
get sh3_down_w4b.q4c       57504546880    942144 "$w4b"
get e1_down_ht_x0x1.ht      1777909632   1228800 "$ht"
get e1_down_suh_x0x1.f16    2092482432      2560 "$ht"
get e1_down_svh_x0x1.f16    2093137792     10240 "$ht"
get e1_down_w4b_full.q4c   53135904704 482345024 "$w4b"
get g0_gu_x0.ht               823782080   1638400 "$ht"
get g0_su.f16               1662642880      5120 "$ht"
get g0_gu_svh_x0.f16        1662648000      2560 "$ht"
get g0_gu_w4b_cb.bin         880484992        64 "$w4b"
get g0_gu_w4b_nib_x0.bin     880485056   1638400 "$w4b"
get g0_gu_w4b_sc_x0.f16    1719345856    204800 "$w4b"
ls -l "$out"
