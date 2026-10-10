#!/usr/bin/env bash
# Build the kernarg capture shim on Ai-core into ~/re-halogen/work/capture/kargs_shim.so.
# usage (on ai-core): bash ~/re-halogen/scripts/capture/build_shim.sh  (paths follow the script location)
# Regenerates kargs_table.h when analysis/function-map-kernel-stubs.tsv and work/fatbin/*.hsaco are present,
# otherwise uses the checked-in scripts/capture/kargs_table.h. Needs gcc, objdump, llvm-readelf, python3.
# Does not touch the engine, podman or halogen-swift.service.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
src="$repo/scripts/capture"
out="$repo/work/capture"
container_glibc=2.41   # libc.so.6 of ghcr.io/peonist-ai/halogen-flash-server:0.17.2 (Debian 13)
mkdir -p "$out"

tsv="$repo/analysis/function-map-kernel-stubs.tsv"
if [ -f "$tsv" ] && ls "$repo"/work/fatbin/*.hsaco >/dev/null 2>&1; then
  objdump -R "$repo/binaries/flash_serve" > "$out/rela.txt"
  notes=()
  for f in "$repo"/work/fatbin/*.hsaco; do
    llvm-readelf --notes "$f" > "$out/notes-$(basename "$f").txt"
    notes+=("$out/notes-$(basename "$f").txt")
  done
  python3 -I "$src/gen_kargs_table.py" "$out/rela.txt" "$tsv" "$out/kargs_table.h" "${notes[@]}"
else
  cp "$src/kargs_table.h" "$src/kernel_names.h" "$out/"
  echo "using checked-in kargs_table.h (analysis tsv or hsaco not present)" >&2
fi
cp "$src/kargs_shim.c" "$out/kargs_shim.c"

gcc -O2 -g0 -Wall -Wextra -Werror -shared -fPIC -fno-plt -o "$out/kargs_shim.so" "$out/kargs_shim.c" -ldl -lpthread
echo "built $out/kargs_shim.so"
nm -D "$out/kargs_shim.so"

# The container's glibc is older than this host's: refuse a .so that needs a newer symbol version.
need=$(objdump -T "$out/kargs_shim.so" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -uV | tail -1)
echo "highest GLIBC symbol version needed: ${need:-none} (container has $container_glibc)"
top=$(printf '%s\n%s\n' "${need:-0}" "$container_glibc" | sort -V | tail -1)
[ "$top" = "$container_glibc" ] || { echo "shim needs glibc $need > $container_glibc" >&2; exit 1; }
