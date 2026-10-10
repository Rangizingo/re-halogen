#!/bin/bash
# Pull flash_serve ELF from ghcr.io/peonist-ai/halogen-flash-server tags into
# work/versions/<tag>/flash_serve. Anonymous ghcr token. Logs to work/pull.status.
set -u
cd /home/rangizingo/re-halogen
S=work/pull.status
echo "start $(date -u +%FT%TZ)" > "$S"
RESP=$(curl -s "https://ghcr.io/token?scope=repository:peonist-ai/halogen-flash-server:pull")
TOKEN=*** "$RESP" | sed -n 's/.*"access_token":"*\([^"]*\)".*/\1/p')
echo "token_len=${#TOKEN}" >> "$S"
for TAG in 0.17.2 0.17.0 0.16.4 0.15.0; do
  OUT=work/versions/$TAG
  if [ -s "$OUT/flash_serve" ]; then echo "$TAG cached" >> "$S"; continue; fi
  mkdir -p "$OUT"
  MAN=$(curl -s -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json" "https://ghcr.io/v2/peonist-ai/halogen-flash-server/manifests/$TAG")
  DIG=$(echo "$MAN" | sed -n '0,/linux\/amd64/s/.*"digest":"\(sha256:[a-f0-9]*\)".*/\1/p')
  if [ -z "$DIG" ]; then echo "$TAG NO-AMD64-MANIFEST" >> "$S"; continue; fi
  MAN=$(curl -s -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json" "https://ghcr.io/v2/peonist-ai/halogen-flash-server/manifests/$DIG")
  echo "$TAG amd64=$DIG" >> "$S"
  echo "$MAN" > "$OUT/manifest.json"
  FOUND=0
  for LAYER in $(echo "$MAN" | grep -o '"digest":"sha256:[a-f0-9]*"' | sed 's/"digest":"//;s/"//'); do
    curl -sL -H "Authorization: Bearer $TOKEN" -o "$OUT/layer.tar.zst" "https://ghcr.io/v2/peonist-ai/halogen-flash-server/blobs/$LAYER"
    SZ=$(stat -c %s "$OUT/layer.tar.zst" 2>/dev/null || echo 0)
    if [ "$SZ" -lt 1000000 ]; then rm -f "$OUT/layer.tar.zst"; continue; fi
    if tar --zstd -tf "$OUT/layer.tar.zst" 2>/dev/null | grep -q 'flash_serve$'; then
      tar --zstd -xf "$OUT/layer.tar.zst" --wildcards '*/flash_serve' -O > "$OUT/flash_serve" 2>/dev/null
      echo "$TAG found layer $LAYER size=$SZ extracted=$(stat -c %s "$OUT/flash_serve")" >> "$S"
      FOUND=1; rm -f "$OUT/layer.tar.zst"; break
    fi
    rm -f "$OUT/layer.tar.zst"
  done
  [ "$FOUND" = 0 ] && echo "$TAG flash_serve NOT FOUND in layers" >> "$S"
done
echo "done $(date -u +%FT%TZ)" >> "$S"
