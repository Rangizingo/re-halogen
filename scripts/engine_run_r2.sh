#!/usr/bin/env bash
# p7-7 runtime run on Ai-core: our 0.17.3 flash_serve in the 0.17.2 image, mode engine, own cache dir.
# Stops the live halogen-swift service (Hermes is down while this runs) and always restores it.
# usage (on ai-core): bash ~/re-halogen/scripts/engine_run_r2.sh <out_dir>
set -euo pipefail
out="${1:?usage: engine_run_r2.sh <out_dir>}"
repo="$HOME/re-halogen"
name=re-halogen-r2
image=ghcr.io/peonist-ai/halogen-flash-server:0.17.2
cache="$repo/work/testcache-r2"
envf="$repo/work/test-r2.env"
mkdir -p "$out" "$cache"

# Own env copy: never prune, own /cache mount, version label 0.17.3.
grep -v -E '^(HALOGEN_CACHE_PRUNE_OLD|HALOGEN_IMAGE_VERSION)=' "$HOME/.config/halogen/halogen-swift.env" > "$envf"
printf 'HALOGEN_CACHE_PRUNE_OLD=0\nHALOGEN_IMAGE_VERSION=0.17.3\n' >> "$envf"
grep -q '^HALOGEN_CACHE_PRUNE_OLD=0$' "$envf"
sha256sum "$repo/binaries/flash_serve" | tee "$out/binary.sha256"

restore() {
  podman logs "$name" > "$out/engine-0.17.3-r2.log" 2>&1 || true
  podman rm -f -i "$name" >/dev/null 2>&1 || true
  echo "restore: starting halogen-swift.service at $(date -u +%FT%TZ)" | tee -a "$out/window.txt"
  systemctl --user start halogen-swift.service
  for _ in $(seq 1 120); do
    if curl -sf -m 5 127.0.0.1:8307/health > "$out/health-after.json"; then
      echo "restore: /health 200 at $(date -u +%FT%TZ)" | tee -a "$out/window.txt"
      return 0
    fi
    sleep 5
  done
  echo "restore: /health NOT ok after 600 s" | tee -a "$out/window.txt"
  return 1
}

curl -sf -m 5 127.0.0.1:8307/health > "$out/health-before.json"
echo "window: stopping halogen-swift.service at $(date -u +%FT%TZ)" | tee "$out/window.txt"
systemctl --user stop halogen-swift.service
trap restore EXIT

podman run -d --name "$name" --rm --ulimit memlock=-1:-1 --group-add keep-groups --ipc=host \
  --device /dev/kfd --device /dev/dri \
  -v /srv/models/halogen-swift:/models:ro \
  -v /srv/models/ukisai_Swift-1.5-Qwen3.8-Flash-Next-GGUF:/srv/models/ukisai_Swift-1.5-Qwen3.8-Flash-Next-GGUF:ro \
  -v "$cache":/cache \
  -v "$repo/binaries/flash_serve":/usr/local/bin/flash_serve:ro \
  -v "$repo/scripts":/probe:ro \
  --env-file "$envf" "$image" engine

for _ in $(seq 1 120); do
  podman logs "$name" 2>&1 | grep -q -i 'listening' && break
  sleep 5
done
podman logs "$name" 2>&1 | grep -i 'listening' | head -1 | tee -a "$out/window.txt"
podman exec "$name" python3 /probe/engine_probe4.py /tmp/probe4.jsonl > "$out/probe4-results.txt" 2>&1 || echo "probe exit $?" >> "$out/probe4-results.txt"
podman cp "$name:/tmp/probe4.jsonl" "$out/probe4-0.17.3.jsonl"
