#!/usr/bin/env bash
# p7-11 runtime run 5 on Ai-core (U12): Tap capture of k_dnc_scan3x launches with SNAP fields (probe 7) in flash_serve 0.17.3 engine mode.
# Same window as engine_run_r3.sh: stops the live halogen-swift service (Hermes is down while this runs) and always restores it.
# The LD_PRELOAD shim (scripts/capture/kargs_shim.c, built by scripts/capture/build_shim.sh) logs to <out>/kargs.jsonl.
# Never mounts /srv/halogen-cache; own cache dir work/testcache-r5.
# usage (on ai-core): bash ~/re-halogen/scripts/engine_run_r5.sh <out_dir>
set -euo pipefail
out="${1:?usage: engine_run_r5.sh <out_dir>}"
repo="$HOME/re-halogen"
name=re-halogen-r5
image=ghcr.io/peonist-ai/halogen-flash-server:0.17.2
cache="$repo/work/testcache-r5"
envf="$repo/work/test-r5.env"
shim="$repo/work/capture/kargs_shim.so"
case "$cache" in /srv/halogen-cache*) echo "refusing live cache dir" >&2; exit 2;; esac
[ -f "$shim" ] || { echo "missing $shim: run scripts/capture/build_shim.sh first" >&2; exit 2; }
# One out dir per attempt: window.txt starts with a plain tee, so a reused dir loses the earlier window.
[ -e "$out" ] && { echo "out dir exists: $out" >&2; exit 2; }
mkdir -p "$out" "$cache"
out="$(cd "$out" && pwd)"

# Own env copy: never prune, own /cache mount, version label 0.17.3, NormFold + Tap paths on, shim loaded.
grep -v -E '^(HALOGEN_CACHE_PRUNE_OLD|HALOGEN_IMAGE_VERSION|HALOGEN_DN_NORM_FOLD|HALOGEN_TAP_FROM_SCAN|LD_PRELOAD|CAPTURE_[A-Z_]+)=' \
  "$HOME/.config/halogen/halogen-swift.env" > "$envf"
cat >> "$envf" <<'EOF'
HALOGEN_CACHE_PRUNE_OLD=0
HALOGEN_IMAGE_VERSION=0.17.3
HALOGEN_DN_NORM_FOLD=1
HALOGEN_TAP_FROM_SCAN=1
LD_PRELOAD=/capture/kargs_shim.so
CAPTURE_OUT=/capout/kargs.jsonl
CAPTURE_MAX=400
CAPTURE_PTR_BYTES=192
CAPTURE_LOG_UNMATCHED=1
CAPTURE_ARM_FILE=/capout/arm
CAPTURE_TRACE=1
CAPTURE_TRACE_MAX=200000
EOF
grep -q '^HALOGEN_CACHE_PRUNE_OLD=0$' "$envf"
sha256sum "$repo/binaries/flash_serve" "$shim" | tee "$out/binary.sha256"

cache_files() {  # every file in the test cache with size; e<N>.rec heads (synthetic token ids only) as hex
  podman unshare find "$cache" -type f -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' | sort -k3 > "$out/cache-files-$1.txt" || true
  podman unshare find "$cache" -type f -name 'e*.rec' | sort | head -8 | while read -r f; do
    echo "== $f"; podman unshare head -c 256 "$f" | od -A x -t x1z -v
  done > "$out/cache-rec-heads-$1.txt" 2>&1 || true
}

restore() {
  cache_files before-stop
  # Graceful stop (SIGTERM, 120 s grace) so the engine can flush its prompt cache to disk (cache-sse-r5 residual).
  echo "stop: podman stop -t 120 at $(date -u +%FT%TZ)" | tee -a "$out/window.txt"
  podman stop -t 120 "$name" >/dev/null 2>&1 || true
  echo "stop: exited at $(date -u +%FT%TZ), state $(podman inspect -f '{{.State.ExitCode}}' "$name" 2>/dev/null)" | tee -a "$out/window.txt"
  cache_files after-stop
  podman logs "$name" > "$out/engine-0.17.3-r5.log" 2>&1 || true
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

podman run -d --name "$name" --ulimit memlock=-1:-1 --group-add keep-groups --ipc=host \
  --device /dev/kfd --device /dev/dri \
  -v /srv/models/halogen-swift:/models:ro \
  -v /srv/models/ukisai_Swift-1.5-Qwen3.8-Flash-Next-GGUF:/srv/models/ukisai_Swift-1.5-Qwen3.8-Flash-Next-GGUF:ro \
  -v "$cache":/cache \
  -v "$repo/binaries/flash_serve":/usr/local/bin/flash_serve:ro \
  -v "$shim":/capture/kargs_shim.so:ro \
  -v "$out":/capout \
  -v "$repo/scripts":/probe:ro \
  --env-file "$envf" "$image" engine

for _ in $(seq 1 120); do
  podman logs "$name" 2>&1 | grep -q -i 'listening' && break
  sleep 5
done
podman logs "$name" 2>&1 | grep -i 'listening' | head -1 | tee -a "$out/window.txt"
# Arm the shim only now so engine warm-up launches do not use up CAPTURE_MAX.
touch "$out/arm"
podman exec "$name" python3 /probe/engine_probe7.py /tmp/probe7.jsonl > "$out/probe7-results.txt" 2>&1 || echo "probe exit $?" >> "$out/probe7-results.txt"
podman cp "$name:/tmp/probe7.jsonl" "$out/probe7-0.17.3.jsonl"
echo "capture lines: $(wc -l < "$out/kargs.jsonl" 2>/dev/null || echo 0)" | tee -a "$out/window.txt"
python3 -I "$repo/scripts/capture/analyse_kargs.py" "$out/kargs.jsonl" > "$out/kargs-table.txt" 2>&1 || true
python3 -I "$repo/scripts/capture/analyse_kargs.py" "$out/kargs.jsonl" --trace --gap 1.5 > "$out/kargs-trace.txt" 2>&1 || true
