#!/usr/bin/env bash
# Offline self-test of kargs_shim.so: a fake "flash_serve" calls hipLaunchKernel in a fake libfakehip.so
# with a scan3x handle slot. No GPU, no engine, no podman. Run after build_shim.sh.
# usage (on ai-core): bash ~/re-halogen/scripts/capture/selftest_shim.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
shim="$repo/work/capture/kargs_shim.so"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp "$repo/work/capture/kargs_table.h" "$repo/work/capture/kernel_names.h" "$tmp/"

cat > "$tmp/fakehip.c" <<'EOF'
#include <string.h>
typedef struct { unsigned x, y, z; } dim3_t;
int hipLaunchKernel(const void *f, dim3_t g, dim3_t b, void **a, unsigned long s, void *st) {
    (void)f; (void)g; (void)b; (void)s; (void)st;
    if (a && a[16]) { unsigned long p0; memcpy(&p0, (char *)a[16] + 16, 8); if (p0) memset((void *)p0, 0x5a, 4096); }
    return 0;
}
int hipMemcpy(void *d, const void *s, unsigned long n, int k) { (void)k; memcpy(d, s, n); return 0; }
int hipMemcpyAsync(void *d, const void *s, unsigned long n, int k, void *st) { (void)st; return hipMemcpy(d, s, n, k); }
int hipDeviceSynchronize(void) { return 0; }
int hipblasLtMatmul(void *a, void *b, const void *c, const void *d, void *e, const void *f, void *g, const void *h,
                    const void *i, void *j, void *k, void *l, const void *m, void *n, unsigned long o, void *p) {
    return (int)(((unsigned long)a + (unsigned long)p + o) == 0x1111 + 3 + 77 ? 0 : 9); /* checks arg order */
}
int hipStreamIsCapturing(void *st, int *c) { (void)st; *c = 0; return 0; }
/* Sticky last error, as in HIP: the real hipMemPtrGetInfo fails on an interior pointer and leaves it set. */
static int last;
int hipGetLastError(void) { int e = last; last = 0; return e; }
int hipPeekAtLastError(void) { return last; }
int hipMemPtrGetInfo(void *p, unsigned long *n) { (void)p; (void)n; last = 1; return 1; }
void fake_set_err(int e) { last = e; }
EOF

cat > "$tmp/flash_serve.c" <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kargs_table.h"
#include "kernel_names.h"
typedef struct { unsigned x, y, z; } dim3_t;
int hipblasLtMatmul(void *, void *, const void *, const void *, void *, const void *, void *, const void *,
                    const void *, void *, void *, void *, const void *, void *, unsigned long, void *);
int hipLaunchKernel(const void *, dim3_t, dim3_t, void **, unsigned long, void *);
int hipMemcpyAsync(void *, const void *, unsigned long, int, void *);
int hipGetLastError(void);
void fake_set_err(int);
char arr[0x400];   /* stands in for flash_serve .bss; GLOBAL_BIAS points the shim here */
int main(void) {
    const struct kentry *e = NULL;
    for (unsigned i = 0; i < KTAB_N; i++) if (KTAB[i].has_tap) { e = &KTAB[i]; break; }
    unsigned long g0 = (unsigned long)malloc(8192), g1 = (unsigned long)malloc(8192), g3 = (unsigned long)malloc(8192);
    memcpy(arr + 0x000, &g0, 8);   /* 18fc638 */
    memcpy(arr + 0x028, &g1, 8);   /* 18fc660 */
    memcpy(arr + 0x180, &g3, 8);   /* 18fc7b8 */
    int mode = 1; memcpy(arr + 0x310, &mode, 4);
    unsigned char *state = malloc(3u << 20), *tapbuf = malloc(3u << 20);
    memset(state, 7, 3u << 20); memset(tapbuf, 0, 3u << 20);
    hipMemcpyAsync(tapbuf, state, 3u << 20, 3, 0);
    unsigned char val[512] = {0}; void *args[32] = {0};
    for (unsigned i = 0; i < e->nargs; i++) args[i] = val + e->aoff[i];
    int slot[3] = {2, -1, -1}; unsigned long p[3] = {(unsigned long)tapbuf, 0, 0};
    memcpy(val + 0x80, slot, 12); memcpy(val + 0x90, p, 24);
    unsigned long n[4] = {0, g0 + 0x40, g1, g3}; int w = 128;
    memcpy(val + 0xa8, n, 32); memcpy(val + 0xc8, &w, 4);
    dim3_t gd = {48, 2, 1}, bk = {128, 1, 1};
    hipLaunchKernel((const void *)e->slot_vma, gd, bk, args, 0, 0);   /* non-PIE: base 0 */
    if (hipGetLastError() != 0) return 4;   /* the shim must not leave its own error for the engine */
    fake_set_err(7);
    hipLaunchKernel((const void *)e->slot_vma, gd, bk, args, 0, 0);   /* engine error pending */
    if (hipGetLastError() != 7) return 5;   /* the shim must not clear the engine's error */
    hipLaunchKernel((const void *)0x1234, gd, bk, args, 0, 0);        /* unmatched */
    hipLaunchKernel((const void *)KNAMES[5].slot_vma, gd, bk, args, 0, 0);   /* named, not scan3: trace only */
    if (hipblasLtMatmul((void *)0x1111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, (void *)3) != 0) return 3;
    printf("ok slot=0x%lx tapbuf=%p arr=%p\n", e->slot_vma, (void *)tapbuf, (void *)arr);
    return 0;
}
EOF

cd "$tmp"
gcc -O1 -shared -fPIC -o libfakehip.so fakehip.c
gcc -O1 -no-pie -o flash_serve flash_serve.c -L. -lfakehip -Wl,-rpath,"$tmp"
arr=$(nm flash_serve | awk '$3=="arr"{print "0x"$1}')
bias=$(( arr - 0x18fc638 ))
CAPTURE_OUT="$tmp/pre.jsonl" CAPTURE_ARM_FILE="$tmp/arm" LD_PRELOAD="$shim" ./flash_serve > /dev/null
grep -q '"captured":0,"matched_before_arm":2' "$tmp/pre.jsonl"   # unarmed: nothing captured
! grep -q '"kind":"launch"' "$tmp/pre.jsonl"
touch "$tmp/arm"
CAPTURE_ARM_FILE="$tmp/arm" CAPTURE_OUT="$tmp/out.jsonl" CAPTURE_GLOBAL_BIAS="$bias" CAPTURE_LOG_UNMATCHED=1 CAPTURE_TRACE=1 LD_PRELOAD="$shim" ./flash_serve
cp "$tmp/out.jsonl" "${1:-$repo/work/capture/selftest.jsonl}"
python3 -I - "$tmp/out.jsonl" <<'EOF'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1])]
kinds = [r['kind'] for r in rows]
assert kinds.count('launch') == 2 and 'memcpy' in kinds and 'unmatched' in kinds and 'summary' in kinds, kinds
L, L2 = [r for r in rows if r['kind'] == 'launch']
assert L['shim_err_pre'] == 1 and L['shim_err_post'] == 0, L   # its own ptrinfo error, cleared
assert 'norm' not in L2 and 'shim_err_pre' not in L2, L2      # engine error pending: no HIP calls
assert len(L['kernarg_hex']) == 2 * L['kernarg_size'] == 416, L['kernarg_size']
m = {p['name']: p for p in L['norm']['ptrs']}
assert m['n0']['equal'] == [] and m['n0']['near'] == [], m
assert m['n1']['equal'] == [] and {'global': 'n1_18fc638', 'delta': 64} in m['n1']['near'], m
assert m['n2']['equal'] == ['n2_18fc660'] and m['n3']['equal'] == ['n3_18fc7b8'], m
assert L['globals']['norm_fold_mode'] == 1 and L['globals']['unreadable'] == 0, L['globals']
r = L['tap']['regions'][0]
assert r['slot'] == 2 and r['bytes'] == 3 << 20 and r['words_changed'] == 1024, r
assert r['vs_state'][0]['dst_is_p'] == 1, r
assert r['post_head'] == '5a' * 64, r            # the fake kernel's tap write, read back after the launch
tr = [r for r in rows if r['kind'] in ('trace', 'matmul')]
assert [r['kind'] for r in tr] == ['trace'] * 4 + ['matmul'] and [r['seq'] for r in tr] == [0, 1, 2, 3, 4], tr
assert sum(1 for r in rows if r['kind'] == 'kname') == 2   # the scan3x handle and the KNAMES[5] handle
assert all(a['ns'] <= b['ns'] for a, b in zip(tr, tr[1:]))
print('selftest PASS')
EOF
