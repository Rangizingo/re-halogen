/*
 * kargs_shim.c: LD_PRELOAD kernel-argument capture for flash_serve 0.17.3 (ledger U12).
 *
 * Interposes hipLaunchKernel and hipMemcpyAsync. A launch whose func handle is a registration
 * slot (or stub address) of a k_dnc_scan3* kernel (kargs_table.h) is logged as one JSON line to
 * $CAPTURE_OUT: kernarg bytes, NormFold pointer matches against host globals, Tap region hashes.
 * Arguments are forwarded untouched. Diagnostic only: it calls hipDeviceSynchronize around each
 * captured launch.
 *
 * Env: CAPTURE_OUT (required), CAPTURE_MAX (64), CAPTURE_TAP_BYTES (3 MiB), CAPTURE_PTR_BYTES (4096),
 *      CAPTURE_MEMCPY_SIZE (0x300000, layer state copies to remember), CAPTURE_LOG_UNMATCHED (0),
 *      CAPTURE_GLOBAL_BIAS (0, signed offset added to the global VMAs),
 *      CAPTURE_ARM_FILE (capture starts once this file exists; skips engine warm-up launches).
 * Only active when the process is named flash_serve and the first matching launch opens the file.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <link.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "kargs_table.h"
#include "kernel_names.h"

typedef struct { unsigned x, y, z; } dim3_t; /* same layout and SysV class as HIP's dim3 */
typedef int (*launch_fn)(const void *, dim3_t, dim3_t, void **, size_t, void *);
typedef int (*memcpy_async_fn)(void *, const void *, size_t, int, void *);
typedef int (*memcpy_fn)(void *, const void *, size_t, int);
typedef int (*sync_fn)(void);
typedef int (*iscap_fn)(void *, int *);
typedef int (*ptrinfo_fn)(void *, size_t *);
typedef int (*lt_matmul_fn)(void *, void *, const void *, const void *, void *, const void *, void *, const void *,
                            const void *, void *, void *, void *, const void *, void *, size_t, void *);

#define HIP_MEMCPY_DEFAULT 4
#define NGLOB 4
#define MAXREC 8

/* flash_serve globals (file VMA; Ghidra name is VMA + 0x100000, e.g. DAT_019fc638). */
static const struct { unsigned long vma; const char *name; } GLOB[NGLOB] = {
    {0x18fc638UL, "n1_18fc638"}, {0x18fc660UL, "n2_18fc660"},
    {0x18fc958UL, "n2buf_18fc958"}, {0x18fc7b8UL, "n3_18fc7b8"}};
#define MODE_VMA 0x18fc948UL /* HALOGEN_DN_NORM_FOLD mode (int) */

static launch_fn real_launch;
static memcpy_async_fn real_memcpy_async;
static memcpy_fn real_memcpy;
static sync_fn real_sync;
static iscap_fn real_iscap;
static ptrinfo_fn real_ptrinfo;
typedef int (*lasterr_fn)(void);
/* hipGetLastError is sticky per thread: every shim HIP call must leave it as the engine left it. */
static lasterr_fn real_getlast, real_peek;
static lt_matmul_fn real_lt_matmul;
static int g_trace; /* CAPTURE_TRACE=1: log every launch and hipblasLtMatmul in order */
static unsigned long g_tmax = 200000, g_seq, g_tlines;
static unsigned char g_seen[KNAMES_N + 1]; /* handle names already written */

static int g_enabled, g_logunm, g_armed;
static const char *g_armfile; /* CAPTURE_ARM_FILE: capture only after this file exists (skips engine warm-up) */
static unsigned long g_prearm;
static unsigned long g_base, g_max = 64, g_cnt, g_total, g_nmem;
static long g_bias;
static size_t g_tapcap = 3u << 20, g_ptrcap = 4096, g_memsz = 0x300000;
static const char *g_path;
static FILE *g_out;
static pthread_mutex_t g_mu = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
static __thread int g_in; /* set while capturing: skip our own memcpy traffic */
static unsigned long g_unm[64];
static int g_nunm;
static struct { const void *dst, *src; size_t n; int kind; unsigned long seq; } g_rec[MAXREC];
static unsigned long g_recseq;
static unsigned char *g_buf[3];

/* ---- small string builder ---- */
typedef struct { char *p; size_t n, cap; } sb_t;

static void sb_add(sb_t *s, const char *fmt, ...) {
    va_list ap;
    for (;;) {
        va_start(ap, fmt);
        int k = vsnprintf(s->p ? s->p + s->n : NULL, s->cap - s->n, fmt, ap);
        va_end(ap);
        if (k < 0) return;
        if ((size_t)k < s->cap - s->n) { s->n += (size_t)k; return; }
        s->cap = s->cap ? s->cap * 2 + (size_t)k : 4096 + (size_t)k;
        s->p = realloc(s->p, s->cap);
        if (!s->p) abort();
    }
}

static void sb_hex(sb_t *s, const unsigned char *b, size_t n) {
    for (size_t i = 0; i < n; i++) sb_add(s, "%02x", b[i]);
}

static uint64_t fnv(const unsigned char *b, size_t n) {
    uint64_t h = 1469598103934665603ULL, w;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) { memcpy(&w, b + i, 8); h = (h ^ w) * 1099511628211ULL; }
    for (; i < n; i++) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}

/* ---- init ---- */
static int phdr_cb(struct dl_phdr_info *i, size_t sz, void *d) {
    (void)sz;
    *(unsigned long *)d = (unsigned long)i->dlpi_addr; /* first entry is the main program */
    return 1;
}

static void *resolve(const char *name) {
    void *p = dlsym(RTLD_NEXT, name);
    if (!p) {
        void *h = dlopen("libamdhip64.so.7", RTLD_LAZY | RTLD_NOLOAD);
        if (h) p = dlsym(h, name);
    }
    return p;
}

__attribute__((constructor)) static void shim_init(void) {
    char exe[512];
    ssize_t k = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (k <= 0) return;
    exe[k] = 0;
    const char *b = strrchr(exe, '/');
    b = b ? b + 1 : exe;
    if (strcmp(b, "flash_serve") != 0) return;
    g_path = getenv("CAPTURE_OUT");
    if (!g_path || !*g_path) return;
    const char *e;
    if ((e = getenv("CAPTURE_MAX"))) g_max = strtoul(e, NULL, 0);
    if ((e = getenv("CAPTURE_TAP_BYTES"))) g_tapcap = strtoul(e, NULL, 0);
    if ((e = getenv("CAPTURE_PTR_BYTES"))) g_ptrcap = strtoul(e, NULL, 0);
    if ((e = getenv("CAPTURE_MEMCPY_SIZE"))) g_memsz = strtoul(e, NULL, 0);
    if ((e = getenv("CAPTURE_GLOBAL_BIAS"))) g_bias = strtol(e, NULL, 0);
    if ((e = getenv("CAPTURE_LOG_UNMATCHED"))) g_logunm = atoi(e);
    if ((e = getenv("CAPTURE_ARM_FILE")) && *e) g_armfile = e;
    if ((e = getenv("CAPTURE_TRACE"))) g_trace = atoi(e);
    if ((e = getenv("CAPTURE_TRACE_MAX"))) g_tmax = strtoul(e, NULL, 0);
    g_armed = g_armfile == NULL;
    dl_iterate_phdr(phdr_cb, &g_base);
    g_enabled = 1;
}

static void resolve_all(void) {
    if (real_launch) return;
    real_launch = (launch_fn)resolve("hipLaunchKernel");
    real_memcpy_async = (memcpy_async_fn)resolve("hipMemcpyAsync");
    real_memcpy = (memcpy_fn)resolve("hipMemcpy");
    real_sync = (sync_fn)resolve("hipDeviceSynchronize");
    real_iscap = (iscap_fn)resolve("hipStreamIsCapturing");
    real_ptrinfo = (ptrinfo_fn)resolve("hipMemPtrGetInfo");
    real_getlast = (lasterr_fn)resolve("hipGetLastError");
    real_peek = (lasterr_fn)resolve("hipPeekAtLastError");
    real_lt_matmul = (lt_matmul_fn)resolve("hipblasLtMatmul");
}

static void emit(sb_t *s) {
    if (!g_out) {
        g_out = fopen(g_path, "a");
        if (!g_out) return;
    }
    fputs(s->p, g_out);
    fputc('\n', g_out);
    fflush(g_out);
}

static void jhead(sb_t *s, const char *kind) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    sb_add(s, "{\"kind\":\"%s\",\"pid\":%d,\"t\":%ld.%06ld", kind, (int)getpid(), (long)ts.tv_sec,
           ts.tv_nsec / 1000);
}

/* ---- device reads ---- */
static int d2h(void *dst, const void *src, size_t n) {
    return real_memcpy ? real_memcpy(dst, src, n, HIP_MEMCPY_DEFAULT) : -1;
}

/* Bytes safe to read at p: allocation size when the runtime knows it, capped. */
static size_t region_size(const void *p, size_t cap, size_t *alloc) {
    size_t sz = 0;
    *alloc = 0;
    if (real_ptrinfo && real_ptrinfo((void *)p, &sz) == 0 && sz) {
        *alloc = sz;
        return sz < cap ? sz : cap;
    }
    return cap;
}

static unsigned char *buf(int i) {
    if (!g_buf[i]) g_buf[i] = malloc(g_tapcap);
    return g_buf[i];
}

static size_t diff_words(const unsigned char *a, const unsigned char *b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i + 4 <= n; i += 4)
        if (memcmp(a + i, b + i, 4)) d++;
    return d;
}

/* Read up to want bytes; on failure retry 4096, then give up. Returns bytes read or 0. */
static size_t read_dev(unsigned char *dst, const void *p, size_t want) {
    if (d2h(dst, p, want) == 0) return want;
    if (want > 4096 && d2h(dst, p, 4096) == 0) return 4096;
    return 0;
}

/* True once capture is allowed: no arm file configured, or the file exists. Called with g_mu held. */
static int arm_check(void) {
    if (g_armed) return 1;
    if (access(g_armfile, F_OK) != 0) return 0;
    g_armed = 1;
    sb_t s = {0};
    jhead(&s, "armed");
    sb_add(&s, ",\"matched_launches_before_arm\":%lu}", g_prearm);
    emit(&s);
    free(s.p);
    return 1;
}

/* ---- capture ---- */
static const struct kentry *match(const void *func) {
    unsigned long f = (unsigned long)func;
    for (unsigned i = 0; i < KTAB_N; i++)
        if (f == g_base + KTAB[i].slot_vma || f == g_base + KTAB[i].stub_vma) return &KTAB[i];
    return NULL;
}

static void log_unmatched(const void *func) {
    unsigned long off = (unsigned long)func - g_base;
    for (int i = 0; i < g_nunm; i++)
        if (g_unm[i] == off) return;
    if (g_nunm >= 64) return;
    g_unm[g_nunm++] = off;
    sb_t s = {0};
    jhead(&s, "unmatched");
    sb_add(&s, ",\"func_minus_base\":\"0x%lx\"}", off);
    emit(&s);
    free(s.p);
}

/* Read host memory without faulting: write() to a pipe returns EFAULT on a bad address. */
static int safe_read(void *dst, unsigned long addr, size_t n) {
    int pf[2];
    if (pipe(pf) != 0) return -1;
    int ok = write(pf[1], (void *)addr, n) == (ssize_t)n && read(pf[0], dst, n) == (ssize_t)n;
    close(pf[0]);
    close(pf[1]);
    return ok ? 0 : -1;
}

static void json_globals(sb_t *s, unsigned long gv[NGLOB]) {
    int bad = 0, mode = -1;
    sb_add(s, ",\"globals\":{");
    for (int i = 0; i < NGLOB; i++) {
        if (safe_read(&gv[i], g_base + GLOB[i].vma + (unsigned long)g_bias, 8)) { gv[i] = 0; bad++; }
        sb_add(s, "%s\"%s\":\"0x%lx\"", i ? "," : "", GLOB[i].name, gv[i]);
    }
    if (safe_read(&mode, g_base + MODE_VMA + (unsigned long)g_bias, 4)) bad++;
    sb_add(s, ",\"norm_fold_mode\":%d,\"unreadable\":%d}", mode, bad);
}

static void json_norm(sb_t *s, const unsigned char *k, const unsigned long gv[NGLOB]) {
    unsigned long n[4];
    int w;
    memcpy(n, k + 0xa8, 32);
    memcpy(&w, k + 0xc8, 4);
    sb_add(s, ",\"norm\":{\"w_i32\":%d,\"ptrs\":[", w);
    for (int i = 0; i < 4; i++) {
        sb_add(s, "%s{\"name\":\"n%d\",\"ptr\":\"0x%lx\",\"equal\":[", i ? "," : "", i, n[i]);
        int first = 1;
        for (int g = 0; g < NGLOB; g++)
            if (n[i] && gv[g] == n[i]) {
                sb_add(s, "%s\"%s\"", first ? "" : ",", GLOB[g].name);
                first = 0;
            }
        sb_add(s, "],\"near\":["); /* nonzero offset from a global's value, within 64 MiB */
        first = 1;
        for (int g = 0; g < NGLOB; g++) {
            long d = (long)(n[i] - gv[g]);
            if (n[i] && gv[g] && d != 0 && d > -(64L << 20) && d < (64L << 20)) {
                sb_add(s, "%s{\"global\":\"%s\",\"delta\":%ld}", first ? "" : ",", GLOB[g].name, d);
                first = 0;
            }
        }
        sb_add(s, "]");
        if (n[i]) {
            unsigned char *b = buf(0);
            size_t want = g_ptrcap < g_tapcap ? g_ptrcap : g_tapcap, got = b ? read_dev(b, (void *)n[i], want) : 0;
            sb_add(s, ",\"read\":%zu", got);
            if (got) {
                sb_add(s, ",\"hash\":\"%016llx\",\"head\":\"", (unsigned long long)fnv(b, got));
                sb_hex(s, b, got < 64 ? got : 64);
                sb_add(s, "\"");
            }
        }
        sb_add(s, "}");
    }
    sb_add(s, "]}");
}

typedef struct { unsigned char *pre; size_t n, alloc; int ok; } tap_t;

static void tap_pre(tap_t t[3], const unsigned char *k) {
    unsigned long p[3];
    memcpy(p, k + 0x90, 24);
    for (int j = 0; j < 3; j++) {
        memset(&t[j], 0, sizeof t[j]);
        if (!p[j]) continue;
        t[j].n = region_size((void *)p[j], g_tapcap, &t[j].alloc);
        t[j].pre = malloc(t[j].n);
        t[j].ok = t[j].pre && read_dev(t[j].pre, (void *)p[j], t[j].n) == t[j].n;
    }
}

static void json_tap(sb_t *s, const unsigned char *k, tap_t t[3]) {
    int slot[3];
    unsigned long p[3];
    memcpy(slot, k + 0x80, 12);
    memcpy(p, k + 0x90, 24);
    sb_add(s, ",\"tap\":{\"slots\":[%d,%d,%d],\"p\":[\"0x%lx\",\"0x%lx\",\"0x%lx\"],\"regions\":[", slot[0], slot[1],
           slot[2], p[0], p[1], p[2]);
    int first = 1;
    for (int j = 0; j < 3; j++) {
        if (!p[j]) continue;
        sb_add(s, "%s{\"k\":%d,\"slot\":%d,\"p\":\"0x%lx\",\"alloc\":%zu,\"bytes\":%zu", first ? "" : ",", j,
               slot[j], p[j], t[j].alloc, t[j].n);
        first = 0;
        if (t[j].ok) {
            unsigned char *post_b = buf(1);
            size_t got = post_b ? read_dev(post_b, (void *)p[j], t[j].n) : 0;
            sb_add(s, ",\"pre_hash\":\"%016llx\"", (unsigned long long)fnv(t[j].pre, t[j].n));
            if (got == t[j].n) {
                sb_add(s, ",\"post_hash\":\"%016llx\",\"words_changed\":%zu",
                       (unsigned long long)fnv(post_b, got), diff_words(t[j].pre, post_b, got));
                sb_add(s, ",\"post_head\":\"");
                sb_hex(s, post_b, got < 64 ? got : 64);
                sb_add(s, "\"");
                /* compare with each remembered layer-state copy */
                sb_add(s, ",\"vs_state\":[");
                int f2 = 1;
                for (int r = 0; r < MAXREC; r++) {
                    if (!g_rec[r].n) continue;
                    unsigned char *sb2 = buf(2);
                    size_t sg = (sb2 && g_rec[r].n <= g_tapcap) ? read_dev(sb2, g_rec[r].src, g_rec[r].n) : 0;
                    sb_add(s, "%s{\"seq\":%lu,\"dst_is_p\":%d,\"src\":\"0x%lx\",\"n\":%zu", f2 ? "" : ",",
                           g_rec[r].seq, (unsigned long)g_rec[r].dst == p[j], (unsigned long)g_rec[r].src,
                           g_rec[r].n);
                    f2 = 0;
                    if (sg) {
                        size_t cmpn = sg < got ? sg : got;
                        sb_add(s, ",\"src_hash_after\":\"%016llx\",\"words_differ_post\":%zu",
                               (unsigned long long)fnv(sb2, sg), diff_words(post_b, sb2, cmpn));
                    }
                    sb_add(s, "}");
                }
                sb_add(s, "]");
            }
        }
        sb_add(s, "}");
        free(t[j].pre);
    }
    sb_add(s, "]}");
}

static int capture_launch(const struct kentry *e, const void *func, dim3_t g, dim3_t b, void **args, size_t shm,
                          void *stream) {
    unsigned char k[512] = {0};
    unsigned long gv[NGLOB];
    g_in = 1;
    for (unsigned i = 0; i < e->nargs && e->aoff[i] + e->asz[i] <= sizeof k; i++)
        if (args && args[i]) memcpy(k + e->aoff[i], args[i], e->asz[i]);

    int capturing = 0;
    if (real_iscap) real_iscap(stream, &capturing);
    /* An error the engine has not read yet must reach it untouched: skip diagnostics then. */
    int pending = real_peek ? real_peek() : 0;
    int heavy = !capturing && real_sync && real_memcpy && real_getlast && real_peek && pending == 0;
    unsigned long idx = g_cnt++;

    sb_t s = {0};
    jhead(&s, "launch");
    sb_add(&s, ",\"idx\":%lu,\"kernel\":\"%s\",\"slot_vma\":\"0x%lx\",\"func\":\"0x%lx\",", idx, e->name, e->slot_vma,
           (unsigned long)func);
    sb_add(&s, "\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shmem\":%zu,\"stream\":\"0x%lx\",", g.x, g.y, g.z, b.x,
           b.y, b.z, shm, (unsigned long)stream);
    sb_add(&s, "\"capturing\":%d,\"kernarg_size\":%u,\"kernarg_hex\":\"", capturing, e->kernarg_size);
    sb_hex(&s, k, e->kernarg_size < sizeof k ? e->kernarg_size : sizeof k);
    sb_add(&s, "\"");
    json_globals(&s, gv);

    tap_t t[3] = {{0}};
    int tapping = 0;
    if (e->has_tap) {
        if (heavy) real_sync();
        if (heavy) json_norm(&s, k, gv);
        unsigned long p[3];
        memcpy(p, k + 0x90, 24);
        tapping = heavy && (p[0] || p[1] || p[2]);
        if (tapping) tap_pre(t, k);
    }
    if (heavy) sb_add(&s, ",\"shim_err_pre\":%d", real_getlast()); /* clears the shim's own errors */

    int rc = real_launch(func, g, b, args, shm, stream);
    /* A launch error belongs to the engine: if one is pending, do no more HIP calls and keep it. */
    if (heavy && real_peek() != 0) heavy = 0, tapping = 0;

    sb_add(&s, ",\"rc\":%d", rc);
    if (e->has_tap && heavy) {
        if (tapping) {
            real_sync();
            json_tap(&s, k, t);
        } else {
            int slot[3];
            memcpy(slot, k + 0x80, 12);
            sb_add(&s, ",\"tap\":{\"slots\":[%d,%d,%d],\"p\":[null,null,null],\"regions\":[]}", slot[0], slot[1],
                   slot[2]);
        }
    }
    if (e->has_tap && heavy) sb_add(&s, ",\"shim_err_post\":%d", real_getlast()); /* clear the shim's own */
    sb_add(&s, "}");
    emit(&s);
    free(s.p);
    g_in = 0;
    return rc;
}

/* ---- launch-order trace (CAPTURE_TRACE=1); call with g_mu held ---- */
static long name_idx(unsigned long off) {
    long lo = 0, hi = (long)KNAMES_N - 1;
    while (lo <= hi) {
        long mid = (lo + hi) / 2;
        if (KNAMES[mid].slot_vma == off) return mid;
        if (KNAMES[mid].slot_vma < off) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

static void emit_nf(sb_t *s) { /* no flush per line; flush every 64 */
    if (!g_out && !(g_out = fopen(g_path, "a"))) return;
    fputs(s->p, g_out);
    fputc('\n', g_out);
    if ((++g_tlines & 63) == 0) fflush(g_out);
}

static unsigned long long mono_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ULL + (unsigned long long)ts.tv_nsec;
}

static void trace_launch(const void *func, dim3_t g, dim3_t b, void *stream) {
    unsigned long off = (unsigned long)func - g_base;
    long ni = name_idx(off);
    sb_t s = {0};
    if (ni >= 0 && !g_seen[ni]) {
        g_seen[ni] = 1;
        sb_add(&s, "{\"kind\":\"kname\",\"h\":\"0x%lx\",\"name\":\"%s\"}", off, KNAMES[ni].name);
        emit_nf(&s);
        s.n = 0;
    }
    sb_add(&s, "{\"kind\":\"trace\",\"seq\":%lu,\"ns\":%llu,\"h\":\"0x%lx\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],"
               "\"stream\":\"0x%lx\"}", g_seq++, mono_ns(), off, g.x, g.y, g.z, b.x, b.y, b.z, (unsigned long)stream);
    emit_nf(&s);
    free(s.p);
}

int hipblasLtMatmul(void *h, void *desc, const void *alpha, const void *A, void *Adesc, const void *B, void *Bdesc,
                    const void *beta, const void *C, void *Cdesc, void *D, void *Ddesc, const void *algo, void *ws,
                    size_t wssz, void *stream) {
    if (!real_lt_matmul) resolve_all();
    if (!real_lt_matmul) return 1; /* HIPBLAS_STATUS_NOT_INITIALIZED */
    if (g_enabled && g_trace) {
        pthread_mutex_lock(&g_mu);
        if (g_seq < g_tmax && arm_check()) {
            sb_t s = {0};
            sb_add(&s, "{\"kind\":\"matmul\",\"seq\":%lu,\"ns\":%llu,\"desc\":\"0x%lx\",\"A\":\"0x%lx\",\"B\":\"0x%lx\","
                       "\"D\":\"0x%lx\",\"ws\":%zu,\"stream\":\"0x%lx\"}", g_seq++, mono_ns(), (unsigned long)desc,
                   (unsigned long)A, (unsigned long)B, (unsigned long)D, wssz, (unsigned long)stream);
            emit_nf(&s);
            free(s.p);
        }
        pthread_mutex_unlock(&g_mu);
    }
    return real_lt_matmul(h, desc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, ws, wssz, stream);
}

int hipLaunchKernel(const void *func, dim3_t grid, dim3_t block, void **args, size_t shmem, void *stream) {
    if (!real_launch) resolve_all();
    if (!g_enabled) return real_launch(func, grid, block, args, shmem, stream);
    const struct kentry *e = NULL;
    pthread_mutex_lock(&g_mu);
    g_total++;
    if (g_trace && g_seq < g_tmax && arm_check()) trace_launch(func, grid, block, stream);
    if (g_cnt < g_max) {
        e = match(func);
        if (!e && g_logunm) log_unmatched(func);
        if (e && !arm_check()) { g_prearm++; e = NULL; }
    }
    int rc;
    if (e) {
        rc = capture_launch(e, func, grid, block, args, shmem, stream);
        pthread_mutex_unlock(&g_mu);
        return rc;
    }
    pthread_mutex_unlock(&g_mu);
    return real_launch(func, grid, block, args, shmem, stream);
}

int hipMemcpyAsync(void *dst, const void *src, size_t n, int kind, void *stream) {
    if (!real_memcpy_async) resolve_all();
    if (g_enabled && !g_in && n == g_memsz && g_cnt < g_max && g_nmem < 100000) {
        pthread_mutex_lock(&g_mu);
        if (!arm_check()) {
            pthread_mutex_unlock(&g_mu);
            return real_memcpy_async(dst, src, n, kind, stream);
        }
        unsigned i = (unsigned)(g_nmem++ % MAXREC);
        g_rec[i].dst = dst;
        g_rec[i].src = src;
        g_rec[i].n = n;
        g_rec[i].kind = kind;
        g_rec[i].seq = ++g_recseq;
        sb_t s = {0};
        jhead(&s, "memcpy");
        sb_add(&s, ",\"seq\":%lu,\"dst\":\"0x%lx\",\"src\":\"0x%lx\",\"n\":%zu,\"copy_kind\":%d,\"after_launch\":%lu}",
               g_recseq, (unsigned long)dst, (unsigned long)src, n, kind, g_cnt);
        emit(&s);
        free(s.p);
        pthread_mutex_unlock(&g_mu);
    }
    return real_memcpy_async(dst, src, n, kind, stream);
}

__attribute__((destructor)) static void shim_fini(void) {
    if (!g_enabled || !g_total) return;
    sb_t s = {0};
    jhead(&s, "summary");
    sb_add(&s, ",\"launches_seen\":%lu,\"captured\":%lu,\"matched_before_arm\":%lu,\"base\":\"0x%lx\",\"table_slots\":%d}",
           g_total, g_cnt, g_prearm, g_base, KTAB_N);
    emit(&s);
    free(s.p);
}
