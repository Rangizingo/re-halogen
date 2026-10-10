# halogen-npu — the XRT/AIE NPU engine (Phase 4 writeup)

Target: `binaries/halogen-npu` (1.1 MB, stripped PIE; Ghidra base = file vaddr + 0x100000).
Evidence: Ghidra decompiles in `tmp/npu/*.c` (+ `work/npu/*.pseudo.c` for this phase's
passes: 0x167960, 0x164090, 0x16ebde, 0x1362d0, 0x12b110, 0x18a5b0, 0x1acdc0),
r2 xref runs (`work/npu/axt_getaddrinfo.txt`, `work/npu/axt_mmap_open.txt`),
`nm -D` import tables, and the vendor's own deployment files in `image/`
(`entrypoint.sh`, `npu-models.txt`, `serve_api.py`).

Scope note: this is the NPU (AMD XDNA/AIE via XRT) engine that serves the small
"decision/embedding/rerank/moderation/generate/image" models; it is a *different
binary* from `flash_serve` (the gfx1151 GPU engine). The front end (`serve_api.py`)
talks to it over a loopback TCP socket (`--npu HOST:PORT`, entrypoint passes
`127.0.0.1:$port`, `image/entrypoint.sh:1298`).

---

## 1. XRT call graph (AIE flow)

All XRT symbols are PLT imports (`nm -D binaries/halogen-npu | grep ' U xrt'`).
Function roles established from decompiled bodies + callee meta:

| Stage | Function(s) | XRT calls | Evidence |
|---|---|---|---|
| Device open | `FUN_00130a20` family (0x130a20/0x130a2d/0x130ac7/0x130baf), `FUN_00143270` (0x1433d5), 0x171dbb, 0x1d1f4b | `xrt::device::device(dev, index)` | `tmp/npu/0x130a20.c:160` (`xrt::device::device(local_380,0)`), meta callee lists |
| ELF load + hw context | `FUN_00157940` (0x15794f) | `xrt::elf::elf(elf, path)` then `xrt::hw_context::hw_context(ctx, device, elf)` | `tmp/npu/0x15794f.c` (whole body, 8 lines) |
| AIE graph kernel | `FUN_00159980` (0x159c03) | `xrt::ext::kernel::kernel(kernel, hw_context, name)` with name built as `"main:" + graph-name` (`0x6e69616d`="main", `':'`) | `tmp/npu/0x159c03.c:257` + string build at :97-99 |
| Run create + args | `FUN_0015a022` (0x15a022), `FUN_0015a0e6` (0x15a0e6) | `xrt::run::run(run, kernel)`, `xrt::run::set_arg_at_index` | meta callees; bodies |
| Run exec | `FUN_00157b70` (0x157b70), `FUN_00157c67` (0x157c67) | `xrt::run::start()`, poll `xrt::run::state()` (states 2/3/7 = running-ish, spin-wait up to `param_6` ms), `xrt::run::wait()`; return-code 4 = timeout | `tmp/npu/0x157b70.c:69-78` |
| BO alloc (host-only) | `FUN_001579a0` (0x157a11) | `xrt::bo::bo(bo, device, size, flags=0x20000000, group=0)`; default size 0x40; then `xrt::bo::map()`, `memset`, `xrt::bo::sync(TO_DEVICE)` | `tmp/npu/0x157a11.c:67-82` |
| BO subview | `FUN_0015a422` (0x15a422) | `xrt::bo::bo(dst, parent_bo, size, offset)` — carves regions out of a parent BO (buddy-ish allocator keyed by (offset,size) in an Rb_tree) | `tmp/npu/0x15a422.c:41` |
| BO sync in (args) | `FUN_001568e0` (0x156b42) | `xrt::bo::sync(bo, dir=0 /*TO_DEVICE*/, len, off)` per registered arg; page-rounding narrowed by env `HALOGEN_NPU_SYNC_NARROW`, whole-buffer mode by `HALOGEN_NPU_SYNC_WHOLE` | `tmp/npu/0x156b42.c:33-123` |
| BO sync out (results) | `FUN_001bfbf0` (0x1bfbf0), `FUN_001bfcc1` (0x1bfcc1) | `xrt::bo::sync(bo, dir=1 /*FROM_DEVICE*/, …)` and `dir=0` | `tmp/npu/0x1bfbf0.c:59,139` |
| Misc sync | `FUN_0013c04e` (0x13c04e) | `xrt::bo::sync(bo, 0, …)` | `tmp/npu/0x13c04e.c:25` |
| Kernel/context teardown | `FUN_001445af` (0x1445af) | `xrt::kernel::~kernel`, `xrt::hw_context::~hw_context` (+ patch-blob memcpy, §2) | meta callees; `0x1445af.c:490` |

**The AIE flow, end to end** (proven from the above):
`xrt::device::device(index)` → read `devices.hnpm` (§2) → pick the `elf` record
("no elf record" error, `0x130a20.c:674`) → `xrt::elf::elf(path)` +
`xrt::hw_context::hw_context(device, elf)` → `xrt::ext::kernel(hw_context,
"main:<graph>")` for the AIE graph → `xrt::run(kernel)` → allocate BOs
(host-only flag, §1.1) and subviews → `set_arg_at_index` per graph arg →
`bo::map()`/fill from the weight container (§2.3) → `bo::sync(TO_DEVICE)` →
`run::start()` → optional state-poll window (`wait_ms`, READY banner field) →
`run::wait()` → `bo::sync(FROM_DEVICE)` → read results from the mapped host
pointers.

### 1.1 The BO flags value 0x20000000

`xrt::bo::bo(bo*, device, size, 0x20000000, 0)` (`0x157a11.c:67`).
Per AMD XRT `xrt_mem.h` (`XCL_BO_FLAGS_*` bit layout, bits 24–31 = BO flags):

- `XCL_BO_FLAGS_CACHEABLE` = 1<<24 = **0x01000000**
- `XCL_BO_FLAGS_KERNBUF`   = 1<<25 = 0x02000000
- `XCL_BO_FLAGS_SVM`       = 1<<27 = 0x08000000
- `XCL_BO_FLAGS_DEV_ONLY`  = 1<<28 = 0x10000000
- `XCL_BO_FLAGS_HOST_ONLY` = 1<<29 = **0x20000000** ← the value used
- `XCL_BO_FLAGS_P2P`       = 1<<30 = 0x40000000

So the raw value is **0x20000000 = XCL_BO_FLAGS_HOST_ONLY** (`xrt::bo::flags::host_only`),
*not* CACHEABLE. Memory group argument = 0. (Verified against
`src/runtime_src/core/include/xrt/detail/xrt_mem.h`, XRT master, 2026-10-10.)
This matches the usage pattern: every BO is `map()`ed and `sync()`ed — host-backed
buffers, the normal choice on a dNPU (Strix Halo) where the AIE tiles DMA to/from
system RAM.

## 2. Container formats

Three file kinds per model (vendor manifest `image/npu-models.txt:18-24`:
`devices/devices.hnpm`, `devices/<shape>.elf`, `devices/consts.hnpb`,
`devices/NOTICE.md`, `<id>.hnpw`, `tokenizer/tokenizer.json`).

### 2.1 `devices.hnpm` — device-set manifest (text)

- Reader: `FUN_00163c50` (0x163c50) — plain `fopen(path,"rb")` + 64 KiB-chunk
  slurp into a `std::string`; failure → `"cannot open <path>"`
  (`tmp/npu/0x163c50.c:21-30,153-158`).
- Parser: `FUN_00164090` (0x164090) — a **line-oriented text format**, not binary:
  split on `\n`, strip trailing `\r`, skip blank lines and `#` comments, tokens
  separated by space/tab, `key=value` pairs, `"quoted values"`; builds records
  (0x58-byte entries: type token + key/value vectors + nested record vectors).
  Errors: `"line N: bad quoted value for …"`, `"line N: a quoted token without a key"`,
  `"line N: a record starts with its type"` (`work/npu/0x164090.pseudo.c`).
- Consumer (`FUN_00145eb0`, file 0x145f0f): the devices manifest must contain a
  record whose type is `"hnpm"` with a `"devices"` field
  (`0x145f0f.c:105-107`), else `"<dir>/devices.hnpm: not an hnpm devices manifest"`.
  Record fields read (all via `FUN_00139a60` type-compare / `FUN_00162ac0`
  key-lookup / `FUN_00162ba0` int-get / `FUN_001631b0` list-get):
  `family` (`qwen35_hybrid`, `qwen3_dense`), `compat`, `name`, `sig`, `elf`,
  `blob`, `patch`, `bytes`, `share`, `layers`, `dims`, `eps`, `hidden`, `types`,
  `f32`, `embed`, `fnorm`, `head`/`kind`, `decision`, `score`/`classify` + `rows`
  (score takes 2 rows "no, yes"; classify 2–16), `route`, `bucket`, `geom`,
  `pass`, `plan`, `role`, `io`, `ops`, `base`, `gbase`, `xw`, `Ts`, `ro`, `keep`,
  `nm`, `kpts`, `zero`, `init`, `src`, `xd`, `rz`, `slots`, `T`, `S`, `CN`, `CM`,
  `M`, `in_len`, `out_len` (strings resolved from `.rodata`, file offsets
  0xef172–0xefa0f).
- Printer (`FUN_0016ebde`, used by `probe`): emits
  `"hnpm 1 entry\nentry P=%d capb=%d nb=%d lin=%d"` — the compact per-entry
  summary with fields **P** (entry count), **capb** (cache capacity bytes),
  **nb**, **lin** (`work/npu/0x16ebde.pseudo.c:115-123`).
- Manifest blob + checksum: the `"manifest"` record is fetched
  (`FUN_00162ec0(devices,"manifest")`) and verified by `FUN_00162fd0`
  (0x162fd0): **xor32** = XOR of the blob's little-endian u32 words (AVX2
  32-byte main loop + scalar tail + byte tail), compared against the stored u32
  at record offset +0x24 (`0x162fd0.c:105`). Failure →
  `"<weights>: no manifest blob, or its checksum fails"` (`0x145f0f.c:126-128`).

### 2.2 `consts.hnpb` — constants/weights blob pack (binary, HNPB v1)

Reader: `FUN_00167960` (0x167960; fully decompiled this phase,
`work/npu/0x167960.pseudo.c`). Callers: 0x145f0f, 0x146a9c (model load),
0x1a8608 (convert/probe), 0x1d6877 (flux2). Load sequence:

1. `open(path, O_RDONLY)` + `fstat`; file must be ≥ 0x40 bytes
   (`": too short for an HNPB header"`).
2. `mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0)` (`": mmap failed"`).
3. Header (little-endian):
   - `+0x00 u32 magic = 'HNPB'` (0x42504e48)
   - `+0x04 u32 version = 1` (else `": not an HNPB v1 file"`)
   - `+0x08 u32` — header field copied to container+0x10 (a flags/format field;
     semantics bounded-open)
   - `+0x20 u64 n_entries`
   - `+0x30 u64 table_off`
   - validation: `table_off + n_entries*0xa8 ≤ file_size`
4. Blob table: `n_entries` records of **0xa8 bytes** at `table_off`, each:
   - `+0x00 char name[80]` (read with `strnlen(…, 0x50)`)
   - `+0x50 u32`, `+0xa0 u32` — two per-blob sub-checksums (kept as a pair in
     the in-memory record; the embed/fnorm path in `FUN_00145eb0` verifies the
     blobs with xor32 and reports `"embed / fnorm checksum fails"`)
   - `+0x70 u64 blob_off` — must be 4096-aligned
   - `+0x78 u64 blob_size` — `blob_off + blob_size ≤ file_size`
     (else `"blob <name> is unaligned or runs past the file"`)
   - duplicate names rejected (`": duplicate blob <name>"`).
5. Result: name→blob map (Rb_tree) + insertion-ordered vector; blobs stay
   mmap'd (zero-copy).

Sizes from the vendor manifest: `consts.hnpb` = 1,645,016 B (decider-0.8b),
5,086,496 B (qwen3-embedding-0.6b) — the 1.6–5 MB range.

### 2.3 `<id>.hnpw` — weight file (same HNPB v1 container)

There is **no `HNPW` magic anywhere in the binary** (byte-scan of
`binaries/halogen-npu` for `HNPW`/`48 4e 50 57`: zero hits; `HNPB` occurs only
as the header-compare immediates at file 0x679f7/0x70856/0x7aae8 and in the
error strings at 0xefc01/0xf1ddb). The weight file is opened by the **same**
`FUN_00167960` HNPB-v1 reader: `FUN_00145eb0(devices_dir, weights_path, …)`
calls `FUN_00167960(container@+0x190, weights_path, err)` at
`0x145f0f.c:109` — i.e. `.hnpw` is an HNPB v1 file whose extension marks its
role. Consequences (all proven at the same call site):

- It carries a `"manifest"` record (model dims/eps/layers/head/decision rows,
  family, compat) — `"the weight file's family (…) is not the device set's"`
  and `"the model manifest lacks dims / eps / layers"` (`0x145f0f.c:306,1054`).
- It carries named blobs checked with the same xor32: `embed` (size a multiple
  of hidden×2 = bf16 vocab rows), `fnorm` (size = hidden×4 = f32), plus
  per-layer `route`/`bucket`/`patch` blobs; `FUN_0016dc30` (0x16dcb5) is the
  generic accessor: `"the weight file's <name> is missing"` /
  `"… has the wrong size or checksum"`.
- `FUN_001445af` (0x1445af) applies `patch` records: memcpy of a checksummed
  weight blob into a graph arg buffer at a byte offset
  (`"patch <name> does not fit … or fails its checksum"`, `0x1445af.c:480-505`).
- Sizes: `decider-0.8b.hnpw` = 2,269,182,920 B (2.27 GB),
  `qwen3-embedding-0.6b.hnpw` = 813,032,248 B, `qwen3.5-2b.hnpw` = 3.6 GB,
  `flux2-klein-4b.hnpw` = 8.0 GB (`image/npu-models.txt`).
- The 2.27 GB file is **not** mmap'd by this reader path (mmap of a 2.27 GB
  file is possible, and `FUN_00167960` does mmap — the whole file is mapped
  PROT_READ and blobs are read through the map; the entrypoint's disk-size
  check `wc -c < "$d/$id.hnpw"` confirms the file is a single flat container).
- Fine-tunes: `entrypoint.sh` runs `halogen-npu probe <dir>` on a user
  checkpoint dir (safetensors: `FUN_0018a5b0`/`FUN_001acdc0` are safetensors
  readers — `dtype`/`shape`/`data_offsets` JSON header, `"is not a safetensors
  file"`) and converts to `halogen-npu.hnpw` beside it.

**Bounded-open:** the exact semantics of the HNPB header u32 at +0x08 and of
the two per-blob u32 sub-checksums at +0x50/+0xa0 (they are stored in the
in-memory record as a pair; only the whole-blob xor32 at record+0x24 is
verified on the paths we decompiled). The `convert` writer (0x1a8608 region,
`fopen(…,"wb")` at `0x1a8608.c:1129`) emits these files; a full writer
decompile would pin the remaining header fields.

## 3. The getaddrinfo question — VERDICT: loopback-only, no outbound network

**Call sites (r2, full `aaa`):** `axt sym.imp.getaddrinfo` → exactly **one**
call site: `0x35042` in `fcn.00033090` (= `FUN_00133090`, the `listen`
subcommand; Ghidra 0x133090). `axt sym.imp.socket` → one site, 0x350b1, same
function (`work/npu/axt_getaddrinfo.txt`).

**The loopback gate** (`analysis/npu_server_pseudo.c:432-470`): after
`getaddrinfo(host, port, hints{ai_socktype=SOCK_STREAM, ai_flags=0x400}, …)`,
the code walks the result list and **refuses every non-loopback address**:

- `AF_INET` (family 2): `if (paVar58->ai_addr->sa_data[2] != '\x7f') goto next;`
  — `sa_data[2]` is the first IPv4 octet, so only `127.0.0.0/8` is accepted
  (line 443).
- `AF_INET6` (family 10): words at `sa_data+6` and `sa_data+10` must be 0, the
  next 10 bytes (`iVar20`, covering `s6_addr[0..9]`) must be 0, and
  `*(int *)(sa_data+2) == 0x1000000` — i.e. `s6_addr[15] == 1`, all else 0:
  exactly `::1` (lines 463-467).
- If no loopback candidate exists: `local_4c0 = -1` and the error path prints
  `"cannot listen on <host>:<port> (a loopback HOST:PORT only)"`
  (`tmp/npu/0x133090.c:749-754`).

Only after the gate: `socket()` + `setsockopt(SO_REUSEADDR)` + `bind()` +
`listen(fd, 64)`.

**No outbound path exists:** the binary's dynamic imports include
`accept, bind, getaddrinfo, listen, recv, send, socket` — and **no `connect`,
no `getnameinfo`, no `inet_*`, no curl/SSL** (`nm -D`). `getaddrinfo` is used
purely as a name→sockaddr resolver for `bind()`. The deployment side agrees:
`entrypoint.sh` starts `halogen-npu listen … 127.0.0.1:$port`
(`NPU_API_ARGS=(--npu "127.0.0.1:$port")`, lines 1298-1314) and `serve_api.py`
connects to that `--npu HOST:PORT`.

**Verdict: loopback-only. halogen-npu cannot and does not open outbound
network connections.**

## 4. The serve/listen protocol

Two transports, one dispatcher:

- `serve DEVICES_DIR WEIGHTS [ORDER] [--model DIR W]…` — commands on **stdin**,
  replies on **stdout** (`std::getline(std::cin, …)` loop,
  `tmp/npu/0x133090.c:854-906`).
- `listen DEVICES_DIR WEIGHTS HOST:PORT [ORDER] [--model DIR W]…` — TCP
  server, loopback-gated (§3). `poll()` on the listen fd + a self-pipe;
  `accept()` → per-connection worker thread `FUN_0012e9d0`
  (`tmp/npu/0x133090.c:536-620`); TCP_NODELAY; connection slots bounded
  (`"ERR 0 busy: N connections open"`, line 690-694).

**READY banner** (printed after models load, flushed):
- listen mode: `READY image listen %s queue %d models 1` (image-only set) or
  `READY hidden %d layers %d order %s threads %d wait_ms %d op %d listen %s contexts %d queue %d batch %d models %zu`
  (`0x133090.c:505-520`).
- serve mode: `READY image` / `READY hidden %d layers %d order %s threads %d
  wait_ms %d op %d` (`:786-793`).

**Commands** (verb = first token of the line; dispatcher `FUN_0012b110`
(0x12b110, decompiled this phase) + per-connection loop `FUN_0012e9d0`):

| Verb | Reply prefix | Notes |
|---|---|---|
| `INFO` | `INFO ` + capability fields + ` ` | capability probe; the front end's `_probe()` reads the `I …` line (`serve_api.py:858`). In `FUN_0012e9d0` the 4-char verb compare immediates are `0x54495551`="QUIT" and `0x4f464e49`="INFO" (`0x12e9d0.c:360`). |
| `GEN` | `GEN ` | generate (the NPU "generate" task models) |
| `DEC` | `DEC ` | decision request (3-char verb: `'D','E','C'`, `0x12e9d0.c:594`); malformed → `"ERR 0 … a malformed DEC line"` |
| `IMG` | `IMG ` | text-to-image (flux2); `"a malformed IMG line"`, `"no text-to-image model is loaded"` |
| `EMB` / `SCO` / `CLS` | `EMB `, `SCO`, `CLS` | embedding / rerank-score / classify-moderation heads |
| `PFX` | `PFX `, `HN ` | prefix (prompt-cache) ops; `HN ` = hidden-state reply |
| `LEDGER` | `LEDGER ` | cache ledger |
| `CANCEL` | — | 6-char verb `0x434e4143`+`0x4c45`="CANCEL" (`0x12e9d0.c:486`) |
| `QUIT` | — | closes the connection (4-char `0x54495551`) |
| (unknown) | `ERR 0 … unknown verb …` | `"(an image model answers IMG)"` hint (`0x12b110.pseudo.c:177-182`) |

Errors are `ERR <code> <text>` (`ERR ` at 0xef39b; `OK ` reply prefix at
0xef0c9). Queue backpressure: `"ERR 0 busy: the queue holds N requests"`
(`0x12e9d0.c:682-685`); line-length cap: `"ERR 0 a line longer than N bytes"`
(`:170`). Shutdown: SIGTERM/SIGINT handler + self-pipe → `"… shutting down"`
replies, `shutdown(fd,2)` on all sockets (`0x133090.c:708-743`).

The per-connection worker `FUN_001362d0` (0x1362d0) **is now decompiled**
(`work/npu/0x1362d0.pseudo.c`, 68 KB): it dequeues admitted requests, resolves
the model's `tokens`/`T` records, runs the graph via the §1 XRT path, and calls
the same `FUN_0012b110` reply formatter. No protocol verbs beyond the table
above appear in it.

## 5. Subcommands (usage string, file 0xf1528)

```
usage: halogen-npu selftest bf16|norm|head|resid ... | replay DIR [REPS] |
serve DEVICES_DIR WEIGHTS [ORDER] [--model DEVICES_DIR WEIGHTS]... |
listen DEVICES_DIR WEIGHTS HOST:PORT [ORDER] [--model DEVICES_DIR WEIGHTS]... |
probe MODEL_DIR |
convert MODEL_DIR DEVICES_DIR OUT [--name ID] [--source SRC] [--if-stale]
```

- **selftest** — on-device kernel self-tests: `bf16`, `norm`, `head` (with
  optional count arg), `resid`, `hostx` (10 args), `gx` (7 args, the gx
  matmul family; strings `"halogen-npu: selftest gx: sizes %zu %zu %zu do not
  match"`), `flux`. Deterministic PRNG inputs (golden-ratio hash,
  `0x133090.c:1090-1101`), bf16 round-trip, results compared and written out.
- **replay DIR [REPS]** — replay a captured call trace:
  `"replay: %zu calls, %zu buffers, ELF %s; load %.1f ms (buffers + program)"`,
  `"replay: %s (%d output ranges differ over %d reps)"` — a bit-exactness
  harness for the AIE program.
- **probe MODEL_DIR** — prints `family=… task=… base=… sig=…` for a HF
  checkpoint dir (safetensors readers §2.3); flux2 special-case prints
  `family=flux2 task=image base=flux2-klein-4b sig=d3072.h24x128.b5x20.m9216.c7680`
  (`0x1a8608.c:273-288`). Used by `entrypoint.sh npu_finetune_ready`.
- **convert MODEL_DIR DEVICES_DIR OUT [--name ID] [--source SRC] [--if-stale]** —
  GGUF/HF checkpoint → device set + `.hnpw` writer: progress strings
  `"halogen-npu convert: layer %d / %d (%.0f s)"`, `"convert: the DiT's %zu
  GEMM weights (%.0f s)"`, `"convert: %s, %zu blobs, %.2f GB, %.1f s"`,
  `"convert: %s is up to date (%.1f s)"` (`--if-stale`),
  `"convert: %s (flux2) -> %s as %s"`.
- Extra utility modes in strings (dispatched from the same main):
  `"usage: halogen-npu cache ecopy IN OUT CAP"` (cache-file re-copy),
  `"usage: halogen-npu flux noise SEED RES OUT | taef2 TAEF2 Z H W OUT_RGB
  [OUT_F32] | dit DEVDIR WEIGHTS RES STEP X CTX OUT [REPS] | image DEVDIR
  WEIGHTS RES SEED CTX OUT_RGB [OUT_LATENTS] [REPS]"` (flux2 pipeline stages:
  noise gen, taef2 VAE decode, DiT step, full image).

## 6. Open items (bounded)

1. HNPB header u32 @+0x08 and per-blob u32 pair @+0x50/+0xa0 semantics —
   writer-side (`convert`, 0x1a8608 region) decompile would pin them.
2. `FUN_0012b110` verb grammar beyond verb names (exact argument lists per
   verb, e.g. `GEN` token-list encoding) — pseudocode is decompiled
   (`work/npu/0x12b110.pseudo.c`) but the per-verb field layouts are not
   fully annotated.
3. `xrt::ext::kernel` graph-name set (which `main:<graph>` names exist) is
   data-driven from `devices.hnpm` `ops`/`plan` records; the graph names
   themselves live in the shipped `.elf` files (not analyzed here).
