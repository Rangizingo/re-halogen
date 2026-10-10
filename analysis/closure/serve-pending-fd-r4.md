# Serve-time pending fd handling in flash_serve 0.17.3 (r4)

## Scope

Question: while connection A is served by FUN_0182ea80, what code accepts other connections, answers INFO and PING inline, and queues the rest? Static only. Ghidra address = file/r2 VMA + 0x100000. Decomp paths are under `work/decomp/flash_serve/`. Struct base `P` = `param_1` of FUN_0182ea80 = `&local_1458` in main (`analysis/main_pseudo.c:1111`, `FUN_0182ea80(&local_1458);`).

Short answer: the code is not in FUN_0182ea80 itself. It is the separate function `FUN_01844d50(P)`, called from the top of FUN_0182ea80's command loop and from a per-step engine callback.

## Verdict: CLOSED

## Evidence

### 1. Poll and accept site: FUN_01844d50

- Listen fd is `P+0xa8` (`01844d50.c:32` `iVar7 = *(int *)(param_1 + 0xa8);`). In main, `local_13b0 = iVar6;` (`main_pseudo.c:1077`) and `local_13b0` is `local_1458+0xa8`, so this is the listen socket. FUN_0182ea80 reads the same field: `uVar35 = *(uint *)((long)param_1 + 0xa8);` (`0182ea80.c:663`).
- Zero-timeout select then non-blocking accept, in a loop until nothing is pending (`01844d50.c:58-62`):
  `local_e8.tv_sec = 0; local_e8.tv_usec = 0; iVar7 = select(iVar7 + 1,&local_c8,...,&local_e8); if ((iVar7 < 1) || (iVar7 = accept(*(int *)(param_1 + 0xa8),...), iVar7 < 0)) break;`
- Disasm: select call `0x01744e0f`, accept call `0x01744e26` (Ghidra 0x01844e0f / 0x01844e26).
- TCP_NODELAY set on the new fd: `setsockopt(iVar7,6,1,&local_cc,4);` (`01844d50.c:65`).
- Queue cap: `if ((ulong)(*(long *)(param_1 + 0xb8) - *(long *)(param_1 + 0xb0)) < 0x7d)` (`01844d50.c:63`). Byte span under 0x7d means at most 31 queued fds before the next is refused with `close(iVar7)` (`01844d50.c:114`). Inference: arithmetic from byte span / sizeof(int).

### 2. Where it is called during A's service

- `0182ea80.c:641-644`: `LAB_0182ee80: ... FUN_01844d50(param_1);` Top of the per-command loop, so it runs between each line/command of A (`goto LAB_0182ee80` at `0182ea80.c:9150`).
- `0184be30.c:42`: `FUN_01844d50(param_1);` FUN_0184be30 is installed as a callback: `*(code **)(lVar96 + 0x318) = FUN_0184be30;` (`0182ea80.c:556`). Inference: the engine invokes this slot during a running GEN so queued fds are still polled mid-generation. The call site of the +0x318 slot was not traced.
- The idle wait in FUN_0182ea80 (`select` at `0182ea80.c:845`, fds = A and `P+0xa8`) returns on the listen fd, exits the read loop with `cVar29 = 1` (`0182ea80.c:851-854, 887`), and control returns to `LAB_0182ee80`, which polls. Inference from control flow.

### 3. Verb check (inline answers)

Only the first 5 bytes of each queued fd are inspected, via MSG_PEEK|MSG_DONTWAIT (flags 0x42):
- `sVar11 = recv(*(int *)(*(long *)(param_1 + 0xb0) + uVar17 * 4),&local_e8,5,0x42); if (sVar11 == 5) {` (disasm `call recv` at `0x01745122`).
- PING: `if (local_e8.tv_sec._4_1_ == '\n' && (int)local_e8.tv_sec == 0x474e4950)`; disasm `mov ecx, 0x474e4950 ; 'PING'` at `0x01745134`. Then it consumes exactly 5 bytes (`recv(...,&local_e8,5,0x40)`) and `send(...,"PONG\n",5,0x4000)`. Trace string: "queued connection pinged while a session was running".
- INFO: `if (local_e8.tv_sec._4_1_ != '\n' || (int)local_e8.tv_sec != 0x4f464e49) goto LAB_0184532b;` (else branch is INFO); disasm `mov ecx, 0x4f464e49 ; 'INFO'` at `0x0174514e`. It consumes 5 bytes, builds a reply from the std::string at `P+0x20` (ptr) / `P+0x28` (len) plus one byte `DAT_0013420f` (inference: newline), and sends it with flags 0x4000 (MSG_NOSIGNAL). Trace string: "queued connection asked INFO while a session was running".
- The match is `VERB\n` exactly. After a successful PING/INFO answer the code sets `iVar6 = 0` and `if (iVar6 == 0) goto LAB_0184510a;`, which re-peeks the next 5 bytes of the same fd. So PING is answered when it is the first line, or follows already-answered PING/INFO lines. PING after CSTAT is not answered: CSTAT fails both compares and goes to `LAB_0184532b`, and the fd is left alone. This matches the probe observation. Verbs with arguments are not matched by the exact 5-byte compare (inference).

### 4. Rest of B's bytes: kept and replayed

- Inline answers consume only their own 5 bytes (flag 0x40, no PEEK). Everything else stays in the kernel socket receive buffer; the queue entry is just the int fd.
- When main pops it (`main_pseudo.c:1089-1101`: `iVar7 = *local_13a8;` then memmove down), FUN_0182ea80 reads from that socket afresh (`recv(..., 0x1000, 0)` at `0182ea80.c:670` / `:867`), so CSTAT, later PING and GEN are processed in order after A closes. Matches probe5.
- A queued fd that returns EOF on `recv(fd,&local_c8,1,0x42)` (`sVar11 == 0`) is closed and erased from the vector; trace "queued connection closed without a request (a health probe does this)". The erase is memmove plus `*(long *)(param_1 + 0xb8) = (long)pvVar13 + -4;` (end of `01844d50.c`).
- Every poll rescans all queued fds from index 0, so a B that sends INFO late is still answered on the next poll.

### 5. Vector push and layout (P = local_1458)

| Offset | Meaning | Cite |
|---|---|---|
| `P+0xa8` | listen fd (int) | `01844d50.c:32`, `main_pseudo.c:1077` |
| `P+0xb0` | vector begin (= `local_13a8`) | `main_pseudo.c:1078,1089` |
| `P+0xb8` | vector end (= `piStack_13a0`) | push: `*piVar1 = iVar7; *(int **)(param_1 + 0xb8) = piVar1 + 1;` |
| `P+0xc0` | vector capacity end | `if (piVar1 == *(int **)(param_1 + 0xc0))` grow path |
| `P+0x20/+0x28` | INFO reply string (ptr, len) | INFO branch of `01844d50.c` |

Push is an inlined `std::vector<int>::push_back`: store at end, end += 4. The grow branch decompiles to `_Znwm(uVar17 * 4)` with no continuation (Ghidra truncation), so growth is standard doubling (inference). Trace line on push: "connection queued behind the running session, %zu waiting" (env `HALOGEN_SERVE_TRACE=1`). FIFO holds because main pops `*local_13a8` and shifts the rest.

## Residuals

- The invoker of the `+0x318` callback (how often during GEN) was not traced; mid-GEN polling rests on the install at `0182ea80.c:556` plus the FUN_01844d50 call inside FUN_0184be30. The loop-top poll is traced.
- The INFO reply content (string at `P+0x20`) was not decoded; only transport is covered.
- The 31-entry cap is derived from the byte-span compare, with no runtime test.
- INFO/PING with trailing arguments is inferred from the 5-byte compare, not probed.
