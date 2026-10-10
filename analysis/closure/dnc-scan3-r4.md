# U12 round 4: scan3 B/C/D/G, NormFold roles, Tap writers, n == 1 decode (flash_serve 0.17.3)

Checked: analysis/closure/dnc-scan3-r3.md (all of it), work/decomp/flash_serve/{018ab5c0,018fa150,01895170,01895ca0,01895270,018951f0,01884460,019406c0}.c, `objdump -d` of VMA 17ab5c0..17b0c90 (gdn_layer_forward, via `ssh ai-core`), `objdump -R` handle map joined with analysis/function-map-kernel-stubs.tsv, the only i3=1 scan3x slice in work/kernels (`k_dnc_scan3x<2,4,T,F,T,1,0,1,F>`, copied as k101.s), per-slice instruction counts over work/kernels/*k_dnc_scan3I*.s.

Conventions as in r3: VMA = Ghidra address minus 0x100000; "proven" = read in disassembly, decompile or kernel ISA; "inferred" is labelled. Stack offsets are `%rsp` offsets in gdn_layer_forward (VMA 17ab5c0).

## Verdict: PARTIAL

Closed this round: (1) Tap non-default writers exist and are fully traced (r3 said none found; it looked at the wrong stack object); Tap is a recurrent-state checkpoint tap. (2) The n == 1 decode path and its stub: `k_dn_conv_ab_rows<1,b>` then `k_dn_recur_step`, not a dnc kernel. (3) NormFold: kernel arithmetic shows it folds the q/k inverse-RMS computation into the scan; per-pointer roles bounded to what the kernel indexing proves; `w` = 128.0 is the divisor of the mean. Still open: semantic labels for scan3 B, C, D, G (no host value source; code effects only, extended below), and the exact buffer identity of NormFold n0, n1, n2.

## 1. Tap writers (DncTapArgs): CLOSED

gdn_layer_forward holds two ScanArgs images: a template at `0x960(%rsp)` (Tap at `0xa00`, defaults `17abbe3: movq $0xffffffffffffffff,0xa00(%rsp)`, the only site r3 found) and the live one at `0x330(%rsp)` (Tap at `0x330+0xa0 = 0x3d0`). The live one is copied to `0x500(%rsp)` by `17af17c: lea 0x500(%rsp),%rdi` / `17af184: lea 0x330(%rsp),%rsi` / `17af18c: mov $0xf0,%edx` and passed to the launcher (`17af4b0: call 18d7b70`).

Defaults at the live copy:
```
17aee67: movl   $0x0,0x3c8(%rsp)                ; ScanArgs+0x98 norm_read = 0
17aee72: movq   $0xffffffffffffffff,0x3d0(%rsp) ; slot0, slot1 = -1
17aee7e: movl   $0xffffffff,0x3d8(%rsp)         ; slot2 = -1
17aee89: movups %xmm0,0x3e0(%rsp)               ; p0, p1 = 0
17aee91: movups %xmm0,0x3f0(%rsp)               ; p2 = 0
```
Non-default writers (one block per tap k = 0, 1, 2):
```
17aefec: mov    0x4bc(%r12),%eax      ; c = ctx.tapcnt[0]   (0x4c0 at 17af07f, 0x4c4 at 17af112)
17aeff4: test   $0x3f,%al             ; c % 64 == 0
17aeff9: cmp    %r14d,%eax ; setg     ; c <= n (r14d = n tokens)
17af012: cmpb   $0x1,0x14d88f(%rip)   ; # 18fc8a8 = HALOGEN_TAP_FROM_SCAN
17af023: lea    0x3f(%rax),%ecx ; 17af02b: sar $0x6,%ecx ; 17af02e: dec %ecx
17af030: mov    %ecx,0x3d0(%rsp)      ; slot0 = c/64 - 1
17af037: mov    0x4e8(%r12),%rax
17af03f: mov    %rax,0x3e0(%rsp)      ; p0 = ctx+0x4e8
```
k = 1: `17af0c3: mov %ecx,0x3d4(%rsp)` / `17af0d2: mov %rax,0x3e8(%rsp)` (ctx+0x4f0). k = 2: `17af156: mov %ecx,0x3d8(%rsp)` / `17af165: mov %rax,0x3f0(%rsp)` (ctx+0x4f8).

Env: `018951f0.c`: `getenv("HALOGEN_TAP_FROM_SCAN")`, unset gives true, a value starting with `0` gives false. When false the Tap stays -1 and the host uses the separate-launch route (`01895270.c` launches the scan three times into `ctx+0x4e8/4f0/4f8`, defaults at `01895270.c:39-40,66-67,94-95`); default true folds the tap into the main scan.

Where `c` comes from (`018fa150.c`, forward(n); `iVar5 = ctx+0x2e8` past length, `uVar9 = iVar5 + n`):
```
uVar10 = iVar2 - iVar5;                         ; c = target_pos - past   (iVar2 = ctx+0x4b0, 0x4b4, 0x4b8)
if (lVar4 == 0 && lVar3 == 0 && uVar10 != 0 && iVar5 <= iVar2
    && (bVar8 & 1) == 0 && iVar2 < (int)uVar9 && (uVar10 & 0x3f) == 0)
  *(uint *)(param_1 + 0x4bc) = uVar10;          ; same for 0x4c0, 0x4c4
```
The three targets are set from an `int[3]` argument, clamped at 0 (`01895170.c:6-27`). Conditions (proven): target position strictly inside the current batch, a multiple of 64 from the batch start, step path off.

Kernel use (r3 section 2, proven by compare): slot k = chunk-block index at which the kernel stores LDS data to p[k]. With slot = c/64 - 1, the store fires at the last full 64-token block of the c-token prefix.

Meaning: before each tap the host runs `hipMemcpyAsync(ctx+0x4e8, 0xb88(layer), 0x300000, 3)` (`17aefc3..17aefdf`, same for 0x4f0 at `17af056`, 0x4f8 at `17af0e9`). 0x300000 = 3 MiB = 48 x 128 x 128 x 4 B, the per-layer GDN recurrent state (48 matches the scan grid.x). `01895ca0.c` then copies from `ctx+0x500` (0x300 B) into per-slot storage at `*(ctx+0x4d8)`. Inferred: Tap = recurrent-state checkpoint at up to three requested absolute positions inside a prefill batch (prefix-cache style restart points). Field semantics (slot = block index, p = destination) are proven; the "checkpoint" label and the exact dump payload are inferred.

Residual: whether the kernel writes the full f32 state or part of it into p[k] is not decoded (LDS bytes, 0x1f7f bound, stride 8 per r3). Test: launch-time capture of kernarg 0x80 plus a diff of ctx+0x4e8 for `forward(n=128)` with a target at position 64.

## 2. n == 1 decode: CLOSED (stub identified)

r3 could not find the stub because the grid constant is built arithmetically. Resolved from disassembly.

Branch (decompile `018ab5c0.c:187`): step path when `n == 1 && past > 0` (iVar4 = `*(int*)(ctx+0x2e8)`) and `ctx[0x408] == 0`; `018fa150.c` sets `ctx[1] = (n < 9 && past > 0 && ctx[0x408] == 0)` for n in 1..8.

Launch at `17abcf7..17abd23`:
```
17abcf7: mov    0x2c0(%rsp),%eax        ; v (0 at 17ab66e/17abafa/17abb5f; 6 at 17abab3 when flag 0x250(%rsp) set)
17abcfe: movabs $0x100000100,%rdx
17abd08: lea    (%rdx,%rax,1),%rdi
17abd0c: add    $0xffffffffffffff28,%rdi ; grid.x = 0x28 + v, y = 1; block = 0x100000100 (256)
17abd23: call   18f4be0 <__hipPushCallConfiguration@plt>
17abd2a: test   %ebx,%ebx ; je 17ad070   ; ebx = DAT 18fe850 = FUN_019406c0 -> getenv("HALOGEN_LQ4W_SH")
17abfaf: lea    ...(%rip),%rdi          ; # 18f64f8
17ad2ef: lea    ...(%rip),%rdi          ; # 18f6500
```
Handle map: `18f64f8 RELATIVE *ABS*+0x17b2950`, `18f6500 RELATIVE *ABS*+0x17b2a30`. analysis/function-map-kernel-stubs.tsv: `018b2950 ... k_dn_conv_ab_rows<1, 1>(unsigned short const*, unsigned short const*, DnRows, unsigned short*, LqGrp, int, unsigned short const*, long)` and `018b2a30 ... k_dn_conv_ab_rows<1, 0>`. So n == 1 launches `k_dn_conv_ab_rows<1,b>`, b = HALOGEN_LQ4W_SH nonzero (ebx != 0 goes to `17abfaf`, ebx == 0 to `17ad2ef`). The first arg is the row count: the sibling site `17ac1ea..17ac247` (`lea 0x28(%rax),%ebp`, `lea -0x2(%r14),%ecx; cmp $0x5; ja`, table at 0x467e0) selects <2..7,b>, default <8,b> (tsv lines 42-55).

The recurrence after it is `k_dn_recur_step<N,b>` (handles 18f8dc0..18f8de0 map to stubs 185d5a0..185da20): launch config `17ac6e2: movabs $0x100000100,%rax`, `17ac6ec: lea -0xd0(%rax),%rdi` = grid (48,1), `17ac6f3: lea -0x80(%rax),%rdx` = block 128 (48 heads, one block each). Preceding kernels in the same function: `k_conv1d_step` (handle 18f65b8, `17ac618`) and `k_dn_step_pre` (handle 18f65c0 maps to stub 17b4110).

Meaning (stub names and argument types proven; kernel arithmetic of conv_ab_rows not decoded): single-token decode runs a causal short conv plus the a/b gate projections per row (`conv_ab_rows<R,b>`) and a per-head single-step state update (`dn_recur_step`), not the 64-token dnc chunk family. This confirms r3 section 8 (dnc = prefill) and names the decode stub. The decompile's `0x100000029` (`018ab5c0.c:526`) is Ghidra's fold of `0x28 + v`; disassembly shows v from a stack slot, so grid.x is 40 + v, not a fixed 41.

## 3. NormFold roles: PARTIAL (kernel roles proven, buffer identity bounded)

Slice: k101.s = `k_dnc_scan3x<2,4,true,false,true,1,0,1,false>`. `:12 s_load_b256 s[24:31], s[0:1], 0xa8` = n0 s[24:25], n1 s[26:27], n2 s[28:29], n3 s[30:31]; `:13 s_load_b32 s8, s[0:1], 0xc8` = w.

Per-pointer indexing:
- **n3** (`:105 s_add_u32 s68, s30, s10`, `:106 s_addc_u32 s69, s31, s11`, with `s[10:11] = s[2:3] << 2` at `:86`): n3 + 4 * s2, s2 = head index. n3 is an f32 array, one entry per head. It is also the only pointer loaded in i3 = 2 (r3 section 3).
- **n2** (`:164 s_add_u32 s9, s28, s12`, `s12` = byte offset of `s2 << 7` elements << 1): n2 + head * 256 B = a per-head vector of 128 u16.
- **n0** (`:151 v_cndmask_b32 v1, s24, v9, vcc_lo`, `:154 v_add_co_ci_u32_e64 v109, null, s25, 0, s7`, offset `s[6:7] << 1` (`:145 s_lshl_b64 s[6:7], s[6:7], 1`)): u16 element address. Host source `17af250: mov %rdx,0x5c8(%rsp)` with `n0 = *(r15+0x268)`, r15 = layer struct (`17ab5f0: add 0x5e0(%rdi),%r15`, stride 0xc68), so n0 is a per-layer u16 weight pointer, not an arena buffer.
- **n1** (`:4621 v_cndmask_b32 v1, s26, v1`, `:4623 ... s27`, repeats at `:4641,:4660,:4907`): loads `global_load_d16_b16 v13..v16, v[1:2], off offset:{0,64,128,192}` with token stride `v_mad_i64_i32 v[1:2], null, 0x1800, v1, v[110:111]` (6144 B = 3072 u16 per token). The s22 stream (`:4630`) and the n1 stream (`:4636`) use the same offset arithmetic: n1 is a second u16 row stream laid out like s22.

Purpose (proven arithmetic, `k101.s:4740-4830`): cross-lane sum by repeated `ds_bpermute_b32 v10, v11, v9` + `v_add_f32`, then `v_div_scale_f32 v10, null, s62, s62, v9` ... `v_div_fixup_f32 v9, v10, s62, v9` (sum / w, with `:205 s_cvt_f32_i32 s62, s8`), then `v_rsq_f32_e32 v9, v9`, then `v_mul_f32_e32 v14, 0x45800000, v9` (4096.0) and `v_mul_f32_e32 v34, 0x4b800000, v9` (2^24). So w = 128.0 (`17af29d: mov %ecx,0x5e8(%rsp)`; `(mode==3 ? -1 : 0) | 0x80` per r3) is the head_dim divisor of a mean of squares and the kernel computes `rsqrt(sum(x^2) / 128)` itself. This replaces the ScanArgs qinv/kinv inputs (+0x88/+0x90, gated by norm_read): the kernel normalizes q and k inline. Mode 3 gives w = -1.0; the effect (negated or inverted scale) is read off the divide only and not otherwise verified.

Host sources (r3, `01884460.c:1259,1276,1280`): n1 = DAT 18fc638, n2 = DAT 18fc660 (or hipMalloc buffer 18fc958 in mode 2), n3 = DAT 18fc7b8, arena pointers. The arena sizes come from unnamed long expressions (`local_748`, `local_790`, `local_8e8`) and were not resolved.

Bounded statement: n0 = per-layer u16 weight pointer; n1 = u16 token-strided stream (6144 B per token) read alongside the s22 stream; n2 = u16[128] per head; n3 = f32[head]; w = 128.0 mean divisor. Not provable statically: whether n2 is an input weight or scratch (no store to n2 searched), and which named buffer n1 is. Test: capture kernarg 0xa8..0xcf at launch with HALOGEN_DN_NORM_FOLD=1 and compare to qinv/kinv.

## 4. scan3 B, C, D, G: OPEN-BOUNDED

Host value source: none. Every S14 case (r3 section 4) fixes B, C, D, G as constants in the `movabs` + handle pair; no env var or data load feeds them (only A = grid.y, E = M10P, F = M10G have host selectors). A name cannot be recovered from host data flow. Code structure over the 70 scan3 slices (counts from this round):

| B | ds_store / ds_load / barrier / global_load / global_store |
|---|---|
| 2 | 11 / 44 / 11 / 9 / 2 (A=8) |
| 4 | 9 / 42 / 9 / 8 / 2 (A=4 and A=8 identical) |
| 8 | 37 / 228 / 37 / 28 / 2 (A=2,4); 44 / 354 / 51 / 42 / 9 (A=8,16) |
| 16 | 28 / 214 / 27 / 26 / 5 (A=4,8,16,32 identical) |
| 32 | 18 / 142 / 15 / 16 / 3 (A=32) |

Proven: with B = 16, A = 4, 8, 16, 32 give identical counts (1806 to 1965 total instructions, differences are address arithmetic), so A is a grid/address constant and B decides structure. Barrier count is non-monotonic in B (11, 9, 27..51, 27, 15), so B is not a plain unroll factor. Inferred, low confidence: B sets the per-block tile held in registers/LDS.

- C: `scan3<8,8,{4,8,16,32},F>`: C = 4, 8, 16 all 23/340/23/14/2 with 1843 instructions; C = 32 has ds_load 244 and 1599 instructions. Proven: C = 4..16 change only LDS offsets (agrees with r3). Inferred label: LDS row stride.
- D: `scan3<8,8,{4,8,16},{F,T}>`: same ds/barrier/global counts (23/340/23/14/2); total 1843 to 1903/1904/1906 (+60 to +63). Proven: D adds no memory operations, only ALU/scalar ones. No label fixable.
- G = 1 (`scan3<8,8,8,T,0,15,1>`): global_load 14 to 16 (+2), total 1904 to 2222 (+318), plus r3's `v_movreld_b32`/`v_movrels_b32` additions. Inferred: runtime-indexed register array with two extra loads.

A label cannot be fixed statically because the arguments are compile-time constants with no host-side name or data flow; closing them needs vendor source or a dynamic experiment (vary S14 and diff outputs).

## Residuals

1. NormFold n0, n1, n2 exact buffers (arena sizes in `01884460.c` unresolved) and whether n2 is a weight or scratch.
2. Tap payload: dump region of the kernel versus the 3 MiB state buffer.
3. scan3 B, C, D, G labels: inferred only.
4. `k_dn_conv_ab_rows<R,b>` arithmetic and the meaning of `HALOGEN_LQ4W_SH`.
5. The writer of v (`0x2c0(%rsp)`) at `17adeb4`, which sets grid.x = 40 + v on the step path.

## Reproduce

```bash
ssh ai-core 'cd ~/re-halogen && objdump -d --no-show-raw-insn --start-address=0x17ab5c0 --stop-address=0x17b0c90 binaries/flash_serve' > gdn.dis
grep -nE '0x3(c8|d0|d4|d8)\(%rsp\)|0x4(bc|c0|c4)\(%r12\)' gdn.dis      # Tap defaults and writers
ssh ai-core 'cd ~/re-halogen && objdump -R binaries/flash_serve | grep RELATIVE' > rela.txt
grep -E '18f64f8|18f6500' rela.txt                                     # 17b2950 / 17b2a30
grep -E '^018b2950|^018b2a30' analysis/function-map-kernel-stubs.tsv   # conv_ab_rows<1,1> / <1,0>
sed -n 20,100p work/decomp/flash_serve/018fa150.c                      # tap target computation
sed -n 4740,4830p k101.s                                               # rsqrt(sum/w) in the i3=1 slice
for f in work/kernels/*k_dnc_scan3I*; do echo "$f $(grep -c ds_store "$f") $(grep -c s_barrier "$f")"; done
```
