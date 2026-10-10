# U11 round 4: probe regions per N, the supplier of N, q5_0 / q5_1 / q8_0 bias

## Scope

Closes the three items left by `analysis/closure/dnftaps-r3.md`: (1) which code region of `k_dn_fused16` each probe N=1..11 timestamps, (2) who supplies N to the launcher at Ghidra `019e2a20`, (3) the dequant bias for q5_0, q5_1, q8_0. Static analysis only. Nothing was run.

Address conventions: Ghidra = r2/objdump + 0x100000. Text segment file offset = vaddr - 0x1000 (`readelf -lW`: `LOAD 0x172a880 0x172b880`, so the scans below add 0x1000). Slice names: `dnf<N>` = `work/kernels-spec/*k_dn_fused16ILi<N>ELb0ELb0ELb0E*.s`, line numbers are `grep -n` numbers of the slice. `DIS53` = `kernels-disasm/codeobj_53000.s`. Inference is labelled "inferred".

## Verdict: PARTIAL

- Item 2 CLOSED as far as static analysis goes: N is a stack argument of `019e2a20`; nothing in the binary calls it; `HALOGEN_DN_PROBE` does not feed it.
- Item 3 CLOSED: q8_0 bias 0, q5_0 bias -16, q5_1 no constant (per-block f16 min).
- Item 1 PARTIAL: N=7..11 map to concrete barrier-delimited regions (high confidence for 8..11). N=1..6 map to concrete instruction ranges whose content is listed, but their logical names are inferred, and N=1, 5, 6 are loop-edge intervals whose exact region depends on the tap loop.

## 1. Probe regions per N (item 1)

### 1.1 How a probe works (evidence)

- Each variant N=1..11 has 3 `s_sendmsg_rtn_b64 ..., sendmsg(MSG_RTN_GET_REALTIME)` reads. The interval is `last read into s[28:29]` to `read into s[54:57]`, accumulated into `s6` at the accumulate pair. dnf1:399-402: `s_sendmsg_rtn_b64 s[54:55]` / `s_sub_i32 s3, s6, s28` / `s_waitcnt lgkmcnt(0)` / `s_add_i32 s6, s3, s54`, i.e. `s6 += t_end - t_start`. Epilogue stores `s6` as one u64 to `Py[N-1]` (dnf1:3644-3648 `s_load_b64 s[0:1], s[0:1], 0x40`, `v_mov_b32_e32 v0, s6`, `global_store_b64 v2, v[0:1], s[0:1]`).
- For N>=2 the entry read at line 195 is overwritten by the second read into the same `s[28:29]` (dnf2:195 and dnf2:392 both `s_sendmsg_rtn_b64 s[28:29]`), so only the second read is the interval start. Only N=1 uses the entry read (dnf1:196) as start.
- Reads are guarded by `s_cbranch_vccnz` on a flag derived from `s38` (dnf6:764 `s_and_not1_b32 vcc_lo, exec_lo, s38`, dnf6:765 `s_cbranch_vccnz 65357`), so a variant skips its clock reads on paths where the flag is off. `s38` source not traced.
- `s6` is the 64-bit accumulator over the tap loop. dnf1:2988 `s_cbranch_scc1 61697` at 0x29E28 branches back (61697-65536 = -3839 dwords, target 0x29E2C - 0x3BFC = 0x26230 = dnf1 line 386, the loop head right before the accumulate at 399). So the `s_cmp_lg_u32 s13/s14` tap tests (dnf1:2985, 2987) feed a back-edge to the loop head, and the accumulate runs once per tap iteration. Inferred: the loop is the 3-tap loop of round 3.

### 1.2 Region landmarks of the production kernel (N=0)

Barriers at dnf0 lines 738, 1505, 2001, 2271, 2836 (grep `s_barrier`). Instruction mix per barrier-delimited segment (python over `s_barrier` cuts, counting instructions that start with `ds_load`, `ds_store`, `global_load`, `global_store`, `v_wmma`, `v_exp_f32`, `v_rsq`, `v_fma*`):

| Seg | dnf0 lines | ds_load | ds_store | g_load | g_store | v_wmma | v_exp | v_rsq | v_fma* | Reading (inferred) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 1-738 | 0 | 5 | 76 | 0 | 0 | 0 | 3 | 0 | input load, tap setup, 3-way v_rsq (norm) with 8 `v_permlane` reductions |
| B | 739-1505 | 49 | 18 | 4 | 0 | 8 | 18 | 0 | 36 | first WMMA group plus 18 `v_exp_f32` |
| C | 1506-2001 | 58 | 10 | 0 | 0 | 0 | 0 | 0 | 151 | FMA-only, no WMMA, no exp |
| D | 2002-2271 | 24 | 8 | 0 | 0 | 4 | 0 | 0 | 2 | small WMMA group, 8 `v_permlane`, 27 cvt |
| E | 2272-2836 | 59 | 0 | 0 | 8 | 20 | 1 | 0 | 72 | main WMMA group (20), 116 `v_perm`, 36 `v_permlane` |
| F | 2837-3610 | 0 | 0 | 0 | 256 | 5 | 0 | 0 | 59 | scale (0x3b800000 muls) and output stores (256 `global_store_b32`) |

The names in the last column are inferred from the mix; the segment boundaries are measured.

### 1.3 Region per N (clock pair lines, then what lies between, from the slice)

Measured per variant with python over the slice: start line, end line, instructions between, and counts in between. Segment letter uses that variant's own barrier lines (dnf5 barriers 624/1391/1894/2161/2835, dnf8 747/1522/2035/2302/2976, dnf11 767/1538/2037/2255/2800, and so on; each from `grep -n s_barrier`).

| N | start..end (dnf lines) | seg | insns | content between | Region (inferred name) | Confidence |
|---|---|---|---|---|---|---|
| 1 | 196 (entry) or 2861 (loop-back) .. 399 | A / F-head | 202 | 4 `global_load`, no ds, no WMMA | prologue wait on the initial loads plus tap-loop turnaround (2861 is re-read each iteration, back-edge 2988 to 386) | medium |
| 2 | 392 .. 416 | A | 23 | 3 `ds_store` (403, 405, 411), 0 loads | LDS stage-in of the loaded tile | medium |
| 3 | 405 .. 543 | A | 137 | 8 `global_load`, 0 ds | second batch of global loads (8) | medium |
| 4 | 538 .. 657 | A | 118 | 2 `v_rsq_f32` (640, 643), `v_mul_f32` x5, 1 `ds_store` (653), 0 loads | norm: rsq, scale, LDS store of the scale | medium |
| 5 | 2871 .. 2981 | F | 109 | 1 `ds_store` (2976), back-edge branches (2874-2910), no WMMA | post-norm tail; in this variant the compiler laid the rsq block at 2852-2858 after the tap tests (2837, 2839), so same logical stage as N=4 placed at the end | low |
| 6 | 766 .. 768 | B | 1 | only `s_branch 65355` (767) | loop back-edge gap just after barrier 1 (dnf6 barrier at 620); not a compute region | low |
| 7 | 759 .. 820 | B head | 60 | 4 `global_load`, 0 WMMA | B-segment head, before the first WMMA group | medium |
| 8 | 812 .. 1531 | B | 718 | 49 `ds_load`, 18 `ds_store`, 8 `v_wmma`, 18 `v_exp_f32`, 35 `v_fma*`, 1 barrier | segment B (WMMA group 1 plus exp) | high |
| 9 | 1525 .. 2027 | C | 501 | 58 `ds_load`, 10 `ds_store`, 0 WMMA, 0 exp, 158 `v_fma*`, 1 barrier | segment C (FMA-only) | high |
| 10 | 2022 .. 2251 | D | 228 | 24 `ds_load`, 8 `ds_store`, 4 `v_wmma`, 1 barrier | segment D (small WMMA group) | high |
| 11 | 2257 .. 2819 | E to F head | 561 | 59 `ds_load`, 25 `v_wmma`, 72 `v_fma*`, 8 `global_store`, 1 barrier | segment E (main WMMA group) plus 19 lines of F | high |

Probe-slot map (r3 offsets, unchanged): `Py[N-1]`. N=8, 9, 10, 11 partition B..E, so `Py[7..10]` together cover the compute body; N=1..4 split segment A. N=12 (r3) is the per-workgroup start time + `HW_ID1` trace and has no interval.

Reproduce:

```bash
cd /Users/peterb/Documents/AI/re-halogen/work/kernels-spec
# clock reads, accumulate pairs, guards for N=8
grep -n -E 's_sendmsg_rtn_b64|s_sub_i32 s[0-9]+, s6, |s_add_i32 s6,|s_cbranch_vccnz' *k_dn_fused16ILi8ELb0ELb0ELb0E*.s | cut -c1-110 | head
# barrier lines for N=0 and N=8
for n in 0 8; do grep -n s_barrier *k_dn_fused16ILi${n}ELb0ELb0ELb0E*.s | cut -d: -f1 | tr '\n' ' '; echo; done
```

## 2. Who supplies N (item 2)

Evidence, all from `binaries/flash_serve` (r2 addresses; add 0x100000 for Ghidra):

1. N is a stack argument. `0x18e2a20` prologue: `sub rsp, 0x3f8`, then `mov r11d, dword [rsp + 0x438]` / `test r11d, r11d` / `jle 0x18e2bea` (count), `cmp qword [rsp + 0x448], 0` / `je 0x18e2b73` (Py), `mov r10d, dword [rsp + 0x450]` / `dec r10d` / `cmp r10d, 0xb` / `ja 0x18e2bea` / `jmp rcx` through the table at 0x48f9c (N). Decomp: `019e2a20.c` `switch(in_stack_00000028)` cases 1..0xc, `if (in_stack_00000020 == 0) FUN_019e4250();` for the N=0 case. Launch config in every case: `__hipPushCallConfiguration(0x100000030,1,0x100000100,1,0)` = grid 48 x 1, block 256 x 1.
2. No caller. Scan of the whole text segment (python, `E8/E9 rel32`, `0F 8x rel32`, rip-relative `lea`) for target 0x18e2a20: no hits. Same scan for 0x18e4b10 (the wrapper `019e4b10`, 192 B, which only calls `019e4250`): no hits. `readelf -rW binaries/flash_serve | grep -i -E "18e2a20|18e4b10"`: no output, so no RELATIVE relocation (no function-pointer or vtable entry) either. Control, same scan: target 0x18e4250 hits at 0x18e2be5, 0x18e4bc0, 0x18e4d4f; target 0x18e4bd0 hits at 0x17af4a9, 0x17aff00 (objdump agrees). Conclusion: `019e2a20` and `019e4b10` are unreferenced in this build.
3. The live path never passes an N. `019e4bd0.c` ends with a single call `FUN_019e4250(...)` (17 arguments, no N); `019e4250.c` has no switch and only reads `HALOGEN_DN_FUSED_GRAM` and `HALOGEN_DN_FUSED_PAIR` (`getenv`, `== '1'`) to choose between ILi0 `Lb` variants (the `ILi0ELb{0,1}ELb{0,1}ELb{0,1}` instances registered in `019e52c0.c:13-39`). So in the shipped runtime only N=0 kernels are launched.
4. `HALOGEN_DN_PROBE` is not N. It is read once (`019cf410.c:8` `getenv("HALOGEN_DN_PROBE")`, stored by `019ca280.c:42` `DAT_019ff760 = FUN_019cf410()`). The only consumers of `DAT_019ff760` (r2 0x18ff760, rip-relative scan) are three loads at r2 0x18cc8fa, 0x18cc92c, 0x18cd3df and the one store at 0x18cee9f, all in the GDN step launcher `019ca280`. That launcher uses grid `0x200000030` and block `0x100000080`, not the `0x100000030 / 0x100000100` of `019e2a20`, and compares the value with `cmp ebp, 5` / `cmp ebp, 4` at r2 0x18cc95b and 0x18cc964 (a runtime kernel selector, not a template argument). `019ca280.c:451` `DAT_019ff760 - 4U < 3` is the same 4..6 test.

Verdict for item 2: N has no producer in this binary. `k_dn_fused16<1..12>` and its launcher `019e2a20` are dead code in the shipped `flash_serve` 0.17.3 (the kernel handles are only registered, `019e52c0.c`). The fatbin still carries the instrumented kernels; a build or harness that calls `019e2a20` with N and a `Py` buffer would produce the probe output. Inferred: the probe is a developer-only timing decomposition for the DN fused kernel, separate from `HALOGEN_DN_PROBE` (which steers GDN step kernels).

Reproduce:

```bash
ssh ai-core 'cd ~/re-halogen && r2 -q -e scr.color=0 -c "pd 22 @ 0x18e2a20" binaries/flash_serve'
ssh ai-core 'cd ~/re-halogen && readelf -rW binaries/flash_serve | grep -i -E "18e2a20|18e4b10"; echo done'
ssh ai-core 'cd ~/re-halogen && objdump -d --no-show-raw-insn binaries/flash_serve | grep -E "(call|jmp).*(18e2a20|18e4250|18e4bd0)"'
```

## 3. q5_0, q5_1, q8_0 bias (item 3)

The three formats have no `k_dn_fused16` instance (all 17 instances are `<N,b,b,b>` with no format parameter, r3 section 1) and no GEMV or `k_deq_*` kernel (`closure/dequant-host.md:45`). Where their decode lives:

### 3.1 Embedding-table gather: `k_gather_ggml<2|3|4>`

The decoder is the `ngram_embedding` row gather. Host proof of the mapping (r2, in the `upload_model_weights` region):

- Error string at file 0x314f2, loaded by `lea -0x175da0f(%rip),%rsi` at r2 0x178eefa: `ngram_embedding: expected fp8g, iq4nl, q8_0, q5_0 or q5_1, got `.
- r2 0x178e2e4-0x178e357: `mov edi, dword [rbx + 0x10]` (dtype), `lea ecx, [rdi - 0x14]`, `cmp ecx, 3` / `jae 0x178e5e4` (accept dtypes 0x14..0x16 = 20, 21, 22). Row bytes in `ebp`: `cmp edi, 0x15` / `mov ebp, 0x78` / `mov ecx, 0x6e` / `cmove ebp, ecx` / `cmp edi, 0x14` / `mov ecx, 0xaa` / `cmove ebp, ecx`. Gather type code in `eax`: `cmp edi, 0x15` / `setne al` / `add eax, 3` / `cmp edi, 0x14` / `mov edx, 2` / `cmove eax, edx`. So code 2 = dtype 20 (q8_0, 0xaa = 170 B per 160-element row = 5 blocks x 34 B), code 3 = dtype 21 (q5_0, 0x6e = 110 B = 5 x 22 B), code 4 = dtype 22 (q5_1, 0x78 = 120 B = 5 x 24 B). The code is stored at r2 0x178e6b0 `mov dword [rcx + 0x79c], eax`. Same site: dtype 13 (iq4nl) stores 1 (0x178e671-0x178e6a9, `imul rdx, rax, 0x5a`), dtype 10 (fp8g) stores 0 (0x178e637-0x178e641).
- Launch switch r2 0x17ef0df-0x17ef112: `mov eax, dword [r12 + 0x79c]` / `dec eax` / `cmp eax, 3` / `ja 0x17ef438` / `jmp rax` via table 0x468bc (python: entries for codes 1..4 -> 0x17ef114, 0x17ef36f, 0x17ef1dd, 0x17ef2a6; default 0x17ef438 uses `movss xmm0, [rdx + 0x788]`, the fp8 float scale). The `lea rdi, [slot]` in each case (r2 0x17ef1b8 slot0, 0x17ef413 slot1, 0x17ef281 slot2, 0x17ef34a slot3, 0x17ef4fc slot4) names the handle table at 0x18f6e00, whose RELATIVE relocations are 17f1a30, 17f1ab0, 17f1b30, 17f1bb0, 17f1c30 (`readelf -rW`), i.e. `k_gather_iq4nl`, `<2>`, `<3>`, `<4>`, `k_gather_fp8` (names: `analysis/function-map-kernel-stubs.tsv:328-332`). So `<2>` = q8_0, `<3>` = q5_0, `<4>` = q5_1. This corrects r3's unsure "0xa0 / 0x78 B row" reading: it is 160 elements per row, and 0x78 = 120 B is exactly 5 q5_1 blocks.

### 3.2 Decode constants (DIS53)

| Format | Kernel (DIS53 header line) | Evidence | Bias |
|---|---|---|---|
| q8_0 (20) | `k_gather_ggml<2>` (2405553) | 2405577 `s_mul_i32 s1, s11, 0xaa` (170 B row); 2405635 `global_load_i8 v13, v[13:14], off offset:2` (signed byte after the 2 B scale); 2405743 `v_bfe_i32 v3, v3, 0, 8`, 2405749 `v_cvt_f32_i32_e32 v3, v3`, 2405751 `v_mul_f32_e32 v3, v6, v3` (scale * int8, nothing added) | 0 |
| q5_0 (21) | `k_gather_ggml<3>` (2405809) | 2405834 `s_mul_i32 s1, s11, 0x6e`; 2405879 `v_mul_i32_i24_e32 v3, 22, v3` (22 B block) followed by `global_load_b32 v9, v3, s[2:3] offset:2` (qh word at +2); 2405902 `global_load_d16_u8 v5, ... offset:6` and 2405892 `... offset:-10` (qs bytes for the low and high element halves); 2405906 `v_lshlrev_b32_e32 v8, 4, v6` and 2405856 `v_and_b32_e32 v3, 16, v8` (qh bit moved to bit 4); **2405860 `v_add3_u32 v3, v3, v4, -16`** (5th bit + nibble - 16); 2405863 `v_cvt_f32_i32_e32 v3, v3`; 2405865 `v_mul_f32_e32 v3, v5, v3` (times f16 scale) | -16 |
| q5_1 (22) | `k_gather_ggml<4>` (2405945) | 2405970 `s_mul_i32 s1, s11, 0x78`; 2406012 `v_mul_i32_i24_e32 v3, 24, v3` (24 B block); 2406013 `global_load_b32 v9, v3, s[2:3] offset:4` (qh at +4); 2406044 `global_load_d16_b16 v6, v[3:4], off` (d) and 2406074 `global_load_d16_b16 v3, v[3:4], off offset:2` (m); 2405992 `v_and_or_b32 v3, v8, 16, v7` (merge 5th bit, 0..31); 2405995 `v_cvt_f32_ubyte0_e32 v3, v3`; **2405998 `v_fmac_f32_e32 v4, v5, v3`** (acc += d * q, accumulator holding m) | no constant; zero point is the per-block f16 min m |

Register roles in q5_1 (v4 = m, v5 = d) are inferred from the two f16 loads at 2406044 and 2406074 and the `v_fmac` operand order; the +0/+2 offsets match the ggml `block_q5_1 { f16 d; f16 m; u32 qh; u8 qs[16]; }` layout.

### 3.3 Cross-check on the host (repack)

`work/decomp/flash_serve/019c37a0.c:533` (`repack_q5_0_rows`, `analysis/function-map.tsv:2675`): error text ` have a scale d whose -16 * d is not an fp16 (non-finite or out of range); refusing rather than rounding it`, built at :524 with ` Q5_0 blocks in repacked row `. So when GGUF q5_0 (ggml type 6, `(param_5 & 0xfffffffe) == 6` at :318) is repacked, the repacker folds the -16 into an f16 min equal to -16 * d. This agrees with the kernel constant -16 and with `hgn-format.md:104-105` (q5_0 22,528 B for [128,256] = 20,480 + 2,048; q5_1 24,576 = 20,480 + 4,096, i.e. the q5_1 form carries d and m). Inferred: the consumer of the repacked tensors on the matmul path (dequant to bf16 outside the handle list) was not traced (open in `dequant-host-r2.md:67`).

Reproduce:

```bash
cd /Users/peterb/Documents/AI/re-halogen
grep -n -E 'v_add3_u32 v3, v3, v4, -16|v_fmac_f32_e32 v4, v5, v3|v_bfe_i32 v3, v3, 0, 8' kernels-disasm/codeobj_53000.s | head
ssh ai-core 'cd ~/re-halogen && r2 -q -e scr.color=0 -c "pd 40 @ 0x178e2e4" binaries/flash_serve'
```

## Residuals

1. N=1, 5, 6 are loop-edge intervals; their logical names (prologue/turnaround, post-norm tail, back-edge gap) are inferred from instruction content only. A dataflow of `s38`, `s53`, `s46` would settle them. N=2/3/4/7 names are medium confidence (the content between the clock reads is listed, the naming is not proven).
2. The flag `s38` that enables the clock reads was not traced to a null test of kernarg `Py` (0x40).
3. The consumer of repacked q5_0 / q5_1 / q8_0 tensors on the matmul path (not the embedding gather) is still unlocated; only the embedding gather decode and the repack constants are proven.
4. `019e2a20` has no static caller; a runtime-only caller is excluded only by absence of references (no call, jmp, lea, or relocation).
