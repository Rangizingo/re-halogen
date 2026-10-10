# Kernel binding: host launch stubs to GPU kernels

Output table: analysis/function-map-kernel-stubs.tsv (stub_addr, mangled_kernel, demangled_kernel, family), 1,738 rows, sorted by stub address. Addresses are Ghidra addresses (ELF VA + 0x100000).

## Result

| Item | Count |
|---|---|
| `__hipRegisterFunction` calls found in the decompile (4 functions) | 1,738 |
| Stub addresses resolved to a function entry | 1,738 of 1,738 (proven) |
| Distinct stubs / distinct kernel names | 1,738 / 1,738 (strict 1:1, no duplicate on either side) |
| Resolved stubs whose function-map role is gpu-dispatch | 1,597 |
| Resolved stubs whose role is hip-runtime (139) or crt-runtime pad_stub (2) | 141 (role mislabels, see below) |
| gpu-dispatch functions NOT bound to a kernel | 83 of 1,680 (1,680 - 1,597) |
| Kernel names with an exact slice file in work/kernels, kernels-dequant, kernels-spec | 1,488 of 1,738 (21 slice names have no registration) |

Registration blocks (grep -c __hipRegisterFunction over every work/decomp/flash_serve/*.c; no other _INIT_N holds any): 01968680 = 1,484, 019e0890 (register_dnc_kernels) = 175, 019f3590 (hip_register_fatbin_kernels_attn) = 62, 019e52c0 (register_dn_fused16_kernels) = 17. Total 1,738. Proven.

## Method

1. Parse each call with a regex. Arg 2 is `&PTR_FUN_<addr>` (1,668), `&DAT_<addr>` (14) or `&LAB_<addr>[_N]` (56, of which 31 carry a decimal offset suffix: slot = hex + N). Arg 3 is the mangled kernel name.
2. The argument is the address of a pointer slot in .data.rel.ro, not the stub. The slot is zero in the file image (PIE). The stub address is the addend of the `R_X86_64_RELATIVE` entry for that slot in `.rela.dyn` (DT_RELA 0x5138, size 0xd548, 2,199 relative relocs in binaries/flash_serve). Ghidra address = ELF VA + 0x100000 (checked: ELF entry 0x172b880 vs map entry 0182b880).
3. Join the relocation target against index.tsv and function-map.tsv by address. All 1,738 land on a `FUN_<addr>` entry (proven: 0 misses).
4. Demangle with `c++filt`. Family = kernel base identifier minus `k_`, first token (two tokens for ht, dnc, attn, linear, exp, route, dn, hc, deq, vis, smpg, i4r, pen); namespace q4moe forced to `q4moe`. This grouping is a heuristic, not a source name (inferred).

Worked case (proven): the call for `k_sample_f32` passes `&PTR_FUN_019f63c0`. The relocation at VA 0x18f63c0 has addend 0x1783720, which is Ghidra 01883720. index.tsv lists `01883720 FUN_01883720 128` bytes. The stub body (e.g. 0190cf40, `k_rope_blocks_yarn`) only packs its arguments into a pointer array and ends in `__hipPopCallConfiguration`.

## Examples (15, evenly spaced through the table)

| stub | family | kernel (demangled, truncated) |
|---|---|---|
| 01883720 | sample | halogen::k_sample_f32(float const*, halogen::SampleRow const*, int*, float*, float*, int) |
| 018cc500 | ht_gemv | k_ht_gemv_dn_ts<4, 8, 1, false> |
| 018d2200 | ht_moe | k_ht_moe_dn<3, 8, 8> |
| 01905f80 | embed | k_embed_image(unsigned short*, unsigned short const*, int const*, int, int, int) |
| 01924130 | linear_q4 | k_linear_q4_mt<16, 2, 2, 128> |
| 01930bc0 | ht_lin | k_ht_lin<3, 8, false, true, true, false> |
| 01937d40 | lq8w | k_lq8w<5, 32, 8> |
| 0193ce60 | lbfw | k_lbfw<1, 2> |
| 01942b10 | route_logits | k_route_logits_rows<4, 6> |
| 0194c290 | lq8s32 | k_lq8s32<2, 32, 1, true> |
| 019530a0 | lq4w | k_lq4w<4, 32, 2, false, 1, 0> |
| 0195a4a0 | lq4w | k_lq4w<2, 32, 1, true, 3, 0> |
| 01961940 | lq4w | k_lq4w<8, 32, 1, false, 5, 1> |
| 019d0a40 | dnc_scan3 | k_dnc_scan3<8, 8, 8, true, 8, 15, 0> |
| 019de780 | dnc_ut5 | k_dnc_ut5<true, true, 1, true, false, false> |

## Kernel families by count (top 24 of 141)

lq4w 488, ht_lin 97, ht_gemv 94, linear_q4 72, dnc_scan3 70, lq8s32 64, lq6k 64, lq6w 56, ht_moe 48, lq8w 40, exp_gu 36, route_logits 32, attn_qs 32, attn_fa2 26, dnc_ut5 24, q4moe 22, deq_q4cp 22, dnc_scan3x 21, lbfw 18, dn_fused16 17, dn_conv 17, i4r_rows 16, hc8 16, hc6 16.
Full list: `cut -f4 analysis/function-map-kernel-stubs.tsv | sort | uniq -c | sort -rn`. 1,716 kernels are in namespace `halogen` (most in the anonymous namespace), 22 in `q4moe` (proven from the names).

## Unresolved

- Registered stubs without a kernel: none.
- 83 gpu-dispatch functions are not registered (proven: no registration call resolves to their address). Their function-map labels (launch_helper 11, hip_checked_op 10, set_launch_cfg_arm 8, launch_gemm_select 4, launch_cfg 4, kernel_launch_wrapper 4, vision_kernel_launch_push 2, matmul_bf16_plan_run 2, others) and the decompile of 01895930 and 01920310 show host launch helpers that call `__hipPushCallConfiguration`, not stubs (inferred). Which stubs each helper calls is not extracted: index.tsv holds callee counts only, and the decompile of these helpers ends at the no-return `__hipPushCallConfiguration`.
- 141 resolved stubs carry role hip-runtime (139) or crt-runtime pad_stub (019f0b50, 019f0c50). The registration proves they are kernel stubs. function-map.tsv roles were NOT edited (reported only).
- 21 slice file names and 250 registered kernel names do not pair by exact file name (cause not checked per file; likely truncated or differently scoped slice names, inferred).

## Runtime selection

Proven: nothing picks a stub by kernel name at runtime. Each template instantiation is its own stub and its own registered kernel (`k_ht_moe_dn2w<3,64,6,0>` and `<4,64,6,0>` are separate rows), so selection happens in host code by choosing which stub to call.

Inferred from 01920310 (launch_gemm_select_f): a switch on `param_5 - 1` (valid 0..7) with grid size computed from `param_4` thresholds (`(n+7)>>3`, `(n+15)>>4`, `(n+3)>>2`) picks the launch config per case. The case-to-stub edges are not extracted, so the selection map from template arguments to stub address stays open.
