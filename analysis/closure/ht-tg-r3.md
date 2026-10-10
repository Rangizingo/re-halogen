# U10 round 3: launch site and meaning of k_ht_tg / k_ht_tg_sum

Binary: binaries/flash_serve (0.17.3). Addresses: Ghidra = ELF VA + 0x100000 where marked G; r2/objdump addresses are ELF VA (0x18xxxxx). Stub 0192bed0 (G) = 0x182bed0 (VA).

## Result

k_ht_tg<3>, k_ht_tg<4> and k_ht_tg_sum ARE launched in 0.17.3. The round-2 "no call site" finding was right about direct `call`s and wrong about the conclusion: FUN_01928680 (0x1828680) launches them by `hipLaunchKernel(&slot, ...)` with the registration slot address inlined, not by calling the stub. The stubs 0192bed0, 0192bfb0, 0192c090 are never called (proven: no rel32 call/jmp targets them), but their slots are used directly.

## Evidence: slot-direct launches in FUN_01928680

Slots (R_X86_64_RELATIVE entries, `readelf -r`): 0x18f7548 -> 0x182bd10 (k_ht_tg3<3>), 0x18f7550 -> 0x182bdf0 (k_ht_tg3<4>), 0x18f7558 -> 0x182bed0 (k_ht_tg<3>), 0x18f7560 -> 0x182bfb0 (k_ht_tg<4>), 0x18f7568 -> 0x182c090 (k_ht_tg_sum). A stub's own `lea rdi, [slot]` proves the slot-to-stub pairing, e.g. `182bf88: lea 0xcb5c9(%rip),%rdi # 18f7558` inside stub 182bed0, `182c122: lea ... # 18f7568` inside 182c090.

`objdump -d | grep` for the slot addresses finds these `lea` sites outside the stubs (all inside 0x1828680..0x1828ea8, plus the module registration code at 0x1869de5..0x1869ecb):

| site | instruction | kernel |
|---|---|---|
| 0x18287cd | `lea rdi, [0x18f7540]` then `call hipLaunchKernel` at 0x18287e4 | k_ht_rot_rows_h (slot 0x18f7540 is written by stub 0x182bc70, `182bce7: lea ... # 18f7540`) |
| 0x1828b21 | `lea rdi, [0x18f7558]`, `jmp 0x1828c4b`, `call hipLaunchKernel` at 0x1828c5b | k_ht_tg<3> |
| 0x1828c44 | `lea rdi, [0x18f7560]`, falls into the same hipLaunchKernel at 0x1828c5b | k_ht_tg<4> |
| 0x1828dc6 | `lea rdi, [0x18f7568]`, `call hipLaunchKernel` at 0x1828ddd | k_ht_tg_sum |
| 0x18289c3 | `call 0x182bd10` (stub) | k_ht_tg3<3> |
| 0x1828cac | `call 0x182bdf0` (stub) | k_ht_tg3<4> |

So the launcher is a compiler-inlined variant: two of the four GEMM kernels go through their stubs, the rest through direct hipLaunchKernel. This also explains why `axt` on the stubs shows nothing. (Method fact worth keeping: kernel-binding.md's "call the stub" model misses inlined launches; search for `lea reg,[slot]` too.)

## Launch order and selection (FUN_01928680)

1. Pre-rotation, always: `__hipPushCallConfiguration` at 0x1828737, launch of k_ht_rot_rows_h (slot 0x18f7540) at 0x18287e4. Grid x = ((arg4>>7) * n + 7) >> 3 (0x18286ec-0x1828716), block 256 (`rdx = 0x100000100`, 0x18286ce). Inferred: each 128-wide row block is handled by 32 lanes, 8 blocks per workgroup.
2. Bits switch: `mov ebx, [obj+0x48]` (0x1828951, 0x18289e1); `cmp ebx, 3` (0x1828975, 0x1828a01). ==3 picks the `<3>` kernel, else `<4>`.
3. Shape A (tg3, 512-thread block, stub calls): taken when 0x18fe6a8 guard set, 0x18fe6a0 nonzero, `(r9 & 1)==0`, and first dim >= 0x18fe6a0 (0x18288ee-0x182891a). Grid (ceil(dim/128), ceil(r8/256)), block 0x200 (`lea rdx,[rbp+0x100]`, 0x182895e).
4. Shape B (tg, 256-thread block, slot launches): every other case (the `jne 0x18289cd` / `jl 0x18289cd` / `je 0x18289cd` exits). Grid (ebp = ceil(arg5/64), r9 = arg4>>7), block 256 (0x18289cd-0x18289fc). This corrects round 2, which assigned shape B to tg3: shape A is tg3, shape B is k_ht_tg.
5. Split-K reduce: `cmp dword [rsp+0xc], 2` / `jl 0x1828de6` (0x1828cba-0x1828cbf). When the split count (esi at 0x18288c0, `ceil(ceil(K/128)/ceil(...))`, capped by the scratch size at 0x18fe368) is >= 2, the GEMM kernel is given the f32 scratch pointer (`r15 = [0x18fe368]` via `cmovge`, 0x18288d7) and k_ht_tg_sum runs after it (0x1828dc6). Below 2, r15 = 0, no scratch, no sum kernel (inferred: GEMM then writes final output directly).
6. Gate (unchanged from round 2): caller 01917610.c:73 and :96 call FUN_01928680; HALOGEN_HT_TG_DEC default 0.

## What the kernels compute

k_ht_tg_sum (work/kernels-dequant/_ZN7halogen12_GLOBAL__N_111k_ht_tg_sumEPKfiPKDF16_Ptii.s, 0x22e600): signature `(const float*, int, const _Float16*, unsigned short*, int, int)`.
- Loop over `s6 = kernarg+8` partials: `s_cmp_lt_i32 s6, 2` (0x22EA1C), `global_load_b128` per partial (0x22EA48, 0x22EAC8), then `v_add_f32` chain (0x22EB18-0x22EC7C): sum of split-K f32 partials, 4 floats per lane.
- Then a 128-point butterfly: `v_mov_b32_dpp ... row_xmask:8` (0x22EC8C), `v_cndmask_b32 -v,v` sign flips, `v_permlanex16_b32 ... 0xfedcba98` (0x22ECF8-0x22ED50), `v_add_f32`. 32 lanes x 4 values = 128.
- `v_mul_f32 v, 0x3db504f3` (0x22ED44-0x22EDB8): 0x3db504f3 = 0.088388 = 1/sqrt(128), the orthonormal Hadamard-128 scale.
- `v_fma_mix_f32` with f16 operands from a `global_load_b64` (0x22EB7C, 0x22ED88-0x22EDC4): multiply by per-group f16 scale.
- Output rounding `v_add3_u32 ..., 0x7fff` with `v_and_b32 1` (0x22EDFC-0x22EE28) = round-to-nearest-even f32 to bf16, then `global_store_b64` (0x22EE44) of 4 bf16 into the `unsigned short*`.
Verdict on meaning (decoded from code, not named by the binary): split-K reduction plus inverse 128-point Hadamard rotation plus f16 scale, written as bf16. The Hadamard step is the counterpart of k_ht_rot_rows_h that the launcher runs first on the input.

k_ht_tg<B> / k_ht_tg3<B> (analysis/kernels-moe.md s9: 80 WMMA f16, 256 `v_sad_u8`, DPP): quantized-weight (`unsigned const*`, B = 3 or 4 bits from `obj+0x48`) times the rotated f16 activations (output of k_ht_rot_rows_h), f16 scales, last argument `float*` = split-K partial buffer. So: a small-batch HT (Hadamard-transformed, B-bit) dense GEMM, rotate in, GEMM out with split-K partials, rotate-out and reduce in k_ht_tg_sum. tg3 is the larger-tile (512-thread, 128x256) variant, tg the 256-thread variant (grids above). Role "dense trunk linear for n < 9" follows from the gate at 01917610.c:61-77 (round 2).

## Still not proven

- Expansion of "tg" (inferred only; do not assert "token gate" or "tied gate").
- Which arguments of FUN_01928680 are K, N, flags; the exact bound semantics of 0x18fe6a0 (inferred HALOGEN_HT_TG_M128, round 2) and `[obj+0x378]`.
- Whether the pointer passed as kernel arg 1 in the sum launch (`[rsp+0x100]`, the 0x18fe368 scratch) is the same buffer the GEMM writes: inferred from the shared register (r15 / [rsp+0x100]), not traced through the kernel's store.
- The kernel-side math of tg/tg3 was not re-read this round (WMMA tile layout), only the host side and tg_sum.

## Verdict

CLOSED for U10's stated gap: launch sites of k_ht_tg<3|4> (0x1828b21, 0x1828c44, via hipLaunchKernel on slots 0x18f7558/0x18f7560) and k_ht_tg_sum (0x1828dc6, slot 0x18f7568) found, all in FUN_01928680, and tg_sum's computation decoded. PARTIAL only on the cosmetic items under "Still not proven" (name expansion, argument naming, tg tile math).

## Reproduce

```bash
cd /Users/peterb/Documents/AI/re-halogen
# slot -> stub relocations
ssh ai-core 'cd ~/re-halogen && readelf -r binaries/flash_serve | grep -i "182bd10\|182bdf0\|182bed0\|182bfb0\|182c090"'
# every code reference to the slots and stubs (stub calls and slot leas)
ssh ai-core 'cd ~/re-halogen && objdump -d --no-show-raw-insn binaries/flash_serve | grep -E "18f754[0-9a-f]|18f755[0-9a-f]|18f756[0-9a-f]|182bd10|182bdf0|182bed0|182bfb0|182c090"'
# launcher disassembly
ssh ai-core 'cd ~/re-halogen && r2 -q -e scr.color=0 -c "pd 230 @ 0x1828680; pd 70 @ 0x1828cb1" binaries/flash_serve'
# tg_sum kernel
grep -n "3db504f3\|row_xmask\|permlanex16\|0x7fff\|global_store_b64" work/kernels-dequant/_ZN7halogen12_GLOBAL__N_111k_ht_tg_sumEPKfiPKDF16_Ptii.s
```
