# U19 HT weight decode, confirmed on real data

Date 2026-10-10. Data: HTTP range reads (Peter approved fetching what is needed) of the vendor checkpoints `qwen38-flash-next-ht43.hgn` (HT, 57,618,253,792 B, 2,015 tensors, of which 409 are HT: 361 at qparam 4616 = 4 bit, 48 at qparam 4614 = 3 bit, counted from its table) and `qwen38-flash-next-w4b.hgn` (q4c references) from `peonist-ai/halogen-qwen3.8-flash-next`. Only the tables and the test slices were fetched (`scripts/fetch_ht_testdata.sh`, about 487 MB). Reproduce: `bash scripts/fetch_ht_testdata.sh <dir> && python3 -I scripts/ht_decode.py <dir>`. Measured output on this Mac, 2026-10-10:

```
layers.3.self_attn.k_proj 4-bit [512,2560] overall 0.9924 row min 0.9914 median 0.9924
layers.3.mlp.shared_expert.down_proj 4-bit [2560,640] overall 0.9931 row min 0.9890 median 0.9932
layers.1.mlp.experts.down_proj expert 0 3-bit [2560,640] overall 0.9839 row min 0.9798 median 0.9839
layers.1.mlp.experts.down_proj expert 1 3-bit [2560,640] overall 0.9848 row min 0.9819 median 0.9848
negative control: expert 1 HT vs expert 0 reference overall -0.0008 row min -0.1435 median -0.0011
```

Tile size in the kernels: `k_ht_deq_t<3>` multiplies the tile index by 0x60 = 96 B (`work/kernels-dequant/_ZN7halogen12_GLOBAL__N_110k_ht_deq_tILi3EEEvPKjiiPt.s:268`), `k_ht_deq_t<4>` shifts it left by 7 = 128 B (`..ILi4EEEvPKjiiPt.s:276`), matching 8*bits words per 16x16 tile.

The reference is a different 4-bit quantization of the same base weights, so about 0.99 (not 1.0) is the expected ceiling. Below is the decode agent's report, re-run and checked by the session author (scores above reproduced exactly). Script paths in it refer to the agent's scratch copies; the repo copy is `scripts/ht_decode.py` (the unified 3 and 4 bit form).


Tensor: `layers.3.self_attn.k_proj.weight`, [512, 2560], ht43 checkpoint (4 bit). Reference: same tensor from w4b (q4c). All numbers below are MEASURED on this one tensor unless marked inferred.

Script: `scripts/ht_decode.py` (the agent's search scripts and log stayed in its scratch area).

## Result

- Overall Pearson, decoded HT vs q4c reference: 0.9924 (measured).
- Per row: min 0.9914, p5 0.9919, median 0.9924, max 0.9933 (measured).
- Relative L2 error ||W_ht - W_ref|| / ||W_ref|| = 0.123 (measured); LS gain ref ~ g * W_ht: g = 0.994 (measured).
- Rotated domain, per 16x16 tile, cols 3..7 of each lane group: rho 0.993 in the window-offset scan (measured). Same figure after the fix for all 8 columns.
- Ablations (measured Pearson): without su -0.02; with abs(sv) 0.02; without sv 0.02. All three are needed, and sv carries sign.
- Residual 0.8% of correlation is the 4-bit error of the q4c reference and of HT itself (two independent 4-bit quantizations of one base weight). Not separable with this data.

## The decoder

```
file layout    : 16x16 tiles, 128 B (32 u32 words) each, tile ids in 128-row blocks
tile id        = (row//128) * (8 * K/16) + (col//16) * 8 + (row%128)//16
lane t (0..31) = 16*((col//8)%2) + row%16        holds u32 word t of the tile; 8 weights, cols 8h..8h+7
pair           = (word[(t-1)%32] << 32) | word[t]          # previous word of the SAME tile, tile-circular
for kk in 0..7 : x = (pair >> 4*kk) & 0xffff
                 p = (x * 0x83dcd12d) mod 2^32
                 S = byte0+byte1+byte2+byte3 of p           # v_sad_u8(p,0,0x6400) minus 1024
                 Wq[row, 8h + 7 - kk] = (S - 510) / 147.8
W = diag(svh) * Hout * Wq * Hin * diag(suh)
     Hin  = block-diag 128 Sylvester Hadamard / sqrt(128), right-multiply (columns, K side)
     Hout = same, left-multiply (rows, N side, blocks of 128 rows)
```
Wq std measured 0.9973, mean 0.0004: unit Gaussian as inferred. Windows kk=0..4 lie fully inside word t; kk=5,6,7 take their top bits from word t-1 (found by scan, hi-word offset 31 gives 0.992, all others <=0.27).

## How it was found (short)

1. q4c layout (measured): 64 B header = 16 float32 codebook (non-uniform, -1.0 to 0.9994), then N*K/2 bytes of nibbles (low nibble first), then N*K/32 fp16 scales (row-major, 80 per row). 64+655360+81920 = 737344.
2. svh row magnitude equals the q4c row std: corr 0.99986, median ratio 1.0006. So svh = sign x per-row scale, Wq has unit variance.
3. Kurtosis of W/sv is 6.66; after the input-side Hadamard 3.02 (Gaussian). Kurtosis did not show that the output Hadamard is also needed.
4. Contiguous row-major stream and naive per-row windows: Pearson 0.00 (all 48 variants). Kernel `k_ht_deq_t<4>` (work/kernels-dequant) showed 16x16 tiles, 3 loads per lane, shifts 0,4,8,12 on one word and 16..28 on a 64-bit pair.
5. Association matrix C^T Y over all 5120 tiles (stream code position vs target position): median row peak 0.969 with tile order blk8, target = Hout Hin (Wref/sv .* su), lsb-first windows; every other target or order is noise (0.04 to 0.05).
6. Scan of window start offset per column and of the high word index fixed cols 0..2 (prev word).

## Open items settled (this tensor, 4-bit)

- Stream bit order: LSB first within little-endian u32; window = (x >> shift) & 0xffff, bit at `shift` is the window LSB. SETTLED.
- Tile order: 16x16 tiles, 128-row blocks, tile ids go row-tile fastest (8), then column tile, then row block. Within a tile lane t = 16*h + r holds row r, cols 8h..8h+7, with the column index DESCENDING as the window shift rises (col = 8h + 7 - kk). SETTLED.
- Trellis closure: tail-biting inside each 128 B tile (lane 0 uses word 31). SETTLED.
- Exact scale s6: (S - 510)/147.8 reproduces the data; fitted gain 0.9945 (rotated domain) and 0.994 (full), i.e. within 0.6% of 1/147.8, partly explained by quantization shrinkage of the reference. Treat s6 = 1/147.8 as confirmed to about 1%; the exact fp16 literal of s6 is not recovered (needs `s2` kernel arg or a lossless reference).
- Additive term: fitted intercept -0.0018 on a unit-std target, i.e. none. SETTLED (zero within 0.2% of std).
- su content: exactly +/-1 random signs, applied BEFORE the input Hadamard (ablation without it: -0.02). SETTLED.
- sv content: signed per-row scale (|sv| = row std of the weight). Applied after the output Hadamard as a diagonal. Sign matters. SETTLED.
- Hout: the output-side 128-point Hadamard is REAL (rows in blocks of 128): the model needs both Hin and Hout; the targets with only Hin gave noise (peak 0.04). Open on the engine side: whether the engine does Hout as `k_ht_rot_out` on the result (it should; consistent with k_ht_rot_out existing). SETTLED for the format.
- Hadamard order: Sylvester natural order, no sequency permutation, orthonormal 1/sqrt(128). SETTLED.

## Not proven (written before Tests A and B; 3-bit and N > K are settled below)

- 3-bit HT (template K=3): not tested; window step and pair layout for K=3 (768 B per 2048 weights) are untested.
- qparam bit 0 (=0 here) and the upper bits 0x1200 (=0x1200 >> 8 = 18): the decode needed no input from them; the binder only reads bits 1..7. Whether 0x1200 encodes tile shape, group size or a version stays unknown with one tensor. Suggest decoding a tensor with a different qparam.
- Ragged shapes: rows not a multiple of 128, cols not a multiple of 16 or 128, 1-D or MoE-expert HT tensors, `k_ht_gemv_gu`/`k_ht_moe_*` layouts (expert strides). Only [512,2560] was tested; tile order for other row counts follows the kernel's division (rows//128 blocks) but is inferred.
- Whether HT tensors with a different transpose/orientation (e.g. down_proj [2560, K]) use the same lane map.
- Exact fp16 literal of s6 and whether the engine path (k_ht_lin with su/sv) applies sv before or after Hout in float precision (same math, rounding differs).
- Residual error source (0.8% decorrelation) is not decomposed into HT vs q4c error.

## Test C: gate_up experts (U27 part)

`layers.0.mlp.experts.gate_up_proj` expert 0, 4 bit [1280, 2560], with the shared `layers.0.mlp.experts.su` [2560] (gate_up has no per-expert suh). Measured with `scripts/ht_decode.py`: Pearson 0.9922 overall, row min 0.9915, median 0.9922 against the w4b reference; the dense tile map gives about 0 on the same bytes (test `test_real_gate_up_expert_matches_reference_only_with_its_own_walk`).

Layout (`decode_codes_gate_up`): tile contents, Hin, Hout, su and svh are as for dense tensors. Only the tile walk differs. Rows are gate 0..N/2-1 then up N/2..N-1. The stream walks blocks b of 128 gate rows plus the matching 128 up rows; inside a block, per column tile ct, the 8 gate tile rows come first, then the 8 up tile rows. Stream tile id = ((b*K/16 + ct)*2 + g)*8 + i, matrix row = g*N/2 + b*128 + i*16 + r. Found from data by the decode agent (association matrix over tile orders, then a per-tile match); the fused kernels `k_ht_gemv_gu` and `k_ht_moe_gu` were not read for it (inferred that they use this walk). Not tested: other experts (stride 1,638,400 B assumed), 3-bit gate_up, N/2 not a multiple of 128.

## Test A: 4-bit, N > K (shared_expert.down_proj [2560, 640])

Decoder: same as above (`scripts/ht_decode.py` `decode_codes(..., bits=4)`). All numbers measured.
- Pearson vs w4b q4c: overall 0.9931, per-row min 0.9890, median 0.9932, max 0.9945. LS gain 0.9968.
- Same lane/tile map, same 128-row tile blocks (20 blocks of 128 rows, 40 tile cols), both-side Hadamard, su before Hin: holds with N > K. suh values are exactly +1/-1.
- q4c scale area for K=640: 24 fp16 per row, only the first 20 (K/32) used, last 4 are zero padding (942144 = 64 + 819200 + 2560*24*2). Check: row std of nibbles x scales vs |svh|: ratio 0.9985, corr 0.9995.

## Test B: 3-bit routed experts (layers.1.mlp.experts.down_proj, [512, 2560, 640])

- Expert-major confirmed. Expert 0: overall 0.9839, row min 0.9798, median 0.9839, max 0.9872. Expert 1 (bytes 614400..1228799): overall 0.9848, row min 0.9819, median 0.9848, max 0.9877. Negative control, expert 1 HT vs expert 0 reference: -0.0008. LS gain (expert 0) 0.9936; decoded Wq std 0.9897 (measured). (S-510)/147.8 unchanged at 3 bit.
- w4b whole-tensor layout (measured by use): 64 B header (codebook), then ALL experts' nibbles (512 * N*K/2 = 419,430,400 B), then scales per expert: N rows x 24 fp16 (48 B per row, first 20 used), 122,880 B per expert. 64 + 419,430,400 + 512*122,880 = 482,345,024.
- Per-expert 3-bit tile: 16x16 weights = 96 B = 24 words (stride 0x60 in `k_ht_deq_t<3>`). Tile order is the same blk8 order (128-row blocks, col tile, row tile fastest), lane group of 8 columns descending. Found by scanning: per target position the best (word a, previous word, shift) over 24 words; all 256 positions reached 0.98 in the 1500-tile scan.

### Unified decoder (4 and 3 bit), delta from the first 4-bit recipe in 3 lines
1. Tile words are read in REVERSE order as one little-endian bit string U (last word = lowest bits). This is the same thing as the 4-bit rule "hi bits come from word t-1": it just means windows run through words backwards.
2. Weight j = 128h + 8r + c (row r, col 8h + c) reads the 16-bit window at bit pos = bits*(255 - j) mod (256*bits), circular within the tile (bits = 3 or 4). For 4 bit this equals pair {w[t-1], w[t]} >> 4*(7-c), t = 16h + r, as before.
3. Hash, byte-sum, (S-510)/147.8, tile order, Hin/Hout, su, sv all unchanged. Only the tile size changes (8*bits words) and the window step (bits).
Run `python3 -I scripts/ht_decode.py <dir>` to reproduce k_proj, A and B scores (the k_proj score is identical, 0.9924, under the unified map).

### Still unproven after A and B
- 3-bit with other shapes (K not multiple of 128 / 16, N not multiple of 128), and gate_up experts (`k_ht_gemv_gu`, fused layouts).
- Experts beyond 0 and 1 (stride 614,400 B per expert assumed constant; only experts 0 and 1 seen).
- Whether expert tensors share su/sv per expert as assumed (verified for experts 0 and 1 only: each expert used its own 640 / 2560 vectors).
- qparam bit 0 and upper bits (0x1200 at 4 bit, 4614 = 0x1206 at 3 bit: same upper bits 0x1200 in both, so they do not encode bits or tile size; meaning still unknown).
- Hout for MoE tensors with N=512 per-expert is not tested (here N=2560).
- 2-bit or other K; the real engine's fp16 rounding.
- Closed form was derived from scan results at 3 bit and checked end to end (0.984), not read off the kernel instruction by instruction.

## i4r (dtype 23), U27 closed

Date 2026-10-10. Data: range reads of `qwen38-flash-next-v2.hgn` (table: 2,015 tensors, data_off 322,560; 96 i4r tensors at qparam 128, 48 `experts.gate_up_proj` plus 48 `experts.down_proj`, counted from its table; the earlier note "all gate_up" was wrong) and the w4b q4c references for layers.0. Slices are in `scripts/fetch_ht_testdata.sh` (`i4r_*`, `d0_dn_w4b_*`).

Layout, from sizes and the loader check (decomp `01996d90.c:215-223`, Ghidra 0x1996d90): i4r needs qparam 0x80, rank 3, dims [512, 1280, 2560] or [512, 2560, 640], and nbytes = K/2·N·E + K/128·N·E·2. So 4.125 bits per weight, not 4.25 (`hgn-format.md` s3 corrected): all experts' nibbles first (expert e at e·N·K/2), then all experts' fp16 scales (expert e at E·N·K/2 + e·N·K/128·2), one scale per 128 weights along K. No per-row term in the payload; `svh` is a separate tensor and holds only +1 and -1 for i4r.

Decode (`decode_codes_i4r`): Wq[n, k] = (nibble - 8) · scale[n, k // 128], nibbles row-major, low nibble first; scales are signed (50.3% negative in expert 0, measured). Then the HT `reconstruct()` unchanged: input and output 128-point Hadamard, rows times `svh`, columns times `su` (gate_up: the shared `mlp.experts.su`; down_proj: `experts.down_proj.suh`/`.svh`).

Measured with `python3 -I scripts/ht_decode.py <dir>` on this Mac:

```
v2 i4r layers.0.mlp.experts.gate_up_proj expert 0 [1280,2560] overall 0.9876 row min 0.9843 median 0.9877
negative control: i4r expert 1 vs expert 0 reference overall -0.0002 row min -0.0680 median 0.0004
v2 i4r layers.0.mlp.experts.down_proj expert 0 [2560,640] overall 0.9885 row min 0.9743 median 0.9882
```

Mutants scored on expert 0 gate_up (scratch run): high nibble first 0.0094; offset 7.5 instead of 8 drops to 0.9724; scales as [K/128, N] 0.0084; input-only or output-only Hadamard below 0.001; no rotation 0.0003. Only the both-side rotation with row-major nibbles and [N, K/128] scales scores.

Refuter mutants on PR #7 (expert 0 gate_up): nibbles shuffled within each 128-group 0.0075; random nibbles with true scales 0.0031; expert 0 nibbles with expert 1 scales 0.0083; scales with signs dropped -0.0065; constant scale -0.0064; scales permuted 0.0009. The score needs the right nibbles in the right order with the right signed scales.

qparam bits (U27 part 3): the same loader validates HT qparam at `01996d90.c:236-238`: `if (((0x1fff < uVar12) || (uVar12 < 0x1000)) || (((uVar12 & 0xf00) != 0x200 || (((char)uVar12 != '\x06' && ((uVar12 & 0xff) != 8)))))) goto LAB_019976c2;` then `uVar12 = uVar12 >> 1 & 0x7f;`. So HT qparam is exactly 0x1206 (3 bit) or 0x1208 (4 bit): 0x1200 is a fixed format tag the decoder never reads, and bit 0 must be 0. The binder at `0190e330.c:357` reads only `uVar1 >> 1 & 0x7f`.

`k_i4r_*_gl`: the 8 `_gl` slices in `work/kernels/` (`rows_gl<1|2>`, `rows_gl_g<1|2,2|4>`, `rows_gl_o<1|2>`) each carry the HT multiply-bytesum decode (`0x83dcd12d` 64 times, `v_sad_u8` 64 times, grep); no other `k_i4r_*` slice has the literal (0 in `rows<2560|640>`, `rows_rot`, `rows_epi`, `rows_fl`, `rows_fl_g`, `rows_fl_o`, `fold`, `fold_rows`, `gu_epi`). So the `_gl` weight decode is the U19 rule at kernel level (static, not run on data). The plain i4r payload above is what the nibble kernels (`v_perm_b32` expansion) read. Which tensors the host routes to `_gl` is a dispatch question under U21, not claimed here.

Not proven: experts other than 0 and 1 (offsets assumed linear in e, consistent with the loader size rule); layers other than 0; the engine's own fp16 rounding.
