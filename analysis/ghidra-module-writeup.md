# Ghidra processor-module probe for AMD GPU code (re-halogen)

Date: 2026-10-09 · Agent: ghidra-module subtask · Ghidra: 12.1.2 (`~/.hermes/tools/ghidra-12.1.2`)

## Question

Can Ghidra (via REA's Ghidra provider, or headless) import/disassemble/decompile the
gfx1151 code objects carved from `flash_serve`'s `.hip_fatbin`
(`work/fatbin/codeobj_*.hsaco`, ELF `EM_AMDGPU`, flags `0x4a` = gfx1151 per
`EF_AMDGPU_MACH_AMDGCN_GFX11_5_1`), and if not, what is the closest open Ghidra module?

## Findings

### 1. No AMDGCN/RDNA shader-ISA module exists for Ghidra 12.1.2

- `Ghidra/Processors/` in the 12.1.2 install ships 40 families (x86, AARCH64, RISCV,
  Hexagon,tricore, …) — **no AMDGPU/AMDGCN/GCN/RDNA entry**.
- REA Ghidra provider on `work/fatbin/codeobj_14aa000.hsaco` →
  `{"error":"Analysis failed","code":"target_unavailable","reason":"unsupported ELF architecture"}`
  (`tmp/amdgpu-rea-analyze.json`). Ghidra's ELF loader has no `EM_AMDGPU` (224) support:
  no language, no relocation handler, no `amd_kernel_code_t`/NT_AMDGPU_METADATA loader.
- GitHub search (repos + code) for `ghidra amdgpu / amdgcn / rdna / gfx1030 / gcn processor`
  returns **zero maintained AMDGCN shader-ISA modules**. The AMDGPU-related Ghidra ecosystem
  is only the F32 microcontroller modules (below). `Sleigh-InSPECtor/InSPECtor` is a
  SLEIGH-veracity research tool (GenSYS/JuxtaPlayer), not an AMD ISA module.
- Conclusion: gfx1151 shader ISA in Ghidra would require authoring a full SLEIGH spec
  (variable 32/64/96-bit encodings, ~1000s of instructions, 204+ register file, custom
  loader for the kernel-code table). Out of scope for this probe; **llvm-objdump 21
  (`--triple=amdgcn-amd-amdhsa --mcpu=gfx1151`) remains the working disassembly path**
  (see `analysis/kernels-disasm/*.s`).

### 2. Installed + tested: cyrozap/ghidra-radeon-f32 (AMD Radeon F32 microcontroller ISA)

This is the real "AMD Radeon processor module" for Ghidra: it decodes the F32 scalar core
that runs GPU management firmware (ASD/SMU-adjacent microcode), based on
fail0verflow/radeon-tools. It is **not** the shader ISA — it covers the GPU's embedded
management CPU, i.e. adjacent silicon, not gfx1151 code objects.

- Module: `cyrozap/ghidra-radeon-f32` @ `6693e360dfd4457dcd0ff311e4bdfac00015619a`
  (2024-10-03, Apache-2.0, WIP per README).
- Cloned to `tmp/ghidra-radeon-f32/`; SLEIGH compiled with the install's own compiler:
  `support/sleigh -a data/languages` → **"2 languages successfully compiled"**
  (`F32:BE:64:default`, `F32:LE:64:default`; only NOP-constructor warnings).
- Installed by directory drop: `cp -r … $GHIDRA_INSTALL_DIR/Ghidra/Processors/radeon-f32`
  (no gradle needed — gradle is absent on this host; the prebuilt `.sla` files are enough).
- Test corpus: real Radeon firmware from the host — `/lib/firmware/amdgpu/navi10_asd.bin.zst`
  decompressed to `tmp/navi10_asd.bin` (205,312 B, Navi 10 / RDNA1 Animation State Data,
  an F32 image).
- Headless run (`scripts/run_f32_test.sh`, via
  `sudo systemd-run --property=MemoryMax=16G --pipe --wait`, BinaryLoader @0x0,
  processor `F32:BE:64:default`, post-script `scripts/F32Report.java`):
  - **Import: OK** — project created, analysis succeeded.
  - **Disassembly: OK** — module decodes real F32 instructions from the image:
    `nop`, `addd r0,r0,#0x0`, `sub r0,r12,#0x0`, `sub r0,r4,#0x21`,
    `stw r12,reg[r7,#0x45c9]`, … (25 ins decoded in the prologue region; the module has
    no file loader, so only the entry region is code — expected for a WIP module).
  - **Decompilation: OK** — `DecompInterface` completed on the entry symbol
    `Reset@IM:00000000` (`decompileCompleted()=true`), emitting C with
    `halt_unimplemented()` at the first unimplemented constructor.
  - Full output: `tmp/f32-headless.log`, `f32report.txt` (script-written).

### 3. Verdict per stage (gfx1151 hsaco, Ghidra path)

| Stage | hsaco (gfx1151) | F32 module on Radeon ASD firmware |
|---|---|---|
| Import | FAIL — `unsupported ELF architecture` (EM_AMDGPU) | OK |
| Disassembly | n/a (never imported) | OK — real mnemonics |
| Decompilation | n/a | OK — decompiler completes |

## Install steps (reproducible)

```bash
cd /home/rangizingo/re-halogen/tmp
git clone https://github.com/cyrozap/ghidra-radeon-f32.git   # @6693e36
cd ghidra-radeon-f32
GHIDRA_INSTALL_DIR=/home/rangizingo/.hermes/tools/ghidra-12.1.2 \
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
/home/rangizingo/.hermes/tools/ghidra-12.1.2/support/sleigh -a data/languages
cp -r . /home/rangizingo/.hermes/tools/ghidra-12.1.2/Ghidra/Processors/radeon-f32
# verify: analyzeHeadless tmp/proj-f32 f32test -import tmp/navi10_asd.bin \
#   -loader BinaryLoader -loader-baseAddr 0x0 -processor "F32:BE:64:default" \
#   -scriptPath scripts -postScript F32Report.java   (run under systemd-run MemoryMax=16G)
```

## Files

- `scripts/run_f32_test.sh` — headless test driver (systemd-run launcher pattern).
- `scripts/F32Report.java` — post-analysis script: instruction/data census + decompiler smoke test.
- `tmp/ghidra-radeon-f32/` — module source + compiled `.sla`.
- `tmp/navi10_asd.bin` — test firmware. `tmp/f32-headless.log`, `f32report.txt` — results.
- `tmp/amdgpu-rea-analyze.json` — REA/Ghidra hsaco import failure evidence.

## Notes / gotchas learned

- Ghidra 12.1.2 Java-script API: `DecompileResults.decompileCompleted()` (not
  `completed()`/`isCompleted()`), `Language.getLanguageID().toString()` (no `getID()`).
- Re-running `-import` into an existing project dir fails with "conflicting program file";
  wipe the project dir (`sudo rm -rf tmp/proj-f32`) between runs.
- Files created by the systemd-run (root) service are root-owned; clean with sudo.
- A real gfx1151 Ghidra path needs: EM_AMDGPU ELF loader + `amd_kernel_code_t` kernel
  entry table + SLEIGH for gfx11 (32/64/96-bit instruction encodings) + p-code semantics.
  Nearest accelerators: `oberoisecurity/ghidra-processor-module-generator` (disasm-only
  from objdump listings, 1–4 byte ISAs — too small for GCN) and `infobyte/isa_recovery`
  (LLM-driven SLEIGH recovery from objdump output; gfx11's 96-bit encodings exceed its
  fixed-length assumptions). llvm-objdump stays the primary path.
