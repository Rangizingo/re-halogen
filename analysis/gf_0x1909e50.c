
/* WARNING: Globals starting with '_' overlap smaller symbols at the same address */

void FUN_01909e50(long param_1,uint param_2,undefined8 param_3,float *param_4,undefined4 param_5,
                 undefined8 param_6,undefined8 param_7,uint param_8)

{
  long lVar1;
  float *pfVar2;
  float fVar3;
  ulong uVar4;
  ulong uVar5;
  ulong uVar6;
  ulong uVar7;
  ulong uVar8;
  int iVar9;
  undefined8 uVar10;
  undefined8 uVar11;
  FILE *__stream;
  ulong uVar12;
  undefined4 uVar13;
  float local_218;
  float local_214;
  ulong local_210;
  undefined8 local_208;
  undefined8 local_200;
  float local_1f8;
  float local_1f4;
  ulong local_1f0;
  undefined8 local_1e8;
  undefined8 local_1e0;
  float local_1d8;
  float local_1d4;
  ulong local_1d0;
  undefined8 local_1c8;
  undefined8 local_1c0;
  float local_1b8;
  float local_1b4;
  ulong local_1b0;
  undefined8 local_1a8;
  undefined8 local_1a0;
  float local_198;
  float local_194;
  ulong local_190;
  undefined8 local_188;
  undefined8 local_180;
  float local_178;
  float local_174;
  ulong local_170;
  undefined8 local_168;
  undefined8 local_160;
  float local_158;
  float local_154;
  undefined8 local_150;
  undefined8 local_148;
  undefined8 local_140;
  float local_138;
  float local_134;
  undefined8 local_130;
  undefined8 local_128;
  undefined8 local_120;
  long local_110;
  undefined8 local_90;
  undefined8 local_38;
  
  uVar12 = (ulong)param_2;
  local_38 = param_6;
  local_90 = std::chrono::_V2::steady_clock::now();
  if (((param_2 - 9 < 0xfffffff8) || ((int)param_8 < 0)) || (8 < (int)(param_8 + param_2))) {
    fprintf(_stderr,"accept_probs: rows %d..%d outside 0..%d\n",(ulong)param_8,
            (ulong)((param_8 + param_2) - 1),7);
                    /* WARNING: Subroutine does not return */
    exit(1);
  }
  local_218 = *param_4;
  if (local_218 <= 0.0) {
    fprintf(_stderr,"%s: temp=%g: temperature 0 is the GREEDY path and must stay on argmax()\n",
            (double)local_218,"accept_probs");
                    /* WARNING: Subroutine does not return */
    exit(1);
  }
  local_214 = param_4[1];
  pfVar2 = param_4 + 2;
  fVar3 = *pfVar2;
  uVar8 = *(ulong *)pfVar2;
  uVar7 = *(ulong *)pfVar2;
  uVar6 = *(ulong *)pfVar2;
  uVar5 = *(ulong *)pfVar2;
  uVar4 = *(ulong *)pfVar2;
  local_210 = *(ulong *)pfVar2;
  local_208 = *(undefined8 *)(param_4 + 4);
  if (param_4[3] <= 1.0) {
    if (param_4[3] < 0.0) {
      local_210 = (ulong)(uint)fVar3;
      if (param_2 == 1) goto LAB_0190a352;
      local_1f0 = (ulong)(uint)fVar3;
      local_1e0 = 0;
      local_1f8 = local_218;
      local_1f4 = local_214;
      local_1e8 = local_208;
      if (param_2 == 2) goto LAB_0190a352;
      local_1d0 = (ulong)(uint)fVar3;
      local_1c0 = 0;
      local_1d8 = local_218;
      local_1d4 = local_214;
      local_1c8 = local_208;
      if (param_2 == 3) goto LAB_0190a352;
      local_1b0 = (ulong)(uint)fVar3;
      local_1a0 = 0;
      local_1b8 = local_218;
      local_1b4 = local_214;
      local_1a8 = local_208;
      if (param_2 == 4) goto LAB_0190a352;
      local_190 = (ulong)(uint)fVar3;
      local_180 = 0;
      local_198 = local_218;
      local_194 = local_214;
      local_188 = local_208;
      if (param_2 == 5) goto LAB_0190a352;
      local_170 = (ulong)(uint)fVar3;
      local_160 = 0;
      local_178 = local_218;
      local_174 = local_214;
      local_168 = local_208;
      if (param_2 == 6) goto LAB_0190a352;
      uVar13 = 0;
      goto LAB_0190a163;
    }
    if (((((param_2 == 1) ||
          (local_1e0 = 0, local_1f8 = local_218, local_1f4 = local_214, local_1f0 = uVar4,
          local_1e8 = local_208, param_2 == 2)) ||
         ((local_1c0 = 0, local_1d8 = local_218, local_1d4 = local_214, local_1d0 = uVar5,
          local_1c8 = local_208, param_2 == 3 ||
          ((local_1a0 = 0, local_1b8 = local_218, local_1b4 = local_214, local_1b0 = uVar6,
           local_1a8 = local_208, param_2 == 4 ||
           (local_180 = 0, local_198 = local_218, local_194 = local_214, local_190 = uVar7,
           local_188 = local_208, param_2 == 5)))))) ||
        (local_160 = 0, local_178 = local_218, local_174 = local_214, local_170 = uVar8,
        local_168 = local_208, param_2 == 6)) ||
       (local_140 = 0, local_158 = local_218, local_154 = local_214,
       local_150 = *(undefined8 *)pfVar2, local_148 = local_208, uVar10 = *(undefined8 *)pfVar2,
       param_2 == 7)) goto LAB_0190a352;
  }
  else {
    uVar13 = 0x3f800000;
    local_210 = CONCAT44(0x3f800000,fVar3);
    if (param_2 == 1) goto LAB_0190a352;
    local_1f0 = CONCAT44(0x3f800000,fVar3);
    local_1e0 = 0;
    local_1f8 = local_218;
    local_1f4 = local_214;
    local_1e8 = local_208;
    if (param_2 == 2) goto LAB_0190a352;
    local_1d0 = CONCAT44(0x3f800000,fVar3);
    local_1c0 = 0;
    local_1d8 = local_218;
    local_1d4 = local_214;
    local_1c8 = local_208;
    if (param_2 == 3) goto LAB_0190a352;
    local_1b0 = CONCAT44(0x3f800000,fVar3);
    local_1a0 = 0;
    local_1b8 = local_218;
    local_1b4 = local_214;
    local_1a8 = local_208;
    if (param_2 == 4) goto LAB_0190a352;
    local_190 = CONCAT44(0x3f800000,fVar3);
    local_180 = 0;
    local_198 = local_218;
    local_194 = local_214;
    local_188 = local_208;
    if (param_2 == 5) goto LAB_0190a352;
    local_170 = CONCAT44(0x3f800000,fVar3);
    local_160 = 0;
    local_178 = local_218;
    local_174 = local_214;
    local_168 = local_208;
    if (param_2 == 6) goto LAB_0190a352;
LAB_0190a163:
    local_160 = 0;
    local_180 = 0;
    local_1a0 = 0;
    local_1c0 = 0;
    local_1e0 = 0;
    local_140 = 0;
    local_1f8 = local_218;
    local_1f4 = local_214;
    local_1e8 = local_208;
    local_1d8 = local_218;
    local_1d4 = local_214;
    local_1c8 = local_208;
    local_1b8 = local_218;
    local_1b4 = local_214;
    local_1a8 = local_208;
    local_198 = local_218;
    local_194 = local_214;
    local_188 = local_208;
    local_178 = local_218;
    local_174 = local_214;
    local_168 = local_208;
    local_158 = local_218;
    local_154 = local_214;
    local_150 = CONCAT44(uVar13,fVar3);
    local_148 = local_208;
    if (param_2 == 7) goto LAB_0190a352;
    uVar10 = CONCAT44(uVar13,fVar3);
  }
  local_130 = uVar10;
  local_140 = 0;
  local_160 = 0;
  local_180 = 0;
  local_1a0 = 0;
  local_1c0 = 0;
  local_1e0 = 0;
  local_120 = 0;
  local_1f8 = local_218;
  local_1f4 = local_214;
  local_1e8 = local_208;
  local_1d8 = local_218;
  local_1d4 = local_214;
  local_1c8 = local_208;
  local_1b8 = local_218;
  local_1b4 = local_214;
  local_1a8 = local_208;
  local_198 = local_218;
  local_194 = local_214;
  local_188 = local_208;
  local_178 = local_218;
  local_174 = local_214;
  local_168 = local_208;
  local_158 = local_218;
  local_154 = local_214;
  local_148 = local_208;
  local_138 = local_218;
  local_134 = local_214;
  local_128 = local_208;
LAB_0190a352:
  local_200 = 0;
  local_110 = param_3;
  iVar9 = hipMemcpy(*(undefined8 *)(param_1 + 0x9a0),&local_218,uVar12 << 5,1);
  __stream = _stderr;
  if (iVar9 == 0) {
    lVar1 = uVar12 * 4;
    iVar9 = hipMemcpy(*(undefined8 *)(param_1 + 0x9b0),local_110,lVar1,1);
    __stream = _stderr;
    if (iVar9 == 0) {
      uVar10 = FUN_01907ff0(param_1,(ulong)param_8,uVar12,param_5);
      local_110 = lVar1;
      if ((DAT_019ff5a0 == '\0') && (iVar9 = __cxa_guard_acquire(&DAT_019ff5a0), iVar9 != 0)) {
        DAT_019ff598 = FUN_0194bd50();
        __cxa_guard_release(&DAT_019ff5a0);
      }
      if (DAT_019ff598 == 0) {
                    /* WARNING: Subroutine does not return */
        __hipPushCallConfiguration(uVar12 | 0x100000000,1,0x100000100,1,0,0);
      }
      FUN_01908ff0(uVar10,*(undefined8 *)(param_1 + 0x9a0),&local_218,uVar12,
                   *(undefined8 *)(param_1 + 0x9c0),*(undefined8 *)(param_1 + 0x9c8),
                   *(undefined8 *)(param_1 + 0x9d0),DAT_019ff598);
                    /* WARNING: Subroutine does not return */
      __hipPushCallConfiguration((uVar12 << 0x20) + 1,1,0x100000100,1,0,0);
    }
    uVar10 = hipGetErrorString(iVar9);
    uVar11 = 0x2418;
  }
  else {
    uVar10 = hipGetErrorString(iVar9);
    uVar11 = 0x2417;
  }
  fprintf(__stream,"HIP %s:%d: %s\n","/src/halogen/src/flash_model.hip",uVar11,uVar10);
                    /* WARNING: Subroutine does not return */
  exit(1);
}

