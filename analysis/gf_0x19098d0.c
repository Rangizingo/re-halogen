
/* WARNING: Globals starting with '_' overlap smaller symbols at the same address */

void FUN_019098d0(float param_1,long param_2,undefined8 param_3,undefined4 param_4,
                 undefined4 param_5,undefined8 param_6,float *param_7,uint param_8)

{
  bool bVar1;
  float fVar2;
  int iVar3;
  long lVar4;
  long lVar5;
  undefined8 uVar6;
  FILE *__stream;
  undefined8 uVar7;
  float local_68 [2];
  ulong local_60;
  undefined8 local_58;
  undefined4 local_50;
  undefined4 local_4c;
  undefined1 local_48 [16];
  
  lVar4 = std::chrono::_V2::steady_clock::now();
  if ((7 < (int)param_8) || ((-1 < (int)param_8 && (*(long *)(param_2 + 0x60) == 0)))) {
    fprintf(_stderr,"draft_sample: q row %d was not kept\n",(ulong)param_8);
                    /* WARNING: Subroutine does not return */
    exit(1);
  }
  if ((*(char *)(param_2 + 0xb7c) != '\x01') || (*(long *)(param_2 + 0xd80) == 0)) {
    fwrite("draft_sample: the MTP head is not staged (HALOGEN_MTP=1)\n",0x39,1,_stderr);
                    /* WARNING: Subroutine does not return */
    exit(1);
  }
  if (param_1 <= 0.0) {
    fprintf(_stderr,"%s: temp=%g: temperature 0 is the GREEDY path and must stay on argmax()\n",
            (double)param_1,"draft_sample");
                    /* WARNING: Subroutine does not return */
    exit(1);
  }
  if ((DAT_019fe1d0 == '\0') && (iVar3 = __cxa_guard_acquire(&DAT_019fe1d0), iVar3 != 0)) {
    DAT_019fe1c8 = (float)FUN_01909da0();
    __cxa_guard_release(&DAT_019fe1d0);
  }
  if ((DAT_019fe1e0 == '\0') && (iVar3 = __cxa_guard_acquire(&DAT_019fe1e0), iVar3 != 0)) {
    DAT_019fe1d8 = (float)FUN_01909dd0();
    __cxa_guard_release(&DAT_019fe1e0);
  }
  if ((DAT_019fe1f0 == '\0') && (iVar3 = __cxa_guard_acquire(&DAT_019fe1f0), iVar3 != 0)) {
    DAT_019fe1e8 = (float)FUN_01909e00();
    __cxa_guard_release(&DAT_019fe1f0);
  }
  fVar2 = DAT_019fe1d8;
  local_48 = ZEXT416(-(uint)(0.0 < DAT_019fe1c8));
  bVar1 = DAT_019fe1e8 != 1.0;
  local_60 = (ulong)(uint)DAT_019fe1c8;
  local_68[0] = param_1 * DAT_019fe1e8;
  local_68[1] = 0.0;
  local_58 = param_3;
  local_50 = param_4;
  local_4c = param_5;
  iVar3 = hipMemcpy(*(undefined8 *)(param_2 + 0x9a0),local_68,0x20,1);
  __stream = _stderr;
  if (iVar3 == 0) {
    if ((int)param_8 < 0) {
      lVar5 = *(long *)(param_2 + 0xd80);
    }
    else {
      lVar5 = (ulong)param_8 * 0xf2800 + *(long *)(param_2 + 0x60);
    }
    uVar6 = 0;
    uVar7 = 0;
    if (0.0 < DAT_019fe1c8) {
      uVar7 = *(undefined8 *)(param_2 + 0x9e8);
    }
    else {
      uVar6 = *(undefined8 *)(param_2 + 0x9f0);
    }
    if (param_7 == (float *)0x0) {
      uVar6 = 0;
      uVar7 = 0;
    }
    FUN_01908c30(param_2,lVar5,local_68,1,uVar7,uVar6);
    iVar3 = hipMemcpy(param_6,*(undefined8 *)(param_2 + 0x9a8),4,2);
    __stream = _stderr;
    if (iVar3 == 0) {
      if (param_7 != (float *)0x0) {
        iVar3 = hipMemcpy(param_7,*(undefined8 *)
                                   (param_2 + 0x9e8 + (ulong)(DAT_019fe1c8 <= 0.0) * 8),4,2);
        __stream = _stderr;
        if (iVar3 != 0) {
          uVar7 = hipGetErrorString(iVar3);
          uVar6 = 0x2400;
          goto LAB_01909d40;
        }
        *param_7 = DAT_019fe1d8 + *param_7;
      }
      if ((bVar1 || ((local_48[0] | -(fVar2 != 0.0)) & 1) != 0) &&
         (*(char *)(param_2 + 0xa88) == '\0')) {
        *(undefined1 *)(param_2 + 0xa88) = 1;
        fprintf(_stderr,
                "draft_sample: FALSIFIER ACTIVE: nucleus %.4f, qoff %+.4f, tskew %.4f. Every proposal from this process is deliberately wrong (METHOD 37).\n"
                ,(double)DAT_019fe1c8,(double)DAT_019fe1d8,(double)DAT_019fe1e8);
      }
      lVar5 = std::chrono::_V2::steady_clock::now();
      *(long *)(param_2 + 0xa90) = *(long *)(param_2 + 0xa90) + 1;
      *(long *)(param_2 + 0xa98) = (lVar5 - lVar4) + *(long *)(param_2 + 0xa98);
      return;
    }
    uVar7 = hipGetErrorString(iVar3);
    uVar6 = 0x23fd;
  }
  else {
    uVar7 = hipGetErrorString(iVar3);
    uVar6 = 0x23f9;
  }
LAB_01909d40:
  fprintf(__stream,"HIP %s:%d: %s\n","/src/halogen/src/flash_model.hip",uVar6,uVar7);
                    /* WARNING: Subroutine does not return */
  exit(1);
}

