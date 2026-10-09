
/* WARNING: Type propagation algorithm not settling */
/* WARNING: Globals starting with '_' overlap smaller symbols at the same address */

undefined8 FUN_01851dc0(long *param_1,uint param_2,long param_3,undefined8 *param_4)

{
  long *plVar1;
  float *pfVar2;
  undefined1 *puVar3;
  bool bVar4;
  char cVar5;
  int iVar6;
  undefined4 uVar7;
  ulong uVar8;
  ulong uVar9;
  undefined8 uVar10;
  int iVar11;
  ulong uVar12;
  uint uVar13;
  int iVar14;
  uint uVar15;
  long lVar16;
  long lVar17;
  int iVar18;
  ulong uVar19;
  long lVar20;
  double dVar21;
  float fVar22;
  double dVar23;
  int local_114;
  undefined4 local_110;
  float local_10c;
  ulong local_108;
  ulong local_100;
  long *local_f8;
  long local_f0;
  uint local_e8 [10];
  long local_c0;
  uint local_b8 [2];
  undefined1 auStack_b0 [24];
  undefined8 local_98;
  undefined8 uStack_90;
  undefined8 local_88;
  undefined8 uStack_80;
  float local_78 [2];
  undefined1 auStack_70 [24];
  undefined4 local_58;
  undefined1 local_54 [36];
  
  lVar16 = param_1[0x21];
  if (*(int *)(lVar16 + 0x68) < 0 || *(int *)(lVar16 + 0x68) != *(int *)(lVar16 + 0x2e8)) {
    bVar4 = *(int *)(lVar16 + 0x58) == *(int *)(lVar16 + 0x2e8) && *(long *)(lVar16 + 0x50) != 0;
  }
  else {
    bVar4 = false;
  }
  FUN_018f9dc0(lVar16,*(undefined4 *)((long)param_1 + 0x19c),0,bVar4);
  lVar16 = param_1[0x21];
  if (*(int *)(lVar16 + 0x5c) < 1) {
    return 0xfffffffd;
  }
  iVar14 = *(int *)(lVar16 + 0x2e8);
  iVar11 = iVar14 + 1;
  FUN_018f9fb0(lVar16,0);
  uVar19 = 1;
  FUN_019098d0(param_1[0x21],*(undefined8 *)(param_1[0x32] + 0x10),iVar11,1,local_b8,local_78,
               0xffffffffffffffff);
  fVar22 = *(float *)((long)param_1 + 0x144);
  if ((fVar22 <= 0.0) || (dVar21 = exp((double)local_78[0]), (double)fVar22 <= dVar21)) {
    FUN_018f9dc0(param_1[0x21],local_b8[0],1,0);
    FUN_018f9fb0(param_1[0x21],1);
    FUN_019098d0(param_1[0x21],*(undefined8 *)(param_1[0x32] + 0x10),iVar14 + 2,1,local_b8 + 1,
                 local_78 + 1,0xffffffffffffffff);
    uVar19 = 2;
    if (param_2 != 2) {
      uVar19 = 2;
      lVar16 = 0;
      do {
        fVar22 = *(float *)(param_1 + 0x29);
        if ((0.0 < fVar22) &&
           (dVar21 = exp((double)*(float *)(auStack_70 + lVar16 + -4)), dVar21 < (double)fVar22))
        goto LAB_01852034;
        FUN_018f9dc0(param_1[0x21],*(undefined4 *)(auStack_b0 + lVar16 + -4),uVar19 & 0xffffffff,0);
        FUN_018f9fb0(param_1[0x21],uVar19 & 0xffffffff);
        FUN_019098d0(param_1[0x21],*(undefined8 *)(param_1[0x32] + 0x10),(int)uVar19 + iVar11,1,
                     auStack_b0 + lVar16,auStack_70 + lVar16,0xffffffffffffffff);
        uVar19 = uVar19 + 1;
        lVar16 = lVar16 + 4;
      } while ((ulong)param_2 * 4 + -8 != lVar16);
      uVar19 = (ulong)param_2;
    }
  }
LAB_01852034:
  uVar13 = (uint)uVar19;
  uVar8 = 3;
  if (uVar13 < 3) {
    uVar8 = uVar19 & 0xffffffff;
  }
  param_1[uVar8 + 0x10] = param_1[uVar8 + 0x10] + 1;
  lVar16 = param_1[0x21];
  if (1 < uVar13) {
    *(undefined4 *)(lVar16 + 0x68) = 0xffffffff;
  }
  local_58 = *(undefined4 *)((long)param_1 + 0x19c);
  uVar8 = uVar19 & 0xffffffff;
  local_108 = uVar19;
  memcpy(local_54,local_b8,uVar8 * 4);
  uVar13 = uVar13 + 1;
  FUN_018fa0e0(lVar16,&local_58);
  local_88 = 0;
  uStack_80 = 0;
  local_98 = 0;
  uStack_90 = 0;
  local_114 = 0;
  local_f0 = param_1[0x21];
  lVar16 = param_1[0x32];
  uVar7 = (undefined4)param_1[0x33];
  cVar5 = (char)param_1[0x49];
  if ((DAT_019fc590 == '\0') && (iVar14 = __cxa_guard_acquire(&DAT_019fc590), iVar14 != 0)) {
    DAT_019fc588 = FUN_01883020(local_e8);
    __cxa_guard_release(&DAT_019fc590);
  }
  iVar14 = DAT_019fc588;
  local_f8 = (long *)&local_110;
  if (cVar5 == '\0') {
    local_f8 = (long *)0x0;
  }
  local_c0 = 1 - uVar8;
  lVar17 = 0;
  lVar20 = 0;
  uVar19 = 0;
  local_100 = uVar8;
  do {
    local_10c = 0.0;
    local_110 = 0;
    FUN_01909e50(local_f0,1,(long)local_b8 + lVar17,lVar16,uVar7,&local_10c,local_f8,uVar19);
    dVar21 = exp((double)local_78[uVar19]);
    dVar23 = 0.0;
    if (0.0 < dVar21) {
      dVar23 = (double)local_10c / dVar21;
    }
    iVar18 = (int)uVar19;
    iVar6 = iVar18;
    if (iVar14 == 3) {
      iVar6 = 0;
    }
    uVar8 = ((ulong)(uint)(iVar6 + iVar11) * -0x2e4ab5cd2e6d12fd ^ *(ulong *)(lVar16 + 0x10) ^
            0x3c6ef372fe94f82a) + 0x9e3779b97f4a7c15;
    uVar8 = (uVar8 >> 0x1e ^ uVar8) * -0x40a7b892e31b1a47;
    if (1.0 <= dVar23) {
      dVar23 = 1.0;
    }
    if (dVar23 <= (double)((float)(uint)((uVar8 >> 0x1b ^ uVar8) * -0x6b2fb644ecceee15 >> 0x28) *
                          5.9604645e-08)) {
      local_f8 = param_1;
      if (uVar19 != 0 && iVar14 == 2) {
        uVar8 = 0;
        do {
          FUN_0184ad90(param_3,local_e8[uVar8]);
          if ((*(char *)(param_3 + 0x208) == '\x01') &&
             (((((*(float *)(param_3 + 0x1e0) != 0.0 || (*(float *)(param_3 + 0x1e4) != 0.0)) ||
                (NAN(*(float *)(param_3 + 0x1e4)))) ||
               ((*(long *)(param_3 + 0x210) != *(long *)(param_3 + 0x218) ||
                (*(float *)(param_3 + 0x1e8) != 1.0)))) || (NAN(*(float *)(param_3 + 0x1e8)))))) {
            FUN_0184a9b0(param_3,*param_4,0);
          }
          uVar8 = uVar8 + 1;
        } while ((uVar19 & 0xffffffff) != uVar8);
      }
      uVar12 = 0;
      uVar8 = uVar19 & 0xffffffff;
      if (iVar14 == 4) {
        uVar8 = uVar12;
      }
      uVar9 = (long)&local_98 - lVar20;
      if (cVar5 == '\0') {
        uVar9 = uVar12;
      }
      *(undefined4 *)((long)&local_98 + uVar19 * 4) = 0;
      if (iVar14 != 1) {
        uVar12 = uVar19 & 0xffffffff;
      }
      FUN_0190aa50(local_f0,uVar8,lVar16,iVar11 + iVar18,uVar7,(long)local_e8 - lVar20,uVar9,
                   &local_114,0xffffffffffffffff,uVar12);
      FUN_0184ad90(param_3,local_e8[uVar19]);
      if ((*(char *)(param_3 + 0x208) == '\x01') &&
         (((*(float *)(param_3 + 0x1e0) != 0.0 || (*(float *)(param_3 + 0x1e4) != 0.0)) ||
          ((NAN(*(float *)(param_3 + 0x1e4)) ||
           (((*(long *)(param_3 + 0x210) != *(long *)(param_3 + 0x218) ||
             (*(float *)(param_3 + 0x1e8) != 1.0)) || (NAN(*(float *)(param_3 + 0x1e8)))))))))) {
        FUN_0184a9b0(param_3,*param_4,0);
      }
      uVar13 = iVar18 + 1;
      iVar14 = (int)local_108;
      param_1 = local_f8;
      goto LAB_01852629;
    }
    local_e8[uVar19] = local_b8[uVar19];
    *(undefined4 *)((long)&local_98 + uVar19 * 4) = local_110;
    if (iVar14 == 2) {
      if (local_c0 + uVar19 == 0) break;
    }
    else {
      FUN_0184ad90(param_3);
      if ((*(char *)(param_3 + 0x208) == '\x01') &&
         ((((*(float *)(param_3 + 0x1e0) != 0.0 || (*(float *)(param_3 + 0x1e4) != 0.0)) ||
           (NAN(*(float *)(param_3 + 0x1e4)))) ||
          (((*(long *)(param_3 + 0x210) != *(long *)(param_3 + 0x218) ||
            (*(float *)(param_3 + 0x1e8) != 1.0)) || (NAN(*(float *)(param_3 + 0x1e8)))))))) {
        FUN_0184a9b0(param_3,*param_4,0);
      }
      if (local_c0 + uVar19 == 0) goto LAB_0185254c;
    }
    uVar19 = uVar19 + 1;
    lVar20 = lVar20 + -4;
    lVar17 = lVar17 + 4;
  } while( true );
  uVar19 = 0;
  do {
    FUN_0184ad90(param_3,local_e8[uVar19]);
    if ((*(char *)(param_3 + 0x208) == '\x01') &&
       (((*(float *)(param_3 + 0x1e0) != 0.0 || (*(float *)(param_3 + 0x1e4) != 0.0)) ||
        (((NAN(*(float *)(param_3 + 0x1e4)) ||
          ((*(long *)(param_3 + 0x210) != *(long *)(param_3 + 0x218) ||
           (*(float *)(param_3 + 0x1e8) != 1.0)))) || (NAN(*(float *)(param_3 + 0x1e8)))))))) {
      FUN_0184a9b0(param_3,*param_4,0);
    }
    uVar19 = uVar19 + 1;
  } while (local_100 != uVar19);
LAB_0185254c:
  uVar8 = local_100;
  uVar19 = local_108;
  *(undefined4 *)((long)&local_98 + local_100 * 4) = 0;
  iVar14 = (int)local_108;
  lVar17 = 0;
  if (cVar5 != '\0') {
    lVar17 = (long)&local_98 + local_100 * 4;
  }
  FUN_01909630(local_f0,local_108 & 0xffffffff,lVar16,iVar11 + iVar14,0,uVar7,local_e8 + local_100,
               lVar17);
  FUN_0184ad90(param_3,local_e8[uVar8]);
  if (*(char *)(param_3 + 0x208) == '\x01') {
    if (((*(float *)(param_3 + 0x1e0) != 0.0) || (*(float *)(param_3 + 0x1e4) != 0.0)) ||
       ((NAN(*(float *)(param_3 + 0x1e4)) ||
        (((*(long *)(param_3 + 0x210) != *(long *)(param_3 + 0x218) ||
          (*(float *)(param_3 + 0x1e8) != 1.0)) || (NAN(*(float *)(param_3 + 0x1e8)))))))) {
      FUN_0184a9b0(param_3,*param_4,0);
    }
    uVar19 = uVar19 & 0xffffffff;
  }
  else {
    uVar19 = uVar19 & 0xffffffff;
  }
LAB_01852629:
  FUN_018fc360(param_1[0x21],uVar13);
  iVar6 = (int)uVar19;
  *param_1 = *param_1 + 1;
  param_1[1] = param_1[1] + (ulong)(0 < iVar6);
  param_1[0xd] = param_1[0xd] + 1;
  param_1[0xe] = (local_100 & 0xffffffff) + param_1[0xe];
  param_1[0xf] = param_1[0xf] + (long)iVar6;
  iVar11 = 1;
  if (1 < iVar6) {
    iVar11 = iVar6;
  }
  param_1[0x10] = param_1[0x10] + (ulong)(iVar11 - 1);
  if (iVar6 < iVar14) {
    param_1[6] = param_1[6] + 1;
  }
  param_1[7] = param_1[7] + (long)local_114;
  *(uint *)(param_1 + 0x34) = local_b8[0];
  if (0 < (int)uVar13) {
    uVar19 = 0;
    do {
      param_1[5] = param_1[5] + 1;
      FUN_01882d20(param_1 + 0x4a,param_1[0x21],local_e8[uVar19],param_1);
      uVar15 = local_e8[uVar19];
      plVar1 = (long *)param_1[0x48];
      if (plVar1 != (long *)0x0) {
        if (((int)uVar15 < 0) ||
           (cVar5 = (**(code **)(*plVar1 + 0x20))(plVar1,uVar15), cVar5 == '\0')) {
          *(uint *)(param_1 + 0x20) = uVar15;
          fprintf(_stderr,
                  "flash_spec: token %d is illegal under the grammar (the mask was not applied); ending the request\n"
                  ,(ulong)uVar15);
          return 0xfffffffe;
        }
        uVar15 = local_e8[uVar19];
      }
      uVar10 = FUN_01851580(*(undefined4 *)((long)&local_98 + uVar19 * 4),&stack0x00000008,uVar15);
      if ((int)uVar10 != 0) {
        *(uint *)((long)param_1 + 0x19c) = local_e8[(ulong)uVar13 - 1];
        return uVar10;
      }
      uVar19 = uVar19 + 1;
    } while (uVar13 != uVar19);
  }
  *(uint *)((long)param_1 + 0x19c) = local_e8[(long)(int)uVar13 + -1];
  if (0 < (int)param_1[0x22]) {
    iVar14 = *(int *)((long)param_1 + 0x1a4) + 1;
    *(int *)((long)param_1 + 0x1a4) = iVar14;
    iVar11 = (uint)(0 < iVar6) + (int)param_1[0x35];
    *(int *)(param_1 + 0x35) = iVar11;
    if ((int)param_1[0x22] <= iVar14) {
      fVar22 = (float)iVar11 / (float)iVar14;
      pfVar2 = (float *)param_1[0x1b];
      if (pfVar2 == (float *)param_1[0x1c]) {
        if ((long)pfVar2 - param_1[0x1a] != 0x7ffffffffffffffc) {
          uVar8 = (long)pfVar2 - param_1[0x1a] >> 2;
          uVar12 = uVar8 + (uVar8 == 0);
          uVar19 = uVar12 + uVar8;
          if (0x1ffffffffffffffe < uVar19) {
            uVar19 = 0x1fffffffffffffff;
          }
          if (CARRY8(uVar12,uVar8)) {
            uVar19 = 0x1fffffffffffffff;
          }
                    /* WARNING: Subroutine does not return */
          operator_new(uVar19 * 4);
        }
LAB_01852aa4:
                    /* WARNING: Subroutine does not return */
        std::__throw_length_error("vector::_M_realloc_append");
      }
      *pfVar2 = fVar22;
      param_1[0x1b] = (long)(pfVar2 + 1);
      puVar3 = (undefined1 *)param_1[0x1e];
      if (puVar3 == (undefined1 *)param_1[0x1f]) {
        uVar19 = (long)puVar3 - param_1[0x1d];
        if (uVar19 != 0x7fffffffffffffff) {
          uVar12 = uVar19 + (uVar19 == 0);
          uVar8 = uVar12 + uVar19;
          if (0x7ffffffffffffffe < uVar8) {
            uVar8 = 0x7fffffffffffffff;
          }
          if (CARRY8(uVar12,uVar19)) {
            uVar8 = 0x7fffffffffffffff;
          }
                    /* WARNING: Subroutine does not return */
          operator_new(uVar8);
        }
        goto LAB_01852aa4;
      }
      *puVar3 = (char)param_1[0x44];
      param_1[0x1e] = param_1[0x1e] + 1;
      *(undefined1 *)(param_1 + 0x44) = 0;
      *(undefined8 *)((long)param_1 + 0x1a4) = 0;
      lVar16 = param_1[0x23];
      if (0 < (int)lVar16 && fVar22 < *(float *)((long)param_1 + 0x114)) {
        lVar17 = param_1[0x21];
        *(undefined1 *)(lVar17 + 0xb7d) = 0;
        *(undefined4 *)(lVar17 + 0x40) = 0xffffffff;
        param_1[3] = param_1[3] + 1;
        *(int *)((long)param_1 + 0x224) = (int)lVar16;
        uVar7 = 0xffffffff;
        goto LAB_01852a2e;
      }
    }
  }
  iVar11 = FUN_018f5260(param_1[0x21]);
  lVar16 = param_1[0x21];
  if ((0 < iVar11) && (iVar11 = (int)param_1[0x4b], *(int *)(lVar16 + 0x78) != iVar11)) {
    if (iVar11 < 0) {
      iVar14 = -1;
      lVar17 = lVar16;
    }
    else {
      iVar6 = FUN_018f5260();
      iVar14 = -1;
      if (iVar11 < iVar6) {
        iVar14 = iVar11;
      }
      lVar17 = param_1[0x21];
    }
    *(int *)(lVar16 + 0x78) = iVar14;
    lVar16 = lVar17;
  }
  uVar7 = FUN_018f9c60(lVar16,local_e8,uVar13);
LAB_01852a2e:
  *(undefined4 *)(param_1 + 0x34) = uVar7;
  return 0;
}

