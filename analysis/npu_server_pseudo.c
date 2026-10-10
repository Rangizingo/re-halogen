
/* WARNING: Type propagation algorithm not settling */
/* WARNING: Globals starting with '_' overlap smaller symbols at the same address */

ulong FUN_00133090(int param_1,long param_2)

{
  pthread_mutex_t *__mutex;
  byte bVar1;
  undefined4 uVar2;
  long ******pppppplVar3;
  sockaddr *psVar4;
  long ****pppplVar5;
  long ******pppppplVar6;
  undefined1 auVar7 [32];
  undefined1 auVar8 [32];
  long ******pppppplVar9;
  undefined1 auVar10 [16];
  undefined1 auVar11 [16];
  undefined8 uVar12;
  ulong *puVar13;
  ulong uVar14;
  pollfd pVar15;
  char cVar16;
  byte bVar17;
  undefined2 uVar18;
  uint uVar19;
  int iVar20;
  undefined4 uVar21;
  uint uVar22;
  uint uVar23;
  uint uVar24;
  uint uVar25;
  int iVar26;
  ulong uVar27;
  FILE *__stream;
  istream *piVar28;
  ulong uVar29;
  ulong uVar30;
  ulong uVar31;
  undefined8 *puVar32;
  void *pvVar33;
  char *pcVar34;
  ulong uVar35;
  string *psVar36;
  void *__ptr;
  void *__ptr_00;
  void *__dest;
  double *pdVar37;
  double *pdVar38;
  long *******ppppppplVar39;
  undefined8 uVar40;
  char *******pppppppcVar41;
  int *piVar42;
  long *******ppppppplVar43;
  long *******ppppppplVar44;
  long *plVar45;
  long lVar46;
  char *******pppppppcVar47;
  long *******ppppppplVar48;
  long lVar49;
  ulong *puVar50;
  undefined1 *puVar51;
  long *****ppppplVar52;
  pthread_mutex_t *ppVar53;
  undefined1 *puVar54;
  undefined4 *puVar55;
  undefined8 *puVar56;
  double *pdVar57;
  addrinfo *paVar58;
  long lVar59;
  void *__src;
  double *pdVar60;
  size_t sVar61;
  undefined1 auVar65 [16];
  double dVar62;
  double dVar63;
  double dVar64;
  undefined1 auVar66 [32];
  undefined1 auVar67 [32];
  undefined1 auVar68 [32];
  undefined1 auVar69 [16];
  undefined1 auVar70 [16];
  float fVar71;
  undefined1 auVar72 [16];
  undefined1 auVar73 [32];
  undefined1 auVar74 [32];
  undefined1 auVar75 [32];
  undefined1 auVar76 [32];
  undefined1 auVar77 [32];
  undefined1 auVar78 [32];
  undefined1 auVar79 [32];
  undefined1 auVar80 [32];
  undefined1 auVar81 [32];
  undefined1 auVar82 [32];
  undefined1 auVar83 [32];
  undefined1 auVar84 [32];
  undefined1 auVar85 [64];
  undefined1 auVar86 [32];
  undefined1 auVar87 [32];
  undefined1 *local_508;
  int local_500;
  undefined1 local_4e0 [32];
  int local_4c0;
  undefined4 uStack_4bc;
  undefined4 local_480;
  undefined4 uStack_47c;
  undefined4 local_420;
  undefined4 uStack_41c;
  undefined8 uStack_418;
  pollfd *local_400;
  long local_3f0;
  int local_3e8;
  int local_3e4 [2];
  undefined4 local_3dc;
  undefined8 local_3d8;
  undefined8 local_3d0;
  undefined8 uStack_3c8;
  undefined1 local_3c0 [16];
  undefined8 local_3b0;
  undefined1 local_3a0 [16];
  long local_390;
  undefined1 local_380 [16];
  undefined8 local_370;
  undefined1 local_360 [16];
  undefined8 local_350;
  string local_340 [32];
  undefined1 *local_320;
  ulong local_318;
  long local_310;
  ulong *local_300;
  undefined8 local_2f8;
  ulong local_2f0 [2];
  pollfd local_2e0;
  undefined8 local_2d8;
  ulong local_2d0 [2];
  char *******local_2c0;
  char *******local_2b8;
  char *******local_2b0 [2];
  ulong *local_2a0;
  undefined8 local_298;
  ulong local_290 [2];
  long ******local_280 [2];
  long ****local_270 [2];
  undefined1 local_260 [16];
  long *******local_250 [10];
  long *****local_200;
  long *****local_1f8;
  undefined1 local_1f0 [8];
  long local_1e8;
  undefined4 local_1e0;
  undefined8 local_1d8;
  undefined8 uStack_1d0;
  undefined8 uStack_1c8;
  undefined8 uStack_1c0;
  undefined8 local_1b8;
  undefined8 uStack_1b0;
  undefined8 uStack_1a8;
  undefined8 uStack_1a0;
  undefined8 local_198;
  undefined8 uStack_190;
  undefined8 uStack_188;
  undefined8 uStack_180;
  undefined8 local_178;
  undefined8 uStack_170;
  undefined8 uStack_168;
  undefined8 uStack_160;
  undefined8 local_158;
  undefined8 uStack_150;
  undefined8 uStack_148;
  undefined8 uStack_140;
  undefined8 local_138;
  undefined8 uStack_130;
  undefined8 local_128;
  undefined4 local_120;
  undefined1 local_118 [16];
  undefined1 local_108 [16];
  undefined8 local_f8;
  undefined8 uStack_f0;
  undefined4 local_e0 [2];
  undefined8 local_d8;
  undefined4 *local_d0;
  undefined4 *local_c8;
  undefined8 local_c0;
  undefined4 local_b0 [2];
  undefined8 local_a8;
  undefined4 *local_a0;
  undefined4 *local_98;
  undefined8 local_90;
  undefined8 local_88;
  undefined4 local_80;
  undefined8 local_78;
  undefined8 uStack_70;
  undefined8 uStack_68;
  undefined8 uStack_60;
  undefined8 local_58;
  undefined8 uStack_50;
  undefined8 local_48;
  
  bVar17 = 0;
  local_3f0 = param_2;
  local_3e4[0] = param_1;
  if (param_1 < 2) {
    fwrite("usage: halogen-npu selftest bf16|norm|head|resid ... | replay DIR [REPS] | serve DEVICES_DIR WEIGHTS [ORDER] [--model DEVICES_DIR WEIGHTS]... | listen DEVICES_DIR WEIGHTS HOST:PORT [ORDER] [--model DEVICES_DIR WEIGHTS]... | probe MODEL_DIR | convert MODEL_DIR DEVICES_DIR OUT [--name ID] [--source SRC] [--if-stale]\n"
           ,1,0x13c,stderr);
    return 1;
  }
  FUN_0011a280(local_340,*(undefined8 *)(param_2 + 8));
  cVar16 = FUN_00139a60(local_340,"selftest");
  if ((cVar16 == '\0') || (local_3e4[0] < 3)) {
    cVar16 = FUN_00139a60(local_340,"cache");
    if (cVar16 == '\0') {
      cVar16 = FUN_00139a60(local_340,"probe");
      if ((cVar16 == '\0') && (cVar16 = FUN_00139a60(local_340,"convert"), cVar16 == '\0')) {
        cVar16 = FUN_00139a60(local_340,&DAT_001ef5d4);
        if (cVar16 == '\0') {
          auVar65._8_8_ = 0;
          auVar65._0_8_ = &local_3f0;
          auVar65 = vpinsrq_avx(auVar65,local_3e4,1);
          uStack_418 = auVar65._8_8_;
          cVar16 = FUN_00139a60(local_340,"replay");
          if (cVar16 == '\0') {
            uStack_3c8 = uStack_418;
            local_3b0 = 0;
            local_3d8 = 0;
            local_3c0 = (undefined1  [16])0x0;
            local_3d0 = auVar65._0_8_;
            cVar16 = FUN_00139a60(local_340,"serve");
            if (cVar16 == '\0') {
LAB_00133f37:
              cVar16 = FUN_00139a60(local_340,"listen");
              if ((cVar16 == '\0') || (local_3e4[0] < 5)) goto LAB_00133ec9;
              cVar16 = FUN_00120f50(&local_3d0,1,local_3c0,&local_3d8);
              uVar40 = local_3d8;
              if (cVar16 != '\0') {
                FUN_0011a280((string *)&local_320,*(undefined8 *)(local_3f0 + 0x20));
                ppppplVar52 = (long *****)&local_200;
                for (lVar46 = 0x14; lVar46 != 0; lVar46 = lVar46 + -1) {
                  *(undefined4 *)ppppplVar52 = 0;
                  ppppplVar52 = (long *****)((long)ppppplVar52 + (ulong)bVar17 * -8 + 4);
                }
                cVar16 = FUN_001278f0(&local_200,local_3c0,uVar40,4);
                uVar27 = 1;
                if (cVar16 != '\0') {
                  auVar69._8_8_ = 0;
                  auVar69._0_8_ = local_380._8_8_;
                  local_380 = auVar69 << 0x40;
                  puVar32 = operator_new(0xf0);
                  __mutex = (pthread_mutex_t *)(puVar32 + 2);
                  puVar32[1] = 0x100000001;
                  *puVar32 = &PTR_FUN_002121b8;
                  ppVar53 = __mutex;
                  for (lVar46 = 0x38; lVar46 != 0; lVar46 = lVar46 + -1) {
                    (ppVar53->__data).__lock = 0;
                    ppVar53 = (pthread_mutex_t *)((long)ppVar53 + (ulong)bVar17 * -8 + 4);
                  }
                  std::condition_variable::condition_variable((condition_variable *)(puVar32 + 7));
                  puVar32[0xd] = 0;
                  puVar32[0xe] = 8;
                  puVar32[0xf] = 0;
                  puVar32[0x10] = 0;
                  puVar32[0x11] = 0;
                  puVar32[0x12] = 0;
                  puVar32[0x13] = 0;
                  puVar32[0x14] = 0;
                  puVar32[0x15] = 0;
                  puVar32[0x16] = 0;
                  pvVar33 = operator_new(0x40);
                  puVar32[0xd] = pvVar33;
                  puVar56 = (undefined8 *)
                            ((long)pvVar33 + (puVar32[0xe] * 4 - 4U & 0xfffffffffffffff8));
                  pvVar33 = operator_new(0x1f0);
                  *puVar56 = pvVar33;
                  auVar72._8_8_ = 0;
                  auVar72._0_8_ = pvVar33;
                  auVar69 = vpunpcklqdq_avx(auVar72,auVar72);
                  *(undefined1 *)(puVar32 + 0x17) = 0;
                  auVar70._8_8_ = 0;
                  auVar70._0_8_ = (long)pvVar33 + 0x1f0;
                  auVar65 = vpinsrq_avx(auVar70,puVar56,1);
                  puVar32[0x1c] = 0;
                  auVar68._0_16_ = ZEXT116(0) * auVar65 + ZEXT116(1) * auVar69;
                  auVar68._16_16_ =
                       ZEXT116(0) * SUB4816((undefined1  [48])0x0,0) + ZEXT116(1) * auVar65;
                  puVar32[0x1d] = 0;
                  *(undefined1 (*) [32])(puVar32 + 0xf) = auVar68;
                  *(undefined1 (*) [32])(puVar32 + 0x13) = auVar68;
                  local_380._0_8_ = __mutex;
                  *(undefined8 *)((long)puVar32 + 0xbc) = 0;
                  *(undefined8 *)((long)puVar32 + 0xc4) = 1;
                  *(undefined1 (*) [16])(puVar32 + 0x1a) = (undefined1  [16])0x0;
                  pcVar34 = getenv("HALOGEN_NPU_QUEUE");
                  iVar26 = 0x40;
                  if ((pcVar34 != (char *)0x0) && (*pcVar34 != '\0')) {
                    iVar26 = __isoc23_strtol(pcVar34,0,10);
                  }
                  if (iVar26 < 1) {
                    iVar26 = 1;
                  }
                  *(int *)(puVar32 + 0x18) = iVar26;
                  pcVar34 = getenv("HALOGEN_NPU_BATCH");
                  uVar19 = 1;
                  if ((pcVar34 != (char *)0x0) && (*pcVar34 != '\0')) {
                    iVar26 = __isoc23_strtol(pcVar34,0,10);
                    uVar19 = (uint)(iVar26 != 0);
                  }
                  *(uint *)(puVar32 + 0x19) = uVar19;
                  *(undefined4 *)((long)puVar32 + 0xc4) = 0x40;
                  FUN_0011b5e0(local_360,&local_200);
                  local_250[0] = (long *******)puVar32[0x1c];
                  local_260._0_8_ = puVar32[0x1a];
                  local_260._8_8_ = puVar32[0x1b];
                  puVar32[0x1c] = local_350;
                  puVar32[0x1a] = local_360._0_8_;
                  puVar32[0x1b] = local_360._8_8_;
                  FUN_0011a600((addrinfo *)local_260);
                  local_350 = 0;
                  local_360 = (undefined1  [16])0x0;
                  FUN_0011a600(local_360);
                  puVar51 = local_320;
                  local_300 = local_2f0;
                  local_2f0[0] = local_2f0[0] & 0xffffffffffffff00;
                  local_2f8 = 0;
                  if (local_318 != 0) {
                    uVar27 = local_318 - 1;
LAB_0013422f:
                    uVar35 = uVar27 - 1;
                    if (local_320[uVar27] != ':') goto code_r0x00134245;
                    if (0xfffffffffffffffd < uVar35) goto LAB_001346e7;
                    local_2e0 = (pollfd)local_2d0;
                    uVar35 = local_318;
                    if (uVar27 <= local_318) {
                      uVar35 = uVar27;
                    }
                    if (uVar35 < 0x10) {
                      if (uVar35 != 1) goto LAB_001355c1;
                      local_2d0[0] = CONCAT71(local_2d0[0]._1_7_,*local_320);
                    }
                    else {
                      if ((long)uVar35 < 0) {
                    /* WARNING: Subroutine does not return */
                        std::__throw_length_error("basic_string::_M_create");
                      }
                      if ((long)(uVar35 + 1) < 0) {
                    /* WARNING: Subroutine does not return */
                        std::__throw_bad_alloc();
                      }
                      local_2e0 = (pollfd)operator_new(uVar35 + 1);
                      local_2d0[0] = uVar35;
LAB_001355c1:
                      memcpy((void *)local_2e0,puVar51,uVar35);
                    }
                    *(char *)((long)local_2e0 + uVar35) = '\0';
                    uVar27 = uVar27 + 1;
                    local_2d8 = uVar35;
                    if (local_318 < uVar27) {
                      iVar26 = 0x1f1180;
                      std::__throw_out_of_range_fmt
                                ("%s: __pos (which is %zu) > this->size() (which is %zu)",
                                 "basic_string::substr");
                      goto LAB_00109ee9;
                    }
                    local_2c0 = (char *******)local_2b0;
                    pppppppcVar47 = (char *******)(local_318 - uVar27);
                    puVar51 = local_320 + uVar27;
                    if (pppppppcVar47 < (char *******)0x10) {
                      if (pppppppcVar47 == (char *******)0x1) {
                        local_2b0[0] = (char *******)CONCAT71(local_2b0[0]._1_7_,*puVar51);
                      }
                      else if (pppppppcVar47 != (char *******)0x0) goto LAB_00135174;
                    }
                    else {
                      if ((long)pppppppcVar47 < 0) {
                    /* WARNING: Subroutine does not return */
                        std::__throw_length_error("basic_string::_M_create");
                      }
                      if ((long)((long)pppppppcVar47 + 1U) < 0) {
                    /* WARNING: Subroutine does not return */
                        std::__throw_bad_alloc();
                      }
                      local_2c0 = operator_new((long)pppppppcVar47 + 1U);
                      local_2b0[0] = pppppppcVar47;
LAB_00135174:
                      memcpy(local_2c0,puVar51,(size_t)pppppppcVar47);
                    }
                    pVar15 = local_2e0;
                    *(undefined1 *)((long)local_2c0 + (long)pppppppcVar47) = 0;
                    local_2b8 = pppppppcVar47;
                    if (((2 < local_2d8) && (*(char *)local_2e0 == '[')) &&
                       (((char *)((long)local_2e0 + -1))[local_2d8] == ']')) {
                      local_260._0_8_ = local_250;
                      ppppppplVar39 = (long *******)(local_2d8 - 1);
                      if ((long *******)(local_2d8 - 2) <= (long *******)(local_2d8 - 1)) {
                        ppppppplVar39 = (long *******)(local_2d8 - 2);
                      }
                      if (ppppppplVar39 < (long *******)0x10) {
                        if (ppppppplVar39 != (long *******)0x1) goto LAB_00135b09;
                        local_250[0] = (long *******)
                                       CONCAT71(local_250[0]._1_7_,*(char *)((long)local_2e0 + 1));
                      }
                      else {
                        if ((long)ppppppplVar39 < 0) {
                    /* WARNING: Subroutine does not return */
                          std::__throw_length_error("basic_string::_M_create");
                        }
                        if ((long)((long)ppppppplVar39 + 1U) < 0) {
                    /* WARNING: Subroutine does not return */
                          std::__throw_bad_alloc();
                        }
                        local_260._0_8_ = operator_new((long)ppppppplVar39 + 1U);
                        local_250[0] = ppppppplVar39;
LAB_00135b09:
                        memcpy((void *)local_260._0_8_,(char *)((long)pVar15 + 1),
                               (size_t)ppppppplVar39);
                      }
                      *(undefined1 *)(local_260._0_8_ + (long)ppppppplVar39) = 0;
                      local_260._8_8_ = ppppppplVar39;
                      FUN_00119de0(&local_2e0,(addrinfo *)local_260);
                      std::__cxx11::string::_M_dispose((string *)local_260);
                    }
                    local_400 = &local_2e0;
                    auVar10._8_8_ = 0;
                    auVar10._0_8_ = local_3a0._8_8_;
                    local_3a0 = auVar10 << 0x40;
                    puVar55 = (undefined4 *)(local_260 + 4);
                    for (lVar46 = 0xb; lVar46 != 0; lVar46 = lVar46 + -1) {
                      *puVar55 = 0;
                      puVar55 = puVar55 + (ulong)bVar17 * -2 + 1;
                    }
                    local_260._8_4_ = 1;
                    local_260._0_4_ = 0x400;
                    iVar26 = getaddrinfo((char *)local_2e0,(char *)local_2c0,(addrinfo *)local_260,
                                         (addrinfo **)local_3a0);
                    if ((iVar26 != 0) || ((addrinfo *)local_3a0._0_8_ == (addrinfo *)0x0)) {
                      std::__cxx11::string::_M_dispose((string *)&local_2c0);
                      std::__cxx11::string::_M_dispose((string *)local_400);
                      goto LAB_001346e7;
                    }
                    paVar58 = (addrinfo *)local_3a0._0_8_;
                    do {
                      iVar26 = paVar58->ai_family;
                      if (iVar26 == 2) {
                        if (paVar58->ai_addr->sa_data[2] != '\x7f') goto LAB_00135124;
LAB_001350a7:
                        local_4c0 = socket(iVar26,paVar58->ai_socktype,paVar58->ai_protocol);
                        if (local_4c0 < 0) goto LAB_00135124;
                        local_3dc = 1;
                        setsockopt(local_4c0,1,2,&local_3dc,4);
                        iVar26 = bind(local_4c0,paVar58->ai_addr,paVar58->ai_addrlen);
                        if ((iVar26 != 0) || (iVar26 = listen(local_4c0,0x40), iVar26 != 0)) {
                          close(local_4c0);
                          goto LAB_00135124;
                        }
                        FUN_00127670((string *)local_280,local_2e0,local_2d8,&DAT_001ef2d8);
                        FUN_00125080((string *)&local_2a0,(string *)local_280,local_2c0,local_2b8);
                        FUN_00119de0(&local_300,(string *)&local_2a0);
                        std::__cxx11::string::_M_dispose((string *)&local_2a0);
                        cVar16 = '\0';
                        std::__cxx11::string::_M_dispose((string *)local_280);
                        goto LAB_0013563e;
                      }
                      if (((iVar26 == 10) &&
                          (psVar4 = paVar58->ai_addr, *(int *)(psVar4->sa_data + 6) == 0)) &&
                         ((*(int *)(psVar4->sa_data + 10) == 0 &&
                          ((iVar20._0_2_ = psVar4[1].sa_family, iVar20._2_1_ = psVar4[1].sa_data[0],
                           iVar20._3_1_ = psVar4[1].sa_data[1], iVar20 == 0 &&
                           (*(int *)(psVar4[1].sa_data + 2) == 0x1000000)))))) goto LAB_001350a7;
LAB_00135124:
                      paVar58 = paVar58->ai_next;
                    } while (paVar58 != (addrinfo *)0x0);
                    local_4c0 = -1;
LAB_0013563e:
                    freeaddrinfo((addrinfo *)local_3a0._0_8_);
                    std::__cxx11::string::_M_dispose((string *)&local_2c0);
                    std::__cxx11::string::_M_dispose((string *)local_400);
                    if (cVar16 != '\0') goto LAB_001346e7;
                    iVar26 = pipe(&DAT_00213548);
                    if (iVar26 != 0) {
                      FUN_0011a280((addrinfo *)local_260,&DAT_001ef646);
                      fprintf(stderr,"halogen-npu: %s\n",local_260._0_8_);
                      std::__cxx11::string::_M_dispose((string *)local_260);
                      goto LAB_00134758;
                    }
                    signal(0xf,FUN_001199e0);
                    signal(2,FUN_001199e0);
                    signal(0xd,(__sighandler_t)0x1);
                    auVar11._8_8_ = 0;
                    auVar11._0_8_ = local_3a0._8_8_;
                    local_3a0 = auVar11 << 0x40;
                    ppppppplVar39 = operator_new(0x28);
                    local_380._8_8_ = puVar32;
                    *ppppppplVar39 = (long ******)&PTR_FUN_00212168;
                    ppppppplVar39[1] = (long ******)__mutex;
                    FUN_00139d70(ppppppplVar39 + 2,local_380 + 8);
                    ppppppplVar39[3] = &local_200;
                    ppppppplVar39[4] = (long ******)FUN_001362d0;
                    local_260._0_8_ = ppppppplVar39;
                    std::thread::_M_start_thread(local_3a0,(addrinfo *)local_260,FUN_00139700);
                    if ((long *******)local_260._0_8_ != (long *******)0x0) {
                      (*(code *)(*(long *******)local_260._0_8_)[1])();
                    }
                    puVar13 = local_300;
                    uVar21 = *(undefined4 *)(puVar32 + 0x18);
                    if (local_1f8 == local_200) {
                      printf("READY image listen %s queue %d models 1\n",local_300);
                    }
                    else {
                      pppplVar5 = *local_200;
                      lVar46 = (long)local_1f8 - (long)local_200;
                      uVar19 = *(uint *)(puVar32 + 0x19);
                      uVar2 = *(undefined4 *)(pppplVar5 + 0xe);
                      bVar1 = *(byte *)((long)pppplVar5 + 0xe4);
                      local_4e0._0_4_ = *(undefined4 *)(pppplVar5 + 0x1c);
                      uVar22 = *(uint *)pppplVar5[0x5b];
                      uVar40 = FUN_0015f8d0(*(undefined4 *)(pppplVar5 + 0x23));
                      printf("READY hidden %d layers %d order %s threads %d wait_ms %d op %d listen %s contexts %d queue %d batch %d models %zu\n"
                             ,(ulong)*(uint *)(pppplVar5 + 0x1b),
                             (ulong)*(uint *)((long)pppplVar5 + 0xdc),uVar40,(ulong)uVar22,
                             (ulong)(uint)local_4e0._0_4_,(ulong)bVar1,puVar13,uVar2,uVar21,
                             (ulong)uVar19,(lVar46 >> 3) + (ulong)(local_1e8 != 0));
                    }
                    fflush(stdout);
                    auVar7._24_8_ = 0;
                    auVar7._0_24_ = local_4e0._8_24_;
                    local_4e0 = auVar7 << 0x40;
                    ppppppplVar39 = (long *******)0x0;
                    local_480 = 0;
                    uStack_47c = 0;
                    do {
                      while( true ) {
                        if (DAT_00213700 != 0) goto LAB_00135bbd;
                        local_2e0.events = 1;
                        local_2e0.revents = 0;
                        local_2e0.fd = local_4c0;
                        local_2d8 = CONCAT44(1,DAT_00213548);
                        iVar26 = poll(local_400,2,-1);
                        if (iVar26 < 0) break;
                        if (local_2d8._6_2_ != 0) goto LAB_00135bbd;
                        if ((((ulong)local_2e0 & 0x1000000000000) != 0) &&
                           (iVar26 = accept(local_4c0,(sockaddr *)0x0,(socklen_t *)0x0), -1 < iVar26
                           )) {
                          local_3dc = 1;
                          setsockopt(iVar26,6,1,&local_3dc,4);
                          local_2c0 = (char *******)0x0;
                          pppppppcVar41 = operator_new(0x208);
                          pppppppcVar41[1] = (char ******)0x100000001;
                          *pppppppcVar41 = (char ******)&PTR_FUN_002121f0;
                          pppppppcVar47 = pppppppcVar41 + 2;
                          for (lVar46 = 0x3f; lVar46 != 0; lVar46 = lVar46 + -1) {
                            *pppppppcVar47 = (char ******)0x0;
                            pppppppcVar47 = pppppppcVar47 + (ulong)bVar17 * -2 + 1;
                          }
                          *(undefined1 *)(pppppppcVar41 + 8) = 1;
                          pppppppcVar47 = pppppppcVar41 + 2;
                          pppppppcVar41[9] = (char ******)(pppppppcVar41 + 0xb);
                          pppppppcVar41[0x2f] = (char ******)(pppppppcVar41 + 0x2d);
                          pppppppcVar41[0x30] = (char ******)(pppppppcVar41 + 0x2d);
                          pppppppcVar41[0x35] = (char ******)(pppppppcVar41 + 0x33);
                          pppppppcVar41[0x36] = (char ******)(pppppppcVar41 + 0x33);
                          *(int *)(pppppppcVar41 + 2) = iVar26;
                          local_2c0 = pppppppcVar47;
                          local_2b8 = pppppppcVar41;
                          FUN_00139a40(__mutex);
                          if (*(int *)((long)puVar32 + 0xbc) < *(int *)((long)puVar32 + 0xc4)) {
                            *(int *)((long)puVar32 + 0xbc) = *(int *)((long)puVar32 + 0xbc) + 1;
                            pthread_mutex_unlock(__mutex);
                            ppppppplVar43 = (long *******)CONCAT44(uStack_47c,local_480);
                            while (ppppppplVar39 != ppppppplVar43) {
                              pppppplVar3 = ppppppplVar43[1];
                              FUN_00139a40();
                              cVar16 = *(char *)(pppppplVar3 + 6);
                              pthread_mutex_unlock((pthread_mutex_t *)(pppppplVar3 + 1));
                              if (cVar16 == '\0') {
                                std::thread::join();
                                close(*(int *)ppppppplVar43[1]);
                                if (ppppppplVar39 != ppppppplVar43 + 3) {
                                  ppppppplVar48 = ppppppplVar43;
                                  for (lVar46 = ((long)ppppppplVar39 - (long)(ppppppplVar43 + 3) >>
                                                3) * -0x5555555555555555; 0 < lVar46;
                                      lVar46 = lVar46 + -1) {
                                    if (*ppppppplVar48 != (long ******)0x0) goto code_r0x00109dd4;
                                    pppppplVar3 = ppppppplVar48[3];
                                    pppppplVar9 = ppppppplVar48[4];
                                    pppppplVar6 = ppppppplVar48[2];
                                    ppppppplVar48[3] = (long ******)0x0;
                                    *ppppppplVar48 = pppppplVar3;
                                    *(undefined1 (*) [16])(ppppppplVar48 + 4) =
                                         (undefined1  [16])0x0;
                                    ppppppplVar48[1] = pppppplVar9;
                                    ppppppplVar48[2] = ppppppplVar48[5];
                                    if (pppppplVar6 != (long ******)0x0) {
                                      FUN_00139cf0();
                                    }
                                    ppppppplVar48 = ppppppplVar48 + 3;
                                  }
                                }
                                if (ppppppplVar39[-1] != (long ******)0x0) {
                                  FUN_00139cf0();
                                }
                                ppppppplVar48 = ppppppplVar39 + -3;
                                ppppppplVar39 = ppppppplVar39 + -3;
                                if (*ppppppplVar48 != (long ******)0x0) goto code_r0x00109dd4;
                              }
                              else {
                                ppppppplVar43 = ppppppplVar43 + 3;
                              }
                            }
                            local_280[0] = (long ******)0x0;
                            ppppppplVar43 = operator_new(0x38);
                            *ppppppplVar43 = (long ******)&PTR_FUN_00212190;
                            ppppppplVar43[1] = (long ******)pppppppcVar47;
                            FUN_00139d70(ppppppplVar43 + 2);
                            ppppppplVar43[3] = (long ******)__mutex;
                            FUN_00139d70(ppppppplVar43 + 4,local_380 + 8);
                            ppppppplVar43[5] = &local_200;
                            ppppppplVar43[6] = (long ******)FUN_0012e9d0;
                            local_260._0_8_ = ppppppplVar43;
                            std::thread::_M_start_thread
                                      (local_280,(addrinfo *)local_260,FUN_00139700);
                            if ((long *******)local_260._0_8_ != (long *******)0x0) {
                              (*(code *)(*(long *******)local_260._0_8_)[1])();
                            }
                            pppppplVar3 = local_280[0];
                            if (ppppppplVar39 == (long *******)local_4e0._0_8_) {
                              lVar46 = (long)ppppppplVar39 - CONCAT44(uStack_47c,local_480);
                              uVar27 = (lVar46 >> 3) * -0x5555555555555555;
                              if (uVar27 == 0x555555555555555) {
                    /* WARNING: Subroutine does not return */
                                std::__throw_length_error("vector::_M_realloc_append");
                              }
                              uVar35 = 1;
                              if (uVar27 != 0) {
                                uVar35 = uVar27;
                              }
                              uVar29 = 0x555555555555555;
                              if ((!CARRY8(uVar27,uVar35)) &&
                                 (uVar29 = 0x555555555555555, uVar27 + uVar35 < 0x555555555555556))
                              {
                                uVar29 = uVar27 + uVar35;
                              }
                              ppppppplVar44 = operator_new(uVar29 * 0x18);
                              pppppplVar3 = local_280[0];
                              local_420 = SUB84(ppppppplVar44,0);
                              uStack_41c = (undefined4)((ulong)ppppppplVar44 >> 0x20);
                              local_280[0] = (long ******)0x0;
                              plVar45 = (long *)((long)ppppppplVar44 + lVar46);
                              *plVar45 = (long)pppppplVar3;
                              plVar45[1] = (long)pppppppcVar47;
                              FUN_00139d70(plVar45 + 2,&local_2b8);
                              ppppppplVar48 = ppppppplVar44;
                              for (ppppppplVar43 = (long *******)CONCAT44(uStack_47c,local_480);
                                  ppppppplVar39 != ppppppplVar43; ppppppplVar43 = ppppppplVar43 + 3)
                              {
                                *ppppppplVar48 = *ppppppplVar43;
                                *ppppppplVar43 = (long ******)0x0;
                                pppppplVar3 = *ppppppplVar43;
                                ppppppplVar48[1] = ppppppplVar43[1];
                                *(undefined1 (*) [16])(ppppppplVar43 + 1) = (undefined1  [16])0x0;
                                ppppppplVar48[2] = ppppppplVar43[2];
                                if (pppppplVar3 != (long ******)0x0) goto code_r0x00109dd4;
                                ppppppplVar48 = ppppppplVar48 + 3;
                              }
                              pvVar33 = (void *)CONCAT44(uStack_47c,local_480);
                              if (pvVar33 != (void *)0x0) {
                                operator_delete(pvVar33,local_4e0._0_8_ - (long)pvVar33);
                              }
                              local_4e0._0_8_ = ppppppplVar44 + uVar29 * 3;
                            }
                            else {
                              local_280[0] = (long ******)0x0;
                              *ppppppplVar39 = pppppplVar3;
                              ppppppplVar39[1] = (long ******)pppppppcVar47;
                              FUN_00139d70(ppppppplVar39 + 2,&local_2b8);
                              local_420 = local_480;
                              uStack_41c = uStack_47c;
                              ppppppplVar48 = ppppppplVar39;
                            }
                            ppppppplVar39 = ppppppplVar48 + 3;
                            if (local_280[0] != (long ******)0x0) {
code_r0x00109dd4:
                    /* WARNING: Subroutine does not return */
                              std::terminate();
                            }
                            FUN_00139cf0(pppppppcVar41);
                            local_480 = local_420;
                            uStack_47c = uStack_41c;
                          }
                          else {
                            FUN_0013a710((string *)&local_2a0);
                            psVar36 = (string *)FUN_00124f70((string *)&local_2a0,"ERR 0 busy: ");
                            std::__cxx11::string::string((string *)local_280,psVar36);
                            psVar36 = (string *)
                                      std::__cxx11::string::append
                                                ((string *)local_280," connections open");
                            std::__cxx11::string::string((string *)local_260,psVar36);
                            FUN_00124e10(pppppppcVar47,(addrinfo *)local_260);
                            std::__cxx11::string::_M_dispose((string *)local_260);
                            std::__cxx11::string::_M_dispose((string *)local_280);
                            std::__cxx11::string::_M_dispose((string *)&local_2a0);
                            close(iVar26);
                            pthread_mutex_unlock(__mutex);
                            FUN_00139cf0(pppppppcVar41);
                          }
                        }
                      }
                      piVar42 = __errno_location();
                    } while (*piVar42 == 4);
LAB_00135bbd:
                    close(local_4c0);
                    FUN_00139a40(__mutex);
                    *(undefined1 *)(puVar32 + 0x17) = 1;
                    pthread_mutex_unlock(__mutex);
                    std::condition_variable::notify_all();
                    std::thread::join();
                    ppppppplVar48 = (long *******)CONCAT44(uStack_47c,local_480);
                    ppppppplVar43 = (long *******)CONCAT44(uStack_47c,local_480);
                    while( true ) {
                      if (ppppppplVar43 == ppppppplVar39) {
                        for (; ppppppplVar48 != ppppppplVar39; ppppppplVar48 = ppppppplVar48 + 3) {
                          std::thread::join();
                          close(*(int *)ppppppplVar48[1]);
                        }
                        local_260._0_8_ = CONCAT44(uStack_47c,local_480);
                        local_250[0] = (long *******)local_4e0._0_8_;
                        local_260._8_8_ = ppppppplVar48;
                        FUN_001201b0((addrinfo *)local_260);
                        uVar27 = 0;
                        if ((addrinfo *)local_3a0._0_8_ != (addrinfo *)0x0) {
                    /* WARNING: Subroutine does not return */
                          std::terminate();
                        }
                        goto LAB_0013475e;
                      }
                      pppppplVar3 = ppppppplVar43[1];
                      iVar26 = pthread_mutex_lock((pthread_mutex_t *)(pppppplVar3 + 1));
                      if (iVar26 != 0) break;
                      if (*(char *)(pppppplVar3 + 6) != '\0') {
                        *(undefined1 *)(pppppplVar3 + 6) = 0;
                        shutdown(*(int *)pppppplVar3,2);
                      }
                      pthread_mutex_unlock((pthread_mutex_t *)(pppppplVar3 + 1));
                      ppppppplVar43 = ppppppplVar43 + 3;
                    }
LAB_00109ee9:
                    /* WARNING: Subroutine does not return */
                    std::__throw_system_error(iVar26);
                  }
LAB_001346e7:
                  FUN_001256c0((string *)local_280,"cannot listen on ",local_320,local_318);
                  psVar36 = (string *)
                            std::__cxx11::string::append
                                      ((string *)local_280," (a loopback HOST:PORT only)");
                  std::__cxx11::string::string((string *)local_260,psVar36);
                  fprintf(stderr,"halogen-npu: %s\n",local_260._0_8_);
                  std::__cxx11::string::_M_dispose((string *)local_260);
                  std::__cxx11::string::_M_dispose((string *)local_280);
LAB_00134758:
                  uVar27 = 1;
LAB_0013475e:
                  std::__cxx11::string::_M_dispose((string *)&local_300);
                  FUN_00139cf0(puVar32);
                }
                FUN_00123250(&local_200);
                std::__cxx11::string::_M_dispose((string *)&local_320);
                goto LAB_00133f08;
              }
LAB_001347c6:
              fwrite("usage: halogen-npu selftest bf16|norm|head|resid ... | replay DIR [REPS] | serve DEVICES_DIR WEIGHTS [ORDER] [--model DEVICES_DIR WEIGHTS]... | listen DEVICES_DIR WEIGHTS HOST:PORT [ORDER] [--model DEVICES_DIR WEIGHTS]... | probe MODEL_DIR | convert MODEL_DIR DEVICES_DIR OUT [--name ID] [--source SRC] [--if-stale]\n"
                     ,1,0x13c,stderr);
              goto LAB_00133f02;
            }
            if (local_3e4[0] < 4) goto LAB_00133ec9;
            cVar16 = FUN_00120f50(&local_3d0,0,local_3c0,&local_3d8);
            if (cVar16 == '\0') goto LAB_001347c6;
            puVar55 = (undefined4 *)local_260;
            for (lVar46 = 0x14; lVar46 != 0; lVar46 = lVar46 + -1) {
              *puVar55 = 0;
              puVar55 = puVar55 + (ulong)bVar17 * -2 + 1;
            }
            cVar16 = FUN_001278f0(local_260,local_3c0,local_3d8);
            if (cVar16 == '\0') {
              uVar27 = 1;
            }
            else {
              if (local_260._8_8_ == local_260._0_8_) {
                puts("READY image");
              }
              else {
                pppppplVar3 = *(long *******)local_260._0_8_;
                FUN_0015f8d0(*(undefined4 *)(pppppplVar3 + 0x23));
                printf("READY hidden %d layers %d order %s threads %d wait_ms %d op %d%s\n",
                       (ulong)*(uint *)(pppppplVar3 + 0x1b),
                       (ulong)*(uint *)((long)pppppplVar3 + 0xdc));
              }
              fflush(stdout);
              local_200 = (long *****)local_1f0;
              local_d0 = local_e0;
              local_a0 = local_b0;
              local_178 = 0;
              uStack_170 = 0;
              uStack_168 = 0;
              uStack_160 = 0;
              local_2e0.fd = 0;
              local_2e0.events = 1;
              local_2e0.revents = 0;
              local_158 = 0;
              uStack_150 = 0;
              uStack_148 = 0;
              uStack_140 = 0;
              local_a8 = 0;
              local_90 = 0;
              local_48 = 0;
              local_1f8 = (long *****)0x0;
              local_1f0[0] = 0;
              local_1e0 = 0;
              local_128 = 0;
              local_120 = 0;
              local_e0[0] = 0;
              local_d8 = 0;
              local_c0 = 0;
              local_b0[0] = 0;
              local_88 = 0;
              local_80 = 0;
              local_1d8 = 0;
              uStack_1d0 = 0;
              uStack_1c8 = 0;
              uStack_1c0 = 0;
              local_1b8 = 0;
              uStack_1b0 = 0;
              uStack_1a8 = 0;
              uStack_1a0 = 0;
              local_198 = 0;
              uStack_190 = 0;
              uStack_188 = 0;
              uStack_180 = 0;
              local_138 = 0;
              uStack_130 = 0;
              local_118 = (undefined1  [16])0x0;
              local_108 = (undefined1  [16])0x0;
              local_f8 = 0;
              uStack_f0 = 0;
              local_78 = 0;
              uStack_70 = 0;
              uStack_68 = 0;
              uStack_60 = 0;
              local_58 = 0;
              uStack_50 = 0;
              local_2d8 = (ulong)local_2d8._4_4_ << 0x20;
              local_2a0 = local_290;
              local_298 = 0;
              local_290[0] = local_290[0] & 0xffffffffffffff00;
              local_c8 = local_d0;
              local_98 = local_a0;
              while( true ) {
                plVar45 = *(long **)(&DAT_002136d0 + *(long *)(std::cin + -0x18));
                if (plVar45 == (long *)0x0) {
                  std::__throw_bad_cast();
                  std::__cxx11::string::_M_dispose((string *)&local_2a0);
                  FUN_0013a310(&local_200);
                  FUN_00123250(local_260);
                  uVar27 = FUN_00109db1();
                  return uVar27;
                }
                if ((char)plVar45[7] == '\0') {
                  std::ctype<char>::_M_widen_init();
                  cVar16 = '\n';
                  if (*(code **)(*plVar45 + 0x30) != std::ctype<char>::do_widen) {
                    cVar16 = (**(code **)(*plVar45 + 0x30))(plVar45,10);
                  }
                }
                else {
                  cVar16 = *(char *)((long)plVar45 + 0x43);
                }
                piVar28 = std::getline<char,std::char_traits<char>,std::allocator<char>>
                                    ((istream *)&std::cin,(string *)&local_2a0,cVar16);
                if (((byte)piVar28[*(long *)(*(long *)piVar28 + -0x18) + 0x20] & 5) != 0) break;
                FUN_00124560(&local_2c0,(string *)&local_2a0);
                if (local_2b8 == local_2c0) {
                  FUN_00139710(&local_2c0);
                }
                else {
                  if ((local_2c0[1] == (char ******)0x4) && (*(int *)*local_2c0 == 0x54495551)) {
                    FUN_00139710(&local_2c0);
                    break;
                  }
                  FUN_0012b110(local_280,local_260,&local_2c0,&local_200,&local_2e0);
                  fputs((char *)local_280[0],stdout);
                  fputc(10,stdout);
                  fflush(stdout);
                  pppppppcVar41 = local_2c0;
                  pppppppcVar47 = local_2b8;
                  if (local_280[0] != (long ******)local_270) {
                    operator_delete(local_280[0],(long)local_270[0] + 1);
                    pppppppcVar41 = local_2c0;
                    pppppppcVar47 = local_2b8;
                  }
                  for (; pppppppcVar47 != pppppppcVar41; pppppppcVar41 = pppppppcVar41 + 4) {
                    if ((char *******)*pppppppcVar41 != pppppppcVar41 + 2) {
                      operator_delete(*pppppppcVar41,(long)pppppppcVar41[2] + 1);
                    }
                  }
                  if (local_2c0 != (char *******)0x0) {
                    operator_delete(local_2c0,(long)local_2b0[0] - (long)local_2c0);
                  }
                }
              }
              uVar27 = 0;
              std::__cxx11::string::_M_dispose((string *)&local_2a0);
              FUN_0013a310(&local_200);
            }
            FUN_00123250(local_260);
          }
          else {
            if (2 < local_3e4[0]) {
              iVar26 = local_3e4[0];
              if (local_3e4[0] != 3) {
                iVar26 = __isoc23_strtol(*(undefined8 *)(local_3f0 + 0x18),0,10);
              }
              FUN_0011a280((string *)&local_200,*(undefined8 *)(local_3f0 + 0x10));
              uVar19 = FUN_00130a20((string *)&local_200,iVar26);
              uVar27 = (ulong)uVar19;
              std::__cxx11::string::_M_dispose((string *)&local_200);
              goto LAB_0013316c;
            }
            uStack_3c8 = uStack_418;
            local_3b0 = 0;
            local_3d8 = 0;
            local_3c0 = (undefined1  [16])0x0;
            local_3d0 = auVar65._0_8_;
            cVar16 = FUN_00139a60(local_340,"serve");
            if (cVar16 == '\0') goto LAB_00133f37;
LAB_00133ec9:
            fwrite("usage: halogen-npu selftest bf16|norm|head|resid ... | replay DIR [REPS] | serve DEVICES_DIR WEIGHTS [ORDER] [--model DEVICES_DIR WEIGHTS]... | listen DEVICES_DIR WEIGHTS HOST:PORT [ORDER] [--model DEVICES_DIR WEIGHTS]... | probe MODEL_DIR | convert MODEL_DIR DEVICES_DIR OUT [--name ID] [--source SRC] [--if-stale]\n"
                   ,1,0x13c,stderr);
LAB_00133f02:
            uVar27 = 1;
          }
LAB_00133f08:
          local_508 = local_3c0;
          FUN_0013ad80(local_508);
          goto LAB_0013316c;
        }
        uVar19 = FUN_001e47f0(local_3e4[0] + -2,local_3f0 + 0x10);
      }
      else {
        uVar19 = FUN_001a7be0(local_3e4[0] + -1,local_3f0 + 8);
      }
    }
    else {
      uVar19 = FUN_00171ca0(local_3e4[0] + -2,local_3f0 + 0x10);
    }
    uVar27 = (ulong)uVar19;
    goto LAB_0013316c;
  }
  FUN_0011a280(&local_2e0,*(undefined8 *)(local_3f0 + 0x10));
  cVar16 = FUN_00139a60(&local_2e0,"bf16");
  if ((cVar16 == '\0') || (local_3e4[0] != 4)) {
    cVar16 = FUN_00139a60(&local_2e0,"norm");
    if ((cVar16 == '\0') || (local_3e4[0] < 8)) {
      cVar16 = FUN_00139a60(&local_2e0,"head");
      if ((cVar16 != '\0') && (5 < local_3e4[0])) {
        local_500 = 0x400;
        if (local_3e4[0] != 6) {
          local_500 = __isoc23_strtol(*(undefined8 *)(local_3f0 + 0x30),0,10);
        }
        FUN_0011a280(local_260,*(undefined8 *)(local_3f0 + 0x28));
        FUN_0011a280((string *)local_280,*(undefined8 *)(local_3f0 + 0x20));
        FUN_0011a280((string *)&local_2a0);
        local_390 = 0;
        local_3a0 = (undefined1  [16])0x0;
        cVar16 = FUN_00120280((string *)&local_2a0,local_3a0);
        lVar46 = local_390;
        uVar40 = local_3a0._0_8_;
        if (cVar16 == '\0') {
          pvVar33 = (void *)0x0;
          local_4c0 = 0;
          uStack_4bc = 0;
LAB_001347f4:
          FUN_0011a280(&local_200,"cannot read hn or rows");
          fprintf(stderr,"halogen-npu: %s\n",local_200);
LAB_00133d13:
          uVar27 = 1;
          std::__cxx11::string::_M_dispose((string *)&local_200);
          if (pvVar33 != (void *)0x0) goto LAB_00133d26;
        }
        else {
          local_370 = 0;
          uVar12 = local_3a0._8_8_;
          local_380 = (undefined1  [16])0x0;
          cVar16 = FUN_00120280((string *)local_280,local_380);
          pvVar33 = (void *)local_380._0_8_;
          local_4c0 = (int)local_370;
          uStack_4bc = (undefined4)((ulong)local_370 >> 0x20);
          if (cVar16 == '\0') goto LAB_001347f4;
          lVar59 = (long)local_500;
          uVar35 = uVar12 - uVar40;
          uVar27 = lVar59 * 4;
          uVar29 = uVar35 / uVar27;
          if (uVar35 % uVar27 != 0) {
LAB_00133ce1:
            FUN_0011a280(&local_200,"sizes do not match H");
            fprintf(stderr,"halogen-npu: %s\n",local_200);
            goto LAB_00133d13;
          }
          uVar30 = local_380._8_8_ - local_380._0_8_;
          uVar31 = uVar30 / uVar27;
          if (uVar30 % uVar27 != 0) goto LAB_00133ce1;
          local_320 = (undefined1 *)0x0;
          local_310 = 0;
          FUN_00139da0(&local_320,uVar29 * lVar59,(string *)&local_200);
          lVar49 = local_310;
          puVar51 = local_320;
          local_300 = (ulong *)0x0;
          local_2f0[0] = 0;
          FUN_00139da0(&local_300,uVar31 * lVar59,(string *)&local_200);
          uVar14 = local_2f0[0];
          puVar13 = local_300;
          local_2c0 = (char *******)0x0;
          local_2b0[0] = (char *******)0x0;
          FUN_00139da0(&local_2c0,uVar29 * uVar31,(string *)&local_200);
          pppppppcVar41 = local_2b0[0];
          pppppppcVar47 = local_2c0;
          memcpy(puVar51,(void *)uVar40,uVar35);
          memcpy(puVar13,pvVar33,uVar30);
          lVar59 = 0;
          puVar54 = puVar51;
          for (uVar35 = 0; uVar29 != uVar35; uVar35 = uVar35 + 1) {
            puVar50 = puVar13;
            for (uVar30 = 0; uVar31 != uVar30; uVar30 = uVar30 + 1) {
              uVar21 = FUN_00161260(puVar54,puVar50,local_500);
              *(undefined4 *)((long)pppppppcVar47 + uVar30 * 4 + lVar59 * 4) = uVar21;
              puVar50 = (ulong *)((long)puVar50 + uVar27);
            }
            puVar54 = puVar54 + uVar27;
            lVar59 = lVar59 + uVar31;
          }
          cVar16 = FUN_00119f30(local_260._0_8_,pppppppcVar47,(long)local_2b8 - (long)pppppppcVar47)
          ;
          if (cVar16 == '\0') {
            FUN_001256c0((string *)&local_200,"cannot write ",local_260._0_8_,local_260._8_8_);
            fprintf(stderr,"halogen-npu: %s\n",local_200);
            std::__cxx11::string::_M_dispose((string *)&local_200);
          }
          uVar27 = (ulong)(cVar16 == '\0');
          if (pppppppcVar47 != (char *******)0x0) {
            operator_delete(pppppppcVar47,(long)pppppppcVar41 - (long)pppppppcVar47);
          }
          operator_delete(puVar13,uVar14 - (long)puVar13);
          operator_delete(puVar51,lVar49 - (long)puVar51);
LAB_00133d26:
          operator_delete(pvVar33,CONCAT44(uStack_4bc,local_4c0) - (long)pvVar33);
        }
        if ((addrinfo *)uVar40 != (addrinfo *)0x0) {
          operator_delete((void *)uVar40,lVar46 - uVar40);
        }
        std::__cxx11::string::_M_dispose((string *)&local_2a0);
        std::__cxx11::string::_M_dispose((string *)local_280);
        goto LAB_0013356d;
      }
      bVar17 = FUN_00139a60(&local_2e0,"resid");
      if ((bVar17 == 0) || (local_3e4[0] != 4)) {
        cVar16 = FUN_00139a60(&local_2e0,"hostx");
        if ((cVar16 == '\0') || (local_3e4[0] != 10)) {
          cVar16 = FUN_00139a60(&local_2e0,&DAT_001ef5c3);
          if ((cVar16 != '\0') && (local_3e4[0] == 7)) {
            FUN_0011a280(&local_200,*(undefined8 *)(local_3f0 + 0x30));
            FUN_0011a280(local_260,*(undefined8 *)(local_3f0 + 0x28));
            FUN_0011a280(local_280,*(undefined8 *)(local_3f0 + 0x20));
            FUN_0011a280(&local_2a0,*(undefined8 *)(local_3f0 + 0x18));
            uVar19 = FUN_0016c0e0(&local_2a0,local_280,local_260,&local_200);
            goto LAB_00133b17;
          }
          fwrite("usage: halogen-npu selftest bf16|norm|head|resid ... | replay DIR [REPS] | serve DEVICES_DIR WEIGHTS [ORDER] [--model DEVICES_DIR WEIGHTS]... | listen DEVICES_DIR WEIGHTS HOST:PORT [ORDER] [--model DEVICES_DIR WEIGHTS]... | probe MODEL_DIR | convert MODEL_DIR DEVICES_DIR OUT [--name ID] [--source SRC] [--if-stale]\n"
                 ,1,0x13c,stderr);
          uVar27 = 1;
        }
        else {
          uVar19 = __isoc23_strtol(*(undefined8 *)(local_3f0 + 0x48),0,10);
          uVar22 = __isoc23_strtol(*(undefined8 *)(local_3f0 + 0x40),0,10);
          uVar23 = __isoc23_strtol(*(undefined8 *)(local_3f0 + 0x38),0,10);
          uVar24 = __isoc23_strtol(*(undefined8 *)(local_3f0 + 0x30),0,10);
          iVar26 = __isoc23_strtol(*(undefined8 *)(local_3f0 + 0x28),0,10);
          uVar25 = __isoc23_strtol(*(undefined8 *)(local_3f0 + 0x20),0,10);
          lVar46 = __isoc23_strtoll(*(undefined8 *)(local_3f0 + 0x18),0,10);
          uVar27 = lVar46 * 2 + 0x3fU & 0xffffffffffffffc0;
          pvVar33 = (void *)aligned_alloc(0x40,lVar46 * 4 + 0x3fU & 0xffffffffffffffc0);
          __ptr = (void *)aligned_alloc(0x40,uVar27);
          __ptr_00 = (void *)aligned_alloc(0x40);
          uVar35 = 0x9e3779b97f4a7c15;
          for (lVar59 = 0; lVar46 != lVar59; lVar59 = lVar59 + 1) {
            uVar35 = uVar35 * 0x5851f42d4c957f2d + 0x14057b7ef767814f;
            *(float *)((long)pvVar33 + lVar59 * 4) =
                 ((float)(int)((uVar35 >> 0x21) % 0x30d41) - 100000.0) / 997.0;
            fVar71 = ((float)(int)((uVar35 >> 0x28) % 0x4e21) - 10000.0) / 313.0;
            uVar18 = (undefined2)((int)fVar71 + 0x7fff + ((uint)fVar71 >> 0x10 & 1) >> 0x10);
            if (0x7f800000 < (uint)ABS(fVar71)) {
              uVar18 = 0x7fc0;
            }
            *(undefined2 *)((long)__ptr + lVar59 * 2) = uVar18;
            *(undefined2 *)((long)__ptr_00 + lVar59 * 2) = 0;
          }
          FUN_00161f20(&local_200,uVar25);
          if (uVar23 != 0) {
            FUN_0015e100(&local_200,uVar23);
          }
          pdVar60 = (double *)0x0;
          pdVar57 = (double *)0x0;
          pdVar37 = (double *)0x0;
          pdVar38 = (double *)0x0;
          local_3e8 = uVar24 + 1;
          puVar32 = (undefined8 *)0x0;
          __src = (void *)0x0;
          for (local_4e0._0_4_ = 0; (int)local_4e0._0_4_ <= local_3e8;
              local_4e0._0_4_ = local_4e0._0_4_ + 1) {
            ppppppplVar39 = (long *******)((long)iVar26 / 1000000);
            ppppppplVar43 = (long *******)(((long)iVar26 % 1000000) * 1000);
            if (0 < iVar26) {
              do {
                local_260._8_8_ = ppppppplVar43;
                local_260._0_8_ = ppppppplVar39;
                iVar20 = nanosleep((timespec *)local_260,(timespec *)local_260);
                if (iVar20 != -1) break;
                piVar42 = __errno_location();
                ppppppplVar39 = (long *******)local_260._0_8_;
                ppppppplVar43 = (long *******)local_260._8_8_;
              } while (*piVar42 == 4);
            }
            dVar62 = (double)FUN_00155ee0();
            if (uVar19 != 0) {
              for (uVar35 = 0; uVar35 < uVar27; uVar35 = uVar35 + 0x40) {
                clflush(*(undefined1 *)((long)__ptr + uVar35));
                clflush(*(undefined1 *)((long)__ptr_00 + uVar35));
              }
            }
            dVar63 = (double)FUN_00155ee0();
            FUN_0015e950(&local_200,pvVar33,__ptr,__ptr_00,lVar46,uVar22 != 0);
            dVar64 = (double)FUN_00155ee0();
            puVar56 = puVar32;
            __dest = __src;
            if (1 < (int)local_4e0._0_4_) {
              if (pdVar38 == pdVar57) {
                sVar61 = (long)pdVar38 - (long)puVar32;
                uVar35 = (long)sVar61 >> 3;
                if (uVar35 == 0xfffffffffffffff) {
                    /* WARNING: Subroutine does not return */
                  std::__throw_length_error("vector::_M_realloc_append");
                }
                uVar29 = 1;
                if (uVar35 != 0) {
                  uVar29 = uVar35;
                }
                uVar30 = 0xfffffffffffffff;
                if ((!CARRY8(uVar35,uVar29)) &&
                   (uVar30 = 0xfffffffffffffff, uVar35 + uVar29 < 0x1000000000000000)) {
                  uVar30 = uVar35 + uVar29;
                }
                puVar56 = operator_new(uVar30 * 8);
                *(double *)((long)puVar56 + sVar61) = dVar64 - dVar63;
                if ((long)sVar61 < 1) {
                  pdVar57 = (double *)((long)puVar56 + sVar61 + 8);
                  if (puVar32 != (undefined8 *)0x0) goto LAB_00135562;
                }
                else {
                  puVar56 = memcpy(puVar56,puVar32,sVar61);
                  pdVar57 = (double *)((long)puVar56 + sVar61 + 8);
LAB_00135562:
                  operator_delete(puVar32,(long)pdVar38 - (long)puVar32);
                }
                pdVar38 = (double *)(puVar56 + uVar30);
              }
              else {
                *pdVar57 = dVar64 - dVar63;
                pdVar57 = pdVar57 + 1;
              }
              if (pdVar37 == pdVar60) {
                sVar61 = (long)pdVar37 - (long)__src;
                uVar35 = (long)sVar61 >> 3;
                if (uVar35 == 0xfffffffffffffff) {
                    /* WARNING: Subroutine does not return */
                  std::__throw_length_error("vector::_M_realloc_append");
                }
                uVar29 = 1;
                if (uVar35 != 0) {
                  uVar29 = uVar35;
                }
                uVar30 = 0xfffffffffffffff;
                if ((!CARRY8(uVar35,uVar29)) &&
                   (uVar30 = 0xfffffffffffffff, uVar35 + uVar29 < 0x1000000000000000)) {
                  uVar30 = uVar35 + uVar29;
                }
                __dest = operator_new(uVar30 * 8);
                *(double *)((long)__dest + sVar61) = dVar63 - dVar62;
                if ((long)sVar61 < 1) {
                  pdVar60 = (double *)((long)__dest + sVar61 + 8);
                  if (__src != (void *)0x0) goto LAB_00135527;
                }
                else {
                  __dest = memcpy(__dest,__src,sVar61);
                  pdVar60 = (double *)((long)__dest + sVar61 + 8);
LAB_00135527:
                  operator_delete(__src,(long)pdVar37 - (long)__src);
                }
                pdVar37 = (double *)(uVar30 * 8 + (long)__dest);
              }
              else {
                *pdVar60 = dVar63 - dVar62;
                pdVar60 = pdVar60 + 1;
              }
            }
            puVar32 = puVar56;
            __src = __dest;
          }
          FUN_0015e100(&local_200,0);
          FUN_0011b010(puVar32,pdVar57);
          FUN_0011b010(__src,pdVar60);
          uVar27 = (long)pdVar57 - (long)puVar32 >> 3;
          printf("hostx n %zu th %d gap %d spin %d nt %d flush %d reps %d: step med %.3f min %.3f p90 %.3f max %.3f ms (%.1f GB/s at 12 B an element); flush med %.3f ms\n"
                 ,puVar32[uVar27 >> 1],*puVar32,puVar32[(uVar27 * 9) / 10],pdVar57[-1],
                 (((double)lVar46 * 12.0) / (double)puVar32[uVar27 >> 1]) / 1000000.0,
                 *(undefined8 *)((long)__src + ((ulong)((long)pdVar60 - (long)__src >> 3) >> 1) * 8)
                 ,lVar46,(ulong)uVar25,iVar26,(ulong)uVar23,(ulong)uVar22,(ulong)uVar19,
                 (ulong)uVar24);
          free(pvVar33);
          free(__ptr);
          free(__ptr_00);
          if (__src != (void *)0x0) {
            operator_delete(__src,(long)pdVar37 - (long)__src);
          }
          uVar27 = 0;
          operator_delete(puVar32,(long)pdVar38 - (long)puVar32);
          FUN_0015dc40(&local_200);
        }
        goto LAB_00133b34;
      }
      lVar59 = __isoc23_strtoll(*(undefined8 *)(local_3f0 + 0x18),0,10);
      local_320 = (undefined1 *)0x0;
      local_310 = 0;
      FUN_00139da0(&local_320,lVar59,&local_200);
      lVar46 = local_310;
      puVar51 = local_320;
      local_300 = (ulong *)0x0;
      local_2f0[0] = 0;
      FUN_00139da0(&local_300,lVar59,&local_200);
      uVar35 = local_2f0[0];
      puVar13 = local_300;
      local_2c0 = (char *******)0x0;
      local_2b0[0] = (char *******)0x0;
      FUN_00139e60(&local_2c0,lVar59,&local_200);
      pppppppcVar41 = local_2b0[0];
      pppppppcVar47 = local_2c0;
      local_2a0 = (ulong *)0x0;
      local_290[0] = 0;
      FUN_00139e60(&local_2a0,lVar59,&local_200);
      uVar29 = local_290[0];
      puVar50 = local_2a0;
      local_280[0] = (long ******)0x0;
      local_270[0] = (long ****)0x0;
      FUN_00139e60(local_280,lVar59,&local_200);
      pppplVar5 = local_270[0];
      pppppplVar3 = local_280[0];
      uVar27 = 0x9e3779b97f4a7c15;
      for (lVar49 = 0; lVar59 != lVar49; lVar49 = lVar49 + 1) {
        uVar27 = uVar27 * 0x5851f42d4c957f2d + 0x14057b7ef767814f;
        fVar71 = ((float)(int)((uVar27 >> 0x21) % 0x30d41) - 100000.0) / 997.0;
        *(float *)((long)puVar13 + lVar49 * 4) = fVar71;
        *(float *)(puVar51 + lVar49 * 4) = fVar71;
        fVar71 = ((float)(int)((uVar27 >> 0x28) % 0x4e21) - 10000.0) / 313.0;
        uVar18 = (undefined2)((int)fVar71 + 0x7fff + ((uint)fVar71 >> 0x10 & 1) >> 0x10);
        if (0x7f800000 < (uint)ABS(fVar71)) {
          uVar18 = 0x7fc0;
        }
        *(undefined2 *)((long)pppppppcVar47 + lVar49 * 2) = uVar18;
      }
      FUN_00161f20(local_260,1);
      FUN_00161f20(&local_200,8);
      dVar62 = (double)FUN_00155ee0();
      FUN_0015e950(local_260,puVar51,pppppppcVar47,puVar50,lVar59,0);
      dVar63 = (double)FUN_00155ee0();
      FUN_0015e950(&local_200,puVar13,pppppppcVar47,pppppplVar3,lVar59,0);
      dVar64 = (double)FUN_00155ee0();
      iVar26 = memcmp(puVar51,puVar13,lVar59 * 4);
      if (iVar26 == 0) {
        iVar26 = memcmp(puVar50,pppppplVar3,lVar59 * 2);
        pcVar34 = "BITWISE";
        if (iVar26 != 0) goto LAB_0013456d;
      }
      else {
LAB_0013456d:
        bVar17 = 0;
        pcVar34 = "DIFFERS";
      }
      printf("resid_copy n %zu: 1 thread %.3f ms, 8 threads %.3f ms, %s\n",dVar63 - dVar62,
             dVar64 - dVar63,lVar59,pcVar34);
      FUN_0015dc40(&local_200);
      uVar27 = (ulong)(bVar17 ^ 1);
      FUN_0015dc40(local_260);
      if (pppppplVar3 != (long ******)0x0) {
        operator_delete(pppppplVar3,(long)pppplVar5 - (long)pppppplVar3);
      }
      if (puVar50 != (ulong *)0x0) {
        operator_delete(puVar50,uVar29 - (long)puVar50);
      }
      if (pppppppcVar47 != (char *******)0x0) {
        operator_delete(pppppppcVar47,(long)pppppppcVar41 - (long)pppppppcVar47);
      }
      operator_delete(puVar13,uVar35 - (long)puVar13);
      operator_delete(puVar51,lVar46 - (long)puVar51);
    }
    else {
      uVar21 = 0x400;
      if (local_3e4[0] != 8) {
        uVar21 = __isoc23_strtol(*(undefined8 *)(local_3f0 + 0x40),0,10);
      }
      FUN_0011a280(&local_200,*(undefined8 *)(local_3f0 + 0x38));
      FUN_0011a280(local_260,*(undefined8 *)(local_3f0 + 0x30));
      FUN_0011a280(local_280,*(undefined8 *)(local_3f0 + 0x28));
      FUN_0011a280(&local_2a0,*(undefined8 *)(local_3f0 + 0x20));
      FUN_0011a280((string *)&local_2c0,*(undefined8 *)(local_3f0 + 0x18));
      uVar19 = FUN_00125f60((string *)&local_2c0,&local_2a0,local_280,local_260,&local_200,uVar21);
      std::__cxx11::string::_M_dispose((string *)&local_2c0);
LAB_00133b17:
      uVar27 = (ulong)uVar19;
      std::__cxx11::string::_M_dispose((string *)&local_2a0);
      std::__cxx11::string::_M_dispose((string *)local_280);
      std::__cxx11::string::_M_dispose((string *)local_260);
      std::__cxx11::string::_M_dispose((string *)&local_200);
    }
  }
  else {
    FUN_0011a280(local_260,*(undefined8 *)(local_3f0 + 0x18));
    __stream = fopen((char *)local_260._0_8_,"w");
    if (__stream == (FILE *)0x0) {
      FUN_001256c0((string *)&local_200,"cannot write ",local_260._0_8_,local_260._8_8_);
      fprintf(stderr,"halogen-npu: %s\n",local_200);
      uVar27 = 1;
      std::__cxx11::string::_M_dispose((string *)&local_200);
    }
    else {
      lVar46 = 0;
      auVar73._8_4_ = 0x80000000;
      auVar73._0_8_ = 0x8000000080000000;
      auVar73._12_4_ = 0x80000000;
      auVar73._16_4_ = 0x80000000;
      auVar73._20_4_ = 0x80000000;
      auVar73._24_4_ = 0x80000000;
      auVar73._28_4_ = 0x80000000;
      auVar80._8_4_ = 0x7f800000;
      auVar80._0_8_ = 0x7f8000007f800000;
      auVar80._12_4_ = 0x7f800000;
      auVar80._16_4_ = 0x7f800000;
      auVar80._20_4_ = 0x7f800000;
      auVar80._24_4_ = 0x7f800000;
      auVar80._28_4_ = 0x7f800000;
      auVar68 = vpsubd_avx2(auVar80,auVar73);
      do {
        iVar26 = (int)lVar46 << 0x18;
        auVar85 = ZEXT1664((undefined1  [16])0x0);
        auVar86._8_4_ = 0x7fffffff;
        auVar86._0_8_ = 0x7fffffff7fffffff;
        auVar86._12_4_ = 0x7fffffff;
        auVar86._16_4_ = 0x7fffffff;
        auVar86._20_4_ = 0x7fffffff;
        auVar86._24_4_ = 0x7fffffff;
        auVar86._28_4_ = 0x7fffffff;
        auVar83._8_4_ = 0xffff;
        auVar83._0_8_ = 0xffff0000ffff;
        auVar83._12_4_ = 0xffff;
        auVar83._16_4_ = 0xffff;
        auVar83._20_4_ = 0xffff;
        auVar83._24_4_ = 0xffff;
        auVar83._28_4_ = 0xffff;
        iVar20 = 0;
        auVar87._4_4_ = iVar26;
        auVar87._0_4_ = iVar26;
        auVar87._8_4_ = iVar26;
        auVar87._12_4_ = iVar26;
        auVar87._16_4_ = iVar26;
        auVar87._20_4_ = iVar26;
        auVar87._24_4_ = iVar26;
        auVar87._28_4_ = iVar26;
        auVar7 = _DAT_001f4420;
        do {
          auVar66._8_8_ = 4;
          auVar66._0_8_ = 4;
          auVar66._16_8_ = 4;
          auVar66._24_8_ = 4;
          auVar80 = vpaddq_avx2(auVar7,auVar66);
          auVar79._8_8_ = 8;
          auVar79._0_8_ = 8;
          auVar79._16_8_ = 8;
          auVar79._24_8_ = 8;
          auVar79 = vpaddq_avx2(auVar7,auVar79);
          auVar82._8_8_ = 0xc;
          auVar82._0_8_ = 0xc;
          auVar82._16_8_ = 0xc;
          auVar82._24_8_ = 0xc;
          auVar66 = vpaddq_avx2(auVar7,auVar82);
          iVar20 = iVar20 + 1;
          auVar67 = vperm2i128_avx2(auVar7,auVar80,0x20);
          auVar82 = vperm2i128_avx2(auVar7,auVar80,0x31);
          auVar84._8_4_ = 1;
          auVar84._0_8_ = 0x100000001;
          auVar84._12_4_ = 1;
          auVar84._16_4_ = 1;
          auVar84._20_4_ = 1;
          auVar84._24_4_ = 1;
          auVar84._28_4_ = 1;
          auVar82 = vpshufd_avx2(auVar82,0xd8);
          auVar67 = vpshufd_avx2(auVar67,0xd8);
          auVar8 = vperm2i128_avx2(auVar79,auVar66,0x20);
          auVar82 = vpunpcklqdq_avx2(auVar67,auVar82);
          auVar77 = vperm2i128_avx2(auVar79,auVar66,0x31);
          auVar67 = vpshufd_avx2(auVar8,0xd8);
          auVar8 = vpshufd_avx2(auVar77,0xd8);
          auVar78 = vpsllq_avx2(auVar80,1);
          auVar82 = vpor_avx2(auVar87,auVar82);
          auVar67 = vpunpcklqdq_avx2(auVar67,auVar8);
          auVar76 = vpsllq_avx2(auVar7,1);
          auVar80 = vpand_avx2(auVar82,auVar86);
          auVar67 = vpor_avx2(auVar87,auVar67);
          auVar75 = vpsllq_avx2(auVar66,1);
          auVar66 = vpsubd_avx2(auVar80,auVar73);
          auVar80 = vpand_avx2(auVar67,auVar86);
          auVar66 = vpcmpgtd_avx2(auVar66,auVar68);
          auVar77._8_8_ = 0x10;
          auVar77._0_8_ = 0x10;
          auVar77._16_8_ = 0x10;
          auVar77._24_8_ = 0x10;
          auVar7 = vpaddq_avx2(auVar7,auVar77);
          auVar80 = vpsubd_avx2(auVar80,auVar73);
          auVar8._8_4_ = 0x7fff;
          auVar8._0_8_ = 0x7fff00007fff;
          auVar8._12_4_ = 0x7fff;
          auVar8._16_4_ = 0x7fff;
          auVar8._20_4_ = 0x7fff;
          auVar8._24_4_ = 0x7fff;
          auVar8._28_4_ = 0x7fff;
          auVar74 = vpsllq_avx2(auVar79,1);
          auVar79 = vpcmpgtd_avx2(auVar80,auVar68);
          auVar80 = vpand_avx2(auVar83,auVar66);
          auVar79 = vpand_avx2(auVar83,auVar79);
          auVar80 = vpackusdw_avx2(auVar80,auVar79);
          auVar66 = vpsrld_avx2(auVar82,0x10);
          auVar79 = vpaddd_avx2(auVar82,auVar8);
          auVar66 = vpand_avx2(auVar66,auVar84);
          auVar82 = vpermq_avx2(auVar80,0xd8);
          auVar80 = vpaddd_avx2(auVar66,auVar79);
          auVar66 = vpsrld_avx2(auVar67,0x10);
          auVar79 = vpaddd_avx2(auVar67,auVar8);
          auVar66 = vpand_avx2(auVar66,auVar84);
          auVar67 = vpsrld_avx2(auVar80,0x10);
          auVar80 = vpaddd_avx2(auVar66,auVar79);
          auVar79 = vpand_avx2(auVar83,auVar67);
          auVar80 = vpsrld_avx2(auVar80,0x10);
          auVar80 = vpand_avx2(auVar83,auVar80);
          auVar80 = vpackusdw_avx2(auVar79,auVar80);
          auVar67._8_4_ = 0x7fc07fc0;
          auVar67._0_8_ = 0x7fc07fc07fc07fc0;
          auVar67._12_4_ = 0x7fc07fc0;
          auVar67._16_4_ = 0x7fc07fc0;
          auVar67._20_4_ = 0x7fc07fc0;
          auVar67._24_4_ = 0x7fc07fc0;
          auVar67._28_4_ = 0x7fc07fc0;
          auVar80 = vpermq_avx2(auVar80,0xd8);
          auVar66 = vpblendvb_avx2(auVar80,auVar67,auVar82);
          auVar81._8_8_ = 1;
          auVar81._0_8_ = 1;
          auVar81._16_8_ = 1;
          auVar81._24_8_ = 1;
          auVar82 = vpmovzxwd_avx2(auVar66._0_16_);
          auVar80 = vpaddq_avx2(auVar78,auVar81);
          auVar79 = vpaddq_avx2(auVar76,auVar81);
          auVar8 = vpmovzxdq_avx2(auVar82._16_16_);
          auVar67 = vpsrlq_avx2(auVar80,0x20);
          auVar77 = vpmovzxdq_avx2(auVar82._0_16_);
          auVar84 = vpsrlq_avx2(auVar8,0x20);
          auVar82 = vpmuludq_avx2(auVar67,auVar8);
          auVar67 = vpmovzxwd_avx2(auVar66._16_16_);
          auVar84 = vpmuludq_avx2(auVar84,auVar80);
          auVar8 = vpmuludq_avx2(auVar80,auVar8);
          auVar80 = vpaddq_avx2(auVar75,auVar81);
          auVar75 = vpsrlq_avx2(auVar79,0x20);
          auVar66 = vpaddq_avx2(auVar74,auVar81);
          auVar74 = vpmuludq_avx2(auVar75,auVar77);
          auVar82 = vpaddq_avx2(auVar82,auVar84);
          auVar84 = vpsrlq_avx2(auVar77,0x20);
          auVar82 = vpsllq_avx2(auVar82,0x20);
          auVar82 = vpaddq_avx2(auVar8,auVar82);
          auVar8 = vpmuludq_avx2(auVar79,auVar77);
          auVar79 = vpmuludq_avx2(auVar84,auVar79);
          auVar77 = vpsrlq_avx2(auVar80,0x20);
          auVar79 = vpaddq_avx2(auVar74,auVar79);
          auVar79 = vpsllq_avx2(auVar79,0x20);
          auVar79 = vpaddq_avx2(auVar8,auVar79);
          auVar8 = vpmovzxdq_avx2(auVar67._0_16_);
          auVar67 = vpmovzxdq_avx2(auVar67._16_16_);
          auVar84 = vpsrlq_avx2(auVar8,0x20);
          auVar79 = vpaddq_avx2(auVar82,auVar79);
          auVar74 = vpsrlq_avx2(auVar67,0x20);
          auVar82 = vpmuludq_avx2(auVar80,auVar67);
          auVar67 = vpmuludq_avx2(auVar77,auVar67);
          auVar80 = vpmuludq_avx2(auVar74,auVar80);
          auVar77 = vpmuludq_avx2(auVar66,auVar8);
          auVar80 = vpaddq_avx2(auVar67,auVar80);
          auVar67 = vpsrlq_avx2(auVar66,0x20);
          auVar67 = vpmuludq_avx2(auVar67,auVar8);
          auVar66 = vpmuludq_avx2(auVar84,auVar66);
          auVar80 = vpsllq_avx2(auVar80,0x20);
          auVar80 = vpaddq_avx2(auVar82,auVar80);
          auVar66 = vpaddq_avx2(auVar67,auVar66);
          auVar66 = vpsllq_avx2(auVar66,0x20);
          auVar66 = vpaddq_avx2(auVar77,auVar66);
          auVar80 = vpaddq_avx2(auVar80,auVar66);
          auVar80 = vpaddq_avx2(auVar79,auVar80);
          auVar80 = vpaddq_avx2(auVar85._0_32_,auVar80);
          auVar85 = ZEXT3264(auVar80);
        } while (iVar20 != 0x100000);
        auVar65 = vpaddq_avx(auVar80._16_16_,auVar80._0_16_);
        auVar69 = vpsrldq_avx(auVar65,8);
        auVar65 = vpaddq_avx(auVar65,auVar69);
        fprintf(__stream,"%llu %llu\n",lVar46,auVar65._0_8_);
        lVar46 = lVar46 + 1;
      } while (lVar46 != 0x100);
      fclose(__stream);
      uVar27 = 0;
    }
LAB_0013356d:
    std::__cxx11::string::_M_dispose((string *)local_260);
  }
LAB_00133b34:
  std::__cxx11::string::_M_dispose((string *)&local_2e0);
LAB_0013316c:
  std::__cxx11::string::_M_dispose(local_340);
                    /* WARNING: Read-only address (ram,0x001f4420) is written */
  return uVar27;
code_r0x00134245:
  uVar27 = uVar35;
  if (uVar35 == 0xffffffffffffffff) goto LAB_001346e7;
  goto LAB_0013422f;
}

