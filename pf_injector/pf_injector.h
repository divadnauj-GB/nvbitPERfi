/*
 * Copyright 2020, NVIDIA CORPORATION.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PF_INJECTOR_H
#define PF_INJECTOR_H

//#define MAX_KNAME_SIZE 1000

#define FL_INSTR 0

#define MemSize 1024*512*720
//#define DBG
typedef struct {
    uint32_t injSMID; // 0 - max SMs
    uint32_t injsubSMID;
    uint32_t injLaneID;
    float    Th;
    uint64_t injNumActivations;
    uint64_t counter_data;
    uint32_t InstrIDx;
    bool errorInjected;
} inj_info_t;



typedef struct {
    uint32_t *KerID;
    uint32_t *SMID;
    uint32_t *InstrID;
    uint32_t *ctaID_x;
    uint32_t *ctaID_y;
    uint32_t *ctaID_z;
    uint32_t *WID; // 0 - max SMs
    uint32_t *LID;
    uint32_t *Mask;
    uint32_t *num_operands;
    uint32_t *TLID;
    uint32_t *Operand1;
    uint32_t *Operand2;
    uint32_t *Operand3;
    uint32_t *Operand4;
    uint32_t *TOperand1;
    uint32_t *TOperand2;
    uint32_t *TOperand3;
    uint32_t *Tmasks;
    uint32_t Table_size;
    uint32_t *T0_instID;
    uint32_t *T0_ctaID_x;
    uint32_t *T0_ctaID_y;
    uint32_t *T0_ctaID_z;
    uint32_t *T0_WID; // 0 - max SMs
    uint32_t *T0_LID;
    uint32_t *T0_Mask;
    uint32_t *T1_Mask;
    uint32_t *T2_Mask;
    uint32_t *flag;
    uint32_t counter;
} muliple_ptr_t;

#endif
