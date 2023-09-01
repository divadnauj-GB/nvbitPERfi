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

#define MAX_KNAME_SIZE 1000

typedef struct {
  uint32_t injSMID; // 0 - max SMs
  int injSubSMID;// 4 Sub SMID retrieved from warpID (0:31)->(0:3)
  uint32_t injLaneID; // core ID 0 - 32
  uint32_t injInstType; // instruction type 
  uint32_t injMask; // injection mask
  uint32_t injReg; // injection mask		
  uint32_t injStuckat; //stuckat model used
  // updated during/after error injection 
  uint64_t injNumActivations;
  bool errorInjected;
} inj_info_t; 

typedef struct {
  uint32_t injSMID; // 0 - max SMs
  uint32_t injSubSMID;
  uint32_t injLaneID;
  uint32_t injThreadMask; //0-32
  uint32_t injMask;
  uint32_t injRegID; // injection mask;
  uint32_t injStuck_at;
  uint32_t injInstType; // instruction type 
  uint32_t injRegOriginal;
  uint32_t injRegReplacement;
  uint64_t injNumActivations;
  uint64_t injNumActivAcc;
  uint32_t injInstrIdx;
  uint32_t injInstPC;
  uint32_t injInstOpcode;  
  uint32_t injIALtype;
  uint32_t blockDimX;
  uint32_t blockDimY;
  uint32_t blockDimZ;
  uint32_t gridDimX;
  uint32_t gridDimY;
  uint32_t gridDimZ;
  uint32_t maxregcount;
  int32_t maxPredReg;
  int32_t KernelPredReg;
  uint32_t TotKerInstr;
  uint32_t TotAppInstr;
  uint32_t num_threads;
  uint32_t MaxWarpsPerSM;
  uint32_t MaxThreadsPerSM;
  uint32_t MaxThreadsPerWarp;  
  char DeviceName[256];
  char KName[256];
  uint32_t kernel_id;
  bool errorInjected;
} inj_info_error_t; 



typedef struct {
    uint32_t *ThrdID;
    uint32_t *WARPID;
    uint32_t *LANEID;
    uint32_t *SMID;
    uint32_t *ctaID_x;
    uint32_t *ctaID_y;
    uint32_t *ctaID_z;

    uint32_t *flag;
    uint32_t counter;
} muliple_ptr_t;


#endif
