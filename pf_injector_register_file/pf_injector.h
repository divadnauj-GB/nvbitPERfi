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
#define WARP_PER_SM 48
#define THREAD_PER_WARP 32





typedef struct {
  uint32_t injSMID; // 0 - max SMs
  uint32_t injLaneID; // 0 - 32
  uint32_t injThreadID;
  uint32_t injReg;
  uint32_t injStuckat;
  //uint32_t injCtaID;	
  uint32_t injMask; // injection mask
  uint32_t injInstType; // instruction type 

  // updated during/after error injection 
  uint64_t injNumActivations;
  bool errorInjected;
} inj_info_t; 


typedef struct {
  uint32_t injSMID; // 0 - max SMs
  uint32_t injScheduler; // 0 - 3
  uint32_t injWarpMaskH;
  uint32_t injWarpMaskL; //
  uint32_t injThreadMask; //0-32
  uint32_t injMaskSeed;
  uint32_t injRegID; // injection mask
  uint32_t injInstType; // instruction type 
  uint32_t injRegOriginal;
  uint32_t injRegReplacement;
  uint64_t injNumActivations;
  bool errorInjected;
} inj_info_error_t; 


typedef struct {
    uint32_t *warp_thread_mask;
    uint32_t *Warp_thread_active;
    uint32_t *register_tmp_recovery;
    uint32_t counter;
    uint32_t num_threads;
} muliple_ptr_t;



#endif
