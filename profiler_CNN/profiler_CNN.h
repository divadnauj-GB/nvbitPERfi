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

#ifndef PROFILER_CNN_H
#define PROFILER_CNN_H
#include <assert.h>
#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream> 
#include <iterator>

#include "arch.h"

//#define MAX_KNAME_SIZE 1000

#define MemSize 1024*512
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
    uint32_t *ThrdID;
    uint32_t *SMID;
    uint32_t *ctaID_x;
    uint32_t *ctaID_y;
    uint32_t *ctaID_z;

    uint32_t *flag;
    uint32_t counter;
} muliple_ptr_t;

#endif
