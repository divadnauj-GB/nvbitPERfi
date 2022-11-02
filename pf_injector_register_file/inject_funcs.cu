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


#include <stdint.h>
#include <stdio.h>

#include "nvbit_reg_rw.h"
#include "utils/utils.h"
#include "pf_injector.h"
#include "arch.h"
//#include "globals.h"
//#include "cuPrintf.cu"

//__shared__ char *injectionOut ;

extern "C" __device__ __noinline__ void inject_error(uint64_t piinfo, uint64_t pverbose_device, int destGPRNum, int regval, int numDestGPRs, int compute_cap) { 
				inj_info_t* inj_info = (inj_info_t*)piinfo; 
				uint32_t verbose_device = *((uint32_t *)pverbose_device);
				
				//check performed on the Straming Multiprocessor ID
				uint32_t smid;
				asm("mov.u32 %0, %smid;" :"=r"(smid));
				if (smid != inj_info->injSMID) 	return; // This is not the selected SM. No need to proceed.
				uint32_t WID;
				asm("mov.u32 %0, %warpid;" :"=r"(WID));
				uint32_t LID;
				asm("mov.u32 %0, %laneid;" :"=r"(LID));
				int warpID;				
				warpID=int(inj_info->injThreadID/32);
				int laneID=int(inj_info->injThreadID%32);
				
				if(WID!=warpID) return;
				if(LID!=laneID) return;

				assert(numDestGPRs > 0);
				uint32_t injAfterVal = 0; 
				uint32_t injBeforeVal = nvbit_read_reg(destGPRNum); // read the register value
									
				if (DUMMY || destGPRNum != inj_info->injReg ) { 
								injAfterVal = injBeforeVal;
				
				} else {
					if(inj_info->injStuckat == 1){
								injAfterVal = injBeforeVal | (inj_info->injMask); //OR
								nvbit_write_reg(destGPRNum, injAfterVal);
					}
					else {	
								injAfterVal = injBeforeVal & (~inj_info->injMask);//AND			
								nvbit_write_reg(destGPRNum, injAfterVal);
											
					}
				if(verbose_device)printf("register=%d, before=0x%x, after=0x%x, expected_after=0x%x, mask =0x%x, stuck at %d\n", destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), injAfterVal,inj_info->injMask,inj_info->injStuckat);
				
				}
				inj_info->errorInjected = true; 
				atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL);  
}

