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

extern "C" __device__ __noinline__ 
int getGlobalIdx_3D_3D(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
 + gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
 + (threadIdx.z * (blockDim.x * blockDim.y))
 + (threadIdx.y * blockDim.x) + threadIdx.x;
return threadId;
}

extern "C" __device__ __noinline__ void inject_error(uint64_t piinfo, uint64_t pverbose_device, uint64_t inj_thread_info, int destGPRNum, int regval, int numDestGPRs, int compute_cap) { 
								
				auto i=getGlobalIdx_3D_3D();
				auto ctaID=get_ctaid();
				auto SMID=get_smid();
				auto WID = get_warpid();
				auto LID = get_laneid();		

				inj_info_error_t* inj_info = (inj_info_error_t*)piinfo; 
				muliple_ptr_t *inj_struct=(muliple_ptr_t *) inj_thread_info;
				uint32_t verbose_device = *((uint32_t *)pverbose_device);
				
				// inj_struct->ThrdID[i]=i;
				// inj_struct->WARPID[i]=WID;
				// inj_struct->LANEID[i]=LID;
				// inj_struct->SMID[i]=SMID;
				// inj_struct->ctaID_x[i]=threadIdx.x;
				// inj_struct->ctaID_y[i]=threadIdx.y;
				// inj_struct->ctaID_z[i]=threadIdx.z;																				
				// inj_struct->flag[i] = 1;

				int warpID=int(inj_info->injThreadMask/32);
				int laneID=int(inj_info->injThreadMask%32);
				//check performed on the Straming Multiprocessor ID
				if ((SMID == inj_info->injSMID) && (WID == warpID) && (LID == laneID)){ // This is not the selected SM. No need to proceed.					
					assert(numDestGPRs > 0);
					uint32_t injAfterVal = 0; 
					uint32_t injBeforeVal = nvbit_read_reg(destGPRNum); // read the register value
										
					if (DUMMY || destGPRNum != inj_info->injRegID ) { 
									injAfterVal = injBeforeVal;
					
					} else {
						if(inj_info->injStuck_at == 1){
									injAfterVal = injBeforeVal | (inj_info->injMask); //OR
						}
						else {	
									injAfterVal = injBeforeVal & (~inj_info->injMask);//AND
						}
						if(verbose_device)printf("register=%d, before=0x%x, after=0x%x, expected_after=0x%x, mask =0x%x, stuck at %d\n", destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), injAfterVal,inj_info->injMask,inj_info->injStuck_at);
					}
					inj_info->errorInjected = true; 
					inj_struct->ThrdID[i]=i;
					inj_struct->WARPID[i]=WID;
					inj_struct->LANEID[i]=LID;
					inj_struct->SMID[i]=SMID;
					inj_struct->ctaID_x[i]=ctaID.x;
					inj_struct->ctaID_y[i]=ctaID.y;
					inj_struct->ctaID_z[i]=ctaID.z;																				
					inj_struct->flag[i] = 1;

					if(injAfterVal!=injBeforeVal){						
						atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL);
					}
					nvbit_write_reg(destGPRNum, injAfterVal);
				}
}

