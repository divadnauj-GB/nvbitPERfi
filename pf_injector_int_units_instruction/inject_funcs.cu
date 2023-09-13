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

extern "C" __device__ __noinline__ 
int getGlobalIdx_3D_3D(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
 + gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
 + (threadIdx.z * (blockDim.x * blockDim.y))
 + (threadIdx.y * blockDim.x) + threadIdx.x;
return threadId;
}

extern "C" __device__ __noinline__ void inject_error_input(uint64_t piinfo, uint64_t pverbose_device, uint64_t inj_thread_info, int sourcereg, int destGPRNum,int regval, int numDestGPRs, int compute_cap) {

				inj_info_error_t* inj_info = (inj_info_error_t*)piinfo; 
				muliple_ptr_t *inj_struct=(muliple_ptr_t *) inj_thread_info;
				uint32_t verbose_device = *((uint32_t *)pverbose_device);
				
				auto i = getGlobalIdx_3D_3D();
				auto ctaID=get_ctaid();

				//check performed on the Straming Multiprocessor ID
				uint32_t smid;
				asm("mov.u32 %0, %smid;" :"=r"(smid));
				if (smid != inj_info->injSMID) return; // This is not the selected SM. No need to proceed.
				
				uint32_t subsmid;
				uint32_t warpid;
				asm("mov.u32 %0, %warpid;" :"=r"(warpid));
				subsmid = (uint32_t)(warpid % 4); 
				if (subsmid != inj_info->injSubSMID) return;						
				uint32_t sublaneid;
				asm("mov.u32 %0, %laneid;" :"=r"(sublaneid));
				uint32_t laneid;
				
				if (compute_cap == 53){ //MAXWELL 5.
					if (inj_info->injInstType == MUFU || inj_info->injInstType == RRO) 
						laneid = (uint32_t)(sublaneid % 8);//8 SFU
					else
						laneid = sublaneid; //32 INT and 32 FP	
				}
				else if(compute_cap == 70 or compute_cap == 75){ //VOLTA 7.
					if (inj_info->injInstType == MUFU || inj_info->injInstType == RRO)
						laneid =(uint32_t)(sublaneid % 4);	 //4 SFU
					else
						laneid = (uint32_t)(sublaneid % 16);//16 INT and 16 FP	
				}
				else if(compute_cap == 86){ //AMPERE 8.
					if (inj_info->injInstType == MUFU || inj_info->injInstType == RRO)
						laneid = (uint32_t)(sublaneid % 4);	 //4 SFU
					else if(inj_info->injInstType >= BFE && inj_info->injInstType <= XMAD) //integer
						laneid = (uint32_t)(sublaneid % 16); //16 INT 
					else //fp					
						laneid = sublaneid % 16; //32 FP
				}
				
				
				if (laneid != inj_info->injLaneID) return;
															
				//assert(numDestGPRs > 0);
				
				uint32_t injAfterVal = 0;
				uint32_t injBeforeVal = nvbit_read_reg(sourcereg); // read the register value
				
								
				if (DUMMY) {
					injAfterVal = injBeforeVal;
				}
				else{
					if(inj_info->injStuck_at == 1){				
						injAfterVal = injBeforeVal | (inj_info->injMask); //OR
					}
					else {	
						injAfterVal = injBeforeVal & (~inj_info->injMask);//AND																					
					}
			
				}
				
				inj_info->errorInjected = true; 
				inj_struct->ThrdID[i]=i;
				inj_struct->WARPID[i]=warpid;
				inj_struct->LANEID[i]=sublaneid;
				inj_struct->SMID[i]=smid;
				inj_struct->ctaID_x[i]=ctaID.x;
				inj_struct->ctaID_y[i]=ctaID.y;
				inj_struct->ctaID_z[i]=ctaID.z;																				
				inj_struct->flag[i] = 1;
				
				if(injAfterVal!=injBeforeVal){						
						atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL);
					}
				nvbit_write_reg(sourcereg, injAfterVal);
				
}

extern "C" __device__ __noinline__ void inject_error_output(uint64_t piinfo, uint64_t pverbose_device, uint64_t inj_thread_info, int destGPRNum, int regval, int numDestGPRs, int compute_cap) {

				inj_info_error_t* inj_info = (inj_info_error_t*)piinfo; 
				muliple_ptr_t *inj_struct=(muliple_ptr_t *) inj_thread_info;
				uint32_t verbose_device = *((uint32_t *)pverbose_device);
				auto i = getGlobalIdx_3D_3D();
				auto ctaID=get_ctaid();
				//check performed on the Straming Multiprocessor ID				
				uint32_t smid;
				asm("mov.u32 %0, %smid;" :"=r"(smid));
				if (smid != inj_info->injSMID) return; // This is not the selected SM. No need to proceed.
				
				uint32_t subsmid;
				uint32_t warpid;
				asm("mov.u32 %0, %warpid;" :"=r"(warpid));
				subsmid = (uint32_t)(warpid % 4); 
				if (subsmid != inj_info->injSubSMID) return;						
				uint32_t sublaneid;
				asm("mov.u32 %0, %laneid;" :"=r"(sublaneid));
				uint32_t laneid;
				

				if (compute_cap == 53){ //MAXWELL 5.
					if (inj_info->injInstType == MUFU || inj_info->injInstType == RRO) 
						laneid = (uint32_t)(sublaneid % 8);//8 SFU
					else
						laneid = sublaneid; //32 INT and 32 FP	
				}
				else if(compute_cap == 70 or compute_cap == 75){ //VOLTA 7.
					if (inj_info->injInstType == MUFU || inj_info->injInstType == RRO)
						laneid =(uint32_t)(sublaneid % 4);	 //4 SFU
					else
						laneid = (uint32_t)(sublaneid % 16);//16 INT and 16 FP	
				}
				else if(compute_cap == 86){ //AMPERE 8.
					if (inj_info->injInstType == MUFU || inj_info->injInstType == RRO)
						laneid = (uint32_t)(sublaneid % 4);	 //4 SFU
					else if(inj_info->injInstType >= BFE && inj_info->injInstType <= XMAD) //integer
						laneid = (uint32_t)(sublaneid % 16); //16 INT 
					else //fp					
						laneid = sublaneid % 16; //32 FP
				}
				

				if (laneid != inj_info->injLaneID) return;
				
				assert(numDestGPRs > 0);
				
				uint32_t injAfterVal = 0; 
				uint32_t injBeforeVal = nvbit_read_reg(destGPRNum); // read the register value
				if (DUMMY) {
								injAfterVal = injBeforeVal;
				}else{ 
					if(inj_info->injStuck_at == 1){
						injAfterVal = injBeforeVal | (inj_info->injMask); //OR						
					}
					else {	
						injAfterVal = injBeforeVal & (~inj_info->injMask);//AND									

					}
				}
				if(verbose_device)printf("register=%d, before=0x%x, after=0x%x, expected_after=0x%x, mask =0x%x, subsmid %d, laneid %d \n", destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), injAfterVal,inj_info->injMask,subsmid,laneid);
				
				inj_info->errorInjected = true; 
				inj_struct->ThrdID[i]=i;
				inj_struct->WARPID[i]=warpid;
				inj_struct->LANEID[i]=sublaneid;
				inj_struct->SMID[i]=smid;
				inj_struct->ctaID_x[i]=ctaID.x;
				inj_struct->ctaID_y[i]=ctaID.y;
				inj_struct->ctaID_z[i]=ctaID.z;																				
				inj_struct->flag[i] = 1;
				
				if(injAfterVal!=injBeforeVal){						
					atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL);
				} 
				nvbit_write_reg(destGPRNum, injAfterVal);
}

