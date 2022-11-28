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

extern "C" __device__ __noinline__ void inject_error_input(uint64_t piinfo, uint64_t pverbose_device, int sourcereg, int destGPRNum,int regval, int numDestGPRs, int compute_cap) {

				inj_info_t* inj_info = (inj_info_t*)piinfo; 
				uint32_t verbose_device = *((uint32_t *)pverbose_device);
				
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
					if (inj_info->injInstType == 15 || inj_info->injInstType == 16) 
						laneid = (uint32_t)(sublaneid % 8);//8 SFU
					else
						laneid = sublaneid; //32 INT and 32 FP	
				}
				else if(compute_cap == 70 or compute_cap == 75){ //VOLTA 7.
					if (inj_info->injInstType == 15 || inj_info->injInstType == 16)
						laneid =(uint32_t)(sublaneid % 4);	 //4 SFU
					else
						laneid = (uint32_t)(sublaneid % 16);//16 INT and 16 FP	
				}
				else if(compute_cap == 86){ //AMPERE 8.
					if (inj_info->injInstType == 15 || inj_info->injInstType == 16)
						laneid = (uint32_t)(sublaneid % 4);	 //4 SFU
					else if(inj_info->injInstType >= 30 && inj_info->injInstType <= 60) //integer
						laneid = (uint32_t)(sublaneid % 16); //16 INT 
					else if(inj_info->injInstType < 30 && inj_info->injInstType != 15 && inj_info->injInstType != 16)//fp					
						laneid = sublaneid ; //32 FP
				}
				
				
				if (laneid != inj_info->injLaneID) return;
															
				//assert(numDestGPRs > 0);
				
				uint32_t injAfterVal = 0;
				uint32_t injBeforeVal = nvbit_read_reg(sourcereg); // read the register value
				
								
				if (DUMMY) {
								injAfterVal = injBeforeVal;
								
				
				//if(verbose_device)printf("register=%d, before=0x%x, after=0x%x, expected_after=0x%x, mask =0x%x,register=%d, before=0x%x, after=0x%x, expected_after=0x%x, mask =0x%x,register=%d, before=0x%x, after=0x%x, expected_after=0x%x, mask =0x%x, subsmid %d, laneid %d \n", sourcereg0, injBeforeVal0, nvbit_read_reg(sourcereg0), injAfterVal0,inj_info->injMask[0],sourcereg1, injBeforeVal1, nvbit_read_reg(sourcereg1), injAfterVal1,inj_info->injMask[1],sourcereg2, injBeforeVal2, nvbit_read_reg(sourcereg2), injAfterVal2,inj_info->injMask[2],subsmid,laneid);
				inj_info->errorInjected = true; 
				atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL); 
				return;
				}
				else{
					if(inj_info->injStuckat == 1){				
								injAfterVal = injBeforeVal | (inj_info->injMask); //OR
								nvbit_write_reg(sourcereg, injAfterVal);
					}
					else {	
								injAfterVal = injBeforeVal & (~inj_info->injMask);//AND			
								nvbit_write_reg(sourcereg, injAfterVal);		
								
					}
					//if(verbose_device)printf("register=%d, before=0x%x, after=0x%x, expected_after=0x%x, mask =0x%x,subsmid %d, laneid %d \n", sourcereg, injBeforeVal, nvbit_read_reg(sourcereg), injAfterVal,inj_info->injMask,subsmid,laneid);
					inj_info->errorInjected = true; 
					atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL); 
					return;				
				}
				
				inj_info->errorInjected = true; 
				atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL); 
				return;	
}

extern "C" __device__ __noinline__ void inject_error_output(uint64_t piinfo, uint64_t pverbose_device, int destGPRNum, int regval, int numDestGPRs, int compute_cap) {

				inj_info_t* inj_info = (inj_info_t*)piinfo; 
				uint32_t verbose_device = *((uint32_t *)pverbose_device);
				
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
					if (inj_info->injInstType == 15 or inj_info->injInstType == 16) 
						laneid = (uint32_t)(sublaneid % 8);//8 SFU
					else
						laneid = sublaneid; //32 INT and 32 FP	
				}
				else if(compute_cap == 70 or compute_cap == 75){ //VOLTA 7.
					if (inj_info->injInstType == 15 or inj_info->injInstType == 16)
						laneid =(uint32_t)(sublaneid % 4);	 //4 SFU
					else
						laneid = (uint32_t)(sublaneid % 16);//16 INT and 16 FP	
				}
				else if(compute_cap == 86){ //AMPERE 8.
					if (inj_info->injInstType == 15 or inj_info->injInstType == 16)
						laneid = (uint32_t)(sublaneid % 4);	 //4 SFU
					else if(inj_info->injInstType >= 30 and inj_info->injInstType <= 60) //integer
						laneid = (uint32_t)(sublaneid % 16); //16 INT 
					else //fp					
						laneid = sublaneid ; //32 FP
				}
				

				if (laneid != inj_info->injLaneID) return;
				
				assert(numDestGPRs > 0);
				
				uint32_t injAfterVal = 0; 
				uint32_t injBeforeVal = nvbit_read_reg(destGPRNum); // read the register value
				if (DUMMY) {
								injAfterVal = injBeforeVal;
				}else{ 
					if(inj_info->injStuckat == 1){
								injAfterVal = injBeforeVal | (inj_info->injMask); //OR
								nvbit_write_reg(destGPRNum, injAfterVal);
					}
					else {	
								injAfterVal = injBeforeVal & (~inj_info->injMask);//AND			
								nvbit_write_reg(destGPRNum, injAfterVal);

					}
				}
				if(verbose_device)printf("register=%d, before=0x%x, after=0x%x, expected_after=0x%x, mask =0x%x, subsmid %d, laneid %d \n", destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), injAfterVal,inj_info->injMask,subsmid,laneid);
				
				inj_info->errorInjected = true; 
				atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL);  
}

