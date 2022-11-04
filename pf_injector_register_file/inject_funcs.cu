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


extern "C" __device__ __noinline__ void inject_error_IRA(uint64_t piinfo, uint64_t Data_arrays, uint64_t pverbose_device, int destGPRNum, int regval, int numDestGPRs, int compute_cap) { 
				
				muliple_ptr_t *inj_struct=(muliple_ptr_t *) Data_arrays;
				inj_info_error_t * inj_info = (inj_info_error_t*)piinfo; 
				uint32_t verbose_device = *((uint32_t *)pverbose_device);
				

				int i = getGlobalIdx_3D_3D();
				auto smid=get_smid();
				auto ctaID=get_ctaid();
				auto WID=get_warpid();
				auto LID=get_laneid();
				auto kidx=WID*32+LID;
				//check performed on the Straming Multiprocessor ID
				//printf("smid %d %d\n",smid, inj_info->injSMID );
				if(inj_info->injSMID != smid) 
							return;
				//printf("two\n");
				if(inj_info->injScheduler != (WID%4)) 
						return;				
				//printf("%d\n",inj_info->injScheduler);
				//if(inj_struct->Warp_thread_active[kidx]==0) return:

				assert(numDestGPRs > 0);
				uint32_t injAfterVal = 0; 
				uint32_t injBeforeVal = nvbit_read_reg(inj_info->injRegID); // read the register value
									
				if (DUMMY || destGPRNum != inj_info->injRegID ) { 
								injAfterVal = injBeforeVal;
				
				} else {
					if(inj_struct->Warp_thread_active[kidx]==1){
								injAfterVal = injBeforeVal ^ (inj_struct->warp_thread_mask[kidx]); 
								nvbit_write_reg(inj_info->injRegID, injAfterVal);
								//printf("four\n");
					}
					else {	
								injAfterVal = injBeforeVal;		
								nvbit_write_reg(inj_info->injRegID, injAfterVal);
								//printf("five\n");
											
					}
				if(verbose_device)printf("register=%d, before=0x%x, after=0x%x, expected_after=0x%x, mask =0x%x, stuck at %d\n", inj_info->injRegID, injBeforeVal, nvbit_read_reg(inj_info->injRegID), injAfterVal,inj_struct->warp_thread_mask[kidx],inj_struct->Warp_thread_active[kidx]);
				
				}
				inj_info->errorInjected = true; 
				atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL);  
}


extern "C" __device__ __noinline__ void inject_error_IRAv2(uint64_t piinfo, uint64_t Data_arrays, uint64_t pverbose_device, int destGPRNum, int regval, int numDestGPRs, int compute_cap) { 
				
				muliple_ptr_t *inj_struct=(muliple_ptr_t *) Data_arrays;
				inj_info_error_t * inj_info = (inj_info_error_t*)piinfo; 
				uint32_t verbose_device = *((uint32_t *)pverbose_device);
				

				int i = getGlobalIdx_3D_3D();
				auto smid=get_smid();
				auto ctaID=get_ctaid();
				auto WID=get_warpid();
				auto LID=get_laneid();
				auto kidx=WID*32+LID;
				//check performed on the Straming Multiprocessor ID
				//printf("smid %d %d\n",smid, inj_info->injSMID );
				if(inj_info->injSMID != smid) 
							return;
				//printf("two\n");
				if(inj_info->injScheduler != (WID%4)) 
						return;				
				//printf("%d\n",inj_info->injScheduler);
				//if(inj_struct->Warp_thread_active[kidx]==0) return:

				assert(numDestGPRs > 0);
				uint32_t injAfterVal = 0; 
				uint32_t injBeforeVal = nvbit_read_reg(inj_info->injRegID); // read the register value
				injAfterVal = nvbit_read_reg((inj_info->injRegID+1));
				if (DUMMY || destGPRNum != inj_info->injRegID ) { 
								injAfterVal = injBeforeVal;
				
				} else {
				if(inj_struct->Warp_thread_active[kidx]==1){
					nvbit_write_reg(inj_info->injRegID, injAfterVal);
				
				if(verbose_device)printf("target_register=%d, before=0x%x, after=0x%x, expected_after=0x%x, ReadReg =0x%x, SMthread %d\n", inj_info->injRegID, injBeforeVal, nvbit_read_reg(inj_info->injRegID), nvbit_read_reg(inj_info->injRegID+1),inj_info->injRegID+1,kidx);
				
				}

				}
				inj_info->errorInjected = true; 
				atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL);  
}



extern "C" __device__ __noinline__ void inject_error_IRA_src_before(uint64_t piinfo, uint64_t Data_arrays, uint64_t pverbose_device, int destGPRNum, int regval, int numDestGPRs, int compute_cap, int instridx) { 
				
				muliple_ptr_t *inj_struct=(muliple_ptr_t *) Data_arrays;
				inj_info_error_t * inj_info = (inj_info_error_t*)piinfo; 
				uint32_t verbose_device = *((uint32_t *)pverbose_device);
				

				int i = getGlobalIdx_3D_3D();
				auto smid=get_smid();
				auto ctaID=get_ctaid();
				auto WID=get_warpid();
				auto LID=get_laneid();
				auto kidx=WID*32+LID;
				//check performed on the Straming Multiprocessor ID
				//printf("smid %d %d\n",smid, inj_info->injSMID );				
				if(inj_info->injSMID == smid && inj_info->injScheduler == (WID%4)){								
						assert(numDestGPRs > 0);
						uint32_t injAfterVal = 0; 
						uint32_t injBeforeVal = nvbit_read_reg((uint64_t)inj_info->injRegOriginal); // read the register value
						inj_struct->register_tmp_recovery[WARP_PER_SM*THREAD_PER_WARP*instridx+kidx]=nvbit_read_reg((uint64_t)inj_info->injRegOriginal);
						injAfterVal = nvbit_read_reg((uint64_t)(inj_info->injRegReplacement));						
						if (DUMMY || destGPRNum != inj_info->injRegOriginal ) { 
										injAfterVal = injBeforeVal;
						
						} else {
						if(inj_struct->Warp_thread_active[kidx]==1){
							nvbit_write_reg((uint64_t)inj_info->injRegOriginal, injAfterVal);
							//inj_struct->register_tmp_recovery[inj_struct->num_threads*instridx+i]=injBeforeVal+instridx;
						
						if(verbose_device)printf("smID=%d, warpID=%d,target_register=%d, before=0x%x, after=0x%x, expected_after=0x%x, ReadReg =0x%x, SMthread %d\n", smid, WID, inj_info->injRegOriginal, injBeforeVal, nvbit_read_reg(inj_info->injRegOriginal), nvbit_read_reg(inj_info->injRegReplacement),inj_info->injRegReplacement,kidx);
						
						}

						}
						inj_info->errorInjected = true; 
						atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL);  					
				}
				//__threadfence();
}


extern "C" __device__ __noinline__ void inject_error_IRAv2_src_after(uint64_t piinfo, uint64_t Data_arrays, uint64_t pverbose_device, int destGPRNum, int regval, int numDestGPRs, int compute_cap, int instridx) { 
				
				muliple_ptr_t *inj_struct=(muliple_ptr_t *) Data_arrays;
				inj_info_error_t * inj_info = (inj_info_error_t*)piinfo; 
				uint32_t verbose_device = *((uint32_t *)pverbose_device);
				
				int i = getGlobalIdx_3D_3D();
				auto smid=get_smid();
				auto ctaID=get_ctaid();
				auto WID=get_warpid();
				auto LID=get_laneid();
				auto kidx=WID*32+LID;
				//check performed on the Straming Multiprocessor ID
				//printf("smid %d %d\n",smid, inj_info->injSMID );
				if(inj_info->injSMID == smid && inj_info->injScheduler == (WID%4)){							
				
					assert(numDestGPRs > 0);
					uint32_t injAfterVal = 0; 
					uint32_t injBeforeVal = nvbit_read_reg((uint64_t)inj_info->injRegOriginal); // read the register value
					if (DUMMY || destGPRNum != inj_info->injRegOriginal ) { 
									injAfterVal = injBeforeVal;					
					} else {
					if(inj_struct->Warp_thread_active[kidx]==1){
						nvbit_write_reg((uint64_t)inj_info->injRegOriginal, inj_struct->register_tmp_recovery[WARP_PER_SM*THREAD_PER_WARP*instridx+kidx]);										
					if(verbose_device)printf("smID=%d, warpID=%d,target_register=%d, before=0x%x, after=0x%x, expected_after=0x%x, ReadReg =0x%x, SMthread %d\n", smid, WID, inj_info->injRegOriginal, injBeforeVal, nvbit_read_reg(inj_info->injRegOriginal), nvbit_read_reg(inj_info->injRegReplacement),inj_struct->register_tmp_recovery[WARP_PER_SM*THREAD_PER_WARP*instridx+kidx],instridx);
					
					}

					}
					inj_info->errorInjected = true; 
					atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL);  
				}
}




extern "C" __device__ __noinline__ void inject_error_IRA_dst(uint64_t piinfo, uint64_t Data_arrays, uint64_t pverbose_device, int destGPRNum, int regval, int numDestGPRs, int compute_cap) { 
				
			muliple_ptr_t *inj_struct=(muliple_ptr_t *) Data_arrays;
			inj_info_error_t * inj_info = (inj_info_error_t*)piinfo; 
			uint32_t verbose_device = *((uint32_t *)pverbose_device);
			

			int i = getGlobalIdx_3D_3D();
			auto smid=get_smid();
			auto ctaID=get_ctaid();
			auto WID=get_warpid();
			auto LID=get_laneid();
			auto kidx=WID*32+LID;
			//check performed on the Straming Multiprocessor ID
			//printf("smid %d %d\n",smid, inj_info->injSMID );
			assert(numDestGPRs > 0);
			uint32_t injAfterVal = 0; 
			uint32_t injBeforeVal = nvbit_read_reg((uint64_t)inj_info->injRegOriginal); // read the register value
			if(inj_info->injSMID == smid && inj_info->injScheduler == (WID%4)){	
				if (DUMMY || inj_info->injRegReplacement == inj_info->injRegOriginal ) { 
								injAfterVal = injBeforeVal;
				} else {
				if(inj_struct->Warp_thread_active[kidx]==1){
					injAfterVal = injBeforeVal ^ inj_struct->warp_thread_mask[kidx];
					nvbit_write_reg((uint64_t)inj_info->injRegReplacement, injBeforeVal);
					nvbit_write_reg((uint64_t)inj_info->injRegOriginal, injAfterVal);
				
				if(verbose_device)printf("smID=%d, warpID=%d,target_register=%d, before=0x%x, after=0x%x, expected_after=0x%x, ReadReg =0x%x, SMthread %d\n", smid, WID, inj_info->injRegOriginal, injBeforeVal, nvbit_read_reg(inj_info->injRegOriginal), nvbit_read_reg(inj_info->injRegReplacement),inj_info->injRegReplacement,kidx);
				
				}

				}
				inj_info->errorInjected = true; 
				atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL); 
			} 
}
