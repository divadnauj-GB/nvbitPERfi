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


extern "C" __device__ __noinline__ void inject_error_IRA(uint64_t piinfo, uint64_t Data_arrays, uint64_t pverbose_device, 
int destGPRNum, int regval, int numDestGPRs, int compute_cap) { 
                
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


extern "C" __device__ __noinline__ void inject_error_IRAv2(uint64_t piinfo, uint64_t Data_arrays, uint64_t pverbose_device, 
int destGPRNum, int regval, int numDestGPRs, int compute_cap) { 
                
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



extern "C" __device__ __noinline__ void inject_error_IRA_src_before(uint64_t piinfo, uint64_t Data_arrays, uint64_t pverbose_device, 
int destGPRNum, int replGPRNum, int regval, int numDestGPRs, int compute_cap, int instridx,int InstOffset, int InstOpcode) { 
                
                muliple_ptr_t *inj_struct=(muliple_ptr_t *) Data_arrays;
                inj_info_error_t * inj_info = (inj_info_error_t*)piinfo; 
                uint32_t verbose_device = *((uint32_t *)pverbose_device);
                

                int i = getGlobalIdx_3D_3D();
                auto smid=get_smid();
                auto ctaID=get_ctaid();
                auto WID=get_warpid();
                auto LID=get_laneid();
                auto kidx=WID*32+LID;
                int ctaIDX =ctaID.x;
                int ctaIDY =ctaID.y;
                int ctaIDZ =ctaID.z;
                uint32_t injAfterVal = 0; 
                uint32_t injBeforeVal = 0;
                //check performed on the Straming Multiprocessor ID
                //printf("smid %d %d\n",smid, inj_info->injSMID );	
                //inj_struct->register_tmp_recovery[inj_struct->num_threads*instridx+i]=nvbit_read_reg(destGPRNum);			
                if(inj_info->injSMID == smid && inj_info->injScheduler == (WID%4)){								                        
                        assert(numDestGPRs > 0);                        
                        injBeforeVal = nvbit_read_reg(destGPRNum); // read the register value                        
                        inj_struct->register_tmp_recovery[kidx]=nvbit_read_reg(destGPRNum);
                        //inj_struct->register_tmp_recovery[inj_info->MaxWarpsPerSM*inj_info->MaxThreadsPerWarp*instridx+kidx]=nvbit_read_reg(destGPRNum);
                        //inj_struct->register_tmp_recovery[inj_struct->num_threads*instridx+i]=nvbit_read_reg(destGPRNum);                        
                        //inj_struct->register_tmp_recovery[kidx]=nvbit_read_reg((uint64_t)destGPRNum);                        						
                        inj_info->injInstrIdx=instridx;
                        inj_info->injInstOpcode=InstOpcode;
                        inj_info->injInstPC=InstOffset;
                        inj_info->injRegOriginal=destGPRNum;
                        inj_info->injRegReplacement=replGPRNum;                        
                        if (DUMMY || replGPRNum==destGPRNum ) { 
                            injAfterVal = injBeforeVal;
                        } else {
                            if(inj_struct->Warp_thread_active[kidx]==1){
                                injAfterVal = nvbit_read_reg(replGPRNum);
                                inj_info->errorInjected = true;                                 
                                atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL);
                                nvbit_write_reg(destGPRNum, injAfterVal);
                                //inj_struct->register_tmp_recovery[inj_struct->num_threads*instridx+i]=injBeforeVal+instridx;                            
                                if(verbose_device)printf("BF: smID=%d, warpID=%d,target_register=%d, before=0x%x, after=0x%x, expected_after=0x%x, ReadReg =0x%x, InstrInst %d\n", smid, WID, destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), injAfterVal, nvbit_read_reg(replGPRNum),instridx);                             
                                //printf("BF: smID=%d, warpID=%d,target_register=%d, before=0x%x, after=0x%x, expected_after=0x%x, ReadReg =0x%x, InstrInst %d\n", smid, WID, destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), injAfterVal, nvbit_read_reg(replGPRNum),instridx);                                     
                                /*
                                printf("BF$ smID: %d; schID: %d; ctaID.x: %d; ctaID.y: %d; ctaID.z: %d; warpID: %d; LaneID: %d; TargOpField: %d; OrgRegID: %d; RepRegID: %d; MskSeed: %d; InstErrID: %d; PCOffset: %d; InstType: %d$ RegID: %d; ValBefore: %d; ValAfter: %d; ExpectAfter: %d; ErrRegID: %d; ERegValBefore: %d; ERegValAfter: %d; ERegExpectAfter: %d$ Kindex: %d \n",
                                smid,inj_info->injScheduler, ctaID.x,ctaID.y,ctaID.z, WID,LID,inj_info->injRegID,destGPRNum,replGPRNum,inj_info->injMaskSeed,instridx, InstOffset, InstOpcode, destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), injAfterVal,replGPRNum,injAfterVal,nvbit_read_reg(replGPRNum),injAfterVal,i);
                                */
                                                            
                            }

                        }
                         					
                }
                //__threadfence_system();
}


extern "C" __device__ __noinline__ void inject_error_IRA_src_after(uint64_t piinfo, uint64_t Data_arrays, uint64_t pverbose_device, 
int destGPRNum, int replGPRNum, int regval, int numDestGPRs, int compute_cap, int instridx,int InstOffset, int InstOpcode) { 
                
                muliple_ptr_t *inj_struct=(muliple_ptr_t *) Data_arrays;
                inj_info_error_t * inj_info = (inj_info_error_t*)piinfo; 
                uint32_t verbose_device = *((uint32_t *)pverbose_device);
                
                int i = getGlobalIdx_3D_3D();
                auto smid=get_smid();
                auto ctaID=get_ctaid();
                auto WID=get_warpid();
                auto LID=get_laneid();
                auto kidx=WID*32+LID;
                int ctaIDX =ctaID.x;
                int ctaIDY =ctaID.y;
                int ctaIDZ =ctaID.z;
                //check performed on the Straming Multiprocessor ID
                //printf("smid %d %d\n",smid, inj_info->injSMID );
                if(inj_info->injSMID == smid && inj_info->injScheduler == (WID%4)){							                    
                    assert(numDestGPRs > 0);
                    uint32_t injAfterVal = 0; 
                    uint32_t injBeforeVal = 0;  // read the register value
                    //uint32_t injBeforeVal=nvbit_read_reg(destGPRNum);
                    //injAfterVal= inj_struct->register_tmp_recovery[WARP_PER_SM*THREAD_PER_WARP+kidx];										
                    //injAfterVal=inj_struct->register_tmp_recovery[inj_struct->num_threads*instridx+i];
                    inj_info->injInstrIdx=instridx;
                    inj_info->injInstOpcode=InstOpcode;
                    inj_info->injInstPC=InstOffset;
                    inj_info->injRegOriginal=destGPRNum;
                    inj_info->injRegReplacement=replGPRNum;
                    if (DUMMY || replGPRNum==destGPRNum) { 
                        injAfterVal = injBeforeVal;					
                    } else {
                        if(inj_struct->Warp_thread_active[kidx]==1){  
                            inj_info->errorInjected = true;                             
                            atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL); 
                            injBeforeVal=nvbit_read_reg(destGPRNum); 
                            //injAfterVal=inj_struct->register_tmp_recovery[inj_struct->num_threads*instridx+i];  
                            injAfterVal= inj_struct->register_tmp_recovery[kidx];	                      
                            //injAfterVal= inj_struct->register_tmp_recovery[inj_info->MaxWarpsPerSM*inj_info->MaxThreadsPerWarp*instridx+kidx];	                      
                            //nvbit_write_reg(destGPRNum, inj_struct->register_tmp_recovery[WARP_PER_SM*THREAD_PER_WARP*instridx+kidx]);										
                            nvbit_write_reg(destGPRNum, injAfterVal);										
                            //nvbit_write_reg(destGPRNum, inj_struct->register_tmp_recovery[kidx]);										
                            if(verbose_device)printf("AF: smID=%d, warpID=%d,target_register=%d, before=0x%x, after=0x%x, expected_after=0x%x, ReadReg =0x%x, InstrInst %d\n", smid, WID, destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), injAfterVal, nvbit_read_reg(replGPRNum),instridx);                             
                            //printf("AF: smID=%d, warpID=%d,target_register=%d, before=0x%x, after=0x%x, expected_after=0x%x, ReadReg =0x%x, InstrInst %d\n", smid, WID, destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), injAfterVal, nvbit_read_reg(replGPRNum),instridx);                             
                            /*printf("AF$ smID: %d; schID: %d; ctaID.x: %d; ctaID.y: %d; ctaID.z: %d; warpID: %d; LaneID: %d; TargOpField: %d; OrgRegID: %d; RepRegID: %d; MskSeed: %d; InstErrID: %d; PCOffset: %d; InstType: %d$ RegID: %d; ValBefore: %d; ValAfter: %d; ExpectAfter: %d; ErrRegID: %d; ERegValBefore: %d; ERegValAfter: %d; ERegExpectAfter: %d$ Kindex: %d\n",
                            smid,inj_info->injScheduler,ctaID.x,ctaID.y,ctaID.z,WID,LID,inj_info->injRegID,destGPRNum,replGPRNum,inj_info->injMaskSeed, instridx, InstOffset, InstOpcode, destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), injAfterVal, replGPRNum,nvbit_read_reg(replGPRNum),nvbit_read_reg(replGPRNum),nvbit_read_reg(replGPRNum),i);
                            */
                        }
                    }                     
                }
                //__threadfence_system();
}




extern "C" __device__ __noinline__ void inject_error_IRA_dst(uint64_t piinfo, uint64_t Data_arrays, uint64_t pverbose_device, 
int destGPRNum, int replGPRNum, int regval, int numDestGPRs, int compute_cap, int instridx, int InstOffset, int InstOpcode) { 
                
            muliple_ptr_t *inj_struct=(muliple_ptr_t *) Data_arrays;
            inj_info_error_t * inj_info = (inj_info_error_t*)piinfo; 
            uint32_t verbose_device = *((uint32_t *)pverbose_device);
                        

            int i = getGlobalIdx_3D_3D();
            auto smid=get_smid();
            auto ctaID=get_ctaid();
            auto WID=get_warpid();
            auto LID=get_laneid();
            auto kidx=WID*32+LID;
            int ctaIDX =ctaID.x;
            int ctaIDY =ctaID.y;
            int ctaIDZ =ctaID.z;
            //check performed on the Straming Multiprocessor ID
            //printf("smid %d %d\n",smid, inj_info->injSMID );
            assert(numDestGPRs > 0);
            uint32_t injAfterVal = 0; 
            uint32_t injBeforeValrep = 0; 
            //uint32_t injBeforeVal = nvbit_read_reg(destGPRNum); // read the register value
            uint32_t injBeforeVal = 0; // read the register value
            if(inj_info->injSMID == smid && inj_info->injScheduler == (WID%4)){	
                inj_info->injInstrIdx=instridx;
                inj_info->injInstOpcode=InstOpcode;
                inj_info->injInstPC=InstOffset;
                inj_info->injRegOriginal=destGPRNum;
                inj_info->injRegReplacement=replGPRNum;
                if (DUMMY || replGPRNum==destGPRNum) { 
                                injAfterVal = injBeforeVal;
                } else {
                    if(inj_struct->Warp_thread_active[kidx]==1){
                        injBeforeVal = nvbit_read_reg(destGPRNum);  
                        injBeforeValrep = nvbit_read_reg(replGPRNum);                                     
                        injAfterVal = injBeforeVal ^ inj_struct->warp_thread_mask[kidx];
                        inj_info->errorInjected = true; 
                        atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL); 
                        nvbit_write_reg(replGPRNum, injBeforeVal);
                        nvbit_write_reg(destGPRNum, (injAfterVal));                                            
                        if(verbose_device)printf("DST: smID=%d, warpID=%d,target_register=%d, before=0x%x, after=0x%x, expected_after=0x%x, ReadReg =0x%x, SMthread %d\n", smid, WID, destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), nvbit_read_reg(replGPRNum),replGPRNum,instridx);                                            
                        //printf("DST: smID=%d, warpID=%d,target_register=%d, before=0x%x, after=0x%x, expected_after=0x%x, ReadReg =0x%x, SMthread %d\n", smid, WID, destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), nvbit_read_reg(replGPRNum),replGPRNum,instridx);                                            
                        /*printf("DST$ smID: %d; schID: %d; ctaID.x: %d; ctaID.y: %d; ctaID.z: %d; warpID: %d; LaneID: %d; TargOpField: %d; OrgRegID: %d; RepRegID: %d; MskSeed: %d; InstErrID: %d; PCOffset: %d; InstType: %d$ RegID: %d; ValBefore: %d; ValAfter: %d; ExpectAfter: %d; ErrRegID: %d; ERegValBefore: %d; ERegValAfter: %d; ERegExpectAfter: %d$ Kindex: %d \n",
                        smid,inj_info->injScheduler,ctaID.x,ctaID.y,ctaID.z,WID,LID,inj_info->injRegID,destGPRNum,replGPRNum,inj_info->injMaskSeed, instridx, InstOffset, InstOpcode, destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), injAfterVal, replGPRNum,injBeforeValrep,nvbit_read_reg(replGPRNum),injBeforeVal,kidx);
                        */
                    }
                    //__threadfence();
                }                 
            } 
}


extern "C" __device__ __noinline__ void inject_error_IAT(uint64_t piinfo, uint64_t Data_arrays, uint64_t pverbose_device, 
int destGPRNum, int regval, int numDestGPRs, int blokDimm, int instridx, int InstOffset, int InstOpcode) { 
                
            muliple_ptr_t *inj_struct=(muliple_ptr_t *) Data_arrays;
            inj_info_error_t * inj_info = (inj_info_error_t*)piinfo; 
            uint32_t verbose_device = *((uint32_t *)pverbose_device);
                        

            int i = getGlobalIdx_3D_3D();
            auto smid=get_smid();
            auto ctaID=get_ctaid();
            auto WID=get_warpid();
            auto LID=get_laneid();
            auto kidx=WID*32+LID;
            int ctaIDX =ctaID.x;
            int ctaIDY =ctaID.y;
            int ctaIDZ =ctaID.z;
            auto blktidx= threadIdx.z * blockDim.y * blockDim.x
            + threadIdx.y * blockDim.x + threadIdx.x;

            //check performed on the Straming Multiprocessor ID
            //printf("smid %d %d\n",smid, inj_info->injSMID );
            assert(numDestGPRs > 0);
            uint32_t injAfterVal = 0; 
            uint32_t injBeforeValrep = 0; 
            //uint32_t injBeforeVal = nvbit_read_reg(destGPRNum); // read the register value
            uint32_t injBeforeVal = 0; // read the register value
            inj_struct->warp_thread_mask[kidx] = blktidx;
            if(inj_info->injSMID == smid && inj_info->injScheduler == (WID%4)){	
                inj_info->injInstrIdx=instridx;
                inj_info->injInstOpcode=InstOpcode;
                inj_info->injInstPC=InstOffset;
                inj_info->injRegOriginal=destGPRNum;               
                injBeforeVal=nvbit_read_reg(destGPRNum); 
                              
                
                //auto tidx=(inj_struct->warp_thread_mask[inj_info->injMaskSeed*inj_info->MaxThreadsPerWarp+LID])%blockDim.x;
                //auto tidy=(inj_struct->warp_thread_mask[inj_info->injMaskSeed*inj_info->MaxThreadsPerWarp+LID]/blockDim.x)%blockDim.y;
                //auto tidz=(inj_struct->warp_thread_mask[inj_info->injMaskSeed*inj_info->MaxThreadsPerWarp+LID]/(blockDim.y*blockDim.x))%blockDim.z;
                //printf("%d %d %d %d %d %d %d %d\n",kidx,WID,inj_struct->warp_thread_mask[inj_info->injMaskSeed*inj_info->MaxThreadsPerWarp+LID],threadIdx.x,injBeforeVal, tidx,tidy,tidz);
                /*
                if(injBeforeVal==blokDimm){
                    injAfterVal = 0;
                }else{
                    injAfterVal = blokDimm;
                }
                */
               /*
                if(blokDimm==0){
                        injAfterVal=tidx;
                }else if(blokDimm==1){
                        injAfterVal=tidy;
                }else if(blokDimm==2){
                        injAfterVal=tidz;
                }
                */
                /*injAfterVal=injBeforeVal+32;
                if(injAfterVal>blktidx)
                {
                    injAfterVal=LID;
                }*/
                if(inj_info->injStuck_at==0){
                    injAfterVal = injBeforeVal & (~inj_info->injMaskSeed);
                }
                else{
                    injAfterVal = injBeforeVal | inj_info->injMaskSeed;
                }
                //injAfterVal = injBeforeVal ^ inj_info->injMaskSeed;
                if(injAfterVal==injBeforeVal){
                    injAfterVal = injBeforeVal ^ inj_info->injMaskSeed;
                }

                if (DUMMY ) { 
                                injAfterVal = injBeforeVal;
                } else {    
                    if(inj_struct->Warp_thread_active[kidx]==1){                    
                        inj_info->errorInjected = true; 
                        atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL);                           
                        nvbit_write_reg(destGPRNum, injAfterVal);                                        
                        if(verbose_device)printf("DST: smID=%d, warpID=%d,target_register=%d, before=0x%x, after=0x%x, expected_after=0x%x, ReadReg =0x%x, SMthread %d\n", smid, WID, destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), nvbit_read_reg(destGPRNum),destGPRNum,instridx);                                                                                            
                    //__threadfence();
                    /*
                    printf("IAT$ smID: %d; schID: %d; ctaID.x: %d; ctaID.y: %d; ctaID.z: %d; warpID: %d; LaneID: %d; TargOpField: %d; OrgRegID: %d; BlockDim: %d; MskSeed: %d; InstErrID: %d; PCOffset: %d; InstType: %d$ RegID: %d; ValBefore: %d; ValAfter: %d; ExpectAfter: %d$ Kindex: %d\n",
                        smid,inj_info->injScheduler,ctaID.x,ctaID.y,ctaID.z,WID,LID,0,destGPRNum,blokDimm,0, instridx, InstOffset, InstOpcode, destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), injAfterVal,kidx);                    
                    */
                    }
                }                 
            } 
            
}


extern "C" __device__ __noinline__ void inject_error_IAC(uint64_t piinfo, uint64_t Data_arrays, uint64_t pverbose_device, 
int destGPRNum, int regval, int numDestGPRs, int gridDimm, int instridx, int InstOffset, int InstOpcode) { 
                
            muliple_ptr_t *inj_struct=(muliple_ptr_t *) Data_arrays;
            inj_info_error_t * inj_info = (inj_info_error_t*)piinfo; 
            uint32_t verbose_device = *((uint32_t *)pverbose_device);
                        

            int i = getGlobalIdx_3D_3D();
            auto smid=get_smid();
            auto ctaID=get_ctaid();
            auto WID=get_warpid();
            auto LID=get_laneid();
            auto kidx=WID*32+LID;
            int ctaIDX =ctaID.x;
            int ctaIDY =ctaID.y;
            int ctaIDZ =ctaID.z;

            auto blktidx= threadIdx.z * blockDim.y * blockDim.x
            + threadIdx.y * blockDim.x + threadIdx.x;

            //check performed on the Straming Multiprocessor ID
            //printf("smid %d %d\n",smid, inj_info->injSMID );
            assert(numDestGPRs > 0);
            uint32_t injAfterVal = 0; 
            uint32_t injBeforeValrep = 0; 
            //uint32_t injBeforeVal = nvbit_read_reg(destGPRNum); // read the register value
            uint32_t injBeforeVal = 0; // read the register value
            if(inj_info->injSMID == smid && inj_info->injScheduler == (WID%4)){	
                inj_info->injInstrIdx=instridx;
                inj_info->injInstOpcode=InstOpcode;
                inj_info->injInstPC=InstOffset;
                inj_info->injRegOriginal=destGPRNum;
                injBeforeVal = nvbit_read_reg(destGPRNum); 

                if(inj_info->injStuck_at==0){
                    injAfterVal = injBeforeVal & (~inj_info->injMaskSeed);
                }
                else{
                    injAfterVal = injBeforeVal | inj_info->injMaskSeed;
                }
                //injAfterVal = injBeforeVal ^ inj_info->injMaskSeed;
                if(injAfterVal==injBeforeVal){
                    injAfterVal = injBeforeVal ^ inj_info->injMaskSeed;
                }
                if(injAfterVal<0){
                    injAfterVal=0x7fffffff;
                }
                //printf("%d %d %d %d %d \n",kidx,WID,threadIdx.x,injBeforeVal,injAfterVal);
                if (DUMMY ) { 
                                injAfterVal = injBeforeVal;
                } else {    
                    if(inj_struct->Warp_thread_active[kidx]==1){                    
                        inj_info->errorInjected = true; 
                        atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL);                           
                        nvbit_write_reg(destGPRNum, injAfterVal);                                        
                        if(verbose_device)printf("DST: smID=%d, warpID=%d,target_register=%d, before=0x%x, after=0x%x, expected_after=0x%x, ReadReg =0x%x, SMthread %d\n", smid, WID, destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), nvbit_read_reg(destGPRNum),destGPRNum,instridx);                                                                                            
                    //__threadfence();
                    /*
                    printf("IAT$ smID: %d; schID: %d; ctaID.x: %d; ctaID.y: %d; ctaID.z: %d; warpID: %d; LaneID: %d; TargOpField: %d; OrgRegID: %d; BlockDim: %d; MskSeed: %d; InstErrID: %d; PCOffset: %d; InstType: %d$ RegID: %d; ValBefore: %d; ValAfter: %d; ExpectAfter: %d$ Kindex: %d\n",
                        smid,inj_info->injScheduler,ctaID.x,ctaID.y,ctaID.z,WID,LID,0,destGPRNum,blokDimm,0, instridx, InstOffset, InstOpcode, destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), injAfterVal,kidx);                    
                    */
                    }
                }                 
            } 
            
}

extern "C" __device__ __noinline__ void inject_error_WV(uint64_t piinfo, uint64_t Data_arrays, uint64_t pverbose_device, 
int PredNum, int predval, int predreg, int numDestGPRs, int gridDimm, int instridx, int InstOffset, int InstOpcode) { 
                
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
            uint32_t injBeforeValrep = 0; 
            //uint32_t injBeforeVal = nvbit_read_reg(destGPRNum); // read the register value
            uint32_t injBeforeVal = 0; // read the register value
            if(inj_info->injSMID == smid && inj_info->injScheduler == (WID%4)){	
                inj_info->injInstrIdx=instridx;
                inj_info->injInstOpcode=InstOpcode;
                inj_info->injInstPC=InstOffset;
                inj_info->injRegOriginal=PredNum;
                injBeforeVal = nvbit_read_pred_reg();
               /* if(inj_info->injStuck_at==0){
                    injAfterVal=injBeforeVal & (~inj_info->injMaskSeed);
                }else{
                    injAfterVal=injBeforeVal | (inj_info->injMaskSeed);
                }
                if(injAfterVal==injBeforeVal){
                   injAfterVal=injBeforeVal ^ (inj_info->injMaskSeed); 
                }*/
                injAfterVal=injBeforeVal ^ (inj_info->injMaskSeed); 
                if (DUMMY ) { 
                                injAfterVal = injBeforeVal;
                } else {    
                    if(inj_struct->Warp_thread_active[kidx]==1){                    
                        inj_info->errorInjected = true; 
                        atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL);                        
                        nvbit_write_pred_reg(injAfterVal);
                        //printf("A:TID: %d; WID: %d; PredNum: %d; PredVal: %d; prev_val: %d; Newval: %d; PredReg: %d\n",i, WID,PredNum, predval, prev_vall, nvbit_read_pred_reg(), predreg);                          
                        //nvbit_write_reg(destGPRNum, injAfterVal);                                        
                        //if(verbose_device)printf("DST: smID=%d, warpID=%d,target_register=%d, before=0x%x, after=0x%x, expected_after=0x%x, ReadReg =0x%x, SMthread %d\n", smid, WID, destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), nvbit_read_reg(destGPRNum),destGPRNum,instridx);                                                                                            
                    //__threadfence();
                    /*
                    printf("IAT$ smID: %d; schID: %d; ctaID.x: %d; ctaID.y: %d; ctaID.z: %d; warpID: %d; LaneID: %d; TargOpField: %d; OrgRegID: %d; BlockDim: %d; MskSeed: %d; InstErrID: %d; PCOffset: %d; InstType: %d$ RegID: %d; ValBefore: %d; ValAfter: %d; ExpectAfter: %d$ Kindex: %d\n",
                        smid,inj_info->injScheduler,ctaID.x,ctaID.y,ctaID.z,WID,LID,0,destGPRNum,blokDimm,0, instridx, InstOffset, InstOpcode, destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), injAfterVal,kidx);                    
                    */
                    }
                }                 
            } 
            
}