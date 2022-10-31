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


#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>
#include <cstdarg>

#include "nvbit_reg_rw.h"
#include "utils/utils.h"
#include "pf_injector.h"
#include "arch.h"
#include <math.h>

extern "C" __device__ __noinline__ 
int getGlobalIdx_3D_3D(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
 + gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
 + (threadIdx.z * (blockDim.x * blockDim.y))
 + (threadIdx.y * blockDim.x) + threadIdx.x;
return threadId;
}


extern "C" __device__ __noinline__ void
inject_error(uint64_t pverbose_device,uint64_t piinf, int destGPRNum, int regval, int numDestGPRs, 
int maxthrds, int instidx, uint64_t todo) {
    /**
     * EDIT FOR FLEX GRIP INJECTION
     * I only need to identify the fault site
     * "instruction", "LANEID", "warp_id", "SMID"
     * The good thing is that I can control most of the things from the host side
     * That is, the instruction and the instrumentation is for each instruction
     */


    muliple_ptr_t *inj_struct=(muliple_ptr_t *) todo;
    uint32_t verbose_device = *((uint32_t *) pverbose_device);
    inj_info_t *inj_info_data =(inj_info_t *)piinf; 
    int i = getGlobalIdx_3D_3D();
    auto sm_id=get_smid();
    auto ctaID=get_ctaid();
    auto WID=get_warpid();
    auto LaneID=get_laneid();
    
    /*
    if(i==0){    
    //printf("%d %d %d \n",maxinst,pSM,injinstr);
        for(int ij=0;ij<maxthrds;++ij){
             if(inj_struct->flag[(maxthrds*instidx)+ij]==1){
                 printf("I: %d T: %d B: %d M: 0x%x\n",instidx,ij, inj_struct->T0_ctaID_x[(maxthrds*instidx)+ij],inj_struct->Mask[(maxthrds*instidx)+ij]);
             }
        }
    }*/
    //__syncthreads();
    //__threadfence_system();
    //if((sm_id==inj_info_data->injSMID) && ((WID%4)==inj_info_data->injsubSMID) && (LaneID==inj_info_data->injLaneID)){
    if((inj_struct->flag[(maxthrds*instidx)+i]==1) && (inj_struct->Mask[(maxthrds*instidx)+i]!=0)){
    //if((inj_struct->flag[(maxthrds*instidx)+i]==1)){

        assert(numDestGPRs > 0);
        uint32_t injAfterVal = 0;
        uint32_t injBeforeVal = nvbit_read_reg(destGPRNum); // read the register value
        if (DUMMY) {
            injAfterVal = injBeforeVal;
        } else {
            injAfterVal = injBeforeVal ^ inj_struct->Mask[(maxthrds*instidx)+i];
            nvbit_write_reg(destGPRNum, injAfterVal);
        }
        // updating counter/flag to check whether the error was injected
        if (verbose_device)
            printf("Instruction: %d, CTAx: %d, CTAy:%d, CTAz: %d, WID: %d, LID: %d, register=%d, before=0x%x, after=0x%x, expected_after=0x%x, mask=0x%x\n", instidx, ctaID.x,ctaID.y,ctaID.z,WID,LaneID,destGPRNum , injBeforeVal,
                nvbit_read_reg(destGPRNum), injAfterVal,inj_struct->Mask[(maxthrds*instidx)+i]);
        
        inj_info_data->errorInjected = true; 
        atomicAdd((unsigned long long*) &inj_info_data->injNumActivations, 1LL);
    }  
    //__syncthreads();
          
}



extern "C" __device__ __noinline__ void
inject_error_prev( uint64_t pverbose_device,uint64_t piinf, int destGPRNum, int regval, int numDestGPRs, int maxthrds, int instidx, 
int mempos,int numoper, int reg0, int reg1,int reg2, uint64_t todo, int32_t cbankval) {
    /**
     * EDIT FOR FLEX GRIP INJECTION
     * I only need to identify the fault site
     * "instruction", "LANEID", "warp_id", "SMID"
     * The good thing is that I can control most of the things from the host side
     * That is, the instruction and the instrumentation is for each instruction
     */

    unsigned active_mask = __ballot_sync(__activemask(), 1);

    muliple_ptr_t *inj_struct=(muliple_ptr_t *) todo;
    uint32_t verbose_device = *((uint32_t *) pverbose_device);
    inj_info_t *inj_info_data =(inj_info_t *)piinf; 
    float a,b,c,dist;
    float inj_table=0.0,mask_table=0.0;
    float inj_table1=0.0,mask_table1=0.0;
    float inj_table2=0.0,mask_table2=0.0;
    int i = getGlobalIdx_3D_3D();
    auto lane_id = get_laneid();
    auto sm_id=get_smid();
    auto ctaID=get_ctaid();
    auto WID=get_warpid();
    auto LaneID=get_laneid();

    /* collect cbank values from other threads */
    //__threadfence_system();
    if(mempos==1){
        inj_struct->T0_Mask[(maxthrds*instidx)+i] =__shfl_sync(active_mask, cbankval, i%32);
        inj_struct->T1_Mask[(maxthrds*instidx)+i]=nvbit_read_reg(reg0);   
        inj_struct->T2_Mask[(maxthrds*instidx)+i]=nvbit_read_reg(reg1);
        }
    else if(mempos==2){
        inj_struct->T1_Mask[(maxthrds*instidx)+i] =__shfl_sync(active_mask, cbankval, i%32);
        inj_struct->T0_Mask[(maxthrds*instidx)+i]=nvbit_read_reg(reg0);   
        inj_struct->T2_Mask[(maxthrds*instidx)+i]=nvbit_read_reg(reg1);
        }
    else if(mempos==3){
        inj_struct->T2_Mask[(maxthrds*instidx)+i] =__shfl_sync(active_mask, cbankval, i%32);
        inj_struct->T0_Mask[(maxthrds*instidx)+i]=nvbit_read_reg(reg0);   
        inj_struct->T1_Mask[(maxthrds*instidx)+i]=nvbit_read_reg(reg1);
        }
    else{
        inj_struct->T0_Mask[(maxthrds*instidx)+i]=nvbit_read_reg(reg0);   
        inj_struct->T1_Mask[(maxthrds*instidx)+i]=nvbit_read_reg(reg1);
        inj_struct->T2_Mask[(maxthrds*instidx)+i]=0;
    } 
    //inj_struct->Mask[(maxthrds*instidx)+i]=0;
        
    //__threadfence_system();
    //inj_struct->T0_ctaID_x[(maxthrds*instidx)+i]=ctaID.x;
    //__threadfence_system();
    //if(sm_id==inj_info_data->injSMID && (WID%4)==inj_info_data->injsubSMID && (LaneID==inj_info_data->injLaneID)){     
    //if(inj_struct->flag[(maxthrds*instidx)+i]==1){ 
    #if(FL_INSTR)
    //if((sm_id==inj_info_data->injSMID) && ((WID%4)==inj_info_data->injsubSMID) && (LaneID==inj_info_data->injLaneID)){
        for(int t=0;t<inj_struct->Table_size;++t){
            if((numoper==3) && (inj_struct->num_operands[t]==3)){                
                inj_table= *((float *)(&inj_struct->T0_Mask[(maxthrds*instidx)+i]));
                mask_table= *((float *)(&inj_struct->Operand1[t]));
                a=(inj_table-mask_table)/((mask_table)+0.00000001);   

                inj_table1= *((float *)(&inj_struct->T1_Mask[(maxthrds*instidx)+i]));
                mask_table1= *((float *)(&inj_struct->Operand2[t]));
                b=(inj_table1-mask_table1)/((mask_table1)+0.00000001);

                inj_table2= *((float *)(&inj_struct->T2_Mask[(maxthrds*instidx)+i]));
                mask_table2= *((float *)(&inj_struct->Operand3[t]));
                b=(inj_table2-mask_table2)/((mask_table2)+0.00000001);

                //dist = 100.0*sqrt(powf(a,2)+powf(b,2)+powf(c,2));              
                dist=100.0*(abs(a)+abs(b)+abs(c))/3;
            }else if(numoper==2){
                //float inj_table=0.0,mask_table=0.0;
                //float inj_table1=0.0,mask_table1=0.0;
                //float inj_table2=0.0,mask_table2=0.0;
                inj_table= *((float *)(&inj_struct->T0_Mask[(maxthrds*instidx)+i]));
                mask_table= *((float *)(&inj_struct->Operand1[t]));
                a=(inj_table-mask_table)/((mask_table)+0.00000001);   

                inj_table1= *((float *)(&inj_struct->T1_Mask[(maxthrds*instidx)+i]));
                mask_table1= *((float *)(&inj_struct->Operand2[t]));
                b=(inj_table1-mask_table1)/((mask_table1)+0.00000001);
        
                //dist = 100.0*sqrt(powf(a,2)+powf(b,2));
                dist=100.0*(abs(a)+abs(b))/2;
            }else{
                continue;
            }
            //__threadfence_system();
            //printf("dist= %f",dist);
            if(dist<inj_info_data->Th){
                inj_struct->Mask[(maxthrds*instidx)+i]=inj_struct->Tmasks[t];
                break;
            }
            // __threadfence_system();  
        }
    //}else{
    //    inj_struct->Mask[(maxthrds*instidx)+i]=0;
    //}
    #else
    //if((sm_id==inj_info_data->injSMID) && ((WID%4)==inj_info_data->injsubSMID) && (LaneID==inj_info_data->injLaneID)){
    for(int t=0;t<inj_struct->Table_size;++t){
        if(numoper==3 && inj_struct->num_operands[t]==3){ 
            a=abs(float(inj_struct->T0_Mask[(maxthrds*instidx)+i])-float(inj_struct->Operand1[t]));   
            b=abs(float(inj_struct->T1_Mask[(maxthrds*instidx)+i])-float(inj_struct->Operand2[t]));   
            c=abs(float(inj_struct->T2_Mask[(maxthrds*instidx)+i])-float(inj_struct->Operand3[t]));   
            dist = 100.0*sqrt(powf(a,2)+powf(b,2)+powf(c,2))/float(0xffffffff);              
            
        }else if(numoper==2){
            a=abs(float(inj_struct->T0_Mask[(maxthrds*instidx)+i])-float(inj_struct->Operand1[t]));   
            b=abs(float(inj_struct->T1_Mask[(maxthrds*instidx)+i])-float(inj_struct->Operand2[t]));    
            dist = 100.0*sqrt(powf(a,2)+powf(b,2))/float(0xffffffff);
        }else{
            continue;
        }
        //__threadfence_system();
        if(dist<inj_info_data->Th){
            inj_struct->Mask[(maxthrds*instidx)+i]=inj_struct->Tmasks[t];
            break;
        }
        //__threadfence_system();            
    }
        //inj_struct->flag[(maxthrds*instidx)+i]=1; 
        
    //}else{
        //inj_struct->Mask[(maxthrds*instidx)+i]=0;
        //inj_struct->flag[(maxthrds*instidx)+i]=0; 
    //}
    #endif
    //__syncthreads();
    //}else{
    //    inj_struct->Mask[(maxthrds*instidx)+i]=0;

    //}
    if((sm_id==inj_info_data->injSMID) && ((WID%4)==inj_info_data->injsubSMID) && (LaneID==inj_info_data->injLaneID)){
        inj_struct->flag[(maxthrds*instidx)+i]=1;                
    }else{
        inj_struct->flag[(maxthrds*instidx)+i]=0; 
        inj_struct->Mask[(maxthrds*instidx)+i]=0;
    }
    //__syncthreads();

    //__threadfence_system();

    
    
    /*if(i==0){    
    //printf("%d %d %d \n",maxinst,pSM,injinstr);
        for(int ij=0;ij<inj_info_data->counter_data;++ij){
             if(inj_struct->flag[(maxthrds*instidx)+ij]==1){
                 printf("T: %d B: %d M: 0x%x\n",ij, inj_struct->ctaID_x[(maxthrds*instidx)+ij],inj_struct->Mask[(maxthrds*instidx)+ij]);
             }
            
        }
    }*/
}


// #include <cstdint>
// #include <cstdio>
// #include <iostream>
// #include <vector>
// 
// #include "nvbit_reg_rw.h"
// #include "utils/utils.h"
// #include "pf_injector.h"
// #include "arch.h"
// 
// extern "C" __device__ __noinline__ void
// inject_error(uint64_t piinfo, uint64_t pverbose_device, int destGPRNum, int regval, int numDestGPRs, int maxRegs, int inst_index) {
//     /**
//      * EDIT FOR FLEX GRIP INJECTION
//      * I only need to identify the fault site
//      * "instruction", "LANEID", "warp_id", "SMID"
//      * The good thing is that I can control most of the things from the host side
//      * That is, the instruction and the instrumentation is for each instruction
//      */
//     
//     kernel_level_inj_t *inj_info = (kernel_level_inj_t *) piinfo;
//     uint32_t verbose_device = *((uint32_t *) pverbose_device);
//     auto sm_id = get_smid();
// 
//     uint32_t value = inj_info->inj_instr_info[inst_index].injSMID;
//     
//     if(sm_id!=(inj_info)->inj_instr_info[inst_index].injSMID)
//         return;
// 
//     int4 ctaID = get_ctaid();
//     
//     //printf("%d (%d,%d,%d) \n",sm_id, ctaID.x,ctaID.y,ctaID.z);
//     printf("%d (%d,%d,%d)\n",inj_info->inj_instr_info[inst_index].NumBlocks, inj_info->inj_instr_info[inst_index].instID,
//     inj_info->inj_instr_info[inst_index].instType,inj_info->inj_instr_info[inst_index].injNumActivations);
// 
//     for (int bidx=0;bidx<inj_info->inj_instr_info[inst_index].NumBlocks;++bidx){
//         printf("%d (%d,%d,%d) \n",(inj_info)->inj_instr_info[inst_index].injSMID, inj_info->inj_instr_info[inst_index].inj_block_info[bidx].ctaID_x,inj_info->inj_instr_info[inst_index].inj_block_info[bidx].ctaID_y,inj_info->inj_instr_info[inst_index].inj_block_info[bidx].ctaID_z);
//         if ((ctaID.x == inj_info->inj_instr_info[inst_index].inj_block_info[bidx].ctaID_x)
//          && (ctaID.y == inj_info->inj_instr_info[inst_index].inj_block_info[bidx].ctaID_y) 
//          && (ctaID.z==inj_info->inj_instr_info[inst_index].inj_block_info[bidx].ctaID_z)){
//             //printf("helllo blocks\n");
//             auto warp_id = get_warpid();
//             printf("blokID %d",bidx);
//             for (int idx=0;idx<inj_info->inj_instr_info[inst_index].inj_block_info[bidx].NumWarps;++idx){
//                 auto lane_id = get_laneid();
//                 if (lane_id != inj_info->inj_instr_info[inst_index].inj_block_info[bidx].inj_warp_info[idx].LaneID)
//                     return;
//                 if (warp_id!=inj_info->inj_instr_info[inst_index].inj_block_info[bidx].inj_warp_info[idx].WarpID){
//                     return;
//                 }else{
//                     assert(numDestGPRs > 0);
//                     uint32_t injAfterVal = 0;
//                     uint32_t injBeforeVal = nvbit_read_reg(destGPRNum); // read the register value
//                     if (DUMMY) {
//                         injAfterVal = injBeforeVal;
//                     } else {
//                         injAfterVal = injBeforeVal ^ inj_info->inj_instr_info[inst_index].inj_block_info[bidx].inj_warp_info[idx].Mask;
//                         nvbit_write_reg(destGPRNum, injAfterVal);
//                     }
//                     // updating counter/flag to check whether the error was injected
//                     if (verbose_device)
//                         printf("register=%d, before=0x%x, after=0x%x, expected_after=0x%x\n", destGPRNum, injBeforeVal,
//                             nvbit_read_reg(destGPRNum), injAfterVal);
//                     inj_info->inj_instr_info[inst_index].errorInjected = true;
//                     atomicAdd((unsigned long long *) &inj_info->inj_instr_info[inst_index].injNumActivations, 1LL);
//                     }                
//                 }
//          }
//          else{
//              return;
//          }
//     //if (warp_id != inj_info->warpID[0])
//     //    return; // This is not the selected Warp ID
//     } 
//                                   
// }

// #include <cstdint>
// #include <cstdio>
// #include <iostream>
// #include <vector>
// 
// #include "nvbit_reg_rw.h"
// #include "utils/utils.h"
// #include "pf_injector.h"
// #include "arch.h"
// 
// extern "C" __device__ __noinline__ void
// inject_error(uint64_t piinfo, uint64_t pverbose_device, int destGPRNum, int regval, int numDestGPRs, int maxRegs) {
//     /**
//      * EDIT FOR FLEX GRIP INJECTION
//      * I only need to identify the fault site
//      * "instruction", "LANEID", "warp_id", "SMID"
//      * The good thing is that I can control most of the things from the host side
//      * That is, the instruction and the instrumentation is for each instruction
//      */
//     
//     kernel_level_inj_t *inj_info = (kernel_level_inj_t *) piinfo;
//     uint32_t verbose_device = *((uint32_t *) pverbose_device);
//     auto sm_id = get_smid();
// 
//     if(sm_id!=inj_info[1].inj_instr_info[0].injSMID)
//         return;
// 
//     printf("helllo SM %d\n",sm_id);
//     int4 ctaID = get_ctaid();
//     
//     printf("helllo blocks\n");
//     for (int bidx=0;bidx<inj_info[1].inj_instr_info[0].NumBlocks;++bidx){
//         if ((ctaID.x == inj_info[1].inj_instr_info[0].inj_block_info[bidx].ctaID_x)
//          && (ctaID.y == inj_info[1].inj_instr_info[0].inj_block_info[bidx].ctaID_y) 
//          && (ctaID.z==inj_info[1].inj_instr_info[0].inj_block_info[bidx].ctaID_z)){
//             printf("helllo blocks\n");
//             auto warp_id = get_warpid();
//             //printf("blokID %d",bidx);
//             for (int idx=0;idx<inj_info[1].inj_instr_info[0].inj_block_info[bidx].NumWarps;++idx){
//                 auto lane_id = get_laneid();
//                 if (lane_id != inj_info[1].inj_instr_info[0].inj_block_info[bidx].inj_warp_info[idx].LaneID)
//                     return;
//                 if (warp_id!=inj_info[1].inj_instr_info[0].inj_block_info[bidx].inj_warp_info[idx].WarpID){
//                     return;
//                 }else{
//                     assert(numDestGPRs > 0);
//                     uint32_t injAfterVal = 0;
//                     uint32_t injBeforeVal = nvbit_read_reg(destGPRNum); // read the register value
//                     if (DUMMY) {
//                         injAfterVal = injBeforeVal;
//                     } else {
//                         injAfterVal = injBeforeVal ^ inj_info[1].inj_instr_info[0].inj_block_info[bidx].inj_warp_info[idx].Mask;
//                         nvbit_write_reg(destGPRNum, injAfterVal);
//                     }
//                     // updating counter/flag to check whether the error was injected
//                     if (verbose_device)
//                         printf("register=%d, before=0x%x, after=0x%x, expected_after=0x%x\n", destGPRNum, injBeforeVal,
//                             nvbit_read_reg(destGPRNum), injAfterVal);
//                     inj_info[1].inj_instr_info[0].errorInjected = true;
//                     atomicAdd((unsigned long long *) &inj_info[1].inj_instr_info[0].injNumActivations, 1LL);
//                     }                
//                 }
//          }
//          else{
//              return;
//          }
//     //if (warp_id != inj_info->warpID[0])
//     //    return; // This is not the selected Warp ID
//     } 
//                                    
// }
