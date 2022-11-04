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
#include <assert.h>
#include <pthread.h>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>
#include <signal.h>
#include <unistd.h>
#include <unordered_set>
#include <cstdlib>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"

#include "globals.h"
#include "pf_injector.h"

using namespace std;

int verbose;
__managed__ int verbose_device;

int limit = INT_MAX;

// injection parameters input filename: This file is created the the script
// that launched error injections
std::string injInputFilename = "nvbitfi-injection-info.txt";
pthread_mutex_t mutex;
__managed__ inj_info_t inj_info; 
__managed__ inj_info_error_t inj_error_info;

__managed__ muliple_ptr_t Injection_masks;

bool read_file=false;

int num_threads;
int max_regcount=0;
std::string kname;

//inj_info_t inj_info;

void reset_inj_info() {
        inj_info.injInstType = 0; 
        inj_info.injSMID = 0; 
        inj_info.injLaneID = 0;

        inj_info.injThreadID = 0;
          inj_info.injReg = 0;
          inj_info.injStuckat = 0;
        //inj_info.injCtaID = 0;
        
        inj_info.injMask = 0;
        inj_info.injNumActivations = 0;
        inj_info.errorInjected = false;

        inj_error_info.injSMID=0;
        inj_error_info.injScheduler=0;
        inj_error_info.injWarpMaskH=0;
        inj_error_info.injWarpMaskL=0;
        inj_error_info.injThreadMask=0;
        inj_error_info.injMaskSeed=0;
        inj_error_info.injRegID=0;
        inj_error_info.injNumActivations = 0;
        inj_error_info.errorInjected = false;
}

// for debugging 
void print_inj_info() {
        assert(fout.good());
        //std::cout << "InstType=" << inj_info.injInstType << ", SMID=" << inj_info.injSMID<< ", LaneID=" << inj_info.injLaneID;
        
        //std::cout << ", Mask=" << inj_info.injMask << std::endl;
}

// Parse error injection site info from a file. This should be done on host side.
void parse_params(std::string filename) {
        static bool parse_flag = false; // file will be parsed only once - performance enhancement
        if (!parse_flag) {
                parse_flag = true;
                reset_inj_info(); 
                float random=0;
                std::ifstream ifs (filename.c_str(), std::ifstream::in);
                if (ifs.is_open()) {
                        
                        ifs >> inj_error_info.injSMID;
                        ifs >> inj_error_info.injScheduler;
                        ifs >> inj_error_info.injWarpMaskH;
                        ifs >> inj_error_info.injWarpMaskL;
                        ifs >> inj_error_info.injThreadMask;
                        ifs >> inj_error_info.injMaskSeed;
                        ifs >> inj_error_info.injRegID;

                        assert(inj_error_info.injSMID < 1000); 
                        inj_error_info.injNumActivations=0;

                } else {
                        printf(" File %s does not exist!", filename.c_str());
                        printf(" This file should contain enough information about the fault site to perform a permanent error injection run: ");
                        printf("(1) SM ID, (2) Lane ID (within a warp), (3) 32-bit mask (as int32), (4) Instruction type (as integer, see maxwell_pascal.h). \n"); 
                        assert(false);
                }
                ifs.close();

                if (verbose) {
                        print_inj_info();
                }

                CUDA_SAFECALL(cudaMallocManaged(&(Injection_masks.Warp_thread_active),(WARP_PER_SM*THREAD_PER_WARP)*sizeof(uint32_t)));
                CUDA_SAFECALL(cudaMallocManaged(&(Injection_masks.warp_thread_mask),(WARP_PER_SM*THREAD_PER_WARP)*sizeof(uint32_t)));
                CUDA_SAFECALL(cudaMallocManaged(&(Injection_masks.register_tmp_recovery),(WARP_PER_SM*THREAD_PER_WARP*MAX_KNAME_SIZE)*sizeof(uint32_t)));
                
                Injection_masks.num_threads=num_threads;

                srand(inj_error_info.injMaskSeed);
                for(int i=0;i<(WARP_PER_SM*THREAD_PER_WARP);++i){
                    random=rand();
                    Injection_masks.warp_thread_mask[i]=*(int *)&random;
                }
                int idx=0;
                int validW=0;
                int validT=0;
                int integer_mask=0;
                for(int i=0;i<WARP_PER_SM;++i){
                    if (i>31){
                        validW=(inj_error_info.injWarpMaskH>>i)&1;
                    }else{
                        validW=(inj_error_info.injWarpMaskL>>i)&1;
                    }
                    for(int j=0;j<THREAD_PER_WARP;++j){   
                        validT= (inj_error_info.injThreadMask>>j)&1;                    
                        random=rand();
                        integer_mask=*(int *)&random;
                        Injection_masks.warp_thread_mask[idx]=integer_mask;
                        Injection_masks.Warp_thread_active[idx]=(validW&validT);
                        idx++;
                    }
                }

        }
}

void update_verbose() {
        static bool update_flag = false; // update it only once - performance enhancement
        if (!update_flag) {
            update_flag = true;
            cudaDeviceSynchronize();
            verbose_device = verbose;
            cudaDeviceSynchronize();
        }
}

int get_maxregs(CUfunction func) {
        int maxregs = -1;
        cuFuncGetAttribute(&maxregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func);
                
        //cuFuncGetAttribute();
        return maxregs;
}

void INThandler(int sig) {
        signal(sig, SIG_IGN); // disable Ctrl-C
        fout << "=================================================================================" << endl;
        fout << "Report for: " << kname << "; kernel Index: "<< kernel_id <<  endl;
        fout << "=================================================================================" << endl;
        fout << ":::NVBit-inject-error; ERROR FAIL Detected Singal SIGKILL;";
        fout << " injNumActivations: " << inj_error_info.injNumActivations << ":::"  << endl;
        fout << "=================================================================================" << endl;
        fout << "Final Report" <<  endl;
        fout << "=================================================================================" << endl;
        fout << "Report_Summary: "
                << "; Target_SM: " << inj_error_info.injSMID
                << "; Target_Scheduler: " << inj_error_info.injScheduler
                << "; Target_reg_field: " << inj_error_info.injRegID
                << "; Max_reg_count: " << max_regcount
                << "; Original_register: " << inj_error_info.injRegOriginal
                <<"; Replacement_register: " << inj_error_info.injRegReplacement
                << "; Error Mask: " << inj_error_info.injMaskSeed
                <<"; Total_Num_Error_Activations: " << inj_info.injNumActivations+inj_error_info.injNumActivations;
                if (max_regcount < inj_error_info.injRegReplacement){
                        fout << "; error register outside the limits..";
                }else{
                        fout << "; error register inside the limits..";
                }
        fout.flush();
        exit(-1);
}


/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
// DO NOT USE UVM (__managed__) variables in this function
void nvbit_at_init() {
        /* just make sure all managed variables are allocated on GPU */
        setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC","1",1);

        /* we get some environment variables that are going to be use to selectively
         * instrument (within a interval of kernel indexes and instructions). By
         * default we instrument everything. */
        if (getenv("TOOL_VERBOSE")) {
                verbose = atoi(getenv("TOOL_VERBOSE"));
        } else {
                verbose = 0;
        }

        if (getenv("INPUT_INJECTION_INFO")) {
                injInputFilename = getenv("INPUT_INJECTION_INFO");
        }
        if (getenv("OUTPUT_INJECTION_LOG")) {
                injOutputFilename = getenv("OUTPUT_INJECTION_LOG");
        }
        if (getenv("INSTRUMENTATION_LIMIT")) {
                limit = atoi(getenv("INSTRUMENTATION_LIMIT"));
        } 

        //GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool (1, 2, 3,..)");

        initInstTypeNameMap();

        signal(SIGINT, INThandler); // install Ctrl-C handler

        open_output_file(injOutputFilename);

        //parse_params(injInputFilename);

        if (verbose) printf("nvbit_at_init:end\n");
        //open_profile_file(injectionOut); 
        //injectionOut = "injection-results.txt";
           //fout3 = fopen("injection-results.txt","a");
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;


void instrument_function_if_needed(CUcontext ctx, CUfunction func) {

        //parse_params(injInputFilename);  // injParams are updated based on injection seed file
        update_verbose();
        
        
        /* Get related functions of the kernel (device function that can be
         * called by the kernel) */
        std::vector<CUfunction> related_functions =
                nvbit_get_related_functions(ctx, func);

        /* add kernel itself to the related function vector */
        related_functions.push_back(func);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties( &devProp, 0) ;
        int archmajor = devProp.major; 
        int archminor = devProp.minor;
        int compute_cap = archmajor*10 + archminor;
        /* iterate on function */
        for (auto f : related_functions) {
                /* "recording" function was instrumented, if set insertion failed
                 * we have already encountered this function */
                if (!already_instrumented.insert(f).second) {
                        continue;
                }

                std::string kname = removeSpaces(nvbit_get_func_name(ctx,f));
                /* Get the vector of instruction composing the loaded CUFunction "func" */
                const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

                int maxregs = get_maxregs(f);
                assert(fout.good());
                //assert(fout3.good());
                int k=0;
                //fout << "Inspecting: " << kname << ";num_static_instrs: " << instrs.size() << ";maxregs: " << maxregs << "(" << maxregs << ")" << std::endl;
                for(auto i: instrs)  {
                        std::string opcode = i->getOpcode(); 
                        std::string instTypeStr = extractInstType(opcode); 
                        int instType = instTypeNameMap[instTypeStr]; 
                        if (verbose) printf("extracted instType: %s, ", instTypeStr.c_str());
                        if (verbose) printf("index of instType: %d\n", instTypeNameMap[instTypeStr]);
                        //if ((uint32_t)instType == inj_info.injInstType || inj_info.injInstType == NUM_ISA_INSTRUCTIONS) {
                        
                        //if ((uint32_t)instType == inj_info.injInstType) {
                                if (verbose) { printf("instruction selected for instrumentation: "); i->print(); }

                                // Tokenize the instruction 
                                std::vector<std::string> tokens;
                                std::string buf; // a buffer string
                                std::stringstream ss(i->getSass()); // Insert the string into a stream
                                while (ss >> buf)
                                        tokens.push_back(buf);

                                int destGPRNum = -1;
                                int numDestGPRs = 0;

                                if (tokens.size() > 1) { // an actual instruction that writes to either a GPR or PR register
                                        if (verbose) printf("num tokens = %ld \n", tokens.size());
                                        int start = 1; // first token is opcode string
                                        if (tokens[0].find('@') != std::string::npos) { // predicated instruction, ignore first token
                                                start = 2; // first token is predicate and 2nd token is opcode
                                        }

                                        // Parse the first operand - this is the first destination
                                        int regnum1 = -1;
                                        int regtype = extractRegNo(tokens[start], regnum1);
                                        if (regtype == 0) { // GPR reg
                                                destGPRNum = regnum1;
                                                numDestGPRs = (getOpGroupNum(instType) == G_FP64) ? 2 : 1;

                                                int szStr = extractSize(opcode); 
                                                if (szStr == 128) {
                                                        numDestGPRs = 4; 
                                                } else if (szStr == 64) {
                                                        numDestGPRs = 2; 
                                                }
                                                
                                                if ((uint32_t)destGPRNum ==inj_error_info.injRegID){
                                                    k++;
                                                    fout <<"Kernel name: "<<kname<<"; kernel Index: "<< kernel_id <<"; Num_Injections: " << k <<";"<< std::endl;

                                                    //printf("instType%d\n",instType);
                                                    nvbit_insert_call(i, "inject_error_IRAv2", IPOINT_AFTER);
                                                    nvbit_add_call_arg_const_val64(i, (uint64_t)&inj_error_info);
                                                    nvbit_add_call_arg_const_val64(i, (uint64_t)&Injection_masks);
                                                    nvbit_add_call_arg_const_val64(i, (uint64_t)&verbose_device);

                                                    nvbit_add_call_arg_const_val32(i, destGPRNum); // destination GPR register number
                                                    if (destGPRNum != -1) {
                                                        nvbit_add_call_arg_reg_val(i, destGPRNum); // destination GPR register val
                                                } else {
                                                        nvbit_add_call_arg_const_val32(i, (unsigned int)-1); // destination GPR register val 
                                                }
                                                nvbit_add_call_arg_const_val32(i, numDestGPRs); // number of destination GPR registers

                                                nvbit_add_call_arg_const_val32(i, compute_cap); // compute_capability
                                                
                                        }
                                        
                                }
                        }
                }
        }
}


void instrument_function_if_neededv2(CUcontext ctx, CUfunction func) {

        //parse_params(injInputFilename);  // injParams are updated based on injection seed file
        update_verbose();        
        /* Get related functions of the kernel (device function that can be
        * called by the kernel) */
        std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

        /* add kernel itself to the related function vector */
        related_functions.push_back(func);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties( &devProp, 0) ;
        int archmajor = devProp.major; 
        int archminor = devProp.minor;
        int compute_cap = archmajor*10 + archminor;
        /* iterate on function */
        for (auto f : related_functions) {                
                /* "recording" function was instrumented, if set insertion failed
                        * we have already encountered this function */
                if (!already_instrumented.insert(f).second) {
                        continue;
                }
                fout << "=================================================================================" << endl;
                fout << "The Instrumentation step Begins Here: " << removeSpaces(nvbit_get_func_name(ctx,f)) << endl;
                fout << "=================================================================================" << endl;

                std::string kname = removeSpaces(nvbit_get_func_name(ctx,f));
                /* Get the vector of instruction composing the loaded CUFunction "func" */
                const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

                int maxregs = get_maxregs(f);
                max_regcount=maxregs;
                assert(fout.good());
                //assert(fout3.good());
                int k=0;
                int instridx=0;
                //fout << "Inspecting: " << kname << ";num_static_instrs: " << instrs.size() << ";maxregs: " << maxregs << "(" << maxregs << ")" << std::endl;
                for(auto i: instrs)  {
                        std::string opcode = i->getOpcode(); 
                        std::string instTypeStr = i->getOpcodeShort();
                        int instType = instTypeNameMap[instTypeStr]; 

                        if (verbose) printf("extracted instType: %s, ", instTypeStr.c_str());
                        if (verbose) printf("index of instType: %d\n", instTypeNameMap[instTypeStr]);
                        //if ((uint32_t)instType == inj_info.injInstType || inj_info.injInstType == NUM_ISA_INSTRUCTIONS) {
                        
                        //if ((uint32_t)instType == inj_info.injInstType) {
                        if (verbose) { printf("instruction selected for instrumentation: "); i->print(); }

                        int destGPRNum = -1;
                        int numDestGPRs = 0;
                        fout << i->getSass() << std::endl;
                        if (i->getNumOperands() > 1) { // an actual instruction that writes to either a GPR or PR register
                        // Parse the first operand - this is the the destination register field
                        const InstrType::operand_t *dst= i->getOperand(0);
                        if(dst->type == InstrType::OperandType::REG ) { // GPR reg as a destination                                      
                                numDestGPRs = (getOpGroupNum(instType) == G_FP64) ? 2 : 1;
                                int szStr = i->getSize()*8; 
                                if (szStr == 128) {
                                        numDestGPRs = 4; 
                                } else if (szStr == 64) {
                                        numDestGPRs = 2; 
                                }                                        
                                if(inj_error_info.injRegID==0){ // inject when it is the destination register as target 
                                destGPRNum = dst->u.reg.num;
                                inj_error_info.injRegOriginal=destGPRNum;
                                inj_error_info.injRegReplacement = inj_error_info.injRegOriginal ^ inj_error_info.injMaskSeed;
                                k++;
                                //fout <<"Kernel name: "<<kname<<"; kernel Index: "<< kernel_id <<"; Num_Injections: " << k <<";"<< std::endl;
                                //fout << "Instr Intrumented: " << i->getSass();
                                fout << i->getSass() << " instrumented; ";
                                fout << "Target_reg_field: "<< inj_error_info.injRegID
                                        << "; Max_reg_count: "<< max_regcount
                                        << "; Original_register: "<< inj_error_info.injRegOriginal
                                        <<"; Replacement_register: "<< inj_error_info.injRegReplacement
                                        << "; Error Mask: " << inj_error_info.injMaskSeed << endl;

                                nvbit_insert_call(i, "inject_error_IRA_dst", IPOINT_AFTER);
                                nvbit_add_call_arg_const_val64(i, (uint64_t)&inj_error_info);
                                nvbit_add_call_arg_const_val64(i, (uint64_t)&Injection_masks);
                                nvbit_add_call_arg_const_val64(i, (uint64_t)&verbose_device);
                                nvbit_add_call_arg_const_val32(i, destGPRNum); // destination GPR register number
                                if (destGPRNum != -1) {
                                        nvbit_add_call_arg_reg_val(i, destGPRNum); // destination GPR register val
                                } else {
                                        nvbit_add_call_arg_const_val32(i, (unsigned int)-1); // destination GPR register val 
                                }
                                nvbit_add_call_arg_const_val32(i, numDestGPRs); // number of destination GPR registers
                                nvbit_add_call_arg_const_val32(i, compute_cap); // compute_capability
                                }else if(inj_error_info.injRegID>0)
                                {
                                int reg_src[5];
                                int cnt = 0;
                                        for (int idx = 1; idx < i->getNumOperands(); idx++) {
                                        const InstrType::operand_t *op = i->getOperand(idx);
                                        if(op->type == InstrType::OperandType::REG){
                                        reg_src[cnt]=op->u.reg.num;
                                        cnt++; 
                                        }                      
                                }
                                if(((uint32_t)cnt)>=inj_error_info.injRegID){
                                        destGPRNum = (uint32_t)reg_src[inj_error_info.injRegID-1];
                                        inj_error_info.injRegOriginal=destGPRNum;
                                        inj_error_info.injRegReplacement = inj_error_info.injRegOriginal ^ inj_error_info.injMaskSeed;
                                        k++; 
                                        //fout <<"Kernel name: "<<kname<<"; kernel Index: "<< kernel_id <<"; Num_Injections: " << k <<";"<< std::endl;
                                        //fout << "Instr Intrumented: " << i->getSass();
                                        fout << i->getSass() << " instrumented; ";
                                        fout << "Target_reg_field: "<< inj_error_info.injRegID
                                        << "; Max_reg_count: "<< max_regcount
                                        << "; Original_register: "<< inj_error_info.injRegOriginal
                                        <<"; Replacement_register: "<< inj_error_info.injRegReplacement
                                        << "; Error Mask: " << inj_error_info.injMaskSeed << endl;
                                        
                                        nvbit_insert_call(i, "inject_error_IRA_src_before", IPOINT_BEFORE);
                                        nvbit_add_call_arg_const_val64(i, (uint64_t)&inj_error_info);
                                        nvbit_add_call_arg_const_val64(i, (uint64_t)&Injection_masks);
                                        nvbit_add_call_arg_const_val64(i, (uint64_t)&verbose_device);
                                        nvbit_add_call_arg_const_val32(i, destGPRNum); // destination GPR register number
                                        if (destGPRNum != -1) {
                                        nvbit_add_call_arg_reg_val(i, destGPRNum); // destination GPR register val
                                        } else {
                                        nvbit_add_call_arg_const_val32(i, (unsigned int)-1); // destination GPR register val 
                                        }
                                        nvbit_add_call_arg_const_val32(i, numDestGPRs); // number of destination GPR registers
                                        nvbit_add_call_arg_const_val32(i, compute_cap); // compute_capability
                                        nvbit_add_call_arg_const_val32(i, instridx); // compute_capability


                                        nvbit_insert_call(i, "inject_error_IRAv2_src_after", IPOINT_AFTER);
                                        nvbit_add_call_arg_const_val64(i, (uint64_t)&inj_error_info);
                                        nvbit_add_call_arg_const_val64(i, (uint64_t)&Injection_masks);
                                        nvbit_add_call_arg_const_val64(i, (uint64_t)&verbose_device);
                                        nvbit_add_call_arg_const_val32(i, destGPRNum); // destination GPR register number
                                        if (destGPRNum != -1) {
                                        nvbit_add_call_arg_reg_val(i, destGPRNum); // destination GPR register val
                                        } else {
                                        nvbit_add_call_arg_const_val32(i, (unsigned int)-1); // destination GPR register val 
                                        }
                                        nvbit_add_call_arg_const_val32(i, numDestGPRs); // number of destination GPR registers
                                        nvbit_add_call_arg_const_val32(i, compute_cap); // compute_capability
                                        nvbit_add_call_arg_const_val32(i, instridx); // compute_capability
                                        instridx++;                                                 
                                }                                
                                }                                                                        
                        }                            
                        }
                }
                fout << "=================================================================================" << endl;
                fout << "The Instrumentation step Stops Here: " << removeSpaces(nvbit_get_func_name(ctx,f)) << endl;
                fout << "=================================================================================" << endl;
        }       
}




/* This call-back is triggered every time a CUDA event is encountered.
 * Here, we identify CUDA kernel launch events and reset the "counter" before
 * th kernel is launched, and print the counter after the kernel has completed
 * (we make sure it has completed by using cudaDeviceSynchronize()). To
 * selectively run either the original or instrumented kernel we used
 * nvbit_enable_instrumented() before launching the kernel. */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                const char *name, void *params, CUresult *pStatus) {
        /* Identify all the possible CUDA launch events */
        if (cbid == API_CUDA_cuLaunch ||
                        cbid == API_CUDA_cuLaunchKernel_ptsz ||
                        cbid == API_CUDA_cuLaunchGrid ||
                        cbid == API_CUDA_cuLaunchGridAsync || 
                        cbid == API_CUDA_cuLaunchKernel) {

                /* cast params to cuLaunch_params since if we are here we know these are
                 * the right parameters type */
                //cuLaunch_params * p = (cuLaunch_params *) params;    
                auto *p = (cuLaunch_params *) params;
                auto *p1 = (cuLaunchKernel_params *) params;
                num_threads  = p1->gridDimX * p1->gridDimY * p1->gridDimZ * p1->blockDimX * p1->blockDimY * p1->blockDimZ;

                if(!is_exit) {
                    if(read_file==false){
                        parse_params(injInputFilename);
                        if (verbose) printf("read file list done..\n");               
                        read_file=true;
                    }               

                        pthread_mutex_lock(&mutex);
                        if (kernel_id < limit) {
                            instrument_function_if_neededv2(ctx, p->f);
                            cudaDeviceSynchronize();
                            fout << "=================================================================================" << endl;
                            fout << "Running instrumented Kernel: " << removeSpaces(nvbit_get_func_name(ctx,p->f)) << "; kernel Index: "<< kernel_id << endl;
                            fout << "..............." << endl;
                            fout << "=================================================================================" << endl;
                            nvbit_enable_instrumented(ctx, p->f, true); // run the instrumented version

                            cudaDeviceSynchronize();
                        } else {
                            nvbit_enable_instrumented(ctx, p->f, false); // do not use the instrumented version
                        }

                }  else {
                        if (kernel_id < limit) {
                                if (verbose) printf("is_exit\n"); 
                                cudaDeviceSynchronize();

                                cudaError_t le = cudaGetLastError();

                                kname = removeSpaces(nvbit_get_func_name(ctx,p->f));
                                //int num_ctas = 0;
                                //int num_threads = 0;//added
                                if ( cbid == API_CUDA_cuLaunchKernel_ptsz ||
                                                cbid == API_CUDA_cuLaunchKernel) {
                                        //cuLaunchKernel_params * p2 = (cuLaunchKernel_params*) params;
                                        //num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
                                        //num_threads = num_ctas * p2->blockDimX * p2->blockDimY * p2->blockDimZ; //added 
                                        //printf ("threads %d\n",num_threads);//added
                                        /*for(int instr=0;instr<6;++instr){
                                            for(int kk=0;kk<WARP_PER_SM*THREAD_PER_WARP;++kk){
                                                if(Injection_masks.register_tmp_recovery[WARP_PER_SM*THREAD_PER_WARP*instr+kk]!=0){
                                                    fout << "instr: " << instr << "thread: " << kk <<" values " << Injection_masks.register_tmp_recovery[WARP_PER_SM*THREAD_PER_WARP*instr+kk] << endl; 
                                                }
                                            }
                                        }*/
                                }
                                assert(fout.good());                            
                                fout << "=================================================================================" << endl;
                                fout << "Report for: " << kname << "; kernel Index: "<< kernel_id <<  endl;
                                fout << "=================================================================================" << endl;
                                if ( cudaSuccess != le ) {
                                        assert(fout.good());                                        
                                        fout << "ERROR FAIL in kernel execution (" << cudaGetErrorString(le) << "); " <<std::endl;
                                        fout <<"Kernel name: "<<kname<<"; kernel Index: "<< kernel_id <<"; Num_Injections: " << inj_error_info.injNumActivations;
                                        fout << "; Target_SM: " << inj_error_info.injSMID
                                        << "; Target_Scheduler: " << inj_error_info.injScheduler
                                        << "; Target_reg_field: "<< inj_error_info.injRegID
                                        << "; Max_reg_count: "<< max_regcount
                                        << "; Original_register: "<< inj_error_info.injRegOriginal
                                        <<"; Replacement_register: "<< inj_error_info.injRegReplacement
                                        << "; Error Mask: " << inj_error_info.injMaskSeed << endl;
                                        exit(1); // let's exit early 
                                }
                                //fout << "inspecting: "<< kname <<"; thread : "<<  inj_info.injThreadID <<"; Register : "<< inj_info.injReg<<";  Mask : "<<inj_info.injMask<<"; SMID : "<<inj_info.injSMID<< "; Stuck at : "<<inj_info.injStuckat  << "; index: " << kernel_id << ";" <<std::endl;
                                fout <<"Kernel name: "<<kname<<"; kernel Index: "<< kernel_id <<"; Num_Injections: " << inj_error_info.injNumActivations;
                                fout << "; Target_SM: " << inj_error_info.injSMID
                                << "; Target_Scheduler: " << inj_error_info.injScheduler
                                << "; Target_reg_field: "<< inj_error_info.injRegID
                                << "; Max_reg_count: "<< max_regcount
                                << "; Original_register: "<< inj_error_info.injRegOriginal
                                <<"; Replacement_register: "<< inj_error_info.injRegReplacement
                                << "; Error Mask: " << inj_error_info.injMaskSeed << endl;

                                inj_info.injNumActivations+= inj_error_info.injNumActivations;
                                inj_error_info.injNumActivations=0;

                                if (verbose) printf("\n index: %d; kernel_name: %s; \n", kernel_id, kname.c_str());
                                kernel_id++; // always increment kernel_id on kernel exit

                                //cudaDeviceSynchronize();
                                pthread_mutex_unlock(&mutex);                            
                            
                        }
                }
        }
}

void nvbit_at_term() { 
    if (verbose) printf("nvbit_at_term:start\n");
    assert(fout.good());
    fout << "=================================================================================" << endl;
    fout << "Final Report" <<  endl;
    fout << "=================================================================================" << endl;
    fout << "Report_Summary: "
        << "; Target_SM: " << inj_error_info.injSMID
        << "; Target_Scheduler: " << inj_error_info.injScheduler
        << "; Target_reg_field: " << inj_error_info.injRegID
        << "; Max_reg_count: " << max_regcount
        << "; Original_register: " << inj_error_info.injRegOriginal
        <<"; Replacement_register: " << inj_error_info.injRegReplacement
        << "; Error Mask: " << inj_error_info.injMaskSeed
        <<"; Total_Num_Error_Activations: " << inj_info.injNumActivations;
        if (max_regcount < inj_error_info.injRegReplacement){
                fout << "; error register outside the limits..";
        }else{
                fout << "; error register inside the limits..";
        }

    if (verbose) printf("nvbit_at_term:end\n");
} 
