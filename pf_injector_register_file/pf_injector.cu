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
#include <cstdio>
#include <cstdlib>

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
//__managed__ inj_info_t inj_info; 
__managed__ inj_info_error_t inj_error_info;

__managed__ muliple_ptr_t Injection_masks;

bool read_file=false;

std::string inj_mode;
std::string kname;
std::string SimEndRes;
int num_threads=0;
//inj_info_t inj_info;

void reset_inj_info() {
        inj_error_info.injSMID=0; // 0 - max SMs
        inj_error_info.injScheduler=0; // 0 - 3
        inj_error_info.injWarpMaskH=0;
        inj_error_info.injWarpMaskL=0; //
        inj_error_info.injThreadMask=0; //0-32
        inj_error_info.injMaskSeed=0;
        inj_error_info.injRegID=0; // injection mask
        inj_error_info.injDimension=0;
        inj_error_info.injStuck_at=0;
        inj_error_info.injInstType=0; // instruction type 
        inj_error_info.injRegOriginal=0;
        inj_error_info.injRegReplacement=0;
        inj_error_info.injNumActivations=0;
        inj_error_info.injNumActivAcc=0;
        inj_error_info.injInstrIdx=0;
        inj_error_info.injInstPC=0;
        inj_error_info.injInstOpcode=NOP;
        inj_error_info.blockDimX=0;
        inj_error_info.blockDimY=0;
        inj_error_info.blockDimZ=0;
        inj_error_info.gridDimX=0;
        inj_error_info.gridDimY=0;
        inj_error_info.gridDimZ=0;
        inj_error_info.maxregcount=0;
        inj_error_info.num_threads=0;    
        inj_error_info.TotKerInstr=0;
        inj_error_info.TotAppInstr=0;
        inj_error_info.maxPredReg=-1;    
        inj_error_info.errorInjected = false;
}

void print_inj_info() {
        assert(fout.good());
        //std::cout << "InstType=" << inj_info.injInstType << ", SMID=" << inj_info.injSMID<< ", LaneID=" << inj_info.injLaneID;
        
        //std::cout << ", Mask=" << inj_info.injMask << std::endl;
}

// Parse error injection site info from a file. This should be done on host side.
void parse_paramsIRA(std::string filename) {
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
                        inj_error_info.injInstrIdx=-1;
                        inj_error_info.injInstOpcode=NOP;
                        inj_error_info.injInstPC=-1;
                        inj_error_info.num_threads=num_threads;

                } else {
                        printf(" File %s does not exist!", filename.c_str());
                        printf(" This file should contain enough information about the fault site to perform a permanent error injection run: ");
                        printf("Documentation to be deifined...\n"); 
                        assert(false);
                }
                ifs.close();

                if (verbose) {
                        print_inj_info();
                }

                CUDA_SAFECALL(cudaMallocManaged(&(Injection_masks.Warp_thread_active),(inj_error_info.MaxWarpsPerSM*inj_error_info.MaxThreadsPerWarp)*sizeof(uint32_t)));
                CUDA_SAFECALL(cudaMallocManaged(&(Injection_masks.warp_thread_mask),(inj_error_info.MaxWarpsPerSM*inj_error_info.MaxThreadsPerWarp)*sizeof(uint32_t)));
                //CUDA_SAFECALL(cudaMallocManaged(&(Injection_masks.register_tmp_recovery),(inj_error_info.MaxWarpsPerSM*inj_error_info.MaxThreadsPerWarp*MAX_KNAME_SIZE)*sizeof(uint32_t)));
                //CUDA_SAFECALL(cudaMallocManaged(&(Injection_masks.register_tmp_recovery),(inj_error_info.num_threads*MAX_KNAME_SIZE)*sizeof(uint32_t)));
                CUDA_SAFECALL(cudaMallocManaged(&(Injection_masks.register_tmp_recovery),(inj_error_info.MaxWarpsPerSM*inj_error_info.MaxThreadsPerWarp)*sizeof(uint32_t)));

                Injection_masks.num_threads=inj_error_info.num_threads;

                srand(inj_error_info.injMaskSeed);
                for(int i=0;i<(inj_error_info.MaxWarpsPerSM*inj_error_info.MaxThreadsPerWarp);++i){
                    random=rand();
                    Injection_masks.warp_thread_mask[i]=*(int *)&random;
                }
                int idx=0;
                int validW=0;
                int validT=0;
                int integer_mask=0;
                for(int i=0;i<inj_error_info.MaxWarpsPerSM;++i){
                    if (i>31){
                        validW=(inj_error_info.injWarpMaskH>>i)&1;
                    }else{
                        validW=(inj_error_info.injWarpMaskL>>i)&1;
                    }
                    for(int j=0;j<inj_error_info.MaxThreadsPerWarp;++j){   
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



// Parse error injection site info from a file. This should be done on host side.
void parse_paramsIAT(std::string filename) {
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
                        ifs >> inj_error_info.injMaskSeed;  // 0: inactive thread 1: active thread   
                        ifs >> inj_error_info.injDimension;
                        ifs >> inj_error_info.injStuck_at;                     

                        assert(inj_error_info.injSMID < 1000); 
                        inj_error_info.injNumActivations=0;
                        inj_error_info.injInstrIdx=-1;
                        inj_error_info.injInstOpcode=NOP;
                        inj_error_info.injInstPC=-1;

                } else {
                        printf(" File %s does not exist!", filename.c_str());
                        printf(" This file should contain enough information about the fault site to perform a permanent error injection run: ");
                        printf("Documentation to be deifined...\n"); 
                        assert(false);
                }
                ifs.close();

                if (verbose) {
                        print_inj_info();
                }

                CUDA_SAFECALL(cudaMallocManaged(&(Injection_masks.Warp_thread_active),(inj_error_info.MaxWarpsPerSM*inj_error_info.MaxThreadsPerWarp)*sizeof(uint32_t)));
                CUDA_SAFECALL(cudaMallocManaged(&(Injection_masks.warp_thread_mask),(inj_error_info.MaxWarpsPerSM*inj_error_info.MaxThreadsPerWarp)*sizeof(uint32_t)));

                int idx=0;
                int validW=0;
                int validT=0;
                int integer_mask=0;
                for(int i=0;i<inj_error_info.MaxWarpsPerSM;++i){
                    if (i>31){
                        validW=(inj_error_info.injWarpMaskH>>i)&1;
                    }else{
                        validW=(inj_error_info.injWarpMaskL>>i)&1;
                    }
                    for(int j=0;j<inj_error_info.MaxThreadsPerWarp;++j){   
                        validT= (inj_error_info.injThreadMask>>j)&1;
                        //printf("valid %d, %d\n",idx,validW&validT) ;                   
                        Injection_masks.Warp_thread_active[idx]=(validW&validT);
                        Injection_masks.warp_thread_mask[idx]=idx;
                        idx++;
                    }
                }        
        }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Parse error injection site info from a file. This should be done on host side.
void parse_paramsIAC(std::string filename) {
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
                        ifs >> inj_error_info.injMaskSeed;  // 0: inactive thread 1: active thread            
                        ifs >> inj_error_info.injDimension;            
                        ifs >> inj_error_info.injStuck_at;


                        assert(inj_error_info.injSMID < 1000); 
                        inj_error_info.injNumActivations=0;
                        inj_error_info.injInstrIdx=-1;
                        inj_error_info.injInstOpcode=NOP;
                        inj_error_info.injInstPC=-1;

                } else {
                        printf(" File %s does not exist!", filename.c_str());
                        printf(" This file should contain enough information about the fault site to perform a permanent error injection run: ");
                        printf("Documentation to be deifined...\n"); 
                        assert(false);
                }
                ifs.close();

                if (verbose) {
                        print_inj_info();
                }

                CUDA_SAFECALL(cudaMallocManaged(&(Injection_masks.Warp_thread_active),(inj_error_info.MaxWarpsPerSM*inj_error_info.MaxThreadsPerWarp)*sizeof(uint32_t)));

                int idx=0;
                int validW=0;
                int validT=0;
                int integer_mask=0;
                for(int i=0;i<inj_error_info.MaxWarpsPerSM;++i){
                    if (i>31){
                        validW=(inj_error_info.injWarpMaskH>>i)&1;
                    }else{
                        validW=(inj_error_info.injWarpMaskL>>i)&1;
                    }
                    for(int j=0;j<inj_error_info.MaxThreadsPerWarp;++j){   
                        validT= (inj_error_info.injThreadMask>>j)&1;
                        //printf("valid %d, %d\n",idx,validW&validT) ;                   
                        Injection_masks.Warp_thread_active[idx]=(validW&validT);
                        idx++;
                    }
                }        
        }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////
// Parse error injection site info from a file. This should be done on host side.
void parse_paramsWV(std::string filename) {
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
                        ifs >> inj_error_info.injRegID;
                        ifs >> inj_error_info.injMaskSeed;  // 0: inactive thread 1: active thread                                    
                        ifs >> inj_error_info.injStuck_at;


                        assert(inj_error_info.injSMID < 1000); 
                        inj_error_info.injNumActivations=0;
                        inj_error_info.injInstrIdx=-1;
                        inj_error_info.injInstOpcode=NOP;
                        inj_error_info.injInstPC=-1;

                } else {
                        printf(" File %s does not exist!", filename.c_str());
                        printf(" This file should contain enough information about the fault site to perform a permanent error injection run: ");
                        printf("Documentation to be deifined...\n"); 
                        assert(false);
                }
                ifs.close();

                if (verbose) {
                        print_inj_info();
                }

                CUDA_SAFECALL(cudaMallocManaged(&(Injection_masks.Warp_thread_active),(inj_error_info.MaxWarpsPerSM*inj_error_info.MaxThreadsPerWarp)*sizeof(uint32_t)));

                int idx=0;
                int validW=0;
                int validT=0;
                int integer_mask=0;
                for(int i=0;i<inj_error_info.MaxWarpsPerSM;++i){
                    if (i>31){
                        validW=(inj_error_info.injWarpMaskH>>i)&1;
                    }else{
                        validW=(inj_error_info.injWarpMaskL>>i)&1;
                    }
                    for(int j=0;j<inj_error_info.MaxThreadsPerWarp;++j){   
                        validT= (inj_error_info.injThreadMask>>j)&1;
                        //printf("valid %d, %d\n",idx,validW&validT) ;                   
                        Injection_masks.Warp_thread_active[idx]=(validW&validT);
                        idx++;
                    }
                }        
                inj_error_info.maxPredReg=-1;
        }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////

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


void report_kernel_results(void){
fout <<"Kernel name: "<<kname<<"; kernel Index: "<< kernel_id
        << "; DeviceName: " << inj_error_info.DeviceName
        << "; MaxThreadsPerSM: " << inj_error_info.MaxThreadsPerSM
        << "; MaxWarpsPerSm: " << inj_error_info.MaxWarpsPerSM
        << "; MaxThreadsPerWarp: " << inj_error_info.MaxThreadsPerWarp
        << "; gridDimX: " << inj_error_info.gridDimX
        << "; gridDimY: " << inj_error_info.gridDimY
        << "; gridDimZ: " << inj_error_info.gridDimZ
        << "; blockDimX: " << inj_error_info.blockDimX
        << "; blockDimY: " << inj_error_info.blockDimY
        << "; blockDimZ: " << inj_error_info.blockDimZ
        << "; NumThreads: " << inj_error_info.num_threads;
        if(inj_error_info.errorInjected==true) 
                fout << "; ErrorInjected: True";
        else
                fout << "; ErrorInjected: False"; 
        fout << "; injSmID: " << inj_error_info.injSMID
        << "; injSchID: " << inj_error_info.injScheduler
        << "; injWarpIDH: " << inj_error_info.injWarpMaskH
        << "; injWarpIDL: " << inj_error_info.injWarpMaskL
        << "; injLaneID: " << inj_error_info.injThreadMask;
        if(inj_mode.compare("IRA")==0 or inj_mode.compare("IR")==0){
                fout << "; injRegField: " << inj_error_info.injRegID
                << "; injMaskSeed: " << inj_error_info.injMaskSeed
                << "; resMaxRegCount: " << inj_error_info.maxregcount
                << "; resRegOrigNum: " << inj_error_info.injRegOriginal
                << "; resRegRepNum: " << inj_error_info.injRegReplacement
                << "; resNumInstr: " << inj_error_info.TotKerInstr;
        }else if(inj_mode.compare("IAT")==0){               
                fout << "; injMaskSeed: " << inj_error_info.injMaskSeed
                << "; InjDimention: " << inj_error_info.injDimension
                << "; injStuck-at: " << inj_error_info.injStuck_at
                << "; resNumInstr: " << inj_error_info.TotKerInstr;
        }else if(inj_mode.compare("IAW")==0){
                fout << "; injMaskSeed: " << inj_error_info.injMaskSeed
                << "; InjDimention: " << inj_error_info.injDimension
                << "; injStuck-at: " << inj_error_info.injStuck_at
                << "; resNumInstr: " << inj_error_info.TotKerInstr;
        }else if(inj_mode.compare("IAC")==0){
                fout << "; injMaskSeed: " << inj_error_info.injMaskSeed
                << "; InjDimention: " << inj_error_info.injDimension
                << "; injStuck-at: " << inj_error_info.injStuck_at
                << "; resNumInstr: " << inj_error_info.TotKerInstr;
        }else if(inj_mode.compare("WV")==0){
                fout << "; injPredReg: " << inj_error_info.injRegID
                << "; injMaskSeed: " << inj_error_info.injMaskSeed
                << "; injStuck-at: " << inj_error_info.injStuck_at
                << "; resMaxPredReg: " << inj_error_info.KernelPredReg
                << "; resNumInstr: " << inj_error_info.TotKerInstr;
        }else {
        }
        fout << "; NumErrInstExeBefStop: " << inj_error_info.injInstrIdx
        << "; LastPCOffset: 0x" << std::hex << inj_error_info.injInstPC  << std::dec
        << "; LastOpcode: " << instTypeNames[inj_error_info.injInstOpcode]
        << "; TotErrAct: " << inj_error_info.injNumActivations << endl; 

}

void report_summary_results(void){
fout << "=================================================================================" << endl;
fout << "Final Report" <<  endl;
fout << "=================================================================================" << endl;
fout << "Report_Summary: " 
        << "; DeviceName: " << inj_error_info.DeviceName
        << "; MaxThreadsPerSM: " << inj_error_info.MaxThreadsPerSM
        << "; MaxWarpsPerSm: " << inj_error_info.MaxWarpsPerSM
        << "; MaxThreadsPerWarp: " << inj_error_info.MaxThreadsPerWarp
        << "; gridDimX: " << inj_error_info.gridDimX
        << "; gridDimY: " << inj_error_info.gridDimY
        << "; gridDimZ: " << inj_error_info.gridDimZ
        << "; blockDimX: " << inj_error_info.blockDimX
        << "; blockDimY: " << inj_error_info.blockDimY
        << "; blockDimZ: " << inj_error_info.blockDimZ;
        if(inj_error_info.errorInjected==true) 
        fout << "; ErrorInjected: True";
        else
        fout << "; ErrorInjected: False"; 

        fout << "; injSmID: " << inj_error_info.injSMID
        << "; injSchID: " << inj_error_info.injScheduler
        << "; injWarpIDH: " << inj_error_info.injWarpMaskH
        << "; injWarpIDL: " << inj_error_info.injWarpMaskL
        << "; injLaneID: " << inj_error_info.injThreadMask;
        if(inj_mode.compare("IRA")==0 or inj_mode.compare("IR")==0){
                fout << "; injRegField: " << inj_error_info.injRegID
                << "; injMaskSeed: " << inj_error_info.injMaskSeed
                << "; resMaxRegCount: " << inj_error_info.maxregcount
                << "; resRegOrigNum: " << inj_error_info.injRegOriginal
                << "; resRegRepNum: " << inj_error_info.injRegReplacement
                << "; resNumInstr: " << inj_error_info.TotAppInstr;
                if (inj_error_info.maxregcount > inj_error_info.injRegReplacement){
                        fout << "; resRegLoc: InsideLims";
                }else{
                        fout << "; resRegLoc: OutsideLims";
                }
        }else if(inj_mode.compare("IAT")==0){               
                fout << "; injMaskSeed: " << inj_error_info.injMaskSeed
                << "; InjDimention: " << inj_error_info.injDimension
                << "; injStuck-at: " << inj_error_info.injStuck_at
                << "; resNumInstr: " << inj_error_info.TotAppInstr;
        }else if(inj_mode.compare("IAW")==0){
                fout << "; injMaskSeed: " << inj_error_info.injMaskSeed
                << "; InjDimention: " << inj_error_info.injDimension
                << "; injStuck-at: " << inj_error_info.injStuck_at
                << "; resNumInstr: " << inj_error_info.TotAppInstr;
        }else if(inj_mode.compare("IAC")==0){
                fout << "; injMaskSeed: " << inj_error_info.injMaskSeed
                << "; InjDimention: " << inj_error_info.injDimension
                << "; injStuck-at: " << inj_error_info.injStuck_at
                << "; resNumInstr: " << inj_error_info.TotAppInstr;
        }else if(inj_mode.compare("WV")==0){
                fout << "; injPredReg: " << inj_error_info.injRegID
                << "; injMaskSeed: " << inj_error_info.injMaskSeed
                << "; injStuck-at: " << inj_error_info.injStuck_at
                << "; resMaxPredReg: " << inj_error_info.maxPredReg
                << "; resNumInstr: " << inj_error_info.TotAppInstr;
        }else {
        }
        fout << "; NumErrInstExeBefStop: " << inj_error_info.injInstrIdx
        << "; LastPCOffset: 0x" << std::hex << inj_error_info.injInstPC  << std::dec
        << "; LastOpcode: " << instTypeNames[inj_error_info.injInstOpcode]
        << "; TotErrAct: " << inj_error_info.injNumActivAcc+inj_error_info.injNumActivations;
        fout << SimEndRes << endl; 
}


void INThandler(int sig) {
        signal(sig, SIG_IGN); // disable Ctrl-C
        fout << "=================================================================================" << endl;
        fout << "Report for: " << kname << "; kernel Index: "<< kernel_id <<  endl;
        fout << "=================================================================================" << endl;
        fout << ":::NVBit-inject-error; ERROR FAIL Detected Singal SIGKILL;" << endl;
        report_kernel_results();
        SimEndRes="; SimEndRes:::ERROR FAIL Detected Singal SIGKILL::: ";
        report_summary_results();
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

        if(getenv("nvbitPERfi")){
                inj_mode=getenv("nvbitPERfi");
        }else{
                inj_mode="IRA";
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
                                                    fout <<"Kernel name: "<<kname<<"; kernel Index: "<< kernel_id <<"; Num_Activations: " << k <<";"<< std::endl;

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


void instrument_function_IRA(CUcontext ctx, CUfunction func) {

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
                inj_error_info.maxregcount=maxregs;
                assert(fout.good());
                //assert(fout3.good());
                int k=0;
                int instridx=0;
                inj_error_info.TotKerInstr=0;
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
                        int replGPRNum = -1;
                        int numDestGPRs = 0;
                        fout << "0x" << std::hex << i->getOffset() << ":::" << i->getSass()  << std::dec << std::endl;
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
                                        if(inj_error_info.injRegID==0 ){ // and instType!=MOV inject when it is the destination register as target 
                                                destGPRNum = dst->u.reg.num;
                                                inj_error_info.injRegOriginal=destGPRNum;
                                                inj_error_info.injRegReplacement = inj_error_info.injRegOriginal ^ inj_error_info.injMaskSeed;
                                                replGPRNum = inj_error_info.injRegReplacement;
                                                k++;
                                                instridx++; 
                                                inj_error_info.TotKerInstr++;
                                                //fout <<"Kernel name: "<<kname<<"; kernel Index: "<< kernel_id <<"; Num_Activations: " << k <<";"<< std::endl;
                                                //fout << "Instr Intrumented: " << i->getSass();

                                                fout << "0x" << std::hex << i->getOffset() << "; " << i->getSass() << std::dec << " instrumented intruction; ";
                                                fout << "Target_reg_field: "<< inj_error_info.injRegID
                                                        << "; Max_reg_count: "<< inj_error_info.maxregcount
                                                        << "; Original_register: "<< inj_error_info.injRegOriginal
                                                        <<"; Replacement_register: "<< inj_error_info.injRegReplacement
                                                        << "; Error Mask: " << inj_error_info.injMaskSeed
                                                        << "; NumThreads: " << inj_error_info.num_threads << endl;

                                                nvbit_insert_call(i, "inject_error_IRA_dst", IPOINT_AFTER);
                                                nvbit_add_call_arg_const_val64(i, (uint64_t)&inj_error_info);
                                                nvbit_add_call_arg_const_val64(i, (uint64_t)&Injection_masks);
                                                nvbit_add_call_arg_const_val64(i, (uint64_t)&verbose_device);
                                                nvbit_add_call_arg_const_val32(i, destGPRNum); // destination GPR register number
                                                nvbit_add_call_arg_const_val32(i, replGPRNum); // destination GPR register number
                                                if (destGPRNum != -1) {
                                                        nvbit_add_call_arg_reg_val(i, destGPRNum); // destination GPR register val
                                                } else {
                                                        nvbit_add_call_arg_const_val32(i, (unsigned int)-1); // destination GPR register val 
                                                }
                                                nvbit_add_call_arg_const_val32(i, numDestGPRs); // number of destination GPR registers
                                                nvbit_add_call_arg_const_val32(i, compute_cap); // compute_capability
                                                nvbit_add_call_arg_const_val32(i, instridx);
                                                nvbit_add_call_arg_const_val32(i, i->getOffset());
                                                nvbit_add_call_arg_const_val32(i, instType);
                                                
                                        }else if(inj_error_info.injRegID>0){
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
                                                        replGPRNum = inj_error_info.injRegReplacement;
                                                        k++; 
                                                        instridx++;  
                                                        inj_error_info.TotKerInstr++;
                                                        //fout <<"Kernel name: "<<kname<<"; kernel Index: "<< kernel_id <<"; Num_Activations: " << k <<";"<< std::endl;
                                                        //fout << "Instr Intrumented: " << i->getSass();
                                                        fout << "0x" << std::hex << i->getOffset() << "; " << i->getSass() << std::dec << " instrumented intruction; ";
                                                        fout << "Target_reg_field: "<< inj_error_info.injRegID
                                                        << "; Max_reg_count: "<< inj_error_info.maxregcount
                                                        << "; Original_register: "<< inj_error_info.injRegOriginal
                                                        <<"; Replacement_register: "<< inj_error_info.injRegReplacement
                                                        << "; Error Mask: " << inj_error_info.injMaskSeed
                                                        << "; NumThreads: " << inj_error_info.num_threads << endl;
                                                        
                                                        nvbit_insert_call(i, "inject_error_IRA_src_before", IPOINT_BEFORE);
                                                        nvbit_add_call_arg_const_val64(i, (uint64_t)&inj_error_info);
                                                        nvbit_add_call_arg_const_val64(i, (uint64_t)&Injection_masks);
                                                        nvbit_add_call_arg_const_val64(i, (uint64_t)&verbose_device);
                                                        nvbit_add_call_arg_const_val32(i, destGPRNum); // destination GPR register number
                                                        nvbit_add_call_arg_const_val32(i, replGPRNum);
                                                        if (destGPRNum != -1) {
                                                        nvbit_add_call_arg_reg_val(i, destGPRNum); // destination GPR register val
                                                        } else {
                                                        nvbit_add_call_arg_const_val32(i, (unsigned int)-1); // destination GPR register val 
                                                        }
                                                        nvbit_add_call_arg_const_val32(i, numDestGPRs); // number of destination GPR registers
                                                        nvbit_add_call_arg_const_val32(i, compute_cap); // compute_capability
                                                        nvbit_add_call_arg_const_val32(i, instridx); // compute_capability
                                                        nvbit_add_call_arg_const_val32(i, i->getOffset());
                                                        nvbit_add_call_arg_const_val32(i, instType);


                                                        nvbit_insert_call(i, "inject_error_IRA_src_after", IPOINT_AFTER);
                                                        nvbit_add_call_arg_const_val64(i, (uint64_t)&inj_error_info);
                                                        nvbit_add_call_arg_const_val64(i, (uint64_t)&Injection_masks);
                                                        nvbit_add_call_arg_const_val64(i, (uint64_t)&verbose_device);
                                                        nvbit_add_call_arg_const_val32(i, destGPRNum); // destination GPR register number
                                                        nvbit_add_call_arg_const_val32(i, replGPRNum);
                                                        if (destGPRNum != -1) {
                                                        nvbit_add_call_arg_reg_val(i, destGPRNum); // destination GPR register val
                                                        } else {
                                                        nvbit_add_call_arg_const_val32(i, (unsigned int)-1); // destination GPR register val 
                                                        }
                                                        nvbit_add_call_arg_const_val32(i, numDestGPRs); // number of destination GPR registers
                                                        nvbit_add_call_arg_const_val32(i, compute_cap); // compute_capability
                                                        nvbit_add_call_arg_const_val32(i, instridx); // compute_capability
                                                        nvbit_add_call_arg_const_val32(i, i->getOffset());
                                                        nvbit_add_call_arg_const_val32(i, instType);
                                                                                                
                                                }                                
                                        }                                                                        
                                }                            
                        }
                }
                inj_error_info.TotAppInstr+=inj_error_info.TotKerInstr;
                fout << "=================================================================================" << endl;
                fout << "The Instrumentation step Stops Here: " << removeSpaces(nvbit_get_func_name(ctx,f)) << endl;
                fout << "=================================================================================" << endl;
        }       
}


/* Instrumentation for IAT and IAW error models */
void instrument_function_IAT(CUcontext ctx, CUfunction func) {

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
                inj_error_info.maxregcount=maxregs;
                assert(fout.good());
                //assert(fout3.good());
                int k=0;
                int instridx=0;
                int blockDimm=0;
                bool injectInstrunc=false;
                inj_error_info.TotKerInstr=0;
                //fout << "Inspecting: " << kname << ";num_static_instrs: " << instrs.size() << ";maxregs: " << maxregs << "(" << maxregs << ")" << std::endl;
                for(auto i: instrs)  {
                        std::string opcode = i->getOpcode(); 
                        std::string instTypeStr = i->getOpcodeShort();
                        std::string GenOperand;
                        int instType = instTypeNameMap[instTypeStr]; 

                        if (verbose) printf("extracted instType: %s, ", instTypeStr.c_str());
                        if (verbose) printf("index of instType: %d\n", instTypeNameMap[instTypeStr]);
                        //if ((uint32_t)instType == inj_info.injInstType || inj_info.injInstType == NUM_ISA_INSTRUCTIONS) {
                        
                        //if ((uint32_t)instType == inj_info.injInstType) {
                        if (verbose) { printf("instruction selected for instrumentation: "); i->print(); }

                        int destGPRNum = -1;
                        int replGPRNum = -1;
                        int numDestGPRs = 0;
                        injectInstrunc=false;
                        fout << "0x" << std::hex << i->getOffset() << ":::" << i->getSass()  << std::dec << std::endl;
                        if (i->getNumOperands() > 1) { // an actual instruction that writes to either a GPR or PR register
                                // Parse the first operand - this is the the destination register field                                
                                for (int idx=0;idx<i->getNumOperands();++idx){
                                        const InstrType::operand_t *dst= i->getOperand(idx);                                        
                                        if(dst->type == InstrType::OperandType::GENERIC ) { // GPR reg as a destination                                                                                      
                                                GenOperand=dst->str;
                                                size_t found = GenOperand.rfind("TID.X");
                                                if (found != string::npos){
                                                        //blockDimm=inj_error_info.blockDimX-1;
                                                        blockDimm=0;
                                                        injectInstrunc=true;
                                                        //printf("Found: %d; \n",found);
                                                }                                                                                        
                                                found = GenOperand.rfind("TID.Y");
                                                if (found != string::npos){
                                                        //blockDimm=inj_error_info.blockDimY-1;
                                                        blockDimm=1;
                                                        injectInstrunc=true;         
                                                        //printf("Found: %d; \n",found); 
                                                }
                                                                                                
                                                found = GenOperand.rfind("TID.Z");
                                                if (found != string::npos){
                                                        //blockDimm=inj_error_info.blockDimZ-1;
                                                        blockDimm=2;
                                                        injectInstrunc=true;
                                                        //printf("Found: %d; \n",found);
                                                }  
                                                /*                                                                                   
                                                found = GenOperand.rfind("CTAID.X");
                                                if (found != string::npos){
                                                        blockDimm=inj_error_info.gridDimX-1;
                                                        injectInstrunc=true;         
                                                        //printf("Found: %d; \n",found); 
                                                }                                                
                                                found = GenOperand.rfind("CTAID.Y");
                                                if (found != string::npos){
                                                        blockDimm=inj_error_info.gridDimY-1;
                                                        injectInstrunc=true;         
                                                        //printf("Found: %d; \n",found); 
                                                }                                                
                                                found = GenOperand.rfind("CTAID.Z");
                                                if (found != string::npos){
                                                        blockDimm=inj_error_info.blockDimZ-1;
                                                        injectInstrunc=true;
                                                        //printf("Found: %d; \n",found);
                                                }*/
                                                                                                                                                                
                                        }        
                                }                                                                                                 
                                if(injectInstrunc==true && inj_error_info.injDimension==blockDimm){
                                        printf("string: %s; blockDimm: %d\n",GenOperand.c_str(),blockDimm); 
                                        fout << "0x" << std::hex << i->getOffset() << "; " << i->getSass() << std::dec << " instrumented intruction; " << endl;
                                        const InstrType::operand_t *dst= i->getOperand(0);
                                        destGPRNum=dst->u.reg.num;
                                        numDestGPRs=1;
                                        instridx++;  
                                        inj_error_info.TotKerInstr++;
                                        nvbit_insert_call(i, "inject_error_IAT", IPOINT_AFTER);
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
                                        nvbit_add_call_arg_const_val32(i, blockDimm); // compute_capability
                                        nvbit_add_call_arg_const_val32(i, instridx); // compute_capability
                                        nvbit_add_call_arg_const_val32(i, i->getOffset());
                                        nvbit_add_call_arg_const_val32(i, instType);
                                        
                                }

                        }

                }
                inj_error_info.TotAppInstr+=inj_error_info.TotKerInstr;
                fout << "=================================================================================" << endl;
                fout << "The Instrumentation step Stops Here: " << removeSpaces(nvbit_get_func_name(ctx,f)) << endl;
                fout << "=================================================================================" << endl;
        }       
}

/* Instrumentation for IAC*/
void instrument_function_IAC(CUcontext ctx, CUfunction func) {

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
                inj_error_info.maxregcount=maxregs;
                assert(fout.good());
                //assert(fout3.good());
                int k=0;
                int instridx=0;
                int gridDimm=0;
                bool injectInstrunc=false;
                inj_error_info.TotKerInstr=0;
                //fout << "Inspecting: " << kname << ";num_static_instrs: " << instrs.size() << ";maxregs: " << maxregs << "(" << maxregs << ")" << std::endl;
                for(auto i: instrs)  {
                        std::string opcode = i->getOpcode(); 
                        std::string instTypeStr = i->getOpcodeShort();
                        std::string GenOperand;
                        int instType = instTypeNameMap[instTypeStr]; 

                        if (verbose) printf("extracted instType: %s, ", instTypeStr.c_str());
                        if (verbose) printf("index of instType: %d\n", instTypeNameMap[instTypeStr]);
                        //if ((uint32_t)instType == inj_info.injInstType || inj_info.injInstType == NUM_ISA_INSTRUCTIONS) {
                        
                        //if ((uint32_t)instType == inj_info.injInstType) {
                        if (verbose) { printf("instruction selected for instrumentation: "); i->print(); }

                        int destGPRNum = -1;
                        int replGPRNum = -1;
                        int numDestGPRs = 0;
                        injectInstrunc=false;
                        fout << "0x" << std::hex << i->getOffset() << ":::" << i->getSass()  << std::dec << std::endl;
                        if (i->getNumOperands() > 1) { // an actual instruction that writes to either a GPR or PR register
                                // Parse the first operand - this is the the destination register field                                
                                for (int idx=0;idx<i->getNumOperands();++idx){
                                        const InstrType::operand_t *dst= i->getOperand(idx);                                        
                                        if(dst->type == InstrType::OperandType::GENERIC ) { // GPR reg as a destination                                                                                      
                                                GenOperand=dst->str;                                                                                 
                                                size_t found = GenOperand.rfind("CTAID.X");
                                                if (found != string::npos){
                                                        gridDimm=0;
                                                        injectInstrunc=true;         
                                                        //printf("Found: %d; \n",found); 
                                                }                                                
                                                found = GenOperand.rfind("CTAID.Y");
                                                if (found != string::npos){
                                                        gridDimm=1;
                                                        injectInstrunc=true;         
                                                        //printf("Found: %d; \n",found); 
                                                }                                                
                                                found = GenOperand.rfind("CTAID.Z");
                                                if (found != string::npos){
                                                        gridDimm=2;
                                                        injectInstrunc=true;
                                                        //printf("Found: %d; \n",found);
                                                }
                                                                                                                                                                
                                        }        
                                }                                                                                                 
                                if(injectInstrunc==true && inj_error_info.injDimension==gridDimm){
                                        printf("string: %s; blockDimm: %d\n",GenOperand.c_str(),gridDimm); 
                                        fout << "0x" << std::hex << i->getOffset() << "; " << i->getSass() << std::dec << " instrumented intruction; " << endl;
                                        const InstrType::operand_t *dst= i->getOperand(0);
                                        destGPRNum=dst->u.reg.num;
                                        numDestGPRs=1;
                                        instridx++; 
                                        inj_error_info.TotKerInstr++;
                                        nvbit_insert_call(i, "inject_error_IAC", IPOINT_AFTER);
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
                                        nvbit_add_call_arg_const_val32(i, gridDimm); // compute_capability
                                        nvbit_add_call_arg_const_val32(i, instridx); // compute_capability
                                        nvbit_add_call_arg_const_val32(i, i->getOffset());
                                        nvbit_add_call_arg_const_val32(i, instType);
                                         
                                }

                        }

                }
                inj_error_info.TotAppInstr+=inj_error_info.TotKerInstr;
                fout << "=================================================================================" << endl;
                fout << "The Instrumentation step Stops Here: " << removeSpaces(nvbit_get_func_name(ctx,f)) << endl;
                fout << "=================================================================================" << endl;
        }       
}


/* Instrumentation for IPP*/
void instrument_function_WV(CUcontext ctx, CUfunction func) {

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
                inj_error_info.maxregcount=maxregs;
                assert(fout.good());
                //assert(fout3.good());
                int k=0;
                int instridx=0;
                int gridDimm=0;
                bool injectInstrunc=false;
                inj_error_info.KernelPredReg=-1;
                inj_error_info.TotKerInstr=0;
                //fout << "Inspecting: " << kname << ";num_static_instrs: " << instrs.size() << ";maxregs: " << maxregs << "(" << maxregs << ")" << std::endl;
                for(auto i: instrs)  {
                        std::string opcode = i->getOpcode(); 
                        std::string instTypeStr = i->getOpcodeShort();
                        std::string GenOperand;
                        int instType = instTypeNameMap[instTypeStr]; 

                        if (verbose) printf("extracted instType: %s, ", instTypeStr.c_str());
                        if (verbose) printf("index of instType: %d\n", instTypeNameMap[instTypeStr]);
                        //if ((uint32_t)instType == inj_info.injInstType || inj_info.injInstType == NUM_ISA_INSTRUCTIONS) {
                        
                        //if ((uint32_t)instType == inj_info.injInstType) {
                        if (verbose) { printf("instruction selected for instrumentation: "); i->print(); }

                        int destGPRNum = -1;
                        int replGPRNum = -1;
                        int numDestGPRs = 0;
                        int predicateNum=-1;
                        int tracePredRegs=0;
                        injectInstrunc=false;
                        fout << "0x" << std::hex << i->getOffset() << ":::" << i->getSass()  << std::dec << std::endl;
                        //i->printDecoded();
                        printf("%s %d\n",i->getSass(),i->hasPred()==true ? i->getPredNum():-1);
                        if (i->getNumOperands() > 1) { // an actual instruction that writes to either a GPR or PR register
                                // Parse the first operand - this is the the destination register field                                
                                printf("%s\n",i->getSass());
                                for (int idx=0;idx<i->getNumOperands();++idx){
                                        const InstrType::operand_t *dst= i->getOperand(idx);                                        
                                        if(dst->type == InstrType::OperandType::PRED && injectInstrunc==false && idx==0) { // GPR reg as a destination                                                                                      
                                                GenOperand=dst->str;  
                                                predicateNum=dst->u.pred.num;
                                                //predicateNum=6;                                                                                                                               
                                                printf("pred:%s; num: %d\n", dst->str, dst->u.pred.num); 
                                                injectInstrunc=true;                                                                                                                                                                                                               
                                        }  
                                        if(dst->type == InstrType::OperandType::PRED && idx==0){
                                                tracePredRegs=dst->u.pred.num;
                                                if(inj_error_info.KernelPredReg<(tracePredRegs)){
                                                        inj_error_info.KernelPredReg=(tracePredRegs);
                                                }
                                        }                                              
                                }                                                                                                 
                                if(injectInstrunc==true and inj_error_info.injRegID==predicateNum){
                                        printf("string: %s; blockDimm: %d\n",GenOperand.c_str(),gridDimm); 
                                        fout << "0x" << std::hex << i->getOffset() << "; " << i->getSass() << std::dec << " instrumented intruction; " << endl;
                                        const InstrType::operand_t *dst= i->getOperand(0);
                                        destGPRNum=dst->u.reg.num;
                                        numDestGPRs=1;
                                        instridx++; 
                                        inj_error_info.TotKerInstr++; 
                                        nvbit_insert_call(i, "inject_error_WV", IPOINT_AFTER);
                                        nvbit_add_call_arg_const_val64(i, (uint64_t)&inj_error_info);
                                        nvbit_add_call_arg_const_val64(i, (uint64_t)&Injection_masks);
                                        nvbit_add_call_arg_const_val64(i, (uint64_t)&verbose_device);
                                        nvbit_add_call_arg_const_val32(i, predicateNum); // destination GPR register number                                        
                                        if (predicateNum != -1) {
                                        nvbit_add_call_arg_pred_val_at(i, 1); // destination GPR register val
                                        } else {
                                        nvbit_add_call_arg_const_val32(i, (unsigned int)-1); // destination GPR register val 
                                        }
                                        nvbit_add_call_arg_pred_reg(i);
                                        nvbit_add_call_arg_const_val32(i, numDestGPRs); // number of destination GPR registers
                                        nvbit_add_call_arg_const_val32(i, gridDimm); // compute_capability
                                        nvbit_add_call_arg_const_val32(i, instridx); // compute_capability
                                        nvbit_add_call_arg_const_val32(i, i->getOffset());
                                        nvbit_add_call_arg_const_val32(i, instType);                                        
                                }

                        }

                }
                
                if(inj_error_info.maxPredReg<inj_error_info.KernelPredReg){
                        inj_error_info.maxPredReg=inj_error_info.KernelPredReg;
                }
                inj_error_info.TotAppInstr+=inj_error_info.TotKerInstr;
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
                        int MaxThreadsPerSM=0;
                        CUdevice device;
                        cuDeviceGet(&device, 0);
                        cuDeviceGetAttribute(&MaxThreadsPerSM,CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,device);
                        cuDeviceGetName(inj_error_info.DeviceName,256,device);
                        inj_error_info.MaxThreadsPerWarp=32;
                        inj_error_info.MaxThreadsPerSM=MaxThreadsPerSM;
                        inj_error_info.MaxWarpsPerSM=MaxThreadsPerSM/inj_error_info.MaxThreadsPerWarp;
                        if(inj_mode.compare("IRA")==0 || inj_mode.compare("IR")==0){
                                parse_paramsIRA(injInputFilename);                                
                        }else if(inj_mode.compare("IAT")==0 || inj_mode.compare("IAW")==0){
                                parse_paramsIAT(injInputFilename); 
                        }else if(inj_mode.compare("IAC")==0){
                                parse_paramsIAC(injInputFilename); 
                        }else if(inj_mode.compare("WV")==0){
                                parse_paramsWV(injInputFilename); 
                        }else{
                                assert(1==0);
                        }         
                                       
                        if (verbose) printf("read file list done..\n");               
                        read_file=true;
                    } 
                        inj_error_info.gridDimX=p1->gridDimX;
                        inj_error_info.gridDimY=p1->gridDimY;
                        inj_error_info.gridDimZ=p1->gridDimZ;
                        inj_error_info.blockDimX=p1->blockDimX;
                        inj_error_info.blockDimY=p1->blockDimY;
                        inj_error_info.blockDimZ=p1->blockDimZ;
                        inj_error_info.num_threads  = num_threads;

                        pthread_mutex_lock(&mutex);
                        if (kernel_id < limit) {
                            cudaDeviceSynchronize();
                            fflush (stdout);
                            fclose (stdout);
                            freopen ("nvbit_stdout.txt", "a", stdout);
                                kname = removeSpaces(nvbit_get_func_name(ctx,p->f));
                                if(inj_mode.compare("IRA")==0){
                                        instrument_function_IRA(ctx, p->f);
                                }else if(inj_mode.compare("IAT")==0 || inj_mode.compare("IAW")==0){
                                        instrument_function_IAT(ctx, p->f);
                                }else if(inj_mode.compare("IAC")==0) {
                                        instrument_function_IAC(ctx, p->f);
                                }else if(inj_mode.compare("WV")==0) {
                                        instrument_function_WV(ctx, p->f);
                                }else{
                                        assert(1==0);
                                }
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
                                fflush (stdout);
                                fclose (stdout);
                                freopen ("stdout.txt", "a", stdout);

                                cudaError_t le = cudaGetLastError();
                                kname = removeSpaces(nvbit_get_func_name(ctx,p->f));
                                //int num_ctas = 0;
                                //int num_threads = 0;//added
                                if ( cbid == API_CUDA_cuLaunchKernel_ptsz ||
                                                cbid == API_CUDA_cuLaunchKernel) {
                                        //cuLaunchKernel_params * p2 = (cuLaunchKernel_params*) params;
                                        //num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
                                        //num_threads = num_ctas * p2->blockDimX * p2->blockDimY * p2->blockDimZ; //added 

                                }
                                assert(fout.good());                            
                                fout << "=================================================================================" << endl;
                                fout << "Report for: " << kname << "; kernel Index: "<< kernel_id <<  endl;
                                fout << "=================================================================================" << endl;
                                if ( cudaSuccess != le ) {
                                        assert(fout.good());                                        
                                        std::string cuerr = cudaGetErrorString(le);
                                        fout << "ERROR FAIL in kernel execution (" << cuerr << "); " <<std::endl;
                                        report_kernel_results();                                       
                                        SimEndRes = "; SimEndRes:::ERROR FAIL in kernel execution (" + cuerr + "):::";                                        
                                        exit(1); // let's exit early 
                                }
                                //fout << "inspecting: "<< kname <<"; thread : "<<  inj_info.injThreadID <<"; Register : "<< inj_info.injReg<<";  Mask : "<<inj_info.injMask<<"; SMID : "<<inj_info.injSMID<< "; Stuck at : "<<inj_info.injStuckat  << "; index: " << kernel_id << ";" <<std::endl;
                                report_kernel_results();
                                SimEndRes = "; SimEndRes:::PASS without fails:::";
                                inj_error_info.injNumActivAcc+= inj_error_info.injNumActivations;
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
    report_summary_results();
    if (verbose) printf("nvbit_at_term:end\n");
} 

