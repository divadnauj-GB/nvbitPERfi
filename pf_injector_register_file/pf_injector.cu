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
// __managed__ inj_info_t inj_info; 
__managed__ inj_info_error_t inj_error_info;
__managed__ muliple_ptr_t inj_thread_info;
//inj_info_t inj_info;


bool read_file=false;

std::string inj_mode;
std::string kname;
std::string SimEndRes;
std::string substr = "gemm";

std::string SASS_filename;
std::ofstream foutSASS;



int num_threads=0;

void reset_inj_info() {
		// inj_info.injInstType = 0; 
		// inj_info.injSMID = 0; 
		// inj_info.injLaneID = 0;
		// inj_info.injThreadID = 0;
  	    // inj_info.injReg = 0;
  		// inj_info.injStuckat = 0;
		// //inj_info.injCtaID = 0;
		// 
		// inj_info.injMask = 0;
		// inj_info.injNumActivations = 0;
		// inj_info.errorInjected = false;

		inj_error_info.injSMID=0; // 0 - max SMs
		inj_error_info.injThreadMask=0; //0-32
		inj_error_info.injMask=0;
		inj_error_info.injRegID=0; // injection mask;
		inj_error_info.injStuck_at=0;
		inj_error_info.injInstType=0; // instruction type 
		inj_error_info.injRegOriginal=0;
		inj_error_info.injRegReplacement=0;
		inj_error_info.injNumActivations=0;
		inj_error_info.injNumActivAcc=0;
		inj_error_info.injInstrIdx=0;
		inj_error_info.injInstPC=0;
		inj_error_info.injInstOpcode=0;  
		inj_error_info.blockDimX=0;
		inj_error_info.blockDimY=0;
		inj_error_info.blockDimZ=0;
		inj_error_info.gridDimX=0;
		inj_error_info.gridDimY=0;
		inj_error_info.gridDimZ=0;
		inj_error_info.maxregcount=0;
		inj_error_info.maxPredReg=-1;
		inj_error_info.KernelPredReg=0;
		inj_error_info.TotKerInstr=0;
		inj_error_info.TotAppInstr=0;
		inj_error_info.num_threads=0;
		inj_error_info.errorInjected=false;
		inj_error_info.kernel_id=0;

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

				std::ifstream ifs (filename.c_str(), std::ifstream::in);
				if (ifs.is_open()) {
						
						ifs >> inj_error_info.injThreadMask;
						ifs >> inj_error_info.injRegID;						
						ifs >> inj_error_info.injMask;
						
						ifs >> inj_error_info.injSMID; 
						assert(inj_error_info.injSMID < 1000); // we don't have a 1000 SM system yet. 
						
						ifs >> inj_error_info.injStuck_at;
			
						//NOT USED 
						//ifs >> inj_info.injLaneID; 
						//assert(inj_info.injLaneID < 32); // Warp-size is 32 or less today. 
						
						
						//printf("inspecting %d %d %d %d %d %d",inj_info.ingThreadID,inj_info.ingCtaID,inj_info.ingReg,inj_info.injMask,inj_info.injSMID,inj_info.injStuckat);
						

						//ifs >> inj_info.injInstType; // instruction type
						//assert(inj_info.injInstType <= NUM_ISA_INSTRUCTIONS); // ensure that the value is in the expected range

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
		}
}

void update_verbose() {
		static bool update_flag = false; // update it only once - performance enhancement
		if (!update_flag) {
			update_flag = true;
			//cudaDeviceSynchronize();
			verbose_device = verbose;
			//cudaDeviceSynchronize();
		}
}

int get_maxregs(CUfunction func) {
		int maxregs = -1;
		cuFuncGetAttribute(&maxregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func);
				
		//cuFuncGetAttribute();
		return maxregs;
}

void report_thread_injections(void){
	// SASS_filename="Finjection_details_threads.csv";
	// foutSASS.open(SASS_filename.c_str(), std::ifstream::out | std::ifstream::app);
	// foutSASS << "SMID; WarpID; LameID; ThreadID; CTAIDx; CTAIDy; CTAIDz; kindex; kname" << std::endl;
	for(int i=0;i<inj_error_info.num_threads;++i){
		if(inj_thread_info.flag[i]!=0){
			foutSASS << inj_thread_info.SMID[i] << "; ";
			foutSASS << inj_thread_info.WARPID[i] << "; ";
			foutSASS << inj_thread_info.LANEID[i] << "; ";
			foutSASS << inj_thread_info.ThrdID[i] << "; ";
			foutSASS << inj_thread_info.ctaID_x[i] << "; ";
			foutSASS << inj_thread_info.ctaID_y[i] << "; ";
			foutSASS << inj_thread_info.ctaID_z[i] << "; ";
			foutSASS << inj_error_info.kernel_id << "; ";
			foutSASS << inj_error_info.KName << std::endl;
		}
	}	
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
        << "; injThreadID: " << inj_error_info.injThreadMask;
        if(inj_mode.compare("REGs")==0){
                fout << "; injRegField: " << inj_error_info.injRegID
                << "; injMaskSeed: " << inj_error_info.injMask
				<< "; injStuck_at: " << inj_error_info.injStuck_at
                << "; resMaxRegCount: " << inj_error_info.maxregcount
                << "; resNumInstr: " << inj_error_info.TotKerInstr;
        }else {
                assert(1==0);
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
        << "; blockDimZ: " << inj_error_info.blockDimZ
		<< "; NumThreads: " << inj_error_info.num_threads;
        if(inj_error_info.errorInjected==true) 
        fout << "; ErrorInjected: True";
        else
        fout << "; ErrorInjected: False"; 

        fout << "; injSmID: " << inj_error_info.injSMID
        << "; injThreadID: " << inj_error_info.injThreadMask;
        if(inj_mode.compare("REGs")==0){
				fout << "; injRegField: " << inj_error_info.injRegID
                << "; injMaskSeed: " << inj_error_info.injMask
				<< "; injStuck_at: " << inj_error_info.injStuck_at
                << "; resMaxRegCount: " << inj_error_info.maxregcount
                << "; resNumInstr: " << inj_error_info.TotAppInstr;
        }else {
                assert(1==0);
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
		report_thread_injections();
        SimEndRes="; SimEndRes:::ERROR FAIL Detected Singal SIGKILL::: ";
        report_summary_results();
        fout.flush();
		foutSASS.flush();
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
		SASS_filename="Finjection_details_threads.csv";
		foutSASS.open(SASS_filename.c_str(), std::ifstream::out | std::ifstream::app);
		foutSASS << "SMID; WarpID; LameID; ThreadID; CTAIDx; CTAIDy; CTAIDz; kindex; kname" << std::endl;
		
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
				std::string kname = removeSpaces(nvbit_get_func_name(ctx,f));

				if (!already_instrumented.insert(f).second){
						continue;
				}

				fout << "=================================================================================" << endl;
                fout << "The Instrumentation step Begins Here: " << removeSpaces(nvbit_get_func_name(ctx,f)) << endl;
                //fout << "=================================================================================" << endl;
				
				/* Get the vector of instruction composing the loaded CUFunction "func" */
				const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

				int maxregs = get_maxregs(f);
				inj_error_info.maxregcount=maxregs;
				assert(fout.good());
				//assert(fout3.good());
				int k=0;
				//fout << "Inspecting: " << kname << ";num_static_instrs: " << instrs.size() << ";maxregs: " << maxregs << "(" << maxregs << ")" << std::endl;
				inj_error_info.TotKerInstr=0;
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
												
												if ((uint32_t)destGPRNum == inj_error_info.injRegID){
													k++;
													//fout <<"Kernel name: "<<kname<<"; kernel Index: "<< kernel_id <<"; Num_Activations: " << k <<";"<< std::endl;
													fout << "0x" << std::hex << i->getOffset() << "; " << i->getSass() << std::dec << " instrumented intruction; "  << endl;
													
													inj_error_info.TotKerInstr++;
													//printf("instType%d\n",instType);
													nvbit_insert_call(i, "inject_error", IPOINT_AFTER);
													nvbit_add_call_arg_const_val64(i, (uint64_t)&inj_error_info);
													nvbit_add_call_arg_const_val64(i, (uint64_t)&verbose_device);
													nvbit_add_call_arg_const_val64(i, (uint64_t)&inj_thread_info);
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
				inj_error_info.TotAppInstr+=inj_error_info.TotKerInstr;
                //fout << "=================================================================================" << endl;
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
				cuLaunch_params * p = (cuLaunch_params *) params;
				auto *p1 = (cuLaunchKernel_params *) params;  
				num_threads  = p1->gridDimX * p1->gridDimY * p1->gridDimZ * p1->blockDimX * p1->blockDimY * p1->blockDimZ; 				                              
				if(!is_exit) {
					if (verbose) printf("is_init\n");									
					if(read_file==false){
                        int MaxThreadsPerSM=0;
                        CUdevice device;
                        cuDeviceGet(&device, 0);
                        cuDeviceGetAttribute(&MaxThreadsPerSM,CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,device);
                        cuDeviceGetName(inj_error_info.DeviceName,256,device);
                        inj_error_info.MaxThreadsPerWarp=32;
                        inj_error_info.MaxThreadsPerSM=MaxThreadsPerSM;
                        inj_error_info.MaxWarpsPerSM=MaxThreadsPerSM/inj_error_info.MaxThreadsPerWarp;						
                        if(inj_mode.compare("REGs")==0){
                            parse_params(injInputFilename);
                        }else{
                            assert(1==0);
                        }                   
                        if (verbose) printf("read file list done..\n");               
                        read_file=true;
                    } 																						
					pthread_mutex_lock(&mutex);						
					if (kernel_id < limit) {		
						inj_error_info.gridDimX=p1->gridDimX;
						inj_error_info.gridDimY=p1->gridDimY;
						inj_error_info.gridDimZ=p1->gridDimZ;
						inj_error_info.blockDimX=p1->blockDimX;
						inj_error_info.blockDimY=p1->blockDimY;
						inj_error_info.blockDimZ=p1->blockDimZ;	
						inj_error_info.num_threads  = num_threads;						
						inj_error_info.kernel_id=kernel_id;		

						CUDA_SAFECALL(cudaMallocManaged(&(inj_thread_info.ThrdID),(num_threads)*sizeof(uint32_t)));
						CUDA_SAFECALL(cudaMallocManaged(&(inj_thread_info.SMID),(num_threads)*sizeof(uint32_t)));
						CUDA_SAFECALL(cudaMallocManaged(&(inj_thread_info.WARPID),(num_threads)*sizeof(uint32_t)));
						CUDA_SAFECALL(cudaMallocManaged(&(inj_thread_info.LANEID),(num_threads)*sizeof(uint32_t)));
						CUDA_SAFECALL(cudaMallocManaged(&(inj_thread_info.ctaID_x),(num_threads)*sizeof(uint32_t)));
						CUDA_SAFECALL(cudaMallocManaged(&(inj_thread_info.ctaID_y),(num_threads)*sizeof(uint32_t)));
						CUDA_SAFECALL(cudaMallocManaged(&(inj_thread_info.ctaID_z),(num_threads)*sizeof(uint32_t)));
						CUDA_SAFECALL(cudaMallocManaged(&(inj_thread_info.flag),(num_threads)*sizeof(uint32_t)));
						//memset(&(inj_thread_info.flag),0,(num_threads)*sizeof(uint32_t));
						//cudaDeviceSynchronize();
						fflush(stdout);
						fclose(stdout);
						freopen("nvbit_stdout.txt", "a", stdout);
						kname = removeSpaces(nvbit_get_func_name(ctx,p->f));
						memset(inj_error_info.KName, 0, sizeof inj_error_info.KName);
						kname.copy(inj_error_info.KName,255);
						if(inj_mode.compare("REGs")==0) {
							instrument_function_if_needed(ctx, p->f);
						}else{                                
							assert(1==0);
						}
						//cudaDeviceSynchronize();
						fout << "=================================================================================" << endl;
						fout << "Running instrumented Kernel: " << removeSpaces(nvbit_get_func_name(ctx,p->f)) << "; kernel Index: "<< kernel_id << endl;
						fout << "..............." << endl;
						//fout << "=================================================================================" << endl;
						//if (kname.find(substr)!=std::string::npos){
						nvbit_enable_instrumented(ctx, p->f, true); // run the instrumented version	
						//cudaDeviceSynchronize(); 
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
									//printf ("threads %d\n",num_threads);//added
							}
							assert(fout.good());                            
							//fout << "=================================================================================" << endl;
							fout << "Report for: " << kname << "; kernel Index: "<< kernel_id <<  endl;
							//fout << "=================================================================================" << endl;
							if ( cudaSuccess != le ) {
									assert(fout.good());                                        
									std::string cuerr = cudaGetErrorString(le);
									fout << "ERROR FAIL in kernel execution (" << cuerr << "); " <<std::endl;
									report_kernel_results(); 
									report_thread_injections();                                      
									SimEndRes = "; SimEndRes:::ERROR FAIL in kernel execution (" + cuerr + "):::";                                        
									exit(1); // let's exit early 
							}
							//fout << "inspecting: "<< kname <<"; thread : "<<  inj_info.injThreadID <<"; Register : "<< inj_info.injReg<<";  Mask : "<<inj_info.injMask<<"; SMID : "<<inj_info.injSMID<< "; Stuck at : "<<inj_info.injStuckat  << "; index: " << kernel_id << ";" <<std::endl;
							report_kernel_results();
							report_thread_injections();
							SimEndRes = "; SimEndRes:::PASS without fails:::";
							inj_error_info.injNumActivAcc+= inj_error_info.injNumActivations;
							inj_error_info.injNumActivations=0;
							inj_error_info.errorInjected = false;
							
							CUDA_SAFECALL(cudaFree(inj_thread_info.SMID));
							CUDA_SAFECALL(cudaFree(inj_thread_info.WARPID));
							CUDA_SAFECALL(cudaFree(inj_thread_info.LANEID));
							CUDA_SAFECALL(cudaFree(inj_thread_info.ThrdID));
							CUDA_SAFECALL(cudaFree(inj_thread_info.ctaID_x));
							CUDA_SAFECALL(cudaFree(inj_thread_info.ctaID_y));
							CUDA_SAFECALL(cudaFree(inj_thread_info.ctaID_z));
							CUDA_SAFECALL(cudaFree(inj_thread_info.flag));								

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
