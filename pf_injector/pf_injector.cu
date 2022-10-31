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
#include <cassert>
#include <pthread.h>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <csignal>
#include <unordered_set>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"
#include "instr_types.h"

#include "globals.h"
#include "pf_injector.h"

/**
 * New functions and variables for FLexGrip Injection
 */
#define FATAL(error) throw std::runtime_error(std::string("ERROR ") + __FILE__ + ":" + std::to_string(__LINE__));

int verbose;
__managed__ int verbose_device;
int limit = INT_MAX;


// injection parameters input filename: This file is created the script
// that launched error injections (MUST BE GLOBAL)
std::string injInputFilename;
std::string injInputFilenameHW;
std::string opcode_table;
pthread_mutex_t mutex;

//__managed__ inj_info_t inj_info;

//__managed__ kernel_level_inj_t *managed_inj_info_array_ptr=nullptr;

//__managed__ int vector_prob[2048*2048*128];
__managed__ muliple_ptr_t vector_todo;
__managed__ inj_info_t inj_info_data;
/*
__managed__ uint32_t KerID[MemSize];
__managed__ uint32_t SMID[MemSize];
__managed__ uint32_t InstrID[MemSize];
__managed__ uint32_t ctaID_x[MemSize];
__managed__ uint32_t ctaID_y[MemSize];
__managed__ uint32_t ctaID_z[MemSize];
__managed__ uint32_t WID[MemSize]; // 0 - max SMs
__managed__ uint32_t LID[MemSize];
__managed__ uint32_t Mask[MemSize];
__managed__ uint32_t num_operands[MemSize];
__managed__ uint32_t TLID[MemSize];
__managed__ uint32_t Operand1[MemSize];
__managed__ uint32_t Operand2[MemSize];
__managed__ uint32_t Operand3[MemSize];
__managed__ uint32_t Operand4[MemSize];
__managed__ uint32_t Tmasks[MemSize];
*/

std::vector<int> host_database_inj_vector;

size_t inj_info_array_size = 0;
int kernel_inj_print=0;
bool read_file=false;
bool kernel_injected=false;
int kernel_inj=-1;
int pidx=0;
int instr_number_prev=0;
int instr_index_vect=0;
unsigned num_threads=0;
// To inject for each kernel
using kernel_tuple = std::tuple<std::string, uint32_t>;
std::vector<kernel_tuple> kernel_vector;

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void Hardware_Fault_locfile(const std::string &filename) {
    
    std::ifstream input_file(filename);
    std::vector<std::string> rows;
    //std::vector<kernel_level_inj_t> host_database_inj_vector;
    
    //printf("reading file started..\n");
    if (input_file.good()) {
        while (!input_file.eof()) {
            std::string line;
            std::getline(input_file, line);
            if (input_file.eof()) {
                break;
            }
            rows.push_back(line);
        }
        inj_info_data.injSMID=std::stoul(rows[0]);
        inj_info_data.injsubSMID=std::stoul(rows[1]);
        inj_info_data.injLaneID=std::stoul(rows[2]);
        inj_info_data.Th=std::stof(rows[3]);
        //printf("threshold %f\n",inj_info_data.Th);

    }else{
        printf("here is an error .....");
        FATAL("Not possible to open the file " + filename)
    }
}



// Parse error injection site info from a file. This should be done on host side.
void parse_flex_grip_file(const std::string &filename) {
    
    std::ifstream input_file(filename);
    std::ifstream input_file1(filename);
    //std::vector<kernel_level_inj_t> host_database_inj_vector;
    
    uint32_t num_of_inst=0,num_of_kernels=0,tableSize=0;
    int iii=0;
    int instid=0;
    int smid=0;
    vector_todo.Table_size=0;
    //printf("reading file started..\n");
    if (input_file1.good()) {
        while (!input_file1.eof()) {
            std::string line;
            std::getline(input_file1, line);
            if (input_file1.eof()) {
                break;
            }
            tableSize++;
            //printf("reading file started %d..\n",tableSize);
        }
    }else{
        printf("here is an error .....");
        FATAL("Not possible to open the file " + filename)
    }
    
    
    CUDA_SAFECALL(cudaMallocManaged(&(vector_todo.num_operands),(tableSize)*sizeof(uint32_t)));
    //CUDA_SAFECALL(cudaMallocManaged(&(vector_todo.InstrIDx),(tableSize)*sizeof(uint32_t)));
    CUDA_SAFECALL(cudaMallocManaged(&(vector_todo.Operand1),(tableSize)*sizeof(uint32_t)));
    CUDA_SAFECALL(cudaMallocManaged(&(vector_todo.Operand2),(tableSize)*sizeof(uint32_t)));
    CUDA_SAFECALL(cudaMallocManaged(&(vector_todo.Operand3),(tableSize)*sizeof(uint32_t)));
    CUDA_SAFECALL(cudaMallocManaged(&(vector_todo.Tmasks),(tableSize)*sizeof(uint32_t)));
    
   /*
    vector_todo.KerID=KerID;
    vector_todo.InstrID=InstrID;
    vector_todo.ctaID_x=ctaID_x;
    vector_todo.ctaID_y=ctaID_y;
    vector_todo.ctaID_z=ctaID_z;
    vector_todo.LID=LID;
    vector_todo.TLID=TLID;
    vector_todo.WID=WID;
    vector_todo.Mask=Mask;
    vector_todo.num_operands=num_operands;
    vector_todo.Operand1=Operand1;
    vector_todo.Operand2=Operand2;
    vector_todo.Operand3=Operand3;
    vector_todo.Operand4=Operand4;
    vector_todo.Tmasks=Tmasks;
    */
   


    //CUDA_SAFECALL(cudaMallocManaged(&(vector_todo.T0_ctaID_x),(MemSize)*sizeof(uint32_t)));
    //CUDA_SAFECALL(cudaMallocManaged(&(vector_todo.T0_ctaID_y),(MemSize)*sizeof(uint32_t)));
    //CUDA_SAFECALL(cudaMallocManaged(&(vector_todo.T0_ctaID_z),(MemSize)*sizeof(uint32_t)));
    //CUDA_SAFECALL(cudaMallocManaged(&(vector_todo.T0_LID),(MemSize)*sizeof(uint32_t)));
    //CUDA_SAFECALL(cudaMallocManaged(&(vector_todo.T0_WID),(MemSize)*sizeof(uint32_t)));
    CUDA_SAFECALL(cudaMallocManaged(&(vector_todo.T0_Mask),(MemSize)*sizeof(uint32_t)));
    CUDA_SAFECALL(cudaMallocManaged(&(vector_todo.T1_Mask),(MemSize)*sizeof(uint32_t)));
    CUDA_SAFECALL(cudaMallocManaged(&(vector_todo.T2_Mask),(MemSize)*sizeof(uint32_t)));
    CUDA_SAFECALL(cudaMallocManaged(&(vector_todo.flag),(MemSize)*sizeof(uint32_t)));
    CUDA_SAFECALL(cudaMallocManaged(&(vector_todo.Mask),(MemSize)*sizeof(uint32_t)));
    
    
    if (input_file.good()) {
        // Read the file that contains the error model from FlexGrip
        while (!input_file.eof()) {
            std::string line, word;
            std::vector<std::string> row,cta_fields,instr,nmemonic;
            
            // read an entire row and
            // store it in a string variable 'line'
            std::getline(input_file, line);
            if (input_file.eof()) {
                break;
            }
            
            // used for breaking words
            std::stringstream s(line);

            // read every column data of a row and
            // store it in a string variable, 'word'
            while (std::getline(s, word, ';')) {
                // add all the column data
                // of a row to a vector
                row.push_back(word);
            }

            std::stringstream fields(row[6]);
            //std::stringstream fields(row[3]);
            while (std::getline(fields, word, ' ')) {
                // add all the column data of a row to a vector
                cta_fields.push_back(word);
            }

            std::stringstream fields1;
            if(cta_fields[0].find('@')!=std::string::npos){
                //std::stringstream fields1(cta_fields[1]);
                fields1 << cta_fields[1];

            }else{
                //std::stringstream fields1(cta_fields[0]);
                fields1 << cta_fields[0];
            }
            
            //std::stringstream fields(row[3]);
            while (std::getline(fields1, word, ' ')) {
                // add all the column data of a row to a vector
                instr.push_back(word);
            }

            std::stringstream fields2(instr[0]);
            //std::stringstream fields(row[3]);
            while (std::getline(fields2, word, '.')) {
                // add all the column data of a row to a vector
                nmemonic.push_back(word);
            }
            //printf("NMN %s\n",nmemonic[0].c_str());
            //printf("%s\n",nmemonic[0].c_str());
            //if(iii==0) inj_info_data.injSMID=std::stoul(row[0]);
            std::string instr_name=nmemonic[0];            
            if(iii==0){
                inj_info_data.InstrIDx=instTypeNameMap[instr_name]; 
                opcode_table=nmemonic[0];
            }
            //printf("InstrID %dMNEMONIC %s\n ",inj_info_data.InstrIDx, nmemonic[0].c_str());
            //vector_todo.TLID[iii]=std::stoul(row[1]);                      
            vector_todo.Operand1[iii]=std::stoul(row[2],nullptr,16);
            vector_todo.Operand2[iii]=std::stoul(row[3],nullptr,16);
            if(row[4].c_str()[0]=='X' || row[4].c_str()[0]=='x'){
                vector_todo.Operand3[iii]=0;
                vector_todo.num_operands[iii]=2;
            }
            else{                 
                vector_todo.Operand3[iii]=std::stoul(row[4],nullptr,16);
                vector_todo.num_operands[iii]=3;
            }
            vector_todo.Tmasks[iii]=std::stoul(row[5]);
            iii++;            
            
        }
        
        inj_info_data.injNumActivations=0;
        inj_info_data.errorInjected=false;
       vector_todo.Table_size=iii;
       inj_info_array_size=iii;
        
        // COPY to gpu the array of injections   
        
        if(verbose) printf("inj_info_array_size %d\n",inj_info_array_size);
        
        //CUDA_SAFECALL(cudaMalloc(&global_mem_info_array_ptr, sizeof(kernel_level_inj_t)));

               
        /*
        // COPY to gpu the array of injections        
        CUDA_SAFECALL(cudaMallocManaged(&managed_inj_info_array,
                                        host_database_inj_info.size() * sizeof(inj_info_t)));
        std::copy(host_database_inj_info.begin(), host_database_inj_info.end(), managed_inj_info_array);
        CUDA_SAFECALL(cudaDeviceSynchronize());
        inj_info_array_size = host_database_inj_info.size();*/
        
    } else {
        printf("here is an error .....");
        FATAL("Not possible to open the file " + filename)
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
    return maxregs;
}

void INThandler(int sig) {
    signal(sig, SIG_IGN); // disable Ctrl-C

    fout << ":::NVBit-inject-error; ERROR FAIL Detected Singal SIGKILL;";    
    fout.flush();
    exit(-1);
}


/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
// DO NOT USE UVM (__managed__) variables in this function
void nvbit_at_init() {
    /* just make sure all managed variables are allocated on GPU */
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    injInputFilename = "nvbitfi-injection-info.txt";
    injInputFilenameHW = "nvbitfi-injection-infoHW.txt";
    /* we get some environment variables that are going to be use to selectively
     * instrument (within a interval of kernel indexes and instructions). By
     * default we instrument everything. */
    if (getenv("TOOL_VERBOSE")) {
        verbose = std::stoi(getenv("TOOL_VERBOSE"));
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
        limit = std::stoi(getenv("INSTRUMENTATION_LIMIT"));
    }

    // GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool (1, 2, 3,..)");

    initInstTypeNameMap();

    signal(SIGINT, INThandler); // install Ctrl-C handler

    open_output_file(injOutputFilename);  

    if (verbose) printf("nvbit_at_init:end\n");
    
}

//std::unordered_set <CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func, int kindex) {

//    parse_params(injInputFilename);  // injParams are updated based on injection seed file
    if(verbose) printf("Kernel_injection %d\n",kindex);
    update_verbose();
    

    // Main flexgrip counter, to iterate over the main injection info
    //uint32_t inj_info_array_it = 0;
    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);
    /* iterate on function */
    for (auto f: related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {            
            continue;
        }   
        std::string kname = removeSpaces(nvbit_get_func_name(ctx, f));
        if (verbose) printf("Kernel name = %s",nvbit_get_func_name(ctx, f));

        /* Get the vector of instruction composing the loaded CUFunction "func" */
        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

        int maxregs = get_maxregs(f);
        assert(fout.good());
        fout << "Inspecting: " << kname << ";num_static_instrs: " << instrs.size() << ";maxregs: " << maxregs << "("
             << maxregs << ")" << std::endl;
        size_t inst_index = 0;
        instr_index_vect=0;
        instr_number_prev=0;
        size_t curr_inst=0;
        unsigned int cbanid=0, mempos=0, numops=0, reg[10]={0,0,0,0,0,0,0,0,0,0};
        unsigned int cbanckoff=0;
        for (auto i: instrs) {
            std::string opcode = i->getOpcode();
            std::string opcodeshort = i->getOpcodeShort();
            std::string instTypeStr = extractInstType(opcode);
            int instType = instTypeNameMap[instTypeStr];
            if (verbose) printf("extracted instType: %s, ", instTypeStr.c_str());
            if (verbose) printf("index of instType: %d\n", instTypeNameMap[instTypeStr]);
            /**
             * MODIFICATION FOR FLEXGRIP PF injection
             */
            numops=i->getNumOperands()-1;
            int regidx=0;
            mempos=0;
            for (int idx = 0; idx < i->getNumOperands(); idx++) {
                const InstrType::operand_t *op = i->getOperand(idx);
                if(op->type == InstrType::OperandType::REG){
                    if (verbose)  printf("R%d ",op->u.reg.num);
                    reg[regidx]=op->u.reg.num;
                    regidx++;
                }
                if(op->type == InstrType::OperandType::IMM_UINT64){
                   if (verbose)   printf("I%d ",op->u.imm_uint64.value);
                }
                if(op->type == InstrType::OperandType::IMM_DOUBLE){
                    if (verbose)  printf("I%d ",op->u.imm_double);
                }
                if(op->type == InstrType::OperandType::CBANK){
                    if (verbose)  printf("c[0x%x][0x%x] ",op->u.cbank.id,
                            op->u.cbank.imm_offset);
                    cbanid=op->u.cbank.id;
                    cbanckoff=op->u.cbank.imm_offset;
                    mempos=idx;
                }
            }
            if (verbose) printf("\n");

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
                    if(opcodeshort.find(opcode_table.c_str())!=std::string::npos){
                    //if (instType == inj_info_data.InstrIDx){
                        if (verbose) {
                            printf("instruction selected for instrumentation: ");                                        
                            i->print();
                            printf("instruction index= %d\n",inst_index);
                        }
                        
                        //---------------------------------------------------------------------------------------------
                        destGPRNum = regnum1;
                        numDestGPRs = (getOpGroupNum(instType) == G_FP64) ? 2 : 1;

                        int szStr = extractSize(opcode);
                        if (szStr == 128) {
                            numDestGPRs = 4;
                        } else if (szStr == 64) {
                            numDestGPRs = 2;
                        }                        
                        //inj_info_data.injNumActivations=0;
                        //inj_info_data.errorInjected=false;
                        

                        if(verbose) printf("instruction index %d %d %d\n",0, inst_index, num_threads);
                        //==============================================================================================================================
                        nvbit_insert_call(i, "inject_error_prev", IPOINT_BEFORE);                     
                        nvbit_add_call_arg_const_val64(i, (uint64_t) &verbose_device);
                        nvbit_add_call_arg_const_val64(i, (uint64_t) &inj_info_data);
                        nvbit_add_call_arg_const_val32(i, destGPRNum); // destination GPR register number
                        if (destGPRNum != -1) {
                            nvbit_add_call_arg_reg_val(i, destGPRNum); // destination GPR register val
                        } else {
                            nvbit_add_call_arg_const_val32(i, (unsigned int) -1); // destination GPR register val
                        }
                        nvbit_add_call_arg_const_val32(i, numDestGPRs); // number of destination GPR registers
                        nvbit_add_call_arg_const_val32(i, num_threads); // max regs used by the inst info
                        nvbit_add_call_arg_const_val32(i, inst_index);
                        nvbit_add_call_arg_const_val32(i, mempos);
                        nvbit_add_call_arg_const_val32(i, numops);
                        nvbit_add_call_arg_const_val32(i, reg[0]);
                        nvbit_add_call_arg_const_val32(i, reg[1]);
                        nvbit_add_call_arg_const_val32(i, reg[2]);
                        nvbit_add_call_arg_const_val64(i, (uint64_t) &vector_todo);
                        nvbit_add_call_arg_cbank_val(i,cbanid,cbanckoff);
                        
                        //==============================================================================================================================
                        nvbit_insert_call(i, "inject_error", IPOINT_AFTER);                        
                        nvbit_add_call_arg_const_val64(i, (uint64_t) &verbose_device);
                        nvbit_add_call_arg_const_val64(i, (uint64_t) &inj_info_data);
                        nvbit_add_call_arg_const_val32(i, destGPRNum); // destination GPR register number
                        if (destGPRNum != -1) {
                            nvbit_add_call_arg_reg_val(i, destGPRNum); // destination GPR register val
                        } else {
                            nvbit_add_call_arg_const_val32(i, (unsigned int) -1); // destination GPR register val
                        }
                        nvbit_add_call_arg_const_val32(i, numDestGPRs); // number of destination GPR registers
                        nvbit_add_call_arg_const_val32(i, num_threads); // max regs used by the inst info
                        nvbit_add_call_arg_const_val32(i, inst_index); // max regs used by the inst info
                        nvbit_add_call_arg_const_val64(i, (uint64_t) &vector_todo);
                        /**********************************************************************************************/
                        if (verbose) printf("instrumentation done!...");
                    }
                    
                }
                // If an instruction has two destination registers, not handled!! (TODO: Fix later)
            }
            inst_index++;
            
        }
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
    //if (verbose) printf("nvbit_at_cuda_event:start\n");
    /* Identify all the possible CUDA launch events */
    //if (verbose) printf("%d\n",cbid);
    if (cbid == API_CUDA_cuLaunch ||
        cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid ||
        cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel) {
        /* cast params to cuLaunch_params since if we are here we know these are
         * the right parameters type */
        auto *p = (cuLaunch_params *) params;
        auto *p1 = (cuLaunchKernel_params *) params;
        num_threads  = p1->gridDimX * p1->gridDimY * p1->gridDimZ * p1->blockDimX * p1->blockDimY * p1->blockDimZ;
        
        if (!is_exit) {
            
            pthread_mutex_lock(&mutex);
            if (verbose) printf("kernel start 1\n");
            if(read_file==false){
                Hardware_Fault_locfile(injInputFilenameHW);
                parse_flex_grip_file(injInputFilename); 
                if (verbose) printf("read file list done..\n");               
                read_file=true;
            }

            kernel_inj=0;
            /*
            for (size_t ki=0;ki<inj_info_array_size;ki++){      
                if (verbose) printf("Kid %d, kidinj %d\n",kernel_id,host_database_inj_vector[ki]);          
                if((uint32_t)kernel_id==host_database_inj_vector[ki]){                    
                    kernel_inj=kernel_id; 
                    kernel_inj_print=ki;                  
                }
            }*/
            if ((kernel_id < limit)) { 
                if (verbose) printf("instrumentation function start\n");               
                
                kernel_injected=true;
                
                //inj_info_array_size = host_database_inj_vector.size();
                
                if (verbose) printf("inj_info_array_size %d\n",inj_info_array_size);

                instrument_function_if_needed(ctx, p->f,kernel_id);
                cudaDeviceSynchronize();
                nvbit_enable_instrumented(ctx, p->f, true); // run the instrumented version
                cudaDeviceSynchronize();
            } else {
                kernel_injected=false;
                nvbit_enable_instrumented(ctx, p->f, false); // do not use the instrumented version
                cudaDeviceSynchronize();
            }
            if (verbose) printf("kernel start\n");

        } else {
            if (kernel_id < limit) {
                if (verbose) printf("is_exit\n");
                cudaDeviceSynchronize();

                cudaError_t le = cudaGetLastError();

                std::string kname = removeSpaces(nvbit_get_func_name(ctx, p->f));
                unsigned num_ctas = 0;
                num_threads=0;
                int threadsperblock=-1;
                if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                    cbid == API_CUDA_cuLaunchKernel) {
                    auto *p2 = (cuLaunchKernel_params *) params;
                    num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
                    num_threads  = num_ctas * p2->blockDimX * p2->blockDimY * p2->blockDimZ;
                    threadsperblock=p2->blockDimX * p2->blockDimY * p2->blockDimZ;
                }
                
                assert(fout.good());
                fout << "Injection data;";
                fout << "SMID: " << inj_info_data.injSMID << ";";
                fout << "subSMID: " << inj_info_data.injsubSMID << ";";
                fout << "LaneID: " << inj_info_data.injLaneID << ";";
                fout << "index: " << kernel_id << ";";
                fout << "kernel_name: " << kname << ";";
                fout << "ctas: " << num_ctas << ";" ;
                fout << "threads/block: " << threadsperblock << ";" ;
                fout << "tot threads: " << num_threads << ";" ;
                fout << "injNumActivations: " << inj_info_data.injNumActivations << std::endl;

                if (cudaSuccess != le) {
                    assert(fout.good());
                    fout << "ERROR FAIL in kernel execution (Error: " << cudaGetErrorString(le) << "); " << std::endl;
                    exit(1); // let's exit early
                }
                /*
                if(kernel_id==0){
                     printf("helllooo!!!\n");
                    //if(i==0){    
                    //printf("%d %d %d \n",maxinst,pSM,injinstr);
                        for(int ij=0;ij<num_threads;++ij){
                            if(vector_todo.flag[(num_threads*4)+ij]==1){
                                printf("I: %d T: %d B: %d M: 0x%x\n",4,ij, vector_todo.T0_ctaID_x[(num_threads*4)+ij],vector_todo.Mask[(num_threads*4)+ij]);
                            }
                        }
                    //}
                }*/


                if (verbose) printf("\n index: %d; kernel_name: %s; \n", kernel_id, kname.c_str());
                kernel_id++; // always increment kernel_id on kernel exit

                pthread_mutex_unlock(&mutex);
            }
        }
    }
    //if (verbose) printf("nvbit_at_cuda_event:end\n");
}



void nvbit_at_term() {
    if (verbose) printf("nvbit_at_term:start\n");
    assert(fout.good());
    fout << "Total injNumActivations: " << inj_info_data.injNumActivations << std::endl;
    if (verbose) printf("nvbit_at_term:end\n");
} //nothing to do here
