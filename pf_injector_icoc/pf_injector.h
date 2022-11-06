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

#ifndef PF_INJECTOR_H
#define PF_INJECTOR_H

#include <fstream>
#include <iostream>

#include "arch.h"

//#define MAX_KNAME_SIZE 1000

#define HOST_FUNCTION_  inline __host__
#define DEVICE_FUNCTION_ __device__ __forceinline__
//#define HOST_DEVICE_FUNCTION_ __host__ DEVICE_FUNCTION_

static void assert_exception(bool condition, const std::string &message, const std::string &file, int line) {
    if (!condition) {
        auto err_message = "Assert false in file " + file + ", in line " + std::to_string(line) +
                           "\nmessage:" + message;
        throw std::runtime_error(err_message);
    }
}

#define assert_condition(condition, message) assert_exception(condition, message, __FILE__, __LINE__)

enum ICOCSubpartition {
    SCHEDULER = 0,
    DECODER,
    FETCH,
    NUM_ICOC_SUBPARTITIONS
};

/**
 * Class that represents all injection information necessary for the instrumenting functions
 * The functions must be clearly defined as HOST_FUNCTION_ or DEVICE_FUNCTION_ as this object is used on
 * device side and host side
 */
struct InjectionInfo {
    uint32_t sm_id; // 0 - max SMs
//    uint32_t lane_id; // 0 - 32
//    uint32_t mask; // injection mask
    uint32_t warp_group;
    InstructionType instruction_type_in; // instruction type in
    InstructionType instruction_type_out; // instruction type in
    // subpartition type
    ICOCSubpartition icoc_subpartition;

    // updated during/after error injection
    uint64_t num_activations;
//    bool error_injected;

    HOST_FUNCTION_
    void reset_injection_info() {
        this->instruction_type_in = static_cast<InstructionType>(0);
        this->sm_id = 0;
//        this->lane_id = 0;
//        this->mask = 0;
        this->num_activations = 0;
        this->warp_group = 0;
//        this->error_injected = false;
        icoc_subpartition = ICOCSubpartition::SCHEDULER;
    }

    // print info for debug
    DEVICE_FUNCTION_
    void print() const {
        printf("InstType=%d, SMID=%d, ICOCSubpartition=%d, warp group %d\n", this->instruction_type_in,
               this->sm_id, this->icoc_subpartition, this->warp_group);
    }

    // print info
    HOST_FUNCTION_
    friend std::ostream &operator<<(std::ostream &os, const InjectionInfo &inj_info) {
        os << "selected SM: " << inj_info.sm_id << ";";
//        os << "selected Lane: " << inj_info.lane_id << ";";
//        os << "selected Mask: " << inj_info.mask << ";";
        os << "selected InstType: " << inj_info.instruction_type_in << ";";
        os << "selected WarpGroup: " << inj_info.warp_group << ";";
        os << "selected ICOCSubpartition: " << inj_info.icoc_subpartition << ";";
        os << "num_activations: " << inj_info.num_activations << ";";
        return os;
    }


    // Parse error injection site info from a file. This should be done on host side.
    HOST_FUNCTION_
    void parse_params(const std::string &filename, bool verbose) {
        static bool parse_flag = false; // file will be parsed only once - performance enhancement
        auto message_str = filename + " should contain enough information about the fault site to perform a"
                                      " permanent error injection run:\nSM ID (int), Instruction type in (int),"
                                      " Instruction type out (int), ICOC subpartition (int), warp group (int).\n";
        if (!parse_flag) {
            parse_flag = true;
            this->reset_injection_info();

            std::ifstream ifs(filename);
            assert_condition(ifs.good(), " File " + filename + " does not exist!\n" + message_str);
            ifs >> this->sm_id;
            assert_condition(this->sm_id <= 1000, "Invalid sm id. We don't have a 1000 SM system yet.\n" + message_str);

//            ifs >> this->lane_id;
//            assert_condition(this->lane_id < 32, "Invalid warp lane id, must be 0 <= warp id < 32");

//            ifs >> this->mask;

            auto inst_type_in = 0, inst_type_out = 0;
            ifs >> inst_type_in; // instruction type
            ifs >> inst_type_out; // instruction type

            this->instruction_type_in = static_cast<InstructionType>(inst_type_in);
            // ensure that the value is in the expected range
            assert_condition(this->instruction_type_in < NUM_ISA_INSTRUCTIONS,
                             std::to_string(this->instruction_type_in) + " Invalid instruction type.\n" + message_str);

            this->instruction_type_out = static_cast<InstructionType>(inst_type_out);
            // ensure that the value is in the expected range
            assert_condition(this->instruction_type_out < NUM_ISA_INSTRUCTIONS,
                             std::to_string(this->instruction_type_out) + " Invalid instruction type.\n" + message_str);

            // Read the syndrome type
            auto subpart = 0;
            ifs >> subpart;
            this->icoc_subpartition = static_cast<ICOCSubpartition>(subpart);
            assert_condition(this->icoc_subpartition < ICOCSubpartition::NUM_ICOC_SUBPARTITIONS,
                             std::to_string(this->icoc_subpartition) + " Invalid syndrome type");

            // Read the warp group
            ifs >> this->warp_group;
            assert_condition(this->warp_group < 4,
                             std::to_string(this->icoc_subpartition) + " Invalid warp group must be 0 <= wg < 4.\n" +
                             message_str);

            if (verbose) {
                std::cout << this << std::endl;
            }
        }
    }

};

//#define MAX_KNAME_SIZE 1000
//
//#define FL_INSTR 0
//
//#define MemSize 1024*512*720
////#define DBG
//typedef struct {
//    uint32_t injSMID; // 0 - max SMs
//    uint32_t injsubSMID;
//    uint32_t injLaneID;
//    float Th;
//    uint64_t injNumActivations;
//    uint64_t counter_data;
//    uint32_t InstrIDx;
//    bool errorInjected;
//} inj_info_t;
//
//
//typedef struct {
//    uint32_t *KerID;
//    uint32_t *SMID;
//    uint32_t *InstrID;
//    uint32_t *ctaID_x;
//    uint32_t *ctaID_y;
//    uint32_t *ctaID_z;
//    uint32_t *WID; // 0 - max SMs
//    uint32_t *LID;
//    uint32_t *Mask;
//    uint32_t *num_operands;
//    uint32_t *TLID;
//    uint32_t *Operand1;
//    uint32_t *Operand2;
//    uint32_t *Operand3;
//    uint32_t *Operand4;
//    uint32_t *TOperand1;
//    uint32_t *TOperand2;
//    uint32_t *TOperand3;
//    uint32_t *Tmasks;
//    uint32_t Table_size;
//    uint32_t *T0_instID;
//    uint32_t *T0_ctaID_x;
//    uint32_t *T0_ctaID_y;
//    uint32_t *T0_ctaID_z;
//    uint32_t *T0_WID; // 0 - max SMs
//    uint32_t *T0_LID;
//    uint32_t *T0_Mask;
//    uint32_t *T1_Mask;
//    uint32_t *T2_Mask;
//    uint32_t *flag;
//    uint32_t counter;
//} muliple_ptr_t;

#endif