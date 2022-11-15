#include <cstdarg>

#include "nvbit_reg_rw.h"
#include "utils/utils.h"
#include "pf_injector.h"
#include "arch.h"

DEVICE_FUNCTION_
void assert_gpu_(bool condition, const char *message, bool verbose, const char *file, int line) {
    if (!condition && verbose) {
        printf("ERROR GPU:%s at %s:%d\n", message, file, line);
        assert(0);
    }
}

#define assert_gpu(condition, message, verbose) assert_gpu_(condition, message, verbose, __FILE__, __LINE__)

DEVICE_FUNCTION_
bool is_fault_injection_necessary(const InjectionInfo *inj_info) {
    // All these parameters are defined at generate_injections_list.py
    auto sm_id = get_smid();
    // This is not the selected SM. No need to proceed.
    if (sm_id != inj_info->sm_id)
        return false;

    auto warp_id = get_warpid();
    if ((warp_id % 4) != inj_info->warp_group) {
        return false;
    }

    if (inj_info->icoc_subpartition == ICOCSubpartition::SCHEDULER && warp_id != inj_info->warp_id) {
        return false;
    }

    return true;
}

DEVICE_FUNCTION_
int32_t define_opcode_behavior_32bits(uint32_t instruction_type, int32_t original_r0,
                                      int32_t r1_int, int32_t r2_int, int32_t r3_int, int32_t r4_int,
                                      bool is_float, bool verbose);

extern "C" __device__ __noinline__
void inject_error(
        uint64_t injection_info_ptr,        //nvbit_add_call_arg_const_val64(i, uint64_t(&inj_info));
        uint64_t verbose_device_ptr,        //nvbit_add_call_arg_const_val64(i, uint64_t(&verbose_device));
        uint64_t count_activations_inst_ptr,//nvbit_add_call_arg_const_val64(i, uint64_t(count_activations_inst));
        uint32_t dest_GPR_num,              //nvbit_add_call_arg_const_val32(i, dest_GPR_num);
        uint32_t num_dest_GPRs,             //nvbit_add_call_arg_const_val32(i, num_dest_GPRs);
        uint32_t is_float,                  //nvbit_add_call_arg_const_val32(i, is_float);
        uint32_t current_opcode,            //nvbit_add_call_arg_const_val32(i, current_opcode);
        uint32_t replace_opcode,            //nvbit_add_call_arg_const_val32(i, next_instruction_opcode);
        int32_t num_operands,               //nvbit_add_call_arg_const_val32(i, i->getNumOperands());
        ...                                 //variadic operands, managed on fill_values_from_variadic
) {
    auto *inj_info = (InjectionInfo *) injection_info_ptr;
    auto *count_activations_inst = (unsigned long long *) count_activations_inst_ptr;

    uint32_t verbose_device = *((uint32_t *) verbose_device_ptr);
    assert_gpu(num_dest_GPRs > 0, "num_dest_GPRs equals to 0", verbose_device);

    int32_t dest_reg_before_val = nvbit_read_reg(dest_GPR_num); // read the register value

    int32_t reg_data[MAX_OPERANDS_NUM];
    for (auto &ri: reg_data) ri = 0;

    va_list vl;
    va_start(vl, num_operands);
    for (uint32_t operand_i = num_dest_GPRs, i = 0; operand_i < num_operands; operand_i++) {
        /** Always put in the following order
         * 1 operand type const 32 bits
         * 2 if the operand is valid const 32bits (0 or 1)
         * 3 operand val, can be 32 bits or mem ref 64 bits
         */
        uint32_t operand_type = va_arg(vl, uint32_t);
        uint32_t is_operand_valid = va_arg(vl, uint32_t);

        if (static_cast<InstrType::OperandType>(operand_type) == InstrType::OperandType::MREF) {
            // do nothing
            reg_data[i] = 0;
        } else if (static_cast<InstrType::OperandType>(operand_type) == InstrType::OperandType::IMM_UINT64) {
            // only what is low in the 64 format
            reg_data[i] = int32_t(va_arg(vl, int64_t));
        } else if (static_cast<InstrType::OperandType>(operand_type) == InstrType::OperandType::IMM_DOUBLE) {
            auto raw = va_arg(vl, int64_t);
            auto val_data = float(*((double *) &raw));
            reg_data[i] = *((int32_t *) &val_data);
        } else {
            reg_data[i] = va_arg(vl, int32_t);
        }
        assert_gpu((is_operand_valid == 0 || is_operand_valid == 1), "is_operand_valid incorrect >1", verbose_device);
        i += is_operand_valid;
    }
    va_end(vl);

    auto dest_reg_after_val = define_opcode_behavior_32bits(replace_opcode, dest_reg_before_val,
                                                            reg_data[0], reg_data[1], reg_data[2], reg_data[3],
                                                            bool(is_float), verbose_device);
    if (DUMMY == 0 && is_fault_injection_necessary(inj_info)) {
        nvbit_write_reg(dest_GPR_num, (int) dest_reg_after_val);
        if (verbose_device)
            printf("register=%d, beforeInst=%d, afterInst=%d, before=0x%x, after=0x%x, expected_after=0x%x\n",
                   dest_GPR_num, current_opcode, replace_opcode, dest_reg_before_val, nvbit_read_reg(dest_GPR_num),
                   dest_reg_after_val);
        atomicAdd((unsigned long long *) &(inj_info->num_activations), 1LL);
        // Count the activations
        atomicAdd(&count_activations_inst[current_opcode], 1);
    }
}


DEVICE_FUNCTION_
int32_t define_opcode_behavior_32bits(uint32_t instruction_type, int32_t original_r0,
                                      int32_t r1_int, int32_t r2_int, int32_t r3_int, int32_t r4_int,
                                      bool is_float, bool verbose) {
    float r1_float = __int_as_float(r1_int);
    float r2_float = __int_as_float(r2_int);
    float r3_float = __int_as_float(r3_int);
    float r4_float = __int_as_float(r4_int);

    int32_t destination_val = original_r0;
    float destination_val_float = __int_as_float(original_r0);

    switch (instruction_type) {
        // Floating point 32 instructions (0-13)
        case FADD:
        case FADD32I:
            destination_val_float = r1_float + r2_float;
            break;
        case FCMP:
            if (r1_float == r2_float)
                destination_val_float = r3_float;
            else
                destination_val_float = r4_float;
            break;
        case FFMA:
        case FFMA32I:
            destination_val_float = r1_float * r2_float + r3_float;
            break;
        case FMUL:
        case FMUL32I:
            destination_val_float = r1_float * r2_float;
            break;
        case FSEL:
            destination_val_float = r1_float;
            break;
        case FSET:
            // compares and set to 0 or 1 the output register
            if (r1_float == r2_float) {
                destination_val_float = 0;
            } else {
                destination_val_float = 1;
            }
            break;
//====== Integer Instructions (35-67)
        case BFE: {
            int32_t mask = ~(0xffffffff << r3_int);
            if (r2_int > 0)
                destination_val = (r1_int >> (r2_int - 1)) & mask;
            else
                destination_val = r1_int & mask;
        }
            break;
        case BFI: {
            int32_t mask = ~(0xffffffff << r4_int) << r3_int;
            mask = ~mask;
            r1_int &= mask;
            destination_val = r1_int | (r2_int << r3_int);
        }
            break;
        case BMSK: {
            int32_t mask = ~r4_int << r3_int;
            mask = ~mask;
            r1_int &= mask;
            destination_val = r1_int | (r2_int << r3_int);
        }
            break;
        case BREV:
            destination_val = __brev(r1_int);
            break;
        case FLO:
            destination_val = 32 - __clz(r1_int);
            break;
        case IABS:
            destination_val = abs(r1_int);
            break;
        case IADD32I:
        case IADD:
            destination_val = r1_int + r2_int;
            break;
        case IADD3:
            destination_val = r1_int + r2_int + r3_int;
            break;
        case ICMP:
            if (r1_int == r2_int)
                destination_val = r3_int;
            else
                destination_val = r4_int;
            break;
        case IDP:
        case IDP4A:
        case IMAD:
        case IMAD32I:
        case IMADSP: // FIXME: find out what is different here
            destination_val = r1_int * r2_int + r3_int;
            break;
        case ISET:
            // compares and set to 0 or 1 the output register
            if (r1_int == r2_int) {
                destination_val = 0;
            } else {
                destination_val = 1;
            }
            break;
        case IMUL:
        case IMUL32I:
            destination_val = r1_int * r2_int;
            break;
        case ISAD:
            destination_val = abs(r1_int - r2_int) + abs(r3_int - r4_int);
            break;
        case ISCADD:
        case ISCADD32I:
            destination_val = r1_int + r2_int;
            break;
        case LOP:
        case LOP32I:
            destination_val = r1_int & r2_int;
            break;
        case LOP3:
            destination_val = r1_int & r2_int & r3_int;
            break;
        case SHF:
            destination_val = __funnelshift_l(r1_int, r2_int, r2_int);
            break;
        case SHL:
            destination_val = r1_int << r2_int;
            break;
        case SHR:
            destination_val = r1_int >> r2_int;
            break;
        case XMAD:
            destination_val = r1_int * r2_int + r3_int;
            break;
        default:
            assert_gpu(false, "Not implemented", verbose);
            break;
    }
    if (is_float) {
        destination_val = __float_as_int(destination_val_float);
    }
    return destination_val;
}


DEVICE_FUNCTION_
int64_t define_opcode_behavior_64bits(InstructionType instruction_type, int64_t original_r0,
                                      int64_t r1_uint, int64_t r2_uint, int64_t r3_uint, int64_t r4_uint,
                                      bool is_float) {
    double destination_val_double = 0;
    double r1_double = __longlong_as_double(r1_uint);
    double r2_double = __longlong_as_double(r2_uint);
    double r3_double = __longlong_as_double(r3_uint);
    int64_t destination_val = original_r0;

    switch (instruction_type) {
        case DADD:
        case DFMA:
        case DMNMX:
        case DMUL:
        case DSET:
        case DSETP:
            break;
    }

    if (is_float) {
        destination_val = __double_as_longlong(destination_val_double);
    }
    return destination_val;
}
