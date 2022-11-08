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
void modify_destination_register(int dest_GPR_num, InjectionInfo *inj_info, uint32_t verbose_device,
                                 uint32_t dest_reg_before_val, uint32_t dest_reg_after_val) {
//    uint32_t dest_reg_before_val = nvbit_read_reg(dest_GPR_num); // read the register value
//    uint32_t dest_reg_after_val = dest_reg_before_val;

//        dest_reg_after_val = dest_reg_before_val ^ inj_info->mask;
    nvbit_write_reg(dest_GPR_num, (int) dest_reg_after_val);

    // updating counter/flag to check whether the error was injected
    if (verbose_device)
        printf("register=%d, before=0x%x, after=0x%x, expected_after=0x%x\n", dest_GPR_num, dest_reg_before_val,
               nvbit_read_reg(dest_GPR_num), dest_reg_after_val);
//    inj_info->error_injected = true;
    atomicAdd((unsigned long long *) &inj_info->num_activations, 1LL);
}

DEVICE_FUNCTION_
void fill_values_from_variadic(InjectionInfo *inj_info, int num_operands, va_list vl, int32_t *data) {
    auto i = 0;
    for (auto &op_descriptor: inj_info->operand_list) {
        // end the list
        if (op_descriptor.is_this_operand_valid) {
            switch (op_descriptor.operand_type) {
                case InstrType::OperandType::IMM_UINT64:
                case InstrType::OperandType::IMM_DOUBLE:
                case InstrType::OperandType::GENERIC:
                    data[i] = 0; // Not managed
                    break;
                case InstrType::OperandType::REG:
                case InstrType::OperandType::UREG:
                case InstrType::OperandType::PRED:
                case InstrType::OperandType::UPRED:
                case InstrType::OperandType::CBANK:
                    data[i] = va_arg(vl, int32_t);
                    break;
                case InstrType::OperandType::MREF: {
                    uint64_t mem_adr = va_arg(vl, uint64_t);
                    data[i] = 0;
                    auto *mem_data = (int32_t *) mem_adr;
                    data[i] = *mem_data;
                }
                    break;
            }
            i++;
        }
        if (num_operands-- < 0)
            break;
    }
}

DEVICE_FUNCTION_
int32_t define_opcode_behavior_32bits(InstructionType instruction_type, int32_t original_r0,
                                      int32_t r1_int, int32_t r2_int, int32_t r3_int, int32_t r4_int,
                                      bool is_float, bool verbose);

extern "C" __device__ __noinline__
void inject_error(uint64_t injection_info_ptr, uint64_t verbose_device_ptr, int dest_GPR_num, int num_dest_GPRs,
                  int is_float, int input_registers_num, ...) {
    auto *inj_info = (InjectionInfo *) injection_info_ptr;
    uint32_t verbose_device = *((uint32_t *) verbose_device_ptr);
    assert_gpu(num_dest_GPRs > 0, "num_dest_GPRs equals to 0", verbose_device);
    assert_gpu(num_dest_GPRs == 1, "num_dest_GPRs > 1", verbose_device);

    int32_t dest_reg_before_val = nvbit_read_reg(dest_GPR_num); // read the register value

    int32_t reg_data[MAX_OPERANDS_NUM] = {0};
    va_list vl;
    va_start(vl, input_registers_num);
    fill_values_from_variadic(inj_info, input_registers_num, vl, reg_data);
    va_end(vl);

    auto dest_reg_after_val = define_opcode_behavior_32bits(inj_info->instruction_type_out, dest_reg_before_val,
                                                            reg_data[0], reg_data[1], reg_data[2], reg_data[3],
                                                            bool(is_float), verbose_device);
    if (DUMMY == 0 && is_fault_injection_necessary(inj_info)) {
        modify_destination_register(dest_GPR_num, inj_info, verbose_device, dest_reg_before_val,
                                    dest_reg_after_val);
    }
}


DEVICE_FUNCTION_
int32_t define_opcode_behavior_32bits(InstructionType instruction_type, int32_t original_r0,
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
//        case FCHK:
//            assert_gpu(false, "Not implemented", verbose);
//            break;
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
//        case FMNMX:
//            assert_gpu(false, "Not implemented", verbose);
//            break;
        case FMUL:
        case FMUL32I:
            destination_val_float = r1_float * r2_float;
            break;
        case FSEL:
            destination_val_float = r1_float;
            break;
//        case FSET:
            //TODO: compares and set to 0 or 1 the output register
//        case FSETP:
//        case FSWZ:
//        case FSWZADD:
//            assert_gpu(false, "Not implemented", verbose);
//            break;
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
//        case IMNMX:
//        case IPA:
//        case ISET:
//        case ISETP:
//            assert_gpu(false, "Not implemented", verbose);
//            break;
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
//        case LEA:
//            assert_gpu(false, "Not implemented", verbose);
//            break;
        case LOP:
        case LOP32I:
            destination_val = r1_int & r2_int;
            break;
        case LOP3:
            destination_val = r1_int & r2_int & r3_int;
            break;
//        case POPC:
//            assert_gpu(false, "Not implemented", verbose);
//            break;
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
