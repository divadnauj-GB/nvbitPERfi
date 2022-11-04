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

    //TODO: Change this function to select if thread is eligible to fault injection
    //      based on the RTL fault types
    auto sm_id = get_smid();
    // This is not the selected SM. No need to proceed.
    if (sm_id != inj_info->sm_id)
        return false;

    auto warp_id = get_warpid();
    if ((warp_id % 4) != inj_info->warp_group) {
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
int32_t define_opcode_behavior_32bits(InstructionType instruction_type, int32_t original_r0,
                                      int32_t r1_uint, int32_t r2_uint, int32_t r3_uint, int32_t r4_uint,
                                      bool is_float, bool verbose);

extern "C" __device__ __noinline__
void inject_error(uint64_t injection_info_ptr, uint64_t verbose_device_ptr,
                  int dest_GPR_num, int reg_val, int num_dest_GPRs, int max_regs,
                  int input_registers_num, ...) {
    auto *inj_info = (InjectionInfo *) injection_info_ptr;
    uint32_t verbose_device = *((uint32_t *) verbose_device_ptr);
    assert_gpu(num_dest_GPRs > 0, "num_dest_GPRs equals to 0", verbose_device);
    assert_gpu(input_registers_num != 4, "More than 4 input registers not managed", verbose_device);

    int32_t dest_reg_before_val = nvbit_read_reg(dest_GPR_num); // read the register value
    // FIXME: get the instrumentation regs here
    va_list vl;
    va_start(vl, input_registers_num);
    int32_t r1 = va_arg(vl, int32_t);
    int32_t r2 = va_arg(vl, int32_t);
    int32_t r3 = va_arg(vl, int32_t);
    int32_t r4 = va_arg(vl, int32_t);
    va_end(vl);

    uint32_t dest_reg_after_val = dest_reg_before_val;

    if (is_fault_injection_necessary(inj_info)) {
        switch (inj_info->icoc_subpartition) {
            case ICOCSubpartition::SCHEDULER: {
                dest_reg_after_val = define_opcode_behavior_32bits(
                        inj_info->instruction_type_out, dest_reg_before_val, r1, r2, r3, r4,
                        (FADD <= inj_info->instruction_type_out) && (inj_info->instruction_type_out <= FSWZADD),
                        verbose_device);
                break;
            }
            case ICOCSubpartition::DECODER:
            case ICOCSubpartition::FETCH:
            default:
//                dest_reg_after_val = dest_reg_before_val ^ inj_info->mask;
                break;
        }
        if (DUMMY == 0) {
            modify_destination_register(dest_GPR_num, inj_info, verbose_device, dest_reg_before_val,
                                        dest_reg_after_val);
        }
    }
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

DEVICE_FUNCTION_
int32_t define_opcode_behavior_32bits(InstructionType instruction_type, int32_t original_r0,
                                      int32_t r1_uint, int32_t r2_uint, int32_t r3_uint, int32_t r4_uint,
                                      bool is_float, bool verbose) {
    float destination_val_float = 0;
    float r1_float = __int_as_float(r1_uint);
    float r2_float = __int_as_float(r2_uint);
    float r3_float = __int_as_float(r3_uint);
    float r4_float = __int_as_float(r4_uint);

    int32_t destination_val = original_r0;

    switch (instruction_type) {
        // Floating point 32 instructions (0-13)
        case FADD:
        case FADD32I:
            destination_val_float = r1_float + r2_float;
            break;
        case FCHK:
            assert_gpu(false, "Not implemented", verbose);
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
        case FMNMX:
            assert_gpu(false, "Not implemented", verbose);
            break;
        case FMUL:
        case FMUL32I:
            destination_val_float = r1_float * r2_float;
            break;
        case FSEL:
            destination_val_float = r1_float;
            break;
        case FSET:
        case FSETP:
        case FSWZ:
        case FSWZADD:
            assert_gpu(false, "Not implemented", verbose);
            break;


            // Integer Instructions (35-67)
        case BFE: {
            int mask = ~(0xffffffff << r3_uint);
            if (r2_uint > 0)
                return (r1_uint >> (r2_uint - 1)) & mask;
            else
                return r1_uint & mask;
        }
            break;
        case BFI: {
            uint mask = ~(0xffffffff << r4_uint) << r3_uint;
            mask = ~mask;
            r1_uint &= mask;
            destination_val = r1_uint | (r2_uint << r3_uint);
        }
            break;
        case BMSK: {
            uint mask = ~r4_uint << r3_uint;
            mask = ~mask;
            r1_uint &= mask;
            destination_val = r1_uint | (r2_uint << r3_uint);
        }
            break;
        case BREV:
            destination_val = __brev(r1_uint);
            break;
        case FLO:
            destination_val = 32 - __clz(r1_uint);
            break;
        case IABS:
            destination_val = abs(r1_uint);
            break;
        case IADD32I:
        case IADD:
            destination_val = r1_uint + r2_uint;
            break;
        case IADD3:
            destination_val = r1_uint + r2_uint + r3_uint;
            break;
        case ICMP:
            if (r1_uint == r2_uint)
                destination_val = r3_uint;
            else
                destination_val = r4_uint;
            break;
        case IDP:
        case IDP4A:
        case IMAD:
        case IMAD32I:
        case IMADSP: // FIXME: find out what is different here
            destination_val = r1_uint * r2_uint + r3_uint;
            break;
        case IMNMX:
        case IPA:
        case ISET:
        case ISETP:
            assert_gpu(false, "Not implemented", verbose);
            break;
        case IMUL:
        case IMUL32I:
            destination_val = r1_uint * r2_uint;
            break;
        case ISAD:
            destination_val = abs(r1_uint - r2_uint) + abs(r3_uint - r4_uint);
            break;
        case ISCADD:
        case ISCADD32I:
            destination_val = r1_uint + r2_uint;
            break;
        case LEA:
            assert_gpu(false, "Not implemented", verbose);
            break;
        case LOP:
        case LOP32I:
            destination_val = r1_uint & r2_uint;
            break;
        case LOP3:
            destination_val = r1_uint & r2_uint & r3_uint;
            break;
        case POPC:
            assert_gpu(false, "Not implemented", verbose);
            break;
        case SHF:
            destination_val = __funnelshift_l(r1_uint, r2_uint, r2_uint);
            break;
        case SHL:
            destination_val = r1_uint << r2_uint;
            break;
        case SHR:
            destination_val = r1_uint >> r2_uint;
            break;
        case XMAD:
            destination_val = r1_uint * r2_uint + r3_uint;
            break;

    }
    if (is_float) {
        destination_val = __float_as_int(destination_val_float);
    }
    return destination_val;
}

