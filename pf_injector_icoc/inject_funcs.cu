#include "nvbit_reg_rw.h"
#include "utils/utils.h"
#include "pf_injector.h"
#include "arch.h"

DEVICE_FUNCTION_
void assert_gpu_(const char *message, bool verbose, const char *file, int line) {
    if (verbose) {
        printf("ERROR GPU:%s at %s:%d\n", message, file, line);
        assert(0);
    }
}

#define assert_gpu(message, verbose) assert_gpu_(message, verbose, __FILE__, __LINE__)

DEVICE_FUNCTION_
bool is_fault_injection_necessary_here(const InjectionInfo *inj_info) {

    //TODO: Change this function to select if thread is eligible to fault injection
    //      based on the RTL fault types
    auto sm_id = get_smid();
    // This is not the selected SM. No need to proceed.
    if (sm_id != inj_info->sm_id)
        return false;

    auto lane_id = get_laneid();
    // This is not the selected Lane ID. No need to proceed.
    if (lane_id != inj_info->lane_id)
        return false;

    return true;
}


DEVICE_FUNCTION_
void default_injection(int dest_GPR_num, InjectionInfo *inj_info, uint32_t verbose_device) {
    uint32_t inj_before_val = nvbit_read_reg(dest_GPR_num); // read the register value
    uint32_t inj_after_val = inj_before_val;

    if (DUMMY == 0) {
        inj_after_val = inj_before_val ^ inj_info->mask;
        nvbit_write_reg(dest_GPR_num, (int) inj_after_val);
    }
    // updating counter/flag to check whether the error was injected
    if (verbose_device)
        printf("register=%d, before=0x%x, after=0x%x, expected_after=0x%x\n", dest_GPR_num, inj_before_val,
               nvbit_read_reg(dest_GPR_num), inj_after_val);
    inj_info->error_injected = true;
    atomicAdd((unsigned long long *) &inj_info->num_activations, 1LL);
}

DEVICE_FUNCTION_
void icoc_injection_scheduler(InjectionInfo *inj_info) {}

DEVICE_FUNCTION_
void icoc_injection_decoder(InjectionInfo *inj_info) {
    assert_gpu("NotImplemented", true);
}

DEVICE_FUNCTION_
void icoc_injection_fetch(InjectionInfo *inj_info) {
    assert_gpu("NotImplemented", true);
}

extern "C" __device__ __noinline__
void inject_error(uint64_t injection_info_ptr, uint64_t verbose_device_ptr,
                  int dest_GPR_num, int reg_val, int num_dest_GPRs, int maxRegs) {

    auto *inj_info = (InjectionInfo *) injection_info_ptr;
    uint32_t verbose_device = *((uint32_t *) verbose_device_ptr);
    assert(num_dest_GPRs > 0);

    if (is_fault_injection_necessary_here(inj_info)) {
        switch (inj_info->icoc_subpartition) {
            case ICOCSubpartition::SCHEDULER: {
                icoc_injection_scheduler(inj_info);
                break;
            }
            case ICOCSubpartition::DECODER: {
                icoc_injection_decoder(inj_info);
                break;
            }
            case ICOCSubpartition::FETCH: {
                icoc_injection_fetch(inj_info);
                break;
            }
            default:
                default_injection(dest_GPR_num, inj_info, verbose_device);
                break;
        }
    }
}
