#include <sstream>
#include <csignal>
#include <unordered_set>
#include <map>
#include <algorithm>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"

#include "globals.h"

#include "pf_injector.h"

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

bool verbose;
__managed__ int verbose_device;
int limit;

// injection parameters input filename: This file is created the script
// that launched error injections
std::string inj_input_filename;
// Output log file
std::string inj_output_filename;

pthread_mutex_t mutex;

__managed__ InjectionInfo inj_info;

template<class ... Ts>
void verbose_printf(Ts &&... inputs) {
    if (verbose) {
        // Do things in your "loop" lambda
        ([&] {
            std::cout << inputs;
        }(), ...);
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

int get_max_regs(CUfunction func) {
    int max_regs = -1;
    CUDA_SAFECALL(cuFuncGetAttribute(&max_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, func))
    return max_regs;
}

void sig_int_handler(int sig) {
    signal(sig, SIG_IGN); // disable Ctrl-C

//    std::ofstream fout(inj_output_filename);
    if (fout.good()) {
        fout << ":::NVBit-inject-error; ERROR FAIL Detected Singal SIGKILL;";
        fout << " num_activations: " << inj_info.num_activations << ":::";
        fout.flush();
    }
    assert_condition(false, "Ctrl-C pressed, stopping execution!");
}


/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
// DO NOT USE UVM (__managed__) variables in this function
void nvbit_at_init() {
    /* just make sure all managed variables are allocated on GPU */
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    // Default values
    verbose = false;
    inj_input_filename = "nvbitfi-injection-info.txt";
    inj_output_filename = "nvbitfi-injection-log-temp.txt";
    limit = INT_MAX;

    /* we get some environment variables that are going to be use to selectively
     * instrument (within a interval of kernel indexes and instructions). By
     * default, we instrument everything. */
    auto env_tool_verbose_ptr = std::getenv("TOOL_VERBOSE");
    if (env_tool_verbose_ptr) {
        verbose = std::stoi(std::string(env_tool_verbose_ptr));
    }

    auto env_inj_info_in_file_ptr = std::getenv("INPUT_INJECTION_INFO");
    if (env_inj_info_in_file_ptr) {
        inj_input_filename = env_inj_info_in_file_ptr;
    }

    auto env_inj_info_out_file_ptr = std::getenv("OUTPUT_INJECTION_LOG");
    if (env_inj_info_out_file_ptr) {
        inj_output_filename = env_inj_info_out_file_ptr;
    }

    auto env_instrumentation_limit = std::getenv("INSTRUMENTATION_LIMIT");
    if (env_instrumentation_limit) {
        limit = std::stoi(std::string(env_instrumentation_limit));
    }

    initInstTypeNameMap();
    fout.open(inj_output_filename);
    assert_condition(fout.good(), "Could not open output file" + inj_output_filename);

    signal(SIGINT, sig_int_handler); // install Ctrl-C handler
    verbose_printf("nvbit_at_init:end\n");
}


void instrument_function_if_needed(CUcontext ctx, CUfunction func) {

    inj_info.parse_params(inj_input_filename, verbose);  // injParams are updated based on injection seed file
    update_verbose();

    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);

    // Open the output file
//    std::ofstream fout(inj_output_filename);

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
        /* Get the vector of instruction composing the loaded CUFunction "func" */
        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

        int max_regs = get_max_regs(f);
        assert_condition(fout.good(), "Output file " + inj_output_filename + " not opened");
        fout << "Inspecting: " << kname << ";num_static_instrs: " << instrs.size() << ";max_regs: " << max_regs << "("
             << max_regs << ")" << std::endl;
        for (auto i: instrs) {
            std::string opcode = i->getOpcode();
            std::string inst_type_str = extractInstType(opcode);
            int inst_type = instTypeNameMap[inst_type_str];
            verbose_printf("extracted inst_type: ", inst_type_str, " index of inst_type: ",
                           instTypeNameMap[inst_type_str], "\n");

//            if (inst_type == inj_info.instruction_type_in || inj_info.instruction_type_in == NUM_ISA_INSTRUCTIONS)
            {
                verbose_printf("instruction selected for instrumentation: ");
                if (verbose) {
                    i->print();
                }

                // Tokenize the instruction
                std::vector<std::string> tokens;
                std::string buf; // a buffer string
                std::stringstream ss(i->getSass()); // Insert the string into a stream
                while (ss >> buf)
                    tokens.push_back(buf);

                int dest_GPR_num = -1;
                int num_dest_GPRs = 0;

                if (tokens.size() > 1) { // an actual instruction that writes to either a GPR or PR register
                    verbose_printf("num tokens = ", tokens.size(), "\n");
                    int start = 1; // first token is opcode string
                    if (tokens[0].find('@') != std::string::npos) { // predicated instruction, ignore first token
                        start = 2; // first token is predicate and 2nd token is opcode
                    }

                    // Parse the first operand - this is the first destination
                    int reg_num_1 = -1;
                    int regtype = extractRegNo(tokens[start], reg_num_1);
                    if (regtype == 0) { // GPR reg
                        dest_GPR_num = reg_num_1;
                        num_dest_GPRs = (getOpGroupNum(inst_type) == G_FP64) ? 2 : 1;

                        int sz_str = extractSize(opcode);
                        if (sz_str == 128) {
                            num_dest_GPRs = 4;
                        } else if (sz_str == 64) {
                            num_dest_GPRs = 2;
                        }

                        nvbit_insert_call(i, "inject_error", IPOINT_AFTER);
                        nvbit_add_call_arg_const_val64(i, (uint64_t) &inj_info);
                        nvbit_add_call_arg_const_val64(i, (uint64_t) &verbose_device);

                        nvbit_add_call_arg_const_val32(i, dest_GPR_num); // destination GPR register number
                        if (dest_GPR_num != -1) {
                            nvbit_add_call_arg_reg_val(i, dest_GPR_num); // destination GPR register val
                        } else {
                            nvbit_add_call_arg_const_val32(i, (unsigned int) -1); // destination GPR register val
                        }
                        nvbit_add_call_arg_const_val32(i, num_dest_GPRs); // number of destination GPR registers

                        nvbit_add_call_arg_const_val32(i, max_regs); // max regs used by the inst info
                        /**************************************************************************
                         * Edit: trying to load all the inst input vals
                         * **************************************************************************/
                        // Add all registers to the function call stack
                        std::vector<int> input_reg_num_vector;
                        /* iterate on the operands */
                        for (auto operand_i = 0; operand_i < i->getNumOperands(); operand_i++) {
                            /* get the operand_i "i" */
                            const InstrType::operand_t *op = i->getOperand(operand_i);
                            switch (op->type) {
                                case InstrType::OperandType::REG:
                                    input_reg_num_vector.push_back(op->u.reg.num);
                                    break;
                                case InstrType::OperandType::IMM_UINT64:
                                    std::cout << "IMM_UINT64 OP:";
                                    i->print();
                                    break;
                                case InstrType::OperandType::IMM_DOUBLE:
                                    std::cout << "IMM_DOUBLE OP:";
                                    i->print();
                                    break;
                                case InstrType::OperandType::PRED:
                                    std::cout << "PRED OP:";
                                    i->print();
                                    break;
                                case InstrType::OperandType::UREG:
                                    std::cout << "UREG OP:";
                                    i->print();
                                    break;
                                case InstrType::OperandType::UPRED:
                                    std::cout << "UPRED OP:";
                                    i->print();
                                    break;
                                case InstrType::OperandType::CBANK:
                                    std::cout << "CBANK OP:";
                                    i->print();
                                    break;
                                case InstrType::OperandType::MREF:
                                    std::cout << "MREF OP:";
                                    i->print();
                                    break;
                                case InstrType::OperandType::GENERIC:
                                    std::cout << "GENERIC OP:";
                                    i->print();
                                    break;
                            }
                        }
                        //  put the size of the operands at the end of the var list
                        nvbit_add_call_arg_const_val32(i, input_reg_num_vector.size());
                        //REGs FIRST as I will read them before the cbank values
                        for (int num: input_reg_num_vector) {
                            /* last parameter tells it is a variadic parameter passed to
                             * the instrument function record_reg_val() */
                            nvbit_add_call_arg_reg_val(i, num, true);
                        }

                    }
                    // If an instruction has two destination registers, not handled!! (TODO: Fix later)
                }
            }
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
    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch ||
        cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid ||
        cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel) {

//        std::ofstream fout(inj_output_filename);

        /* cast params to cuLaunch_params since if we are here we know these are
         * the right parameters type */
        auto *p = (cuLaunch_params *) params;

        if (!is_exit) {
            pthread_mutex_lock(&mutex);
            if (kernel_id < limit) {
                instrument_function_if_needed(ctx, p->f);
                // cudaDeviceSynchronize();

                nvbit_enable_instrumented(ctx, p->f, true); // run the instrumented version
                // cudaDeviceSynchronize();
            } else {
                nvbit_enable_instrumented(ctx, p->f, false); // do not use the instrumented version
            }

        } else {
            if (kernel_id < limit) {
                verbose_printf("is_exit\n");
                cudaDeviceSynchronize();

                cudaError_t le = cudaGetLastError();

                std::string kname = removeSpaces(nvbit_get_func_name(ctx, p->f));
                unsigned num_ctas = 0;
                if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                    cbid == API_CUDA_cuLaunchKernel) {
                    auto *p2 = (cuLaunchKernel_params *) params;
                    num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
                }
                assert_condition(fout.good(), "Output file " + inj_output_filename + " not opened");

                fout << "Injection data; ";
                fout << "index: " << kernel_id << ";";
                fout << "kernel_name: " << kname << ";";
                fout << "ctas: " << num_ctas << ";";
                fout << inj_info << std::endl;
//                fout << "selected SM: " << inj_info.sm_id << ";";
//                fout << "selected Lane: " << inj_info.lane_id << ";";
//                fout << "selected Mask: " << inj_info.mask << ";";
//                fout << "selected InstType: " << inj_info.instruction_type_in << ";";
//                fout << "num_activations: " << inj_info.num_activations << std::endl;

                if (cudaSuccess != le) {
                    assert_condition(fout.good(), "Output file " + inj_output_filename + " not opened");

                    fout << "ERROR FAIL in kernel execution (" << cudaGetErrorString(le) << "); " << std::endl;
                    assert_condition(false,
                                     "ERROR FAIL in kernel execution (" + std::string(cudaGetErrorString(le)) + "); ");
                }
                verbose_printf("\n index: ", kernel_id, "; kernel_name: ", kname, "\n");
                kernel_id++; // always increment kernel_id on kernel exit

                // cudaDeviceSynchronize();
                pthread_mutex_unlock(&mutex);
            }
        }
    }
}

void nvbit_at_term() {
    // nothing to do here.
}
