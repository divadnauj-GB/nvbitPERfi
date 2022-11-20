#include <sstream>
#include <csignal>
#include <unordered_set>
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
std::string simulation_end_result;

pthread_mutex_t mutex;

__managed__ InjectionInfo inj_info;
__managed__ unsigned long long count_activations_inst[NUM_ISA_INSTRUCTIONS];

std::string last_kernel, last_instruction_sass_str;
uint64_t last_pc_offset;
uint32_t current_instruction_opcode;

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

void update_inst_counters() {
    static bool update_flag = false;
    if (!update_flag) {
        update_flag = true;
        cudaDeviceSynchronize();
        std::fill(count_activations_inst, count_activations_inst + NUM_ISA_INSTRUCTIONS, 0);
        cudaDeviceSynchronize();
    }
}

uint32_t generate_current_instruction_type(const uint32_t current_opcode) {
    constexpr InstructionType possible_ops[] = {
            FADD, FADD32I, FCMP, FFMA, FFMA32I, FMUL, FMUL32I,
            BFE, BFI, BMSK, BREV, FLO, IABS, IADD, IADD32I,
            IADD3, ICMP, IDP, IDP4A, IMAD, IMAD32I, IMADSP,
            IMUL, IMUL32I, ISAD, ISCADD, ISCADD32I, LOP, LOP32I,
            LOP3, SHF, SHL, SHR, XMAD, ISET, FSET
    };
    std::vector<uint32_t> weights;
    for (auto i: possible_ops) {
        if (i == current_opcode)
            weights.push_back(0);
        else
            weights.push_back(1);
    }

    // gets 'entropy' from device that generates random numbers itself
    // to seed a mersenne twister (pseudo) random generator
    static std::mt19937 generator(std::random_device{}());
    static std::discrete_distribution<> distribution(weights.begin(), weights.end());
    return possible_ops[distribution(generator)];
}

int get_max_regs(CUfunction func) {
    int max_regs = -1;
    CUDA_SAFECALL(cuFuncGetAttribute(&max_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, func))
    return max_regs;
}


void report_kernel_results() {
    assert_condition(fout.good(), "Output file " + inj_output_filename + " not good for writing");
    fout << "Kernel name: " << last_kernel << "; kernel Index: " << kernel_id;
    if (inj_info.num_activations == 0)
        fout << "; ErrorInjected: False";
    else
        fout << "; ErrorInjected: True";
    fout << "; SmID: " << inj_info.sm_id
         << "; SchID: " << inj_info.warp_group
         << "; WarpID: " << inj_info.warp_id
         << "; LastPCOffset: 0x" << std::hex << last_pc_offset << std::dec
         << "; LastOpcode: " << current_instruction_opcode
         << "; TotErrAct: " << inj_info.num_activations << std::endl;

    // Write the activation per instructions
    std::string activation_string = "perInstructionActivations:";
    for (auto i = 0; i < NUM_ISA_INSTRUCTIONS; i++) {
        if (count_activations_inst[i]) {
            activation_string += " #" + std::string(instTypeNames[i]) + ":" + std::to_string(count_activations_inst[i]);
        }
    }
    activation_string += ";\n";
    verbose_printf(activation_string, "\n");
    fout << activation_string;
    fout << simulation_end_result << std::endl;
}

void report_summary_results() {
    fout << "=================================================================================" << std::endl;
    fout << "Final Report" << std::endl;
    fout << "=================================================================================" << std::endl;
    fout << "Report_Summary: ";
    report_kernel_results();
}


void sig_int_handler(int sig) {
    signal(sig, SIG_IGN); // disable Ctrl-C
    assert_condition(fout.good(), "Output file " + inj_output_filename + " not good for writing");
    fout << "=================================================================================" << std::endl;
    fout << "Report for: " << last_kernel << "; kernel Index: " << kernel_id << std::endl;
    fout << "=================================================================================" << std::endl;
    fout << ":::NVBit-inject-error; ERROR FAIL Detected Singal SIGKILL;" << std::endl;
    report_kernel_results();
    simulation_end_result = "; SimEndRes:::ERROR FAIL Detected Singal SIGKILL::: ";
    fout << inj_info << std::endl;
    fout.flush();
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
    open_output_file(inj_output_filename);
    assert_condition(fout.good(), "Could not open output file" + inj_output_filename);

    signal(SIGINT, sig_int_handler); // install Ctrl-C handler
    verbose_printf("nvbit_at_init:end\n");
}


void insert_instrumentation_for_icoc(Instr *i, int dest_GPR_num, int num_dest_GPRs, uint32_t is_float,
                                     uint32_t replace_instruction_opcode, int num_operands) {
//    fout << "; current opcode:" << i->getSass()
//         << "; replace_instruction_opcode_num:" << replace_instruction_opcode
//         << "; num_operands:" << num_operands
//         << "; last_inst:" << last_instruction_sass_str
//         << "; last_pc_offset:" << last_pc_offset << std::endl;
    nvbit_insert_call(i, "inject_error_icoc", IPOINT_AFTER);
    nvbit_add_call_arg_const_val64(i, uint64_t(&inj_info));
    nvbit_add_call_arg_const_val64(i, uint64_t(&verbose_device));
    nvbit_add_call_arg_const_val64(i, uint64_t(count_activations_inst));
    // destination GPR register number
    nvbit_add_call_arg_const_val32(i, dest_GPR_num);
    // number of destination GPR registers
    nvbit_add_call_arg_const_val32(i, num_dest_GPRs);
    // Put if it is float or not
    nvbit_add_call_arg_const_val32(i, is_float);
    // Put last opcode index
    nvbit_add_call_arg_const_val32(i, current_instruction_opcode);
    // Put the next opcode index
    nvbit_add_call_arg_const_val32(i, replace_instruction_opcode);
    //  put the size of the operands at the end of the var list
    nvbit_add_call_arg_const_val32(i, num_operands);
    assert_condition(num_operands <= MAX_OPERANDS_NUM,
                     "More than " + std::to_string(MAX_OPERANDS_NUM) + "operands not managed");

    /* iterate on the operands */
//                        auto mem_id = 0;
    for (auto operand_i = num_dest_GPRs; operand_i < num_operands; operand_i++) {
        /* get the operand_i "i" */
        const InstrType::operand_t *op = i->getOperand(operand_i);
        InstrType::OperandType operand_type = op->type;
        /** Always put in the following order
         * 1 if the operand is valid const 32bits (0 or 1)
         * 2 operand val, can be 32 bits or mem ref 64 bits
         */
        switch (operand_type) {
            case InstrType::OperandType::REG: {
                nvbit_add_call_arg_const_val32(i, 1, true);
                nvbit_add_call_arg_reg_val(i, op->u.reg.num, true);
                break;
            }
            case InstrType::OperandType::CBANK: {
                nvbit_add_call_arg_const_val32(i, 1, true);
                if (op->u.cbank.has_imm_offset) {
                    nvbit_add_call_arg_cbank_val(i, op->u.cbank.id, op->u.cbank.imm_offset, true);
                } else {
                    nvbit_add_call_arg_cbank_val(i, op->u.cbank.id, op->u.cbank.reg_offset, true);
                }
                break;
            }
            case InstrType::OperandType::IMM_UINT64: {
                nvbit_add_call_arg_const_val32(i, 1, true);
                nvbit_add_call_arg_const_val32(i, uint32_t(op->u.imm_uint64.value), true);
                break;
            }
            case InstrType::OperandType::IMM_DOUBLE: {
                nvbit_add_call_arg_const_val32(i, 1, true);
                auto data_val = float((*(double *) &op->u.imm_double.value));
                auto *const_to_variadic = (uint32_t *) &data_val;
                nvbit_add_call_arg_const_val32(i, *const_to_variadic, true);
                break;
            }
            case InstrType::OperandType::MREF:
            case InstrType::OperandType::GENERIC:
            case InstrType::OperandType::UREG:
            case InstrType::OperandType::UPRED:
            case InstrType::OperandType::PRED: {
                nvbit_add_call_arg_const_val32(i, 0, true);
                nvbit_add_call_arg_const_val32(i, 0, true);
                break;
            }
        }
    }
}

void insert_instrumentation_for_iio(Instr *i, int dest_GPR_num, int num_dest_GPRs, int num_operands) {
    static std::mt19937 generator(std::random_device{}());
    static std::uniform_int_distribution<uint32_t> distribution(0, 0xFFFFFFFF);
    for (auto operand_i = num_dest_GPRs; operand_i < num_operands; operand_i++) {
        /* get the operand_i "i" */
        const InstrType::operand_t *op = i->getOperand(operand_i);
        if (op->type == InstrType::OperandType::IMM_UINT64 || op->type == InstrType::OperandType::IMM_DOUBLE) {
            // Generates a random mask for the injection in the output
            uint32_t destination_mask = distribution(generator);
            // instrument this instruction and stops the loop
            nvbit_insert_call(i, "inject_error_iio", IPOINT_AFTER);
            nvbit_add_call_arg_const_val64(i, uint64_t(&inj_info));
            nvbit_add_call_arg_const_val64(i, uint64_t(&verbose_device));
            nvbit_add_call_arg_const_val64(i, uint64_t(count_activations_inst));
            // destination GPR register number
            nvbit_add_call_arg_const_val32(i, dest_GPR_num);
            // number of destination GPR registers
            nvbit_add_call_arg_const_val32(i, num_dest_GPRs);
            // Put last opcode index
            nvbit_add_call_arg_const_val32(i, current_instruction_opcode);
            // Put the final value destination
            nvbit_add_call_arg_const_val32(i, destination_mask);
//            fout << "; destination_mask:" << destination_mask
//                 << "; num_operands:" << num_operands
//                 << "; last_inst:" << last_instruction_sass_str
//                 << "; last_pc_offset:" << last_pc_offset << std::endl;
            break;
        }
    }
}

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    // For this version of the injection we don't cover uniform instructions
    static const auto forbidden_instructions = {
            R2UR, REDUX, S2UR, UBMSK, UBREV, UCLEA, UF2FP, UFLO, UIADD3, UIADD3_64, UIMAD, UISETP, ULDC, ULEA, ULOP,
            ULOP3, ULOP32I, UMOV, UP2UR, UPLOP3, UPOPC, UPRMT, UPSETP, UR2UP, USEL, USGXT, USHF, USHL, USHR, VOTEU
    };
    inj_info.parse_params(inj_input_filename, verbose);  // injParams are updated based on injection seed file
    update_verbose();
    update_inst_counters();

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
        assert_condition(fout.good(), "Output file " + inj_output_filename + " not opened");

        fout << "=================================================================================" << std::endl;
        fout << "The Instrumentation step Begins Here: " << kname << std::endl;
        fout << "=================================================================================" << std::endl;
        /* Get the vector of instruction composing the loaded CUFunction "func" */
        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

        int max_regs = get_max_regs(f);
        fout << "Inspecting: " << kname << ";num_static_instrs: " << instrs.size() << ";max_regs: " << max_regs << "("
             << max_regs << ")" << std::endl;
        for (auto i: instrs) {
            std::string opcode = i->getOpcode();
            std::string inst_type_str = extractInstType(opcode);
            int inst_type = instTypeNameMap[inst_type_str];
            verbose_printf("extracted inst_type: ", inst_type_str, " index of inst_type: ",
                           instTypeNameMap[inst_type_str], "\n");

//            if (inst_type == inj_info.instruction_type_in || inj_info.instruction_type_in == NUM_ISA_INSTRUCTIONS)
            if (std::count(forbidden_instructions.begin(), forbidden_instructions.end(), inst_type) == 0) {
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
                        auto op_group = getOpGroupNum(inst_type);
                        num_dest_GPRs = (op_group == G_FP64) ? 2 : 1;

                        int sz_str = extractSize(opcode);
                        if (sz_str == 128) {
                            num_dest_GPRs = 4;
                        } else if (sz_str == 64) {
                            num_dest_GPRs = 2;
                        }

                        // Save last instrumented instruction info
                        current_instruction_opcode = inst_type;
                        last_pc_offset = i->getOffset();
                        last_instruction_sass_str = i->getSass();
                        auto is_float = uint32_t(op_group == G_FP32);
                        auto replace_instruction_opcode = generate_current_instruction_type(current_instruction_opcode);
                        auto num_operands = i->getNumOperands();
                        if (inj_info.is_iio_fault_model) {
                            insert_instrumentation_for_iio(i, dest_GPR_num, num_dest_GPRs, num_operands);
                        } else {
                            insert_instrumentation_for_icoc(i, dest_GPR_num, num_dest_GPRs, is_float,
                                                            replace_instruction_opcode, num_operands);
                        }
                    }
                    // If an instruction has two destination registers, not handled!! (TODO: Fix later)
                }
            }
        }
        fout << "=================================================================================" << std::endl;
        fout << "The Instrumentation step Stops Here: " << kname << std::endl;
        fout << "=================================================================================" << std::endl;
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
                fout << "================================================================================="
                     << std::endl;
                fout << "Running instrumented Kernel: " << removeSpaces(nvbit_get_func_name(ctx, p->f))
                     << "; kernel Index: " << kernel_id << std::endl;
                fout << "..............." << std::endl;
                fout << "================================================================================="
                     << std::endl;
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
                fout << "================================================================================="
                     << std::endl;
                fout << "Report for: " << kname << "; kernel Index: " << kernel_id << std::endl;
                fout << "================================================================================="
                     << std::endl;
                fout << "Injection data; ";
                fout << "index: " << kernel_id << ";";
                fout << "kernel_name: " << kname << ";";
                fout << "ctas: " << num_ctas << ";";
                fout << " LastPCOffset: " << last_pc_offset << ";";
                fout << " LastOpcode: " << current_instruction_opcode << ";";
                fout << " LastInstSASS: " << last_instruction_sass_str << ";";
                fout << inj_info << std::endl;
//                dump_count_activations_inst();
                last_kernel = kname;
                report_kernel_results();

                if (cudaSuccess != le) {
                    assert_condition(fout.good(), "Output file " + inj_output_filename + " not opened");

                    simulation_end_result =
                            "; SimEndRes:::ERROR FAIL in kernel execution (" + std::string(cudaGetErrorString(le)) +
                            "); ";
                    assert_condition(false,
                                     "; SimEndRes:::ERROR FAIL in kernel execution (" +
                                     std::string(cudaGetErrorString(le)) + "); ");
                }
                verbose_printf("\n index: ", kernel_id, "; kernel_name: ", kname, "\n");
                simulation_end_result = "; SimEndRes:::PASS without fails:::";
                kernel_id++; // always increment kernel_id on kernel exit

                // cudaDeviceSynchronize();
                pthread_mutex_unlock(&mutex);
            }
        }
    }
}

void nvbit_at_term() {
    fout << "Report_Summary: ;";
    fout << "kernel_index: " << kernel_id << ";";
    fout << "kernel_name: " << last_kernel << ";";
    fout << " LastPCOffset: " << last_pc_offset << ";";
    fout << " LastOpcode: " << current_instruction_opcode << ";";
    fout << " LastInstSASS: " << last_instruction_sass_str << ";";
    fout << inj_info << std::endl;

    cudaDeviceSynchronize();
//    dump_count_activations_inst();
    report_summary_results();
}
