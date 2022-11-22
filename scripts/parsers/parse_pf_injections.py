#!/usr/bin/python3
import datetime
import os
import re

import pandas as pd

NVBIT_PER_FI_ORIGINAL_PATH = "/home/carol/nvbit_release/tools/nvbitPERfi"
DATA_PATH = "data"
OUTPUT_PARSED_FILE = f"{DATA_PATH}/parsed_data.csv"
DEFAULT_LOG_HELPER_PATH = f"{DATA_PATH}/log_helper_dest/radiation-benchmarks/log"
DEFAULT_NVBITPERFI_PATH = f"{DATA_PATH}/logs"
NUM_INJECTIONS = 1000
ERROR_MODELS = ["ICOC", "IIO", "IRA"]
BENCHMARKS = ["accl", "bfs", "cfd", "gaussian", "gemm", "hotspot", "lava", "lud", "mergesort", "mxm", "nw", "quicksort"]

DUE_CAUSE_NAMES = {
    'NoDUE': "NoDUE",
    '::ERROR FAIL in kernel execution (an illegal memory access was encountered); ': "IllegalMemAccess",
    '::ERROR FAIL in kernel execution (misaligned address); ': "MisalignedAddress",
    'TIMEOUT': "Timeout",
    '::ERROR FAIL in kernel execution (operation not supported on global/shared address space); ': "OpNotSuppAddrSpace",
    '::ERROR FAIL in kernel execution (invalid program counter); ': "InvalidPC",
    '(an illegal memory access was encountered); ': "IllegalMemAccess",
    '(an illegal instruction was encountered); ': "IllegalInstruction",
    '(misaligned address); ': "MisalignedAddress"
}


def execute_cmd(cmd):
    print("EXECUTING:", cmd)
    if os.system(cmd) != 0:
        raise ValueError(f"Could not execute {cmd}")


def parse_lib_log_helper_file(log_path: str, error_model: str, app: str) -> dict:
    # ...log/2022_09_15_16_00_43_PyTorch-c100_res44_test_02_relu6-bn_200_epochs_ECC_OFF_carolinria.log
    pattern = r".*/(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_\S+_ECC_(\S+)_(\S+).log"
    m = re.match(pattern, log_path)
    if m:
        year, month, day, hour, minute, seconds, ecc, hostname = m.groups()
        with open(log_path) as log_fp:
            log_fp.readline()  # Skip the header line
            # header = re.match(r"#HEADER (.*)", log_fp.readline()).group(1)
            data_dict = dict(app=app, hostname=hostname, error_model=error_model, has_end=1, sdc=0, it=0, ker_time=0.0,
                             cuda_framework_error=0)
            has_end, has_err = False, False
            for line in log_fp:
                if "#END" in line:
                    has_end = True
                if "#ERR" in line and "CUDA Framework error" not in line:
                    has_err = True
                if "CUDA Framework error" in line:
                    data_dict["cuda_framework_error"] = 1
                sdc_m = re.match(r"#SDC Ite:(\d+) KerTime:(\S+) AccTime:(\S+) KerErr:(\d+) AccErr:(\d+)", line)

                if sdc_m:
                    assert data_dict["sdc"] == 0, f"More than one SDC per log {log_path}"
                    it, ker_time, acc_time, ker_err, acc_err = sdc_m.groups()
                    data_dict["sdc"] = 1
                    data_dict["ker_time"] = float(ker_time)
                    data_dict["it"] = int(it)
                # for masked faults
                it_m = re.match(r"#IT.*KerTime:(\S+) AccTime:.*", line)
                if it_m:
                    data_dict["ker_time"] = float(it_m.group(1))
            data_dict["has_end"] = int(has_end)
            if data_dict["cuda_framework_error"] == 1:
                data_dict["has_end"] = 0
        if data_dict["sdc"] == 0 and has_err is True:
            raise ValueError(f"Incorrect parsing {log_path}")

        return data_dict


def get_log_file_name(fi_dir):
    config_stdout = os.path.join(fi_dir, "selected_output.txt")
    # if selected_output_file is not available search in the stdout
    if os.path.isfile(config_stdout) is False:
        config_stdout = os.path.join(fi_dir, "stdout.txt")

    pattern = r"Log file path " + NVBIT_PER_FI_ORIGINAL_PATH + r"/test-apps/(\S+) - FILE:.*"
    with open(config_stdout) as fp:
        for line in fp:
            m = re.match(pattern, line)
            if m:
                return m.group(1)
    raise ValueError(config_stdout)


def get_due_cause(fi_dir: str, has_end: bool) -> str:
    nvbit_log = os.path.join(fi_dir, "nvbitfi-injection-log-temp.txt")
    with open(nvbit_log) as nv_fp:
        for line in nv_fp:
            if "ERROR FAIL Detected Singal SIGKILL" in line:
                return "TIMEOUT"
            m = re.match(r"ERROR FAIL in kernel execution (.*)", line)
            if m:
                return m.group(1)

    stderr_diff_log = os.path.join(fi_dir, "stderr.txt")
    with open(stderr_diff_log) as fp:
        for line in fp:
            m = re.match(r".*SimEndRes:(.*)", line)
            if m:
                return m.group(1)

    if has_end is False:
        raise ValueError(f"The libLoghelper file does not have end and DUE cause is not found, {stderr_diff_log}")
    return "NoDUE"


def main():
    data_list = list()
    start = datetime.datetime.now()
    for error_model in ERROR_MODELS:
        for app in BENCHMARKS:
            print("Parsing error model", error_model, "for", app)
            for injection_count in range(1, NUM_INJECTIONS + 1):
                nvbit_perfi_log_i_path = os.path.join(DEFAULT_NVBITPERFI_PATH, app, error_model, "logs",
                                                      f"{app}-mode{error_model}-icount{injection_count}")
                assert os.path.isdir(nvbit_perfi_log_i_path), f"Not a path {nvbit_perfi_log_i_path}"
                log_helper_file = get_log_file_name(fi_dir=nvbit_perfi_log_i_path)
                respective_log_helper_file = os.path.join(DATA_PATH, log_helper_file)

                data_i = parse_lib_log_helper_file(log_path=respective_log_helper_file, error_model=error_model,
                                                   app=app)
                data_i["log_file"] = log_helper_file
                data_i["due_cause"] = get_due_cause(fi_dir=nvbit_perfi_log_i_path, has_end=data_i["has_end"] == 1)
                data_list.append(data_i)

    end = datetime.datetime.now()
    print("Time spent", end - start)
    df = pd.DataFrame(data_list)
    print(df)
    df.to_csv(OUTPUT_PARSED_FILE, index=False)


if __name__ == '__main__':
    main()
