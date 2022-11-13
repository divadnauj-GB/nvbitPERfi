#!/usr/bin/python3
import json
import os
import shutil

REAL_WORKLOADS_DIR = os.path.abspath(os.getcwd())
LOG_HELPER_SOURCE_PATH = f"{REAL_WORKLOADS_DIR}/libLogHelper"
# Also used to set the makefiles
LOG_HELPER_LIB_PATH = f"{LOG_HELPER_SOURCE_PATH}/build"
LOG_HELPER_INCLUDE_PATH = f"{LOG_HELPER_SOURCE_PATH}/include"

LOG_HELPER_LOGS_DIR = f"{REAL_WORKLOADS_DIR}/log_helper_dest"

# NVBitPermaFI configuration file
REAL_WORKLOADS_PARAMETERS_FILE = f"{REAL_WORKLOADS_DIR}/../../scripts/real_workloads_parameters.py"

NVBITFI_HOME = os.environ["NVBITFI_HOME"]


def execute_cmd(cmd: str, err_message: str):
    print("EXECUTING:", cmd)
    if os.system(cmd) != 0:
        raise ValueError(f"Failed to execute CMD:{cmd}, message:{err_message}")


def build_and_set_lib_log_helper():
    rad_logs_dir = f"{LOG_HELPER_LOGS_DIR}/radiation-benchmarks/log"
    # Make sure that the logs dir exists
    if os.path.isdir(rad_logs_dir) is False:
        os.makedirs(rad_logs_dir)

    print("Removing and recreating", LOG_HELPER_SOURCE_PATH)
    if os.path.isdir(LOG_HELPER_LIB_PATH) is True:
        shutil.rmtree(LOG_HELPER_LIB_PATH, ignore_errors=True)

    os.mkdir(LOG_HELPER_LIB_PATH)
    os.chdir(LOG_HELPER_LIB_PATH)

    cmake_cmd = f"cmake -DLOGGING_TYPE=LOCAL -DWITH_PYTHON_SUPPORT=OFF"
    cmake_cmd += f" -DLOG_DIR={LOG_HELPER_LOGS_DIR} -DPATH_TO_CONFIG_FILE={LOG_HELPER_LIB_PATH} .."
    execute_cmd(cmd=cmake_cmd, err_message="PROBLEM ON CMAKE")
    execute_cmd(cmd=f"make", err_message="PROBLEM ON MAKE LOG HELPER")

    os.chdir(REAL_WORKLOADS_DIR)


def build_benchmark_with_fi_parameters(app: str, parameters: dict):
    app_full_path = f"{REAL_WORKLOADS_DIR}/{parameters['APP_DIR']}"
    os.chdir(app_full_path)
    make_cmd = f"make LOG_HELPER_PATH={LOG_HELPER_SOURCE_PATH} clean all"
    execute_cmd(cmd=make_cmd, err_message=f"Making {app}")
    # Generate golden
    app_make_parameters = " ".join([f"{k}={v}" for k, v in parameters["MAKE_PARAMETERS"].items()])
    generate_cmd = f"make {app_make_parameters} generate"
    execute_cmd(cmd=generate_cmd, err_message=f"Generate golden for {app}")


def main():
    # Build libLogHelper first
    build_and_set_lib_log_helper()
    real_workloads_dict_out = dict()
    for workload_name, workload_parameters in REAL_WORKLOADS.items():
        print("Building and setting", workload_name)
        # build_benchmark_with_fi_parameters(app=workload_name, parameters=workload_parameters)
        app_dir, app_bin = workload_parameters["APP_DIR"], workload_parameters["APP_BIN"]
        real_workloads_dict_out[workload_name] = [
            NVBITFI_HOME + f'/test-apps/real_workloads/{app_dir}',  # workload directory
            app_bin,  # binary name
            NVBITFI_HOME + f'/test-apps/real_workloads/{app_dir}',  # path to the binary file
            5,  # expected runtime secs
            " ".join(map(str, workload_parameters["MAKE_PARAMETERS"].values()))  # additional parameters to the run.sh
        ]

    with open(REAL_WORKLOADS_PARAMETERS_FILE, 'w') as handle:
        handle.write("REAL_WORKLOAD_DICT = ")
        handle.write(json.dumps(real_workloads_dict_out, indent=4))


REAL_WORKLOADS = {
    # ----------------------------------------------------------------------------------------------------------------
    # LAVA
    "FLAVA": {
        # "MAKE_PARAMETERS": dict(PRECISION="float", SIZE=15, STREAMS=1),
        "MAKE_PARAMETERS": dict(PRECISION="single", SIZE=2, STREAMS=1),
        "APP_DIR": "lava", "APP_BIN": "cuda_lava_single"
    },
    # ----------------------------------------------------------------------------------------------------------------
    # MXM AND GEMM USE BOTH 2KX2K MATRICES
    "FMXM": {
        # "MAKE_PARAMETERS": dict(PRECISION="float", SIZE=2048, CUBLAS=0, TENSOR_CORES=0),
        "MAKE_PARAMETERS": dict(PRECISION="float", SIZE=512, CUBLAS=0, TENSOR_CORES=0), "APP_DIR": "gemm",
        "APP_BIN": "gemm"
    },
    "FGEMM": {
        # "MAKE_PARAMETERS": dict(PRECISION="float", SIZE=2048, CUBLAS=0, TENSOR_CORES=0),
        "MAKE_PARAMETERS": dict(PRECISION="float", SIZE=512, CUBLAS=1, TENSOR_CORES=0), "APP_DIR": "gemm",
        "APP_BIN": "gemm"
    },
    # ----------------------------------------------------------------------------------------------------------------
    # HOTSPOT
    "FHOTSPOT": {
        # "MAKE_PARAMETERS": dict(PRECISION="float", STREAMS=4, SIM_TIME=1000),
        "MAKE_PARAMETERS": dict(PRECISION="float", STREAMS=1, SIM_TIME=100), "APP_DIR": "hotspot",
        "APP_BIN": "cuda_hotspot"
    },
    # ------------------------------------------------------------------------------------------------------------------
    # GAUSSIAN - 1K
    "FGAUSSIAN": {
        # "MAKE_PARAMETERS": dict(SIZE=1024),
        "MAKE_PARAMETERS": dict(SIZE=512), "APP_DIR": "gaussian", "APP_BIN": "cudaGaussian"
    },
    # ------------------------------------------------------------------------------------------------------------------
    # BFS
    "BFS": {
        # "MAKE_PARAMETERS": dict(),
        "MAKE_PARAMETERS": dict(), "APP_DIR": "bfs", "APP_BIN": "cudaBFS"
    },
    # ------------------------------------------------------------------------------------------------------------------
    # LUD - 8K
    "FLUD": {
        # "MAKE_PARAMETERS": dict(SIZE=8192),
        "MAKE_PARAMETERS": dict(SIZE=2048), "APP_DIR": "lud", "APP_BIN": "cudaLUD"
    },
    # ------------------------------------------------------------------------------------------------------------------
    # CCL
    "ACCL": {
        # "MAKE_PARAMETERS": dict(SIZE=7, FRAMES=7),
        "MAKE_PARAMETERS": dict(SIZE=7, FRAMES=7), "APP_DIR": "accl", "APP_BIN": "cudaACCL"
    },
    # ------------------------------------------------------------------------------------------------------------------
    # NW
    "NW": {
        # "MAKE_PARAMETERS": dict(SIZE=16384, PENALTY=10),
        "MAKE_PARAMETERS": dict(SIZE=16384, PENALTY=10), "APP_DIR": "nw", "APP_BIN": "nw",
    },
    # ------------------------------------------------------------------------------------------------------------------
    # CFD
    "FCFD": {
        # "MAKE_PARAMETERS": dict(STREAMS=100),
        "MAKE_PARAMETERS": dict(STREAMS=1), "APP_DIR": "cfd", "APP_BIN": "cudaCFD",
    },
    # ----------------------------------------------------------------------------------------------------------------
    # SORTS USE 128K, for fault injection 1k
    # QUICKSORT
    "QUICKSORT": {
        # "MAKE_PARAMETERS": dict(SIZE=134217728),
        "MAKE_PARAMETERS": dict(SIZE=1048576), "APP_DIR": "quicksort", "APP_BIN": "quicksort",
    },
    # MERGESORT
    "MERGESORT": {
        # "MAKE_PARAMETERS": dict(SIZE=134217728),
        "MAKE_PARAMETERS": dict(SIZE=1048576), "APP_DIR": "mergesort", "APP_BIN": "mergesort"
    },
    # Maybe radix sort
}

if __name__ == '__main__':
    main()