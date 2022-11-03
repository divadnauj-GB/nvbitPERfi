#!/bin/bash

# environment variables for NVBit
#export NOBANNER=1
# set TOOL_VERBOSE=1 to print debugging information during profling and injection runs
export TOOL_VERBOSE=1
export VERBOSE=1

export NVBITFI_HOME=$CWD
export CUDA_BASE_DIR=/usr/local/cuda
export PATH=$PATH:$CUDA_BASE_DIR/bin
export LD_LIBRARY_PATH=$CUDA_BASE_DIR/lib64/:$CUDA_BASE_DIR/extras/CUPTI/lib64/

eval LD_PRELOAD=../../pf_injector_icoc/pf_injector.so ./gemm


