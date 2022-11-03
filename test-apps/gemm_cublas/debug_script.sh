#!/bin/bash

# environment variables for NVBit
#export NOBANNER=1
# set TOOL_VERBOSE=1 to print debugging information during profling and injection runs
export TOOL_VERBOSE=1
export VERBOSE=1

export NVBITFI_HOME=$CWD
export CUDA_BASE_DIR=/usr/local/cuda-10.2
export PATH=$PATH:$CUDA_BASE_DIR/bin
export LD_LIBRARY_PATH=$CUDA_BASE_DIR/lib64/:$CUDA_BASE_DIR/extras/CUPTI/lib64/

#eval LD_PRELOAD=/home/carol/git_research/nvbitfi_ml/nvbitfi/profiler/profiler.so ./test_cublas
eval LD_PRELOAD=/home/carol/git_research/nvbitfi_ml/nvbitfi/injector/injector.so ./test_cublas


