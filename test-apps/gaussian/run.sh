#!/bin/bash

CUDAPATH=$1
LIB_LOG_HELPER=$2
SIZE=$3

DATA_DIR=${BIN_DIR}

eval LD_LIBRARY_PATH="${LIB_LOG_HELPER}":"${CUDAPATH}"/lib64:"$LD_LIBRARY_PATH" "${PRELOAD_FLAG}" \
"${BIN_DIR}"/cudaGaussian --size "${SIZE}" --input "${DATA_DIR}"/input_"${SIZE}".data \
--gold "${DATA_DIR}"/gold_"${SIZE}".data --iterations 1 --verbose >stdout.txt 2>stderr.txt

sed -i '/time/c\' stdout.txt 
