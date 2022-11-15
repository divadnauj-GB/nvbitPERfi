#!/bin/bash

CUDAPATH=$1
LIB_LOG_HELPER=$2
DATA_DIR=${BIN_DIR}

eval LD_LIBRARY_PATH="${LIB_LOG_HELPER}":"${CUDAPATH}"/lib64:"$LD_LIBRARY_PATH" "${PRELOAD_FLAG}" \
"${BIN_DIR}"/cudaBFS --input "${DATA_DIR}"/graph1MW_6.txt \
--gold "${DATA_DIR}"/gold.data --iterations 1 --verbose  > stdout.txt 2> stderr.txt

sed -i '/Time/c\' stdout.txt
sed -i '/time/c\' stdout.txt 
