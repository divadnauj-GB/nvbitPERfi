#!/bin/bash


CUDAPATH=$1
LIB_LOG_HELPER=$2
STREAMS=$3

DATA=${BIN_DIR}
INPUT_BASE=missile.domn.0.2M
INPUT=${DATA}/${INPUT_BASE}
GOLD=${DATA}/cfd_gold_${INPUT_BASE}

eval LD_LIBRARY_PATH="${LIB_LOG_HELPER}":"${CUDAPATH}"/lib64:"$LD_LIBRARY_PATH" "${PRELOAD_FLAG}" \
"${BIN_DIR}"/cudaCFD --streams "${STREAMS}" --input "${INPUT}" \
--gold "${GOLD}" --iterations 1 --verbose > stdout.txt 2> stderr.txt

sed -i '/Kernel time/c\REPLACED.' stdout.txt
sed -i '/read../c\REPLACED.' stdout.txt
