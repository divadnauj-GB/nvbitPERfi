#!/bin/bash

CUDAPATH=$1
LIB_LOG_HELPER=$2
SIZE=$3
PENALTY=$4

DATA=${BIN_DIR}

eval LD_LIBRARY_PATH="${LIB_LOG_HELPER}":"${CUDAPATH}"/lib64:"$LD_LIBRARY_PATH" "${PRELOAD_FLAG}" \
"${BIN_DIR}"/nw "${SIZE}" "${PENALTY}" "${DATA}"/input_"${SIZE}"_"${PENALTY}" \
"${DATA}"/gold_"${SIZE}"_"${PENALTY}" 1 0 >stdout.txt 2> stderr.txt

sed -i '/kernel time/c\REPLACED.' stdout.txt 
sed -i '/read../c\REPLACED.' stdout.txt
