#!/bin/bash

CUDAPATH=$1
LIB_LOG_HELPER=$2
SIZE=$3
DATADIR=${BIN_DIR}

eval LD_LIBRARY_PATH="${LIB_LOG_HELPER}":"${CUDAPATH}"/lib64:"$LD_LIBRARY_PATH" "${PRELOAD_FLAG}" \
"${BIN_DIR}"/quicksort -size="${SIZE}" -input="${DATADIR}"/quicksort_input_134217728 \
-gold="${DATADIR}"/quicksort_gold_"${SIZE}" -iterations=1 -verbose > stdout.txt 2> stderr.txt

sed -i '/Time/c\' stdout.txt
sed -i '/time/c\' stdout.txt 
sed -i '/Perf/c\' stdout.txt
sed -i '/Starting/c\' stdout.txt
sed -i '/Done in/c\' stdout.txt
