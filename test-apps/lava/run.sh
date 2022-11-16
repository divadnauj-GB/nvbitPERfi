#!/bin/bash

CUDAPATH=$1
LIB_LOG_HELPER=$2
PRECISION=$3
SIZE=$4
STREAMS=$5

ITERATIONS=1
DATA_DIR=${BIN_DIR}
DEF_CHARGES_INPUT=${DATA_DIR}/lava_${PRECISION}_charges_${SIZE}
DEF_DISTANCES_INPUT=${DATA_DIR}/lava_${PRECISION}_distances_${SIZE}
DEF_GOLD=${DATA_DIR}/lava_${PRECISION}_gold_${SIZE}


eval LD_LIBRARY_PATH="${LIB_LOG_HELPER}":"${CUDAPATH}"/lib64:"$LD_LIBRARY_PATH"  "${PRELOAD_FLAG}" \
"${BIN_DIR}"/cuda_lava_"${PRECISION}" -boxes="${SIZE}" -streams="${STREAMS}" -iterations="${ITERATIONS}" \
-verbose -input_distances="${DEF_DISTANCES_INPUT}" \
-input_charges="${DEF_CHARGES_INPUT}" \
-output_gold="${DEF_GOLD}" >stdout.txt 2>stderr.txt

sed -i '/Time/c\' stdout.txt
sed -i '/time/c\' stdout.txt
sed -i '/FLOPS:/c\' stdout.txt
