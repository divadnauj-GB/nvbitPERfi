#!/bin/bash

PRECISION=$1
SIZE=$2
STREAMS=$3
DATA_DIR=.
DEF_CHARGES_INPUT=${DATA_DIR}/lava_${PRECISION}_charges_${SIZE}
DEF_DISTANCES_INPUT=${DATA_DIR}/lava_${PRECISION}_distances_${SIZE}
DEF_GOLD=${DATA_DIR}/lava_${PRECISION}_gold_${SIZE}


eval LD_LIBRARY_PATH=../libLogHelper/build:"$LD_LIBRARY_PATH"  "${PRELOAD_FLAG}" "${BIN_DIR}"/cuda_lava_"${PRECISION}" \
-boxes="${SIZE}" -streams="${STREAMS}" -iterations="${ITERATIONS}" \
-verbose -input_distances="${DEF_DISTANCES_INPUT}" \
-input_charges="${DEF_CHARGES_INPUT}" \
-output_gold="${DEF_GOLD}" >stdout.txt 2>stderr.txt

