#!/bin/bash


CUDAPATH=$1
LIB_LOG_HELPER=$2
PRECISION=$3
STREAMS=$4
SIM_TIME=$5

DATADIR=${BIN_DIR}
SIZE=1024

eval LD_LIBRARY_PATH="${LIB_LOG_HELPER}":"${CUDAPATH}"/lib64:"$LD_LIBRARY_PATH"  "${PRELOAD_FLAG}" \
"${BIN_DIR}"/cuda_hotspot -verbose -precision="${PRECISION}" -size="${SIZE}" -sim_time="${SIM_TIME}" \
-streams="${STREAMS}" -input_temp="${DATADIR}"/temp_"${SIZE}" -input_power="${DATADIR}"/power_"${SIZE}" \
-gold_temp="${DATADIR}"/gold_"${PRECISION}"_"${SIZE}"_"${SIM_TIME}" \
-iterations=1 > stdout.txt 2> stderr.txt

sed -i '/Time/c\' stdout.txt
sed -i '/time/c\' stdout.txt 
sed -i '/Performance/c\' stdout.txt
