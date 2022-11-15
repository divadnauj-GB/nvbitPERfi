#!/bin/bash


CUDAPATH=$1
LIB_LOG_HELPER=$2
SIZE=$3
FRAMES=$4

DATADIR=${BIN_DIR}

eval LD_LIBRARY_PATH="${LIB_LOG_HELPER}":"${CUDAPATH}"/lib64:"$LD_LIBRARY_PATH" "${PRELOAD_FLAG}" \
"${BIN_DIR}"/cudaACCL --size "${SIZE}" --frames "${FRAMES}" --input "${DATADIR}"/"${FRAMES}"Frames.pgm \
--gold "${DATADIR}"/gold_"${SIZE}"_"${FRAMES}".data --iterations 1 --verbose > stdout.txt 2>stderr.txt

sed -i '/Time/c\' stdout.txt
sed -i '/time/c\' stdout.txt 

