#!/bin/bash

CUDAPATH=$1
LIB_LOG_HELPER=$2

IMG_LIST="${BIN_DIR}"/coco2017_img_list.txt
GOLD_DIR="${BIN_DIR}"/test_float_tiny_0.csv

EXEC=darknet_v3_float


eval LD_LIBRARY_PATH="${LIB_LOG_HELPER}":"${CUDAPATH}"/lib64:"$LD_LIBRARY_PATH" "${PRELOAD_FLAG}" \
"${BIN_DIR}"/"${EXEC}" detector test_radiation \
"${BIN_DIR}"/cfg/coco.data "${BIN_DIR}"/yolov3-spp.cfg "${BIN_DIR}"/yolov3-spp.weights \
"${IMG_LIST}" -generate 0 -gold "${GOLD_DIR}" \
-iterations 1	-norm_coord 0 -tensor_cores 0 -thresh 0.5 >stdout.txt 2>stderr.txt

sed -i '/Time/c\' stdout.txt
sed -i '/time/c\' stdout.txt
sed -i '/seconds/c\' stdout.txt
sed -i '/^$$/d' stdout.txt