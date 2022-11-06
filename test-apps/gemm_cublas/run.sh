#!/bin/bash
eval ${PRELOAD_FLAG} ${BIN_DIR}/gemm float ${BIN_DIR}/a.data ${BIN_DIR}/b.data ${BIN_DIR}/gold.data 0 > stdout.txt 2> stderr.txt
