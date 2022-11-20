#!/bin/bash
APP_ARGS=$*
eval ${PRELOAD_FLAG} ${BIN_DIR}/gemm ${APP_ARGS} > stdout.txt 2> stderr.txt
