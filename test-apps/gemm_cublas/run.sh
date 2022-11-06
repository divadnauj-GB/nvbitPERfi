#!/bin/bash
eval ${PRELOAD_FLAG} ${BIN_DIR}/${APP_BIN} ${APP_ARGS} > stdout.txt 2> stderr.txt
