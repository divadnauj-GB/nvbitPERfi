#!/bin/bash
APP_ARGS=$*
eval ${PRELOAD_FLAG} ${BIN_DIR}/vectorAdd ${APP_ARGS} > stdout.txt 2> stderr.txt
