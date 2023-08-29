
#!/bin/bash
APP_ARGS=$*

#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate Pytorch_nvbitPERfi

eval ${PRELOAD_FLAG} python ${BIN_DIR}/Run_Layer.py ${APP_ARGS} > stdout.txt 2> stderr.txt