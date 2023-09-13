
#!/bin/bash
APP_ARGS=$*

#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate Pytorch_nvbitPERfi
#~/archiconda3/bin/conda env list
eval ${PRELOAD_FLAG} python3 ${BIN_DIR}/${APP_BIN} ${APP_ARGS} > stdout.txt 2> stderr.txt

#CUDA_INJECTION64_PATH=/home/jetson/Documents/GitHub/nvbit_release/tools/nvbitPERfi/profiler_CNN/profiler.so python /home/jetson/Documents/GitHub/nvbit_release/tools/nvbitPERfi/test-apps/Pytorch_CNNs/LeNet.py -ln 0 -bs 1 > stdout.txt 2> stderr.txt
