#!/bin/bash
#
# Copyright 2020, NVIDIA CORPORATION.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

# stop after first error 
#set -e 

# Uncomment for verbose output
# set -x 

CWD=`pwd`
echo "Current working directory: $CWD"

###############################################################################
# Step 0: Set-up 
#
# (1) One-time only: Copy nvbitfi tool package to NVBit/tools/ directory
# (2) Everytime we run an injection campaign: Setup environment
# (3) One-time only: Build the inject_error and igprofiler NVBit tools 
# (4) One-time only: Run and collect golden stdout and stderr files for each of the applications
###############################################################################

###############################################################################
# Step 0 (1): Provide execute permissions to *.sh scripts 
###############################################################################
find . -name "*.sh" | xargs chmod +x 

###############################################################################
# Step 0 (2): Setup environment 
###############################################################################
printf "\nStep 0 (2): Setting environment variables\n"

# environment variables for NVBit
export NOBANNER=1
# set TOOL_VERBOSE=1 to print debugging information during profling and injection runs 
export TOOL_VERBOSE=0
export VERBOSE=0

export NVBITFI_HOME=$CWD
export CUDA_BASE_DIR=/usr/local/cuda
export PATH=$PATH:$CUDA_BASE_DIR/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_BASE_DIR/lib64/:$CUDA_BASE_DIR/extras/CUPTI/lib64/
export DARKNET_HOME=/home/juancho/Documents/GitHub/darknet_jd_v1

# Change here the name of the application simple_add convolution2d
#export APP=simple_add

###############################################################################
# Step 0 (3): Build the nvbitfi injector 
###############################################################################
printf "\nStep 0 (3): Build the nvbitfi injector tool\n"
cd profiler_CNN/
#make clean 
make all
cd $CWD

###############################################################################
# Step 0 (4): Run the app without instrumentation. Collect golden stdout and
# stderr files. User must generate this before starting the injection campaign.
###############################################################################
printf "\nStep 0 (4): Run and collect output without instrumentation\n"


cd test-apps/LeNet/
rm *.txt

LD_PRELOAD=/$NVBITFI_HOME/profiler_CNN/profiler.so $DARKNET_HOME/darknet classifier predict $DARKNET_HOME/LeNet/cfg/mnist.data $DARKNET_HOME/LeNet/cfg/mnist_lenet.cfg $DARKNET_HOME/LeNet/mnist_lenet.weights $DARKNET_HOME/LeNet/mnist_images/test/25045_One.png -t 10


new_folder="$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p $new_folder
mv *.txt $new_folder
#cuobjdump -sass $DARKNET_HOME/darknet > LeNet.sass
cd $CWD






