#!/bin/bash
#export NVBITFI_HOME=/home/juancho/Documents/GitHub/Ampere_NVBit/nvbit_release/tools/Ampere_nvbitfi
#export DARKNET_HOME=/home/juancho/Documents/GitHub/darknet_jd_v1

cd $NVBITFI_HOME/pf_injector/


#LD_PRELOAD=$NVBITFI_HOME/pf_injector/pf_injector.so $DARKNET_HOME/darknet classifier predict $DARKNET_HOME/LeNet/cfg/mnist.data $DARKNET_HOME/LeNet/cfg/mnist_lenet.cfg $DARKNET_HOME/LeNet/mnist_lenet.weights $DARKNET_HOME/LeNet/mnist_images/test/0_Five.png -t 10 > $NVBITFI_HOME/test-apps/LeNet/stdout.txt 2> $NVBITFI_HOME/test-apps/LeNet/stderr.txt cat nvbitfi-injection-log-temp.txt 
LD_PRELOAD=$NVBITFI_HOME/pf_injector/pf_injector.so $DARKNET_HOME/darknet classifier test $DARKNET_HOME/LeNet/cfg/mnist.data $DARKNET_HOME/LeNet/cfg/mnist_lenet.cfg $DARKNET_HOME/LeNet/mnist_lenet.weights $DARKNET_HOME/LeNet/mnist_images/test/25045_One.png -t 10 > $NVBITFI_HOME/test-apps/LeNet/stdout.txt 2> $NVBITFI_HOME/test-apps/LeNet/stderr.txt cat nvbitfi-injection-log-temp.txt 
