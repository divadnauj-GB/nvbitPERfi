#!/bin/bash


LD_PRELOAD=/$NVBITFI_HOME/profiler/profiler.so $DARKNET_HOME/darknet classifier predict $DARKNET_HOME/LeNet/cfg/mnist.data $DARKNET_HOME/LeNet/cfg/mnist_lenet.cfg $DARKNET_HOME/LeNet/mnist_lenet.weights $DARKNET_HOME/LeNet/mnist_images/test/25045_One.png -t 10


