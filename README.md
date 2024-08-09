
# Effective Fault Injection Techniques for Permanent Faults in GPUs running DNNs

# Prerequisites
 * [NVBit v1.5.5](https://github.com/NVlabs/NVBit/releases/tag/1.5.5)
 * System requirements
   * SM compute capability: >= 3.5 && <= 8.6
   * Host CPU: x86_64
   * OS: Linux
   * GCC version: >= 5.3.0
   * CUDA version: >= 8.0 && <= 11.x
   * CUDA driver version: <= 510.85
   * nvcc version for tool compilation >= 10.2
   * CMake >= 3.16
   * python >= 3.7   
   * pandas==1.3.4
   * TensorRT==8.4.3.1
 
# Getting started on a Linux x86\_64 PC
```bash
# NVBit-v1.5.5
wget https://github.com/NVlabs/NVBit/releases/download/1.5.5/nvbit-Linux-x86_64-1.5.5.tar.bz2
tar xvfj nvbit-Linux-x86_64-1.5.5.tar.bz2
cd nvbit_release/tools/

# NVbitPERfi 
git clone https://github.com/divadnauj-GB/nvbitPERfi.git
cd nvbitPERfi
git checkout Dev
find . -name "*.sh" | xargs chmod +x

# Prepare the benchmarks 
cd test-apps/
git clone https://github.com/divadnauj-GB/pytorch-DNNs.git
cd pytorch-DNNs
```
Follow the instructions inside the README.md file of [pytorch-DNNs](https://github.com/divadnauj-GB/pytorch-DNNs/tree/main) repository

# How to use this framework?
The previous setup configured the framework and generated golden outputs for different DNN models. If you want to perform any fault injection as described in the paper; run the following script. 

```bash
export FAULT_MODE=FUs
# for each benchmark in test-apps
for bench in DNNs-LeNet DNNs-AlexNet DNNs-MobileNetv3 DNNs-ResNet50; do
   bash ./runPERfi.sh $bench $FAULT_MODE> log_$bench.log
done
```

If you want to target the register files, just change the **FAULT_MODE** variable to REGs and configure the details of the fault injection in [fault_models_config.yaml](https://github.com/divadnauj-GB/nvbitPERfi/blob/Dev/scripts/fault_models_config.yaml).
