
# Understanding the Effects of Permanent Faults in GPU's Parallelism Management and Control Units

[![DOI](https://zenodo.org/badge/560037637.svg)](https://zenodo.org/badge/latestdoi/560037637)

```bibtex
@article{guerrero2023sc3,
  title={Understanding the Effects of Permanent Faults in GPU's Parallelism Management and Control Units},
  author={Guerrero-Balaguera, Juan-David and Rodriguez Condia, Josie E. and F. dos Santos, Fernando and Sonza Reorda, Matteo and Rech, Paolo},
  journal={arXiv preprint arXiv:2306.10856},
  year={2023}
}
```


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
 
# Getting started on a Linux x86\_64 PC
```bash
# NVBit-v1.5.5
wget https://github.com/NVlabs/NVBit/releases/download/1.5.5/nvbit-Linux-x86_64-1.5.5.tar.bz2
tar xvfj nvbit-Linux-x86_64-1.5.5.tar.bz2
cd nvbit_release/tools/

# NVbitPERfi 
git clone https://github.com/divadnauj-GB/nvbitPERfi.git
cd nvbitPERfi
find . -name "*.sh" | xargs chmod +x

# Prepare the benchmarks  
cd test-apps/
python3 configure_real_workloads.py
```
# How to use this framework?
The previous setup configure and run all the application benchmarchs and generate the golden outputs.
If you want to execute the IOC error model for all apps that were used in the paper; run the following script. 

```bash
export FAULT_MODE=ICOC
# for each benchmark in test-apps
for bench in accl bfs cfd darknet_v3 gaussian gemm hotspot lava LeNet lud mergesort nw quicksort VectorAdd; do
   ./runPERfi.sh $bench $FAULT_MODE> log_$bench.log
done
```

If you want to use any other error model just change the **FAULT_MODE** variable whith one of the following descriptors:
```console
[IAT, IAW, IAC, WV, IRA, IMS, IMD, IAL, ICOC, IIO]
```

Assuming that the tool and its dependencies have been appropriately installed and configured, you can also run the NVBitPERfi for a single benchmark such as GEMM with IOC fault model, execute the following command:

```bash
export FAULT_MODE=ICOC
./runPERfi.sh gemm $FAULT_MODE> gemm.log
```
After completing the fault injection procedure, the results can be extracted by running the parsers to generate a CSV file containing the parsed fault injection details. It is important to note that each parser has configuration variables in its first lines that must be set prior to execution. The following command can be used to parse the data:

```bash
cd scripts/parsers/
# - Set the configuration variables
# inside the parser and execute it
python3 parse_pf_injections.py
# Results will be stored in the path
# set in the OUTPUT_PARSED_FILE var
```
