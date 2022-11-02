# TOWARDS DSN :). How to use this environmet?
run the following command. the `VectorAdd` parameter is the application folder where the executable binary is located

```console
./runPERfi.sh VectorAdd > log.log

```
you will get a file called `log.log` with something like the below dump results. if not, probably the target register is not in the SASS file, so change the second colum of the fault list file `modeIRA1000.txt` located in `test-apps/logs/VectorAdd/injection-list` 

If everything was working as it suposed to be, here the results. 
```console
Current working directory: /home/juancho/Documents/GitHub/Ampere_NVBit/nvbit_release/tools/nvbitPERfi

Step 0 (2): Setting environment variables

Step 0 (3): Build the nvbitfi injector and profiler tools
rm -f *.so *.o
nvcc -ccbin=`which gcc` -D_FORCE_INLINES -I../../../core -I../common -maxrregcount=16 -Xptxas -astoolspatch --keep-device-functions -arch=sm_86 -DDUMMY=0 -Xcompiler -Wall -Xcompiler -fPIC -c inject_funcs.cu -o inject_funcs.o
#nvcc -ccbin=`which gcc` -D_FORCE_INLINES -I../../../core -I../common -maxrregcount=16 -Xptxas -arch=sm_86 -DDUMMY=0 -Xcompiler -Wall -Xcompiler -fPIC -c inject_funcs.cu -o inject_funcs.o
nvcc -ccbin=`which gcc` -D_FORCE_INLINES -dc -c -std=c++11 -I../../../core -I../common -Xptxas -cloning=no -Xcompiler -Wall -arch=sm_86 -O3 -Xcompiler -fPIC pf_injector.cu -o pf_injector.o
nvcc -ccbin=`which gcc` -D_FORCE_INLINES -arch=sm_86 -O3 inject_funcs.o pf_injector.o -L../../../core -lnvbit -L/usr/local/cuda/lib64 -lcuda -lcudart_static -shared -o pf_injector.so
make: Nothing to be done for 'all'.

Step 0 (4): Run and collect output without instrumentation
>>> GCC Version is greater or equal to 5.1.0 <<<
make: Nothing to be done for 'all'.

Step 2: Run the error injection campaign
Inj_count=1, App=VectorAdd, Mode=IRA, Time=0.894535, Outcome: SDC: Standard output is different
Inj_count=2, App=VectorAdd, Mode=IRA, Time=0.536717, Outcome: Masked: other reasons
Inj_count=3, App=VectorAdd, Mode=IRA, Time=0.914253, Outcome: SDC: Standard output is different
Inj_count=4, App=VectorAdd, Mode=IRA, Time=0.879296, Outcome: SDC: Standard output is different
Inj_count=5, App=VectorAdd, Mode=IRA, Time=0.890800, Outcome: SDC: Standard output is different
Inj_count=6, App=VectorAdd, Mode=IRA, Time=0.890167, Outcome: SDC: Standard output is different
Inj_count=7, App=VectorAdd, Mode=IRA, Time=0.887980, Outcome: SDC: Standard output is different
Inj_count=8, App=VectorAdd, Mode=IRA, Time=0.891328, Outcome: SDC: Standard output is different
Inj_count=9, App=VectorAdd, Mode=IRA, Time=0.892257, Outcome: SDC: Standard output is different
Inj_count=10, App=VectorAdd, Mode=IRA, Time=0.909858, Outcome: SDC: Standard output is different
Inj_count=11, App=VectorAdd, Mode=IRA, Time=0.887454, Outcome: SDC: Standard output is different
Inj_count=12, App=VectorAdd, Mode=IRA, Time=0.923082, Outcome: SDC: Standard output is different
Inj_count=13, App=VectorAdd, Mode=IRA, Time=0.907690, Outcome: SDC: Standard output is different
Inj_count=14, App=VectorAdd, Mode=IRA, Time=0.916286, Outcome: SDC: Standard output is different
Inj_count=15, App=VectorAdd, Mode=IRA, Time=0.892563, Outcome: SDC: Standard output is different
Inj_count=16, App=VectorAdd, Mode=IRA, Time=0.535005, Outcome: Masked: other reasons
Inj_count=17, App=VectorAdd, Mode=IRA, Time=0.890140, Outcome: SDC: Standard output is different
Inj_count=18, App=VectorAdd, Mode=IRA, Time=0.884540, Outcome: SDC: Standard output is different
Inj_count=19, App=VectorAdd, Mode=IRA, Time=0.901644, Outcome: SDC: Standard output is different
Inj_count=20, App=VectorAdd, Mode=IRA, Time=0.535725, Outcome: Masked: other reasons
Inj_count=21, App=VectorAdd, Mode=IRA, Time=0.905243, Outcome: SDC: Standard output is different
Inj_count=22, App=VectorAdd, Mode=IRA, Time=0.535942, Outcome: Masked: other reasons
Inj_count=23, App=VectorAdd, Mode=IRA, Time=0.904790, Outcome: SDC: Standard output is different
Inj_count=24, App=VectorAdd, Mode=IRA, Time=0.902430, Outcome: SDC: Standard output is different
Inj_count=25, App=VectorAdd, Mode=IRA, Time=0.912683, Outcome: SDC: Standard output is different
Inj_count=26, App=VectorAdd, Mode=IRA, Time=0.537524, Outcome: Masked: other reasons
Inj_count=27, App=VectorAdd, Mode=IRA, Time=0.535737, Outcome: Pot DUE: Standard output is different, but dmesg recorded

VectorAdd

```

Now you should have two new folders `logs`, `sdcs` and one report file `results-modeIRA1000.txt` inside `test-apps/logs/VectorAdd`

that's it, now you can continue having fun|


# NVBitFI: An Architecture-level Fault Injection Tool for GPU Application Resilience Evaluations

NVBitFI provides an automated framework to perform error injection campaigns for GPU application resilience evaluation.  NVBitFI builds on top of [**NV**IDIA **Bi**nary **I**nstrumentation **T**ool (NVBit)](https://github.com/NVlabs/NVbit), which is a research prototype of a dynamic binary instrumentation library for NVIDIA GPUs. NVBitFI offers functionality that is similar to a prior tool called [SASSIFI](https://github.com/NVlabs/sassifi).  

# Summary of NVBitFI's capabilities 

NVBitFI injects errors into the destination register values of a dynamic thread-instruction by instrumenting instructions after they are executed.  A dynamic instruction is selected at random from all dynamic kernels of a program for error injection.  Only one error is injected per run.  This mode was referred to as IOV in SASSIFI.  As of now (4/1/2020), NVBitFI allows us to select the following instruction groups to study how errors in them can propagate to the application output.

 * Instructions that write to general purpose registers
 * Single precision floating point instructions
 * Double precision floating point instructions 
 * Load instructions 

NVBitFI can be extended to include custom instruction groups. See below for more details. 

For a selected destination register, following errors can be injected. 

 * Single bit-flip: one bit-flip in one register in one thread
 * Double bit-flip: bit-flips in two adjacent bits in one register in one thread
 * Random value: random value in one register in one thread
 * Zero value: zero out the value of one register in one thread

New bit-flip models can be added by modifying common/arch.h and injector/inject\_func.cu and scripts/params.py. 


# Prerequisites
 * [NVBit v1.3](https://github.com/NVlabs/NVBit/releases/tag/1.3) or newer
 * [System requirements](https://github.com/NVlabs/NVbit#requirements)

# Getting started on a Linux x86\_64 PC
```console
# NVBit-v1.3
wget https://github.com/NVlabs/NVBit/releases/download/1.3/nvbit-Linux-x86_64-1.3.tar.bz2
tar xvfj nvbit-Linux-x86_64-1.3.tar.bz2
cd nvbit_release/tools/

# NVBitFI 
git clone https://github.com/NVlabs/nvbitfi
cd nvbitfi
find . -name "*.sh" | xargs chmod +x
./test.sh
```
On an ARM-based device (e.g., Jetson Nano)
```console
# NVBit-1.5.3
wget https://github.com/NVlabs/NVBit/releases/download/1.5.3/nvbit-Linux-aarch64-1.5.3.tar.bz2
tar xvfj nvbit-Linux-aarch64-1.5.3.tar.bz2
cd nvbit_release/tools/

# NVBitFI 
git clone https://github.com/NVlabs/nvbitfi
cd nvbitfi
find . -name "*.sh" | xargs chmod +x
./test.sh
```

If these commands complete without errors, you just completed your first error injection campaign using NVBitFI. The printed output should say where the results are stored. Summary of the campaign is stored in a tab-separated file, `results_*NVbitFI_details.tsv`. It can be opened using a spreadsheet program (e.g., Excel) for visualization and analysis. 

# Detailed steps 

There are three main steps to run NVBitFI. We provide a sample script (test.sh) that automates nearly all these steps.

## Step 0: Setup

 * One-time only: Copy NVBitFI package to tool directory in NVBit installation (see the above commands) 
 * Every time we run an injection campaign: Setup environment (see Step 0 (2) in test.sh)
 * One-time only: Build the injector and profiler tools (see Step 0 (3) in test.sh)
 * One-time only: Run and collect golden stdout and stderr files for each of the applications (see Step 0 (4) in test.sh). 
    * Record fault-free outputs: Record golden output file (as golden.txt), stdout (as golden\_stdout.txt), and stderr (as golden\_stderr.txt) in the workload directory (e.g., nvbitfi/test-apps/simple\_add).
    * Create application-specific scripts: Create run.sh and sdc\_check.sh scripts in workload directory. Instead of using absolute paths, please use environment variables for paths such as BIN\_DIR, APP\_DIR, and DATASET\_DIR. These variables are set in set\_env function in scripts/common\_functions.py. See the scripts in the nvbitfi/test-apps/simple\_add directory for examples.
    * Workloads will be run from logs/workload-name/run-name directory. It would be great if the workload can run from this directory. If the program requires input files to be in a specific location, either update the workload or provide soft links to the input files in appropriate locations. 
    * The program output should be deterministic. Please exclude non-deterministic values (e.g., runtimes) from the file if they are present in one of the output files (see test-apps/simple\_add/sdc\_check.sh for more details).

## Step 1: Profile and generate injection list

 * Profile the application: Run the program once by using profiler/profiler.so. We provide scripts/run\_profiler.py script for this step. A new file named nvbitfi-igprofile.txt will be generated in logs/workload-name directory. This file contains the instruction counts for all the instruction groups and opcodes defined in common/arch.h. One line is created per dynamic kernel invocation.
   Profiling is often slow as it instruments every instruction in every dynamic kernel. Using an approximate profile can speed it up by orders of magnitude. There are many ways to approximate a profile and trade-off accuracy for speed. In this release we implement a method that approximates the profiles of all dynamic invocations of a static kernel with the profile of the first invocation of the static kernel. It essentially profiles all static kernels just ones, which can make the profiling very fast if a program has few static kernels and many dynamic involutions per kernel. This approximation can be enabled by using the `SKIP_PROFILED_KERNELS` flag while building the profiler. 
 * Generate injection sites:
    * Ensure that the parameters are set correctly in scripts/params.py.  Following are some of the parameters that need user attention: 
		* Setting maximum number of error injections to perform per instruction group and bit-flip model combination. See NUM\_INJECTION and THRESHOLD\_JOBS in scripts/params.py. 
		* Selecting instruction groups and bit-flip models (more details in scripts/params.py). 
		* Listing the applications, benchmark suite name, application binary file name, and the expected runtime on the system where the injection job will be run. See the apps dictionary in scripts/params.py for an example. The expected runtime defined here is used later to determine when to timeout injection runs (based on the TIMEOUT\_THRESHOLD defined in scripts/params.py).
    * Run scripts/generate\_injection\_list.py to generate a file that contains a list of errors to be injected during the injection campaign. Instructions are selected randomly from the instructions of the selected instruction group. 

## Step 2: Run the error injection campaign

Run scripts/run\_injections.py to launch the error injection campaign. This script will run one injection run at a time in the standalone mode.  If you plan to run multiple injection runs in parallel, please take special care to ensure that the output file is not clobbered. As of now, we support running multiple jobs on a multi-GPU system. Please see scripts/run\_one\_injection.py for more details. 

Tip: Perform a few dummy injections before proceeding with full injection campaign (by setting DUMMY flag in injector/Makefile. Setting this flag will allow you to go through most of the SASSI handler code but skip the error injection. This is to ensure that you are not seeing crashes/SDCs that you should not see.

## Step 3: Parse the results

Use the scripts/parse\_results.py script to parse the results. This script generates three tab-separated values (tsv) files. The first file shows the fraction of executed instructions for different instruction groups and opcodes. The second file shows the outcomes of the error injections.  Refer to CAT\_STR in scripts/params.py for the list of error outcome categories. The third file shows the average runtime for the injection runs for different applications and selected error models. These files can be opened using a spreadsheet program (e.g., Excel) for plotting and analysis.



# NVBitFI vs. SASSIFI

NVBitFI benefits from the featured offered by NVBit. It can run on newer GPUs (e.g., Turing and Volta GPUs). It works with pre-compiled libraries also, unlike SASSIFI. NVBitFI is expected to be faster than SASSIFI as it instruments just a single chosen dynamic kernel (SASSIFI, as it was implemented, instrumented all dynamic kernels) for the injection runs.  As of now (April 14, 2020), NVBitFI implements a subset of the error injection models and we may be expanding this over time (users are more than welcome to contribute). 



# Contributing to NVBitFI

If you are interested in contributing to NVBitFI, please initialize a [Pull Request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) and complete the [Contributor License Agreement](https://www.apache.org/licenses/icla.pdf).
