This is an instruction manula about how to intall TensorRT on ubuntu 2004 with an specific version of CUDA. I'll put my example here:

1: intall CUDA and the nvidia drivers for that cuda version; make sure they are working properly

2: Install tensorRT using the apt package manager, for this follow the instruction intalation presented by nvidia but be sure
you are installing the exact version for your cuda version. to do so type tis command:

    sudo apt-cache policy tensorrt

this command should give you the list of options available to be installed, like this 

tensorrt:
  Installed: 8.4.3.1-1+cuda11.6
  Candidate: 8.6.1.6-1+cuda12.0
  Version table:
     8.6.1.6-1+cuda12.0 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
     8.6.1.6-1+cuda11.8 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
     8.6.0.12-1+cuda12.0 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
     8.6.0.12-1+cuda11.8 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
     8.5.3.1-1+cuda11.8 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
     8.5.2.2-1+cuda11.8 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
     8.5.1.7-1+cuda11.8 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
 *** 8.4.3.1-1+cuda11.6 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
        100 /var/lib/dpkg/status
     8.4.2.4-1+cuda11.6 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
     8.4.1.5-1+cuda11.6 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages


if you have cuda 11.6, you can install any of the versions using this command:

sudo apt-get install tensorrt=8.4.3.1-1+cuda11.6

it might rise an error about other dependeces, so force the instalation of every of those ones adding the same tensorrt version you want to install


if you can't find the right version in this repository I sugest you to follow the tar file instalaction porcedure, this allows you to intall tensorrt native on your OS but also allows you to add the python packages on your environment like conda (as I did)

Download the TensorRT tar file that matches the CPU architecture and CUDA version you are using.
Choose where you want to install TensorRT. This tar file will install everything into a subdirectory called TensorRT-8.x.x.x.
Unpack the tar file.

version="8.x.x.x"
arch=$(uname -m)
cuda="cuda-x.x"
tar -xzvf TensorRT-${version}.Linux.${arch}-gnu.${cuda}.tar.gz

Where:
    8.x.x.x is your TensorRT version
    cuda-x.x is CUDA version 11.8 or 12.0
This directory will have sub-directories like lib, include, data, and so on.

ls TensorRT-${version}
bin  data  doc  graphsurgeon  include  lib  onnx_graphsurgeon  python  samples  targets  uff

Add the absolute path to the TensorRT lib directory to the environment variable LD_LIBRARY_PATH:

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>

Install the Python TensorRT wheel file (replace cp3x with the desired Python version, for example, cp310 for Python 3.10). (here you can use conda environmets or anyother you required)

cd TensorRT-${version}/python

python3 -m pip install tensorrt-*-cp3x-none-linux_x86_64.whl

Optionally, install the TensorRT lean and dispatch runtime wheel files:

python3 -m pip install tensorrt_lean-*-cp3x-none-linux_x86_64.whl
python3 -m pip install tensorrt_dispatch-*-cp3x-none-linux_x86_64.whl

Install the Python UFF wheel file. This is only required if you plan to use TensorRT with TensorFlow in UFF format.

cd TensorRT-${version}/uff

python3 -m pip install uff-0.6.9-py2.py3-none-any.whl

Check the installation with:

which convert-to-uff

Install the Python graphsurgeon wheel file.

cd TensorRT-${version}/graphsurgeon

python3 -m pip install graphsurgeon-0.4.6-py2.py3-none-any.whl

Install the Python onnx-graphsurgeon wheel file.

cd TensorRT-${version}/onnx_graphsurgeon
    
python3 -m pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl