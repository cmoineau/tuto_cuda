# CUDA

- Tested on Ubuntu 22
- CUDA 12.1

## Installation

The first step is to install the CUDA toolkit, which includes the necessary tools for creating and running CUDA kernels.
Download the latest version of the CUDA toolkit from the NVIDIA website.
Make sure to choose the appropriate version for your operating system and hardware.
Ressource : https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

## Set up environment

Update bashrc with path : 

```bash
export PATH=/usr/local/cuda-X.X/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-X.X/lib64:$LD_LIBRARY_PATH
```

## Matrix multiplication
TODO : https://tcuvelier.developpez.com/tutoriels/gpgpu/cuda/introduction/?page=conclusions#LVI-C

## Tools : 

- [nsight-systems](https://developer.nvidia.com/nsight-systems)

## :bookmark_tabs: Usefull links 

- [Doc NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)

- [CUDA C/C++ Basics](https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf)