#!/bin/bash
set -x
set -e

CUDA_PATH=/usr/lib/x86_64-linux-gnu

echo "Compiling crop_and_resize kernels by nvcc..."
cd roi_align/src/cuda
$CUDA_PATH/bin/nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_37

cd ../../
python build.py
