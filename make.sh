#!/bin/bash
set -x

echo "Compiling crop_and_resize kernels by nvcc..."
cd roi_align/src/cuda
nvcc -ccbin clang-3.8 -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_37
cd ../../
python3 build.py
cd ../

cd nms/src/cuda
echo "Compiling nms kernels by nvcc..."
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_37
cd ../../
python build.py
cd ../
