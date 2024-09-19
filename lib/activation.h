#include <cuda_runtime.h>
#include <iostream>


__global__ void reluKernel(const float* inputImage, float* outputImage, int width, int height);

__global__ void reluKernelMultiple(const float* inputImage, float* outputImage, int width, int height, int numChannels);