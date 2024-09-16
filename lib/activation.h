#include <cuda_runtime.h>
#include <iostream>


__global__ void reluKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height);