#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <float.h>  // For FLT_MAX

#define TILE_WIDTH 16  // Example tile width for shared memory optimization

__global__ void MaxPoolingKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height);

__global__ void MaxPoolingKernelShared(const float* input, float* output, 
                                        int inputHeight, int inputWidth, 
                                        int kernelHeight, int kernelWidth, 
                                        int strideHeight, int strideWidth, 
                                        int paddingHeight, int paddingWidth);