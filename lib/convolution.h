#include <cuda_runtime.h>
#include <iostream>

#define TILE_WIDTH 16  // Example tile width for shared memory optimization

// __global__ void convolutionKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, const float* filter, const int filterwidth, int padding, int stride);
__global__ void convolutionKernel(unsigned char* input, unsigned char* output, int inputWidth, int inputHeight, const float* kernel,
                       int kernelHeight, int kernelWidth, 
                       int strideHeight, int strideWidth, 
                       int paddingHeight, int paddingWidth);

__global__ void convolutionKernelShared(unsigned char* input, float* output, float* kernel, 
                                        int inputHeight, int inputWidth, 
                                        int kernelHeight, int kernelWidth, 
                                        int strideHeight, int strideWidth, 
                                        int paddingHeight, int paddingWidth);

__global__ void initializeWeights(float* weights, int size, unsigned long long seed, float min, float max);