#include <cuda_runtime.h>
#include <iostream>

#define FILTER_WIDTH 3 // Example filter size (3x3 filter)
#define TILE_WIDTH 16  // Example tile width for shared memory optimization

__global__ void convolutionKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, const float* filter);