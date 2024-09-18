#include <../lib/activation.h>


// CUDA kernel for ReLU activation function
__global__ void reluKernel(const float* inputImage, float* outputImage, int width, int height) {
    // 2D thread index
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // perform relu function
    if (row < height && col < width) {
        outputImage[row * width + col] = inputImage[row * width + col] > 0 ? inputImage[row * width + col] : 0;
    }
}
