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


// CUDA kernel for ReLU activation function
__global__ void reluKernelMultiple(const float* inputImage, float* outputImage, int width, int height, int numChannels) {
    // 3D thread index
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channelIdx = blockIdx.z;

    // perform relu function
    if (row < height && col < width && channelIdx < numChannels) {
        outputImage[channelIdx * width * height + row * width + col] = inputImage[channelIdx * width * height + row * width + col] > 0 ? inputImage[channelIdx * width * height + row * width + col] : 0;
    }
}