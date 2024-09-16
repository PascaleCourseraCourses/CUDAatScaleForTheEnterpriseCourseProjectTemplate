#include <../lib/activation.h>


// CUDA kernel for ReLU activation function
__global__ void reluKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height) {
    // 2D thread index
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // perform relu function
    if (row < height && col < width) {
        unsigned char result = inputImage[row * width + col] > 0 ? inputImage[row * width + col] : 0;
        outputImage[row * width + col] = min(max(int(result), 0), 255); // Clamp result between 0 and 255
    }
}
