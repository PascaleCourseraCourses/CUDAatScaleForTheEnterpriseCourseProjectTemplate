#include <../lib/pooling.h>


// CUDA kernel for max pooling operation
__global__ void MaxPoolingKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height) {
    // 2D thread index
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Allocate shared memory for the tile because convolution on each pixel requires values of neighbor pixels
    // shared memory can be accessed by threads in a block
    // Shared memory size should be enough for number of threads per block or shared memory overflow will happen.
    __shared__ unsigned char sharedMem[TILE_WIDTH + FILTER_WIDTH - 1][TILE_WIDTH + FILTER_WIDTH - 1];

    // Calculate the shared memory coordinates
    int sharedRow = threadIdx.y + FILTER_WIDTH / 2;
    int sharedCol = threadIdx.x + FILTER_WIDTH / 2;

    // Load the input image data into shared memory
    // check if thread is within the image boundaries
    if (row < height && col < width) {
        sharedMem[sharedRow][sharedCol] = inputImage[row * width + col];
    }

    // Load halo elements (extra pixels needed for the convolution)
    // Handle boundaries by padding with zeros 
    if (threadIdx.x < FILTER_WIDTH / 2) {
        sharedMem[sharedRow][sharedCol - FILTER_WIDTH / 2] = (col >= FILTER_WIDTH / 2) ? inputImage[row * width + (col - FILTER_WIDTH / 2)] : -INFINITY;
    }
    if (threadIdx.x >= blockDim.x - FILTER_WIDTH / 2) {
        sharedMem[sharedRow][sharedCol + FILTER_WIDTH / 2] = (col + FILTER_WIDTH / 2 < width) ? inputImage[row * width + (col + FILTER_WIDTH / 2)] : -INFINITY;
    }
    if (threadIdx.y < FILTER_WIDTH / 2) {
        sharedMem[sharedRow - FILTER_WIDTH / 2][sharedCol] = (row >= FILTER_WIDTH / 2) ? inputImage[(row - FILTER_WIDTH / 2) * width + col] : -INFINITY;
    }
    if (threadIdx.y >= blockDim.y - FILTER_WIDTH / 2) {
        sharedMem[sharedRow + FILTER_WIDTH / 2][sharedCol] = (row + FILTER_WIDTH / 2 < height) ? inputImage[(row + FILTER_WIDTH / 2) * width + col] : -INFINITY;
    }

    __syncthreads();

    // Apply the convolution filter (if within image boundaries)
    if (row < height && col < width) {
        float max_result = 0.0f;
        for (int i = -FILTER_WIDTH / 2; i <= FILTER_WIDTH / 2; i++) {
            for (int j = -FILTER_WIDTH / 2; j <= FILTER_WIDTH / 2; j++) {
                max_result = max(max_result, static_cast<float>(sharedMem[sharedRow + i][sharedCol + j]));
            }
        }
        outputImage[row * width + col] = min(max(int(max_result), 0), 255); // Clamp result between 0 and 255
    }
}

