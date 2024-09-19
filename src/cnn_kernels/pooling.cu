#include <../lib/pooling.h>

// CUDA kernel for max pooling operation
__global__ void MaxPoolingKernelShared(const float* input, float* output, 
                              int inputHeight, int inputWidth, 
                              int kernelHeight, int kernelWidth, 
                              int strideHeight, int strideWidth, 
                              int paddingHeight, int paddingWidth) {

    // Shared memory for the input tile and kernel
    extern __shared__ float sharedMem[];

    // Pointers to shared memory segments
    float* sharedTile = sharedMem;

    // Calculate output pixel's row and column index
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the thread's index for the input tile including padding (halo region)
    int inRow = outRow * strideHeight - paddingHeight + threadIdx.y;
    int inCol = outCol * strideWidth - paddingWidth + threadIdx.x;

    // Load input tile into shared memory (with boundary checks)
    if (inRow >= 0 && inRow < inputHeight && inCol >= 0 && inCol < inputWidth) {
        sharedTile[threadIdx.y * (TILE_WIDTH + kernelWidth - 1)  + threadIdx.x] = input[inRow * inputWidth + inCol];
    } else {
        sharedTile[threadIdx.y * (TILE_WIDTH + kernelWidth - 1)  + threadIdx.x] = 0.0;  // Assign -INFINITY for padding
    }
    
    // Synchronize threads to ensure all shared memory is loaded
    __syncthreads();

    // Perform convolution (only valid output threads compute)
    if (threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH) {
        float max_value = 0.0;
        for (int kh = 0; kh < kernelHeight; kh++) {
            for (int kw = 0; kw < kernelWidth; kw++) {
                max_value = max(max_value, sharedTile[(threadIdx.y + kh) * (TILE_WIDTH + kernelWidth - 1) + (threadIdx.x + kw)]);
            }
        }

        // Store the result in the output matrix if within valid output bounds
        if (outRow < (inputHeight + 2 * paddingHeight - kernelHeight) / strideHeight + 1 &&
            outCol < (inputWidth + 2 * paddingWidth - kernelWidth) / strideWidth + 1) {
            output[outRow * ((inputWidth + 2 * paddingWidth - kernelWidth) / strideWidth + 1) + outCol] = max_value;
        }
    }

    // Synchronize threads before ending the kernel
    __syncthreads();
}


__global__ void MaxPoolingKernelSharedMultiple(const float* input, float* output, 
                              int inputHeight, int inputWidth, 
                              int kernelHeight, int kernelWidth, 
                              int strideHeight, int strideWidth, 
                              int paddingHeight, int paddingWidth, int numChannels) {

    // Shared memory for the input tile and kernel
    extern __shared__ float sharedMem[];

    float* sharedTile = sharedMem;

    // Calculate output pixel's row and column index
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int channelIdx = blockIdx.z; 

    // Calculate the thread's index for the input tile including padding (halo region)
    int inRow = outRow * strideHeight - paddingHeight + threadIdx.y;
    int inCol = outCol * strideWidth - paddingWidth + threadIdx.x;

    // Initialize the shared memory tile for each channel
    int sharedMemTileWidth = TILE_WIDTH + kernelWidth - 1;
    int sharedMemTileSize = sharedMemTileWidth * sharedMemTileWidth;

    int outputHeight = (inputHeight + 2 * paddingHeight - kernelHeight) / strideHeight + 1;
    int outputWidth = (inputWidth + 2 * paddingWidth - kernelWidth) / strideWidth + 1;

    // Load input into shared memory
    if (inRow >= 0 && inRow < inputHeight && inCol >= 0 && inCol < inputWidth && channelIdx < numChannels) {
        sharedTile[threadIdx.y * sharedMemTileWidth + threadIdx.x] = input[channelIdx * inputHeight * inputWidth + inRow * inputWidth + inCol];
    } else if (channelIdx < numChannels) {
        sharedTile[threadIdx.y * sharedMemTileWidth + threadIdx.x] = 0.0f;  // Padding
    }
    
    // Synchronize threads to ensure all shared memory is loaded
    __syncthreads();

    // Perform convolution (only valid output threads compute)
    if (threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH && channelIdx < numChannels) {

        float max_value = 0.0;
        for (int kh = 0; kh < kernelHeight; ++kh) {
            for (int kw = 0; kw < kernelWidth; ++kw) {
                max_value = max(max_value, sharedTile[(threadIdx.y + kh) * sharedMemTileWidth + (threadIdx.x + kw)]);
            }
        }

        // Store the result in the output matrix if within valid output bounds
        if (outRow < outputHeight && outCol < outputWidth) {
            output[channelIdx * outputHeight * outputWidth + outRow * outputWidth + outCol] = max_value;
        }
    }

    // Synchronize threads before ending the kernel
    __syncthreads();
}




