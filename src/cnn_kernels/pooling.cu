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
    int inRow = outRow * strideHeight - paddingHeight + kernelHeight / 2;
    int inCol = outCol * strideWidth - paddingWidth + kernelWidth / 2;

    // Initialize the shared memory tile for each channel
    int sharedMemTileWidth = TILE_WIDTH + kernelWidth - 1;
    int sharedMemTileSize = sharedMemTileWidth * sharedMemTileWidth;

    int outputHeight = (inputHeight + 2 * paddingHeight - kernelHeight) / strideHeight + 1;
    int outputWidth = (inputWidth + 2 * paddingWidth - kernelWidth) / strideWidth + 1;

    // Load input into shared memory
    if (inRow >= 0 && inRow <= inputHeight - 1 && inCol >= 0 && inCol <= inputWidth - 1 && channelIdx < numChannels) {
        sharedTile[(threadIdx.y + kernelHeight / 2) * sharedMemTileWidth + (threadIdx.x + kernelWidth / 2)] = input[channelIdx * inputHeight * inputWidth + inRow * inputWidth + inCol];
    }

    if (inRow < kernelHeight / 2 && channelIdx < numChannels){
        sharedTile[(threadIdx.y + kernelHeight / 2 - paddingHeight) * sharedMemTileWidth + (threadIdx.x + kernelWidth / 2)] = -FLT_MAX;

    }

    if (inCol < kernelWidth / 2 && channelIdx < numChannels){
        sharedTile[(threadIdx.y + kernelHeight / 2) * sharedMemTileWidth + (threadIdx.x + kernelWidth / 2 - paddingWidth)] = -FLT_MAX;

    }

    if (inRow >= inputHeight - kernelHeight / 2 && channelIdx < numChannels){
        sharedTile[(threadIdx.y + kernelHeight / 2 + paddingHeight) * sharedMemTileWidth + (threadIdx.x + kernelWidth / 2)] = -FLT_MAX;

    }

    if (inCol >= inputWidth - kernelWidth / 2 && channelIdx < numChannels){
        sharedTile[(threadIdx.y + kernelHeight / 2) * sharedMemTileWidth + (threadIdx.x + kernelWidth / 2 + paddingWidth)] = -FLT_MAX;

    }
    
    // Synchronize threads to ensure all shared memory is loaded
    __syncthreads();

    // Perform convolution (only valid output threads compute)
    if (threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH && channelIdx < numChannels) {

        float max_value = -FLT_MAX;
        for (int kh = - kernelHeight / 2; kh <= kernelHeight / 2; ++kh) {
            for (int kw = - kernelWidth / 2; kw <= kernelWidth / 2; ++kw) {
                max_value = max(max_value, sharedTile[(threadIdx.y + kernelHeight / 2 + kh) * sharedMemTileWidth + (threadIdx.x + kernelWidth / 2 + kw)]);
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

__global__ void MaxPoolingBackwardKernelSharedMultiple(const float* input, const float* d_output, float* d_input, 
                                                       int inputHeight, int inputWidth, 
                                                       int kernelHeight, int kernelWidth, 
                                                       int strideHeight, int strideWidth, 
                                                       int paddingHeight, int paddingWidth, int numChannels) {

    // Shared memory for the input tile
    extern __shared__ float sharedMem[];

    float* sharedTile = sharedMem;

    // Calculate output pixel's row and column index
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int channelIdx = blockIdx.z;

    // Calculate the thread's index for the input tile including padding
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
        sharedTile[threadIdx.y * sharedMemTileWidth + threadIdx.x] = -FLT_MAX;  // Padding
    }
    

    // Synchronize threads to ensure all shared memory is loaded
    __syncthreads();

    // Perform the backward pass (only valid output threads compute)
    if (threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH && channelIdx < numChannels) {

        float max_value = -FLT_MAX;
        int max_idx = -1;

        // Find the maximum value in the pooling window and its index
        for (int kh = 0; kh < kernelHeight; ++kh) {
            for (int kw = 0; kw < kernelWidth; ++kw) {

                int sharedIdx = (threadIdx.y + kh) * sharedMemTileWidth + (threadIdx.x + kw);

                if (sharedTile[sharedIdx] > max_value) {

                    max_value = sharedTile[sharedIdx];
                    max_idx = sharedIdx;

                }
            }
        }

        // Store the gradient to the position of the max element
        if (outRow < outputHeight && outCol < outputWidth && max_idx != -1) {
            // Retrieve the gradient from the output and propagate it to the max index
            float grad = d_output[channelIdx * outputHeight * outputWidth + outRow * outputWidth + outCol];
            int inRowIdx = inRow + (max_idx / sharedMemTileWidth);
            int inColIdx = inCol + (max_idx % sharedMemTileWidth);

            if (inRowIdx >= 0 && inRowIdx < inputHeight && inColIdx >= 0 && inColIdx < inputWidth) {
                atomicAdd(&d_input[channelIdx * inputHeight * inputWidth + inRowIdx * inputWidth + inColIdx], grad);
            }
        }
    }

    // Synchronize threads before ending the kernel
    __syncthreads();
}




