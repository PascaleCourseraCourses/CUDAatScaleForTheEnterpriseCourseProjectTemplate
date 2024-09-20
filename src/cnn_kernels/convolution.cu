#include <../lib/convolution.h>
#include <curand_kernel.h>
#include <random>

// CUDA kernel for 2D convolution
__global__ void convolutionKernel(unsigned char* input, unsigned char* output, int inputWidth, int inputHeight, const float* kernel,
                       int kernelHeight, int kernelWidth, 
                       int strideHeight, int strideWidth, 
                       int paddingHeight, int paddingWidth) {
    
    // Calculate output pixel's row and column index
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate padded input dimensions
    int paddedHeight = inputHeight + 2 * paddingHeight;
    int paddedWidth = inputWidth + 2 * paddingWidth;

    // Initialize convolution sum for the output pixel
    float sum = 0.0f;

    // Perform convolution for each output pixel
    for (int kh = 0; kh < kernelHeight; kh++) {
        for (int kw = 0; kw < kernelWidth; kw++) {
            // Calculate input pixel's row and column index considering stride and padding
            int inRow = outRow * strideHeight + kh - paddingHeight;
            int inCol = outCol * strideWidth + kw - paddingWidth;

            // Boundary check for the input matrix (handle padding)
            if (inRow >= 0 && inRow < inputHeight && inCol >= 0 && inCol < inputWidth) {
                sum += input[inRow * inputWidth + inCol] * kernel[kh * kernelWidth + kw];
            }
        }
    }

    // Write result to output matrix if within valid range
    if (outRow < (paddedHeight - kernelHeight) / strideHeight + 1 && 
        outCol < (paddedWidth - kernelWidth) / strideWidth + 1) {
        output[outRow * ((paddedWidth - kernelWidth) / strideWidth + 1) + outCol] = sum;
    }
}


__global__ void convolutionKernelShared(unsigned char* input, float* output, float* kernel, 
                              int inputHeight, int inputWidth, 
                              int kernelHeight, int kernelWidth, 
                              int strideHeight, int strideWidth, 
                              int paddingHeight, int paddingWidth) {

    // Shared memory for the input tile and kernel
    extern __shared__ float sharedMem[];

    // Pointers to shared memory segments
    float* sharedTile = sharedMem;
    float* sharedKernel = sharedMem + (TILE_WIDTH + kernelHeight - 1) * (TILE_WIDTH + kernelWidth - 1);

    // Calculate output pixel's row and column index
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the thread's index for the input tile including padding (halo region)
    int inRow = outRow * strideHeight - paddingHeight + threadIdx.y;
    int inCol = outCol * strideWidth - paddingWidth + threadIdx.x;

    // Load kernel into shared memory (only by one thread)
    if (threadIdx.y < kernelHeight && threadIdx.x < kernelWidth) {
        sharedKernel[threadIdx.y * kernelWidth + threadIdx.x] = kernel[threadIdx.y * kernelWidth + threadIdx.x];
    }

    // Load input tile into shared memory (with boundary checks)
    if (inRow >= 0 && inRow < inputHeight && inCol >= 0 && inCol < inputWidth) {
        sharedTile[threadIdx.y * (TILE_WIDTH + kernelWidth - 1) + threadIdx.x] = input[inRow * inputWidth + inCol];
    } else {
        sharedTile[threadIdx.y * (TILE_WIDTH + kernelWidth - 1) + threadIdx.x] = 0.0f; // Padding
    }

    // Synchronize threads to ensure all shared memory is loaded
    __syncthreads();

    // Perform convolution (only valid output threads compute)
    if (threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH) {
        float sum = 0.0f;
        for (int kh = 0; kh < kernelHeight; kh++) {
            for (int kw = 0; kw < kernelWidth; kw++) {
                sum += sharedTile[(threadIdx.y + kh) * (TILE_WIDTH + kernelWidth - 1) + (threadIdx.x + kw)] * sharedKernel[kh * kernelWidth + kw];
            }
        }

        // Store the result in the output matrix if within valid output bounds
        if (outRow < (inputHeight + 2 * paddingHeight - kernelHeight) / strideHeight + 1 &&
            outCol < (inputWidth + 2 * paddingWidth - kernelWidth) / strideWidth + 1) {
            output[outRow * ((inputWidth + 2 * paddingWidth - kernelWidth) / strideWidth + 1) + outCol] = sum;
        }
    }

    // Synchronize threads before ending the kernel
    __syncthreads();
}


__global__ void convolutionKernelSharedMultiple(unsigned char* input, float* output, float* kernels, 
                              int inputHeight, int inputWidth, 
                              int kernelHeight, int kernelWidth, 
                              int strideHeight, int strideWidth, 
                              int paddingHeight, int paddingWidth,
                              int numFilters, int numChannels) {

    // Shared memory for the input tile and kernel
    extern __shared__ float sharedMem[];

    // Calculate output pixel's row and column index
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int filterIdx = blockIdx.z; 

    int outputHeight = (inputHeight + 2 * paddingHeight - kernelHeight) / strideHeight + 1;
    int outputWidth = (inputWidth + 2 * paddingWidth - kernelWidth) / strideWidth + 1;

    // Calculate the thread's index for the input tile including padding (halo region)
    int inRow = outRow * strideHeight - paddingHeight + kernelHeight / 2;
    int inCol = outCol * strideWidth - paddingWidth + kernelWidth / 2;
    int channelIdx = threadIdx.z;

    // Initialize the shared memory tile for each channel
    int sharedMemTileWidth = TILE_WIDTH + kernelWidth - 1;
    int sharedMemTileSize = sharedMemTileWidth * sharedMemTileWidth;
    
    float* sharedTile = &sharedMem[channelIdx * sharedMemTileSize];

    // Load input into shared memory
    if (inRow >= 0 && inRow <= inputHeight - 1 && inCol >= 0 && inCol <= inputWidth - 1 && channelIdx < numChannels) {
        sharedTile[(threadIdx.y + kernelHeight / 2) * sharedMemTileWidth + (threadIdx.x + kernelWidth / 2)] = input[channelIdx * inputHeight * inputWidth + inRow * inputWidth + inCol];
    }

    if (inRow < kernelHeight / 2 && channelIdx < numChannels){
        sharedTile[(threadIdx.y + kernelHeight / 2 - paddingHeight) * sharedMemTileWidth + (threadIdx.x + kernelWidth / 2)] = 0;

    }

    if (inCol < kernelWidth / 2 && channelIdx < numChannels){
        sharedTile[(threadIdx.y + kernelHeight / 2) * sharedMemTileWidth + (threadIdx.x + kernelWidth / 2 - paddingWidth)] = 0;

    }

    if (inRow >= inputHeight - kernelHeight / 2 && channelIdx < numChannels){
        sharedTile[(threadIdx.y + kernelHeight / 2 + paddingHeight) * sharedMemTileWidth + (threadIdx.x + kernelWidth / 2)] = 0;

    }

    if (inCol >= inputWidth - kernelWidth / 2 && channelIdx < numChannels){
        sharedTile[(threadIdx.y + kernelHeight / 2) * sharedMemTileWidth + (threadIdx.x + kernelWidth / 2 + paddingWidth)] = 0;

    }

    // Synchronize threads to ensure all shared memory is loaded
    __syncthreads();

     // Perform convolution (only valid output threads compute)
    if (threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH && filterIdx < numFilters && channelIdx < numChannels) {

        float sum = 0.0f;
        // Access the kernel for the current filter (on register memory)
        float* filterKernel = kernels + filterIdx * numChannels * kernelHeight * kernelWidth + channelIdx * kernelHeight * kernelWidth;
            
        for (int kh = - kernelHeight / 2; kh <= kernelHeight / 2; kh++) {
            for (int kw = - kernelWidth / 2; kw <= kernelWidth / 2; kw++) {
                sum += sharedTile[(threadIdx.y + kernelHeight / 2 + kh) * sharedMemTileWidth + (threadIdx.x + kernelWidth / 2 + kw)] * filterKernel[kh * kernelWidth + kw];
            }
        }

        // Store the result in the output array for the corresponding filter
        if (outRow < outputHeight && outCol < outputWidth) {
            int outputIdx = filterIdx * outputHeight * outputWidth + outRow * outputWidth + outCol;
            atomicAdd(&output[outputIdx], sum);
        }
    }

    // Synchronize threads before ending the kernel
    __syncthreads();
}



__global__ void convolutionBackwardKernelSharedMultiple(unsigned char* input, float* dOutput, float* dInput,
                                                        float* dKernels, float* kernels,
                                                        int inputHeight, int inputWidth, 
                                                        int kernelHeight, int kernelWidth, 
                                                        int strideHeight, int strideWidth, 
                                                        int paddingHeight, int paddingWidth,
                                                        int numFilters, int numChannels) {
    
    // Shared memory for the input tile and kernel
    extern __shared__ float sharedMem[];

    // Calculate output pixel's row and column index
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int filterIdx = blockIdx.z; 

    int outputHeight = (inputHeight + 2 * paddingHeight - kernelHeight) / strideHeight + 1;
    int outputWidth = (inputWidth + 2 * paddingWidth - kernelWidth) / strideWidth + 1;

    // Calculate the thread's index for the input tile including padding (halo region)
    int inRow = outRow * strideHeight - paddingHeight + threadIdx.y;
    int inCol = outCol * strideWidth - paddingWidth + threadIdx.x;
    int channelIdx = threadIdx.z;

    // Initialize the shared memory tile for each channel
    int sharedMemTileWidth = TILE_WIDTH + kernelWidth - 1;
    int sharedMemTileSize = sharedMemTileWidth * sharedMemTileWidth;

    float* sharedTile = &sharedMem[channelIdx * sharedMemTileSize];

    // Load input into shared memory
    if (inRow >= 0 && inRow < inputHeight && inCol >= 0 && inCol < inputWidth && channelIdx < numChannels) {
        sharedTile[threadIdx.y * sharedMemTileWidth + threadIdx.x] = input[channelIdx * inputHeight * inputWidth + inRow * inputWidth + inCol];
    } else if (channelIdx < numChannels) {
        sharedTile[threadIdx.y * sharedMemTileWidth + threadIdx.x] = 0.0f;  // Padding
    }

    // Synchronize threads to ensure all shared memory is loaded
    __syncthreads();

    // Backpropagation step to compute the gradient w.r.t input
    if (threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH && filterIdx < numFilters && channelIdx < numChannels) {

        float sum = 0.0f;
        float* filterKernel = kernels + filterIdx * numChannels * kernelHeight * kernelWidth + channelIdx * kernelHeight * kernelWidth;

        for (int kh = 0; kh < kernelHeight; kh++) {
            for (int kw = 0; kw < kernelWidth; kw++) {
                int h_in = outRow * strideHeight - paddingHeight + kh;
                int w_in = outCol * strideWidth - paddingWidth + kw;

                if (h_in >= 0 && h_in < inputHeight && w_in >= 0 && w_in < inputWidth) {
                    int flipped_kh = kernelHeight - 1 - kh;
                    int flipped_kw = kernelWidth - 1 - kw;
                    sum += dOutput[filterIdx * outputHeight * outputWidth + (outRow + kh) * outputWidth + (outCol+ kw)] * 
                           filterKernel[flipped_kh * kernelWidth + flipped_kw];
                }
            }
        }

        // Store the gradient w.r.t the input
        if (inRow >= 0 && inRow < inputHeight && inCol >= 0 && inCol < inputWidth) {
            atomicAdd(&dInput[channelIdx * inputHeight * inputWidth + inRow * inputWidth + inCol], sum);
        }
    }

    // Synchronize threads to ensure all computations are done before finishing
    __syncthreads();

    // Compute gradient w.r.t kernel weights
    if (outRow < outputHeight && outCol < outputWidth && filterIdx < numFilters && channelIdx < numChannels) {
        
        float gradKernel = 0.0f;
        for (int kh = 0; kh < kernelHeight; kh++) {
            for (int kw = 0; kw < kernelWidth; kw++) {
                int h_in = outRow * strideHeight - paddingHeight + kh;
                int w_in = outCol * strideWidth - paddingWidth + kw;

                if (h_in >= 0 && h_in < inputHeight && w_in >= 0 && w_in < inputWidth) {
                    gradKernel += dOutput[filterIdx * outputHeight * outputWidth + (outRow + kh) * outputWidth + (outCol + kw)] * 
                                  sharedTile[(threadIdx.y + kh) * sharedMemTileWidth + (threadIdx.x + kw)];
                }
            }
        }

        atomicAdd(&dKernels[filterIdx * numChannels * kernelHeight * kernelWidth + channelIdx * kernelHeight * kernelWidth], gradKernel);
    }
}



__global__ void initializeWeights(float* weights, int size, unsigned long long seed, float min, float max) {
    
    // Define the CUDA random state
    curandState state;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Initialize the random state with the seed
    curand_init(seed, idx, 0, &state);

    // Make sure to not go out of bounds
    if (idx < size) {
        // Generate a random float in the range [min, max]
        float randValue = curand_uniform(&state) * (max - min) + min;
        weights[idx] = randValue;
    }
}