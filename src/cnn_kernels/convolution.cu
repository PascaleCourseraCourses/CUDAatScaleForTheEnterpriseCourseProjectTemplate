#include <../lib/convolution.h>
#include <curand_kernel.h>
#include <random>

// CUDA kernel for 2D convolution
// __global__ void convolutionKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, const float* filter, const int filterwidth, int padding, int stride) {
//     // 2D thread index
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     int row = blockIdx.y * blockDim.y + threadIdx.y;

//     // Allocate shared memory for the tile because convolution on each pixel requires values of neighbor pixels
//     // shared memory can be accessed by threads in a block
//     // Shared memory size should be enough for number of threads per block or shared memory overflow will happen.
//     __shared__ unsigned char sharedMem[TILE_WIDTH + MAX_FILTER_WIDTH - 1][TILE_WIDTH + MAX_FILTER_WIDTH - 1];

//     // Calculate the output image size
//     int outputWidth = (width - filterwidth + 2 * padding) / stride + 1;
//     int outputHeight = (height - filterwidth + 2 * padding) / stride + 1;

//     // Calculate the shared memory coordinates
//     int sharedRow = threadIdx.y + filterwidth / 2;
//     int sharedCol = threadIdx.x + filterwidth / 2;

//     int temp_row = row;
//     int temp_col = col;

//     if (row < height && col < width) {
//         sharedMem[sharedRow][sharedCol] = inputImage[temp_row * width + col];
//     } 

//     // Top edge padding
//     if (threadIdx.y < filterwidth / 2){

//         if (row < filterwidth / 2){

//             if(row - padding < 0){

//                 sharedMem[sharedRow - padding][sharedCol] = 0;

//             }
//         } else{

//             sharedMem[sharedRow - padding][sharedCol] = inputImage[(temp_row - filterwidth / 2) * width + temp_col];;

//         }

//     }

//     // Bottom edge padding
//     if (threadIdx.y > blockDim.y - filterwidth / 2 - 1){

//         if (row > height - filterwidth / 2 - 1){

//             if(row + padding > height - 1){

//                 sharedMem[sharedRow + padding][sharedCol] = 0;

//             }
//         } else{

//             sharedMem[sharedRow + padding][sharedCol] = inputImage[(temp_row + filterwidth / 2) * width + temp_col];

//         }

//     }

//     // Left edge padding
//     if (threadIdx.x < filterwidth / 2){

//         if (col < filterwidth / 2){

//             if(col - padding < 0){

//                 sharedMem[sharedRow][sharedCol - padding] = 0;

//             }
//         } else{

//             sharedMem[sharedRow][sharedCol - padding] = inputImage[temp_row * width + (temp_col - filterwidth / 2)];

//         }

//     }


//     // Right edge padding
//     if (threadIdx.x > blockDim.x - filterwidth / 2 - 1){

//         if (col > width - filterwidth / 2 - 1){

//             if(col + padding > width - 1){

//                 sharedMem[sharedRow][sharedCol + padding] = 0;

//             }

//         } else{

//             sharedMem[sharedRow][sharedCol + padding] = inputImage[temp_row * width + (temp_col + filterwidth / 2)];

//         }

//     }

//     __syncthreads();

//     // Apply the convolution filter (if within image boundaries)
//     if (row >= filterwidth / 2 - padding && row < outputHeight - filterwidth / 2 + padding && col >= filterwidth / 2 - padding && col < outputWidth - filterwidth / 2 + padding && row % stride == 0 && col % stride == 0 ) {
//         float result = 0.0f;


//         for (int i = -filterwidth / 2; i <= filterwidth / 2; i++) {
//             for (int j = -filterwidth / 2; j <= filterwidth / 2; j++) {
                
//                 result += sharedMem[sharedRow + i][sharedCol + j] * filter[(i + filterwidth / 2) * filterwidth + (j + filterwidth / 2)];
//             }
//         }
//         outputImage[row / stride * width + col / stride] = min(max(int(result), 0), 255); // Clamp result between 0 and 255
//     }
// }



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