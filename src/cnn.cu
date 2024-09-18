#include "../lib/cnn.h"

// Constructor
CNN::CNN(int inputHeight, int inputWidth,
         int dstHeight, int dstWidth, 
         int filterHeight, int filterWidth,
         int strideHeight, int strideWidth,
         int paddingHeight, int paddingWidth,
         int numFilters)
    : inputHeight(inputHeight), inputWidth(inputWidth),
      dstHeight(dstHeight), dstWidth(dstWidth), 
      filterHeight(filterHeight), filterWidth(filterWidth),
      strideHeight(strideHeight), strideWidth(strideWidth),
      paddingHeight(paddingHeight), paddingWidth(paddingWidth),
      numFilters(numFilters) {
    AllocateMemory();
    SetFilters();
}

// Destructor
CNN::~CNN() {
    FreeMemory();
}

// Allocate memory for GPU data
void CNN::AllocateMemory() {
    size_t size_input = inputWidth * inputHeight * sizeof(unsigned char);
    cudaMalloc(&deviceInput, size_input);

    size_t size_resized = dstHeight * dstWidth * sizeof(unsigned char);
    cudaMalloc(&deviceResized, size_resized);

    size_t filter_size = filterHeight * filterWidth * sizeof(float);
    cudaMalloc(&deviceFilters, filter_size);

    convHeight = (dstHeight + 2 * paddingHeight - filterHeight) / strideHeight + 1;
    convWidth = (dstWidth + 2 * paddingWidth - filterWidth) / strideWidth + 1;
    size_t conv_size = convWidth * convHeight * sizeof(float);
    cudaMalloc(&deviceConv, conv_size);

    size_t act_size = convWidth * convHeight * sizeof(float);
    cudaMalloc(&deviceAct, act_size);

    poolHeight = (convHeight + 2 * paddingHeight - filterHeight) / strideHeight + 1;
    poolWidth = (convWidth + 2 * paddingWidth - filterWidth) / strideWidth + 1;
    size_t pool_size = poolWidth * poolHeight * sizeof(float);
    cudaMalloc(&devicePool, pool_size);

    // cudaMalloc(&deviceGradInput, /* size */);

    // cudaMalloc(&deviceGradFilters, /* size */);

    // cudaMalloc(&deviceGradOutput, /* size */);

    blockSizeconv = dim3(TILE_WIDTH, TILE_WIDTH); 
    gridSizeconv = dim3((convWidth + TILE_WIDTH - 1) / TILE_WIDTH, (convHeight + TILE_WIDTH - 1) / TILE_WIDTH);
    sharedMemSizeconv = (TILE_WIDTH + filterWidth  - 1) * (TILE_WIDTH + filterHeight  - 1) * sizeof(float) + filterHeight * filterWidth * sizeof(float);

    blockSizepool = dim3(TILE_WIDTH, TILE_WIDTH); 
    gridSizepool = dim3((poolWidth + TILE_WIDTH - 1) / TILE_WIDTH, (poolHeight + TILE_WIDTH - 1) / TILE_WIDTH);
    sharedMemSizepool = (TILE_WIDTH + filterWidth  - 1) * (TILE_WIDTH + filterHeight  - 1) * sizeof(float);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " in File " << __FILE__
                  << " in line " << __LINE__
                  << std::endl;
        exit(EXIT_FAILURE);
    }

}

// Free GPU memory
void CNN::FreeMemory() {
    cudaFree(deviceInput);
    cudaFree(deviceResized);
    cudaFree(deviceFilters);
    cudaFree(deviceConv);
    cudaFree(devicePool);
    cudaFree(deviceAct);

    // cudaFree(deviceGradInput);
    // cudaFree(deviceGradFilters);
    // cudaFree(deviceGradOutput);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " in File " << __FILE__
                  << " in line " << __LINE__
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Forward pass
void CNN::ForwardPass(unsigned char* hostInput) {

    // Copy from host to device
    cudaError_t err = cudaMemcpy(deviceInput, hostInput, inputWidth * inputHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                << " in File " << __FILE__
                << " in line " << __LINE__
                << std::endl;
        exit(EXIT_FAILURE);
    }

    // Resize the image
    resizeImageGPU();

    // deviceInput = hostInput;
    LaunchConvolutionKernel();
    LaunchActivationKernel();
    LaunchMaxPoolingKernel();
}

// // Backward pass
// void CNN::BackwardPass(float* deviceGradOutput) {
//     cudaMemcpy(this->deviceGradOutput, deviceGradOutput, /* size */, cudaMemcpyHostToDevice);
//     LaunchMaxPoolingBackwardKernel();
//     LaunchActivationBackwardKernel();
//     LaunchConvolutionBackwardKernel();
//     UpdateFilters();
// }

// Implement convolution kernel launch
void CNN::LaunchConvolutionKernel() {

    convolutionKernelShared<<<gridSizeconv, blockSizeconv, sharedMemSizeconv>>>(deviceResized, deviceConv, deviceFilters,
                                                                                dstHeight, dstWidth,
                                                                                filterHeight, filterWidth,
                                                                                strideHeight, strideWidth,
                                                                                paddingHeight, paddingWidth);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " in File " << __FILE__
                  << " in line " << __LINE__
                  << std::endl;
        exit(EXIT_FAILURE);
    }

}

// Implement activation kernel launch
void CNN::LaunchActivationKernel() {

    reluKernel<<<gridSizeconv, blockSizeconv>>>(deviceConv, deviceAct, convWidth, convHeight);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " in File " << __FILE__
                  << " in line " << __LINE__
                  << std::endl;
        exit(EXIT_FAILURE);
    }

}

// Implement max pooling kernel launch
void CNN::LaunchMaxPoolingKernel() {
    MaxPoolingKernelShared<<<gridSizepool, blockSizepool, sharedMemSizepool>>>(deviceAct, devicePool,
                                                                                convHeight, convWidth,
                                                                                filterHeight, filterWidth,
                                                                                strideHeight, strideWidth,
                                                                                paddingHeight, paddingWidth);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " in File " << __FILE__
                  << " in line " << __LINE__
                  << std::endl;
        exit(EXIT_FAILURE);
    }

}




// //Implement convolution backward kernel
// void CNN::LaunchConvolutionBackwardKernel() {
//     ConvolutionBackwardKernel<<<gridSize, blockSize>>>(deviceGradInput, deviceGradFilters, deviceGradOutput,
//                                                        inputHeight, inputWidth,
//                                                        filterHeight, filterWidth,
//                                                        strideHeight, strideWidth,
//                                                        paddingHeight, paddingHeight);
//     cudaDeviceSynchronize();
// }

// // Implement activation backward kernel
// void CNN::LaunchActivationBackwardKernel() {
//     int outputSize = /* calculate size */;
//     ActivationBackwardKernel<<<gridSize, blockSize>>>(deviceGradOutput, deviceGradOutput, outputSize);
//     cudaDeviceSynchronize();
// }

// // Implement max pooling backward kernel
// void CNN::LaunchMaxPoolingBackwardKernel() {
//     MaxPoolingBackwardKernel<<<gridSize, blockSize>>>(deviceGradOutput, deviceGradInput,
//                                                       /* pooling parameters */);
//     cudaDeviceSynchronize();
// }

// // Update filters using gradient descent
// void CNN::UpdateFilters() {
//     float learningRate = 0.01f;
//     UpdateFiltersKernel<<<gridSize, blockSize>>>(deviceFilters, deviceGradFilters, learningRate,
//                                                  filterHeight, filterWidth, numFilters);
//     cudaDeviceSynchronize();
// }

// Set filters from host to device
void CNN::SetFilters() {
    int filter_num_elements = filterHeight * filterWidth;
    initializeWeights<<<1, filter_num_elements>>>(deviceFilters, filter_num_elements, 1234ULL, -0.5f, 0.5f);
}

void CNN::resizeImageGPU() {

    NppiSize srcSize = {inputWidth, inputHeight}; // Source size
    NppiSize dstSize = {dstWidth, dstHeight}; // Destination size
    NppiRect srcRectROI = {0, 0, inputWidth, inputHeight}; // Source ROI
    NppiRect dstRectROI = {0, 0, dstWidth, dstHeight}; // Destination ROI

    size_t srcStep = inputWidth * sizeof(unsigned char); // Row step for source image
    size_t dstStep = dstWidth * sizeof(unsigned char); // Row step for destination image

    NppStatus status = nppiResize_8u_C1R(
        deviceInput, srcStep, srcSize, srcRectROI,
        deviceResized, dstStep, dstSize, dstRectROI,
        NPPI_INTER_LINEAR
    );

    if (status != NPP_SUCCESS) {
        std::cerr << "NPP error: " << status << std::endl;
    }

    cudaDeviceSynchronize();

}


// Get output from device to host
std::tuple<int, int, float*> CNN::GetOutput() {

    // int output_size = poolWidth * poolHeight;
    // float* hostOutputfloat = new float[output_size]; 
    // unsigned char* hostOutputuchar = new unsigned char[output_size];

    // cudaError_t err = cudaMemcpy(hostOutputfloat, devicePool, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(err)
    //               << " in File " << __FILE__
    //               << " in line " << __LINE__
    //               << std::endl;
    //     exit(EXIT_FAILURE);
    // }

    // float minRange = *std::min_element(hostOutputfloat, hostOutputfloat + output_size);
    // float maxRange = *std::max_element(hostOutputfloat, hostOutputfloat + output_size);

    // std::cout << minRange << std::endl;
    // std::cout << maxRange << std::endl;

    // if (maxRange == minRange) {
    //     std::fill(hostOutputuchar, hostOutputuchar + output_size, 0);  
    // } else {
    //     for (int i = 0; i < output_size; ++i) {
    //         unsigned char scaledValue = static_cast<unsigned char>(255.0f * (hostOutputfloat[i] - minRange) / (maxRange - minRange));
    //         hostOutputuchar[i] = scaledValue;
    //     }
    // }

    // cv::Mat convMat(poolHeight, poolWidth, CV_8UC1, hostOutputuchar);
    // // cv::imwrite("./output/output.png", convMat);


    // delete[] hostOutputfloat;
    // delete[] hostOutputuchar;


    return {poolWidth, poolHeight, devicePool};
}
