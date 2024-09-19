#include "../lib/cnnlayer.h"

// Constructor
CNNLayer::CNNLayer(int inputHeight, int inputWidth,
         int dstHeight, int dstWidth, 
         int filterHeight, int filterWidth,
         int strideHeight, int strideWidth,
         int paddingHeight, int paddingWidth,
         int numFilters, int numChannels)
    : inputHeight(inputHeight), inputWidth(inputWidth),
      dstHeight(dstHeight), dstWidth(dstWidth), 
      filterHeight(filterHeight), filterWidth(filterWidth),
      strideHeight(strideHeight), strideWidth(strideWidth),
      paddingHeight(paddingHeight), paddingWidth(paddingWidth),
      numFilters(numFilters), numChannels(numChannels) {
    AllocateMemory();
    SetFilters();
}

// Destructor
CNNLayer::~CNNLayer() {
    FreeMemory();
}

// Allocate memory for GPU data
void CNNLayer::AllocateMemory() {
    size_t size_input = inputWidth * inputHeight * numChannels * sizeof(unsigned char);
    cudaMalloc(&deviceInput, size_input);

    size_t size_resized = dstHeight * dstWidth * numChannels * sizeof(unsigned char);
    cudaMalloc(&deviceResized, size_resized);

    size_t filter_size = filterHeight * filterWidth * numFilters * numChannels * sizeof(float);
    cudaMalloc(&deviceFilters, filter_size);

    convHeight = (dstHeight + 2 * paddingHeight - filterHeight) / strideHeight + 1;
    convWidth = (dstWidth + 2 * paddingWidth - filterWidth) / strideWidth + 1;
    size_t conv_size = convWidth * convHeight * numFilters * sizeof(float);
    cudaMalloc(&deviceConv, conv_size);

    size_t act_size = convWidth * convHeight * numFilters * sizeof(float);
    cudaMalloc(&deviceAct, act_size);

    poolHeight = (convHeight + 2 * paddingHeight - filterHeight) / strideHeight + 1;
    poolWidth = (convWidth + 2 * paddingWidth - filterWidth) / strideWidth + 1;
    size_t pool_size = poolWidth * poolHeight * numFilters * sizeof(float);
    cudaMalloc(&devicePool, pool_size);

    // cudaMalloc(&deviceGradInput, /* size */);

    // cudaMalloc(&deviceGradFilters, /* size */);

    // cudaMalloc(&deviceGradOutput, /* size */);

    blockSizeconv = dim3(TILE_WIDTH, TILE_WIDTH, numChannels); 
    gridSizeconv = dim3((convWidth + TILE_WIDTH - 1) / TILE_WIDTH, (convHeight + TILE_WIDTH - 1) / TILE_WIDTH, numFilters);
    sharedMemSizeconv = (TILE_WIDTH + filterWidth  - 1) * (TILE_WIDTH + filterHeight  - 1) * numChannels * sizeof(float);

    blockSizeact = dim3(TILE_WIDTH, TILE_WIDTH, 1); 
    gridSizeact = dim3((convWidth + TILE_WIDTH - 1) / TILE_WIDTH, (convHeight + TILE_WIDTH - 1) / TILE_WIDTH, numFilters);

    blockSizepool = dim3(TILE_WIDTH, TILE_WIDTH, 1); 
    gridSizepool = dim3((poolWidth + TILE_WIDTH - 1) / TILE_WIDTH, (poolHeight + TILE_WIDTH - 1) / TILE_WIDTH, numFilters);
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
void CNNLayer::FreeMemory() {
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
void CNNLayer::ForwardPass(unsigned char* hostInput) {

    // Copy from host to device
    cudaError_t err = cudaMemcpy(deviceInput, hostInput, inputWidth * inputHeight * numChannels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                << " in File " << __FILE__
                << " in line " << __LINE__
                << std::endl;
        exit(EXIT_FAILURE);
    }

    // Resize the image
    resizeImageGPU();

    LaunchConvolutionKernel();
    LaunchActivationKernel();
    LaunchMaxPoolingKernel();
}

// Backward pass
// void CNNLayer::BackwardPass(float* deviceGradOutput) {
//     cudaMemcpy(this->deviceGradOutput, deviceGradOutput, /* size */, cudaMemcpyHostToDevice);
//     LaunchMaxPoolingBackwardKernel();
//     LaunchActivationBackwardKernel();
//     LaunchConvolutionBackwardKernel();
//     UpdateFilters();
// }

// Implement convolution kernel launch
void CNNLayer::LaunchConvolutionKernel() {

    convolutionKernelSharedMultiple<<<gridSizeconv, blockSizeconv, sharedMemSizeconv>>>(deviceResized, deviceConv, deviceFilters,
                                                                                        dstHeight, dstWidth,
                                                                                        filterHeight, filterWidth,
                                                                                        strideHeight, strideWidth,
                                                                                        paddingHeight, paddingWidth,
                                                                                        numFilters, numChannels);

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
void CNNLayer::LaunchActivationKernel() {

    reluKernelMultiple<<<gridSizeact, blockSizeact>>>(deviceConv, deviceAct, convWidth, convHeight, numFilters);

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
void CNNLayer::LaunchMaxPoolingKernel() {
    MaxPoolingKernelSharedMultiple<<<gridSizepool, blockSizepool, sharedMemSizepool>>>(deviceAct, devicePool,
                                                                                        convHeight, convWidth,
                                                                                        filterHeight, filterWidth,
                                                                                        strideHeight, strideWidth,
                                                                                        paddingHeight, paddingWidth, numFilters);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " in File " << __FILE__
                  << " in line " << __LINE__
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // float* singlefitler = devicePool + 3 * poolHeight * poolWidth;
    // int output_size = poolWidth * poolHeight;
    // float* hostOutputfloat = new float[output_size]; 
    // unsigned char* hostOutputuchar = new unsigned char[output_size];

    // err = cudaMemcpy(hostOutputfloat, singlefitler, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(err)
    //                 << " in File " << __FILE__
    //                 << " in line " << __LINE__
    //                 << std::endl;
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

    // cv::Mat convMat(poolHeight, poolWidth, CV_MAKETYPE(CV_8U, 1), hostOutputuchar);
    // cv::imwrite("./output/temp.png", convMat);


    // delete[] hostOutputfloat;
    // delete[] hostOutputuchar;
}




// //Implement convolution backward kernel
// void CNNLayer::LaunchConvolutionBackwardKernel() {
//     ConvolutionBackwardKernel<<<gridSize, blockSize>>>(deviceGradInput, deviceGradFilters, deviceGradOutput,
//                                                        inputHeight, inputWidth,
//                                                        filterHeight, filterWidth,
//                                                        strideHeight, strideWidth,
//                                                        paddingHeight, paddingHeight);
//     cudaDeviceSynchronize();
// }

// // Implement activation backward kernel
// void CNNLayer::LaunchActivationBackwardKernel() {
//     int outputSize = /* calculate size */;
//     ActivationBackwardKernel<<<gridSize, blockSize>>>(deviceGradOutput, deviceGradOutput, outputSize);
//     cudaDeviceSynchronize();
// }

// Implement max pooling backward kernel
// void CNNLayer::LaunchMaxPoolingBackwardKernel() {
//     MaxPoolingBackwardKernel<<<gridSize, blockSize>>>(deviceGradOutput, deviceGradInput,
//                                                       /* pooling parameters */);
//     cudaDeviceSynchronize();
// }

// // Update filters using gradient descent
// void CNNLayer::UpdateFilters() {
//     float learningRate = 0.01f;
//     UpdateFiltersKernel<<<gridSize, blockSize>>>(deviceFilters, deviceGradFilters, learningRate,
//                                                  filterHeight, filterWidth, numFilters);
//     cudaDeviceSynchronize();
// }

// Set filters from host to device
void CNNLayer::SetFilters() {
    int filter_num_elements = filterHeight * filterWidth * numFilters * numChannels;
    initializeWeights<<<1, filter_num_elements>>>(deviceFilters, filter_num_elements, 1234ULL, -0.5f, 0.5f);
}

void CNNLayer::resizeImageGPU() {

    NppiSize srcSize = {inputWidth, inputHeight}; // Source size
    NppiSize dstSize = {dstWidth, dstHeight}; // Destination size
    NppiRect srcRectROI = {0, 0, inputWidth, inputHeight}; // Source ROI
    NppiRect dstRectROI = {0, 0, dstWidth, dstHeight}; // Destination ROI
    size_t srcStep = inputWidth * numChannels * sizeof(unsigned char); // Row step for source image
    size_t dstStep = dstWidth * numChannels * sizeof(unsigned char); // Row step for destination image
    if (numChannels == 3){

        NppStatus status = nppiResize_8u_C3R(
            deviceInput, srcStep, srcSize, srcRectROI,
            deviceResized, dstStep, dstSize, dstRectROI,
            NPPI_INTER_LINEAR
        );

        if (status != NPP_SUCCESS) {
            std::cerr << "NPP error: " << status << std::endl;
        }

    } else if (numChannels == 4){

        NppStatus status = nppiResize_8u_C4R(
            deviceInput, srcStep, srcSize, srcRectROI,
            deviceResized, dstStep, dstSize, dstRectROI,
            NPPI_INTER_LINEAR
        );
    
        if (status != NPP_SUCCESS) {
            std::cerr << "NPP error: " << status << std::endl;
        }
    } else if (numChannels == 1){

        NppStatus status = nppiResize_8u_C1R(
            deviceInput, srcStep, srcSize, srcRectROI,
            deviceResized, dstStep, dstSize, dstRectROI,
            NPPI_INTER_LINEAR
        );
    
        if (status != NPP_SUCCESS) {
            std::cerr << "NPP error: " << status << std::endl;
        }
    }

    cudaDeviceSynchronize();


    // int output_size = dstHeight * dstWidth * numChannels;
    // unsigned char* hostOutputuchar = new unsigned char[output_size];

    // cudaError_t err = cudaMemcpy(hostOutputuchar, deviceResized, output_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(err)
    //               << " in File " << __FILE__
    //               << " in line " << __LINE__
    //               << std::endl;
    //     exit(EXIT_FAILURE);
    // }

    // cv::Mat convMat(dstHeight, dstWidth, CV_MAKETYPE(CV_8U, numChannels), hostOutputuchar);
    // cv::imwrite("./output/temp.png", convMat);


    // delete[] hostOutputuchar;

}


// Get output from device to host
std::tuple<int, int, float*> CNNLayer::GetOutput() {

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

    float* output = deviceConv + 0 * convHeight * convWidth;


    return {convWidth, convHeight, output};
}
