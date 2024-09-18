#ifndef CNN_H
#define CNN_H

#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pooling.h>
#include <activation.h>
#include <convolution.h>
#include <npp.h>
#include <nppi.h>

#define TILE_WIDTH 16  // Example tile width for shared memory optimization

class CNN {
public:
    // Constructor
    CNN(int inputHeight, int inputWidth,
        int dstHeight, int dstWidth, 
        int filterHeight, int filterWidth,
        int strideHeight, int strideWidth,
        int paddingHeight, int paddingWidth,
        int numFilters);

    // Destructor
    ~CNN();

    // Forward pass
    void ForwardPass(unsigned char* hostInput);

    // Backward pass
    void BackwardPass(float* deviceGradOutput);



    // Setters and Getters for parameters
    std::tuple<int, int, float*> GetOutput();

private:
    int inputHeight, inputWidth;
    int dstWidth, dstHeight;
    int filterHeight, filterWidth;
    int strideHeight, strideWidth;
    int paddingHeight, paddingWidth;
    int numFilters;
    int poolWidth;
    int poolHeight;
    int convHeight;
    int convWidth;

    unsigned char* deviceInput;
    unsigned char* deviceResized;
    float* deviceFilters;
    float* deviceConv;
    float* devicePool;
    float* deviceAct;
    // float* deviceIntermediate;
    // float* deviceGradInput;
    // float* deviceGradFilters;
    // float* deviceGradOutput;

    dim3 gridSizeconv;
    dim3 blockSizeconv;
    dim3 gridSizepool;
    dim3 blockSizepool;
    size_t sharedMemSizeconv;
    size_t sharedMemSizepool;

    void resizeImageGPU();
    void AllocateMemory();
    void FreeMemory();
    void SetFilters();
    void LaunchConvolutionKernel();
    void LaunchActivationKernel();
    void LaunchMaxPoolingKernel();
    // void LaunchConvolutionBackwardKernel();
    // void LaunchActivationBackwardKernel();
    // void LaunchMaxPoolingBackwardKernel();
    // void UpdateFilters();
};

#endif // CNN_H
