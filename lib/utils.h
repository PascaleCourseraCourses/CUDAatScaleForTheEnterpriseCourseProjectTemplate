#ifndef UTILS_H
#define UTILS_H

// Platform-specific includes
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

// Standard and third-party includes
#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>

namespace fs = std::filesystem;

// Template function definitions
void handleCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " in File " << __FILE__
                  << " in line " << __LINE__
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <typename T>
T* AllocateDeviceMemory(size_t size) {
    T* d_device = nullptr;
    handleCudaError(cudaMalloc(&d_device, size));
    return d_device;
}

template <typename T>
T* AllocateHostMemory(size_t size, const std::string& mem_type) {
    T* d_host = nullptr;
    if (mem_type == "paged") {
        d_host = static_cast<T*>(malloc(size));
        if (!d_host) {
            std::cerr << "Error allocating pageable memory using malloc." << std::endl;
        }
    } else if (mem_type == "pinned") {
        handleCudaError(cudaHostAlloc(reinterpret_cast<void**>(&d_host), size, cudaHostAllocDefault));
    } else if (mem_type == "mapped") {
        handleCudaError(cudaHostAlloc(reinterpret_cast<void**>(&d_host), size, cudaHostAllocMapped));
    } else if (mem_type == "unified") {
        handleCudaError(cudaMallocManaged(reinterpret_cast<void**>(&d_host), size));
    } else {
        std::cerr << "Invalid memory type specified." << std::endl;
    }
    return d_host;
}

template <typename T>
void CopyToHost(T* d_host, const T* d_device, const size_t size) {
    handleCudaError(cudaMemcpy(d_host, d_device, size, cudaMemcpyDeviceToHost));
}

template <typename T>
void CopyToDevice(const T* d_host, T* d_device, const size_t size) {
    handleCudaError(cudaMemcpy(d_device, d_host, size, cudaMemcpyHostToDevice));
}

template <typename T>
void FreeDevice(T* d_device) {
    handleCudaError(cudaFree(d_device));
}

#endif // UTILS_H
