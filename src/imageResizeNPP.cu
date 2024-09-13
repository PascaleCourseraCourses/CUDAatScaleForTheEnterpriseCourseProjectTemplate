#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <filesystem>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <opencv2/opencv.hpp> 

namespace fs = std::filesystem;

void handleCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


template<typename T>
T* AllocateDeviceMemory(size_t size) {

    T* d_device = nullptr;
    handleCudaError(cudaMalloc(&d_device, size));

    return d_device;  
}


template<typename T>
T* AllocateHostMemory(size_t size, const std::string& mem_type) {
    
    T* d_host = nullptr;

    if (mem_type == "paged") {
        d_host = static_cast<T*>(malloc(size));
        if (!d_host) {
            std::cerr << "Error allocating pageable memory using malloc." << std::endl;
        }
    }

    else if (mem_type == "pinned") {
        handleCudaError(cudaHostAlloc(reinterpret_cast<void**>(&d_host), size, cudaHostAllocDefault));
    }

    else if (mem_type == "mapped") {
        handleCudaError(cudaHostAlloc(reinterpret_cast<void**>(&d_host), size, cudaHostAllocMapped));
    }

    else if (mem_type == "unified") {
        handleCudaError(cudaMallocManaged(reinterpret_cast<void**>(&d_host), size));

    }

    else {
        std::cerr << "Invalid memory type specified." << std::endl;
    }

    return d_host;
}

template<typename T>
void CopyToHost(T* d_host, const T* d_device, const size_t size) {
    handleCudaError(cudaMemcpy(d_host, d_device, size, cudaMemcpyDeviceToHost));
}

template<typename T>
void CopyToDevice(const T* d_host, T* d_device, const size_t size) {
    handleCudaError(cudaMemcpy(d_device, d_host, size, cudaMemcpyHostToDevice));
}

template<typename T>
void FreeDevice(T* d_device){

    handleCudaError(cudaFree(d_device));

}


void resizeImage(unsigned char* d_image, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {

    size_t dstSize_1 = dstWidth * dstHeight * sizeof(unsigned char);

    unsigned char* d_resized_image = AllocateDeviceMemory<unsigned char>(dstSize_1); // Allocate GPU memory for the resized image

    NppiSize srcSize = {srcWidth, srcHeight}; // Source size
    NppiSize dstSize = {dstWidth, dstHeight}; // Destination size
    NppiRect srcRectROI = {0, 0, srcWidth, srcHeight}; // Source ROI
    NppiRect dstRectROI = {0, 0, dstWidth, dstHeight}; // Destination ROI

    int srcStep = srcWidth * sizeof(unsigned char); // Row step for source image
    int dstStep = dstWidth * sizeof(unsigned char); // Row step for destination image

    NppStatus status = nppiResize_8u_C1R(
        d_image, srcStep, srcSize, srcRectROI,
        d_resized_image, dstStep, dstSize, dstRectROI,
        NPPI_INTER_LINEAR
    );

    if (status != NPP_SUCCESS) {
        std::cerr << "NPP error: " << status << std::endl;
    }

    // Copy the resized image from device to host
    unsigned char* h_resized_image = new unsigned char[dstSize_1]; // "new" for malloc
    CopyToHost<unsigned char>(h_resized_image, d_resized_image, dstSize_1);

    // Use OpenCV to save the resized image
    cv::Mat resizedMat(dstHeight, dstWidth, CV_8UC1, h_resized_image);
    cv::imwrite("./data/resized.png", resizedMat);

    FreeDevice<unsigned char>(d_resized_image);
}

__host__ std::tuple<std::vector<unsigned char*>, int, int> read_images(const fs::path& directory) {
    std::vector<unsigned char*> d_images;
    std::vector<cv::Mat> images;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            // Read the image in grayscale
            npp::ImageCPU_8u_C1 oHostSrc;

            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
            
            if (!img.empty()) {
                images.push_back(img);
            } else {
                std::cerr << "Failed to load image: " << entry.path() << std::endl;
            }
        }
    }

    int width = images[0].cols;
    int height = images[0].rows;
    int num_pixels = width * height;

    // Allocate memory on device and copy image data
    for (const auto& img : images) {
        unsigned char* d_image;
        size_t imageSize = num_pixels * sizeof(unsigned char);
        
        handleCudaError(cudaMalloc(&d_image, imageSize));
        handleCudaError(cudaMemcpy(d_image, img.data, imageSize, cudaMemcpyHostToDevice));

        d_images.push_back(d_image);
    }

    return {d_images, width, height};
}


std::tuple<std::string, int, int, int> parseArguments(int argc, char* argv[]) {
    // Initialize default values
    std::string directory = "./data/train/mnist_images";
    int index = 0;
    int dstWidth = 320;
    int dstHeight = 240;

    // Iterate through command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Check for the directory flag
        if (arg == "-d" && i + 1 < argc) {
            directory = argv[++i];
        }
        // Check for the index flag
        else if (arg == "-i" && i + 1 < argc) {
            try {
                index = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid index value provided. Using default value 0." << std::endl;
            }
        }
        // Check for the index flag
        else if (arg == "-w" && i + 1 < argc) {
            try {
                dstWidth = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid width value provided. Using default value 320." << std::endl;
            }
        }
        // Check for the index flag
        else if (arg == "-h" && i + 1 < argc) {
            try {
                dstHeight = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid height value provided. Using default value 240." << std::endl;
            }
        }
    }

    std::cout << "Data Path: " << directory << std::endl;
    std::cout << "Image Index: " << index << std::endl;
    std::cout << "Width: " << dstWidth << std::endl;
    std::cout << "Height: " << dstHeight << std::endl;


    return {directory, index, dstWidth, dstHeight};
}


int main(int argc, char* argv[]) {

    auto[directory, index, dstWidth, dstHeight] = parseArguments(argc, argv);

    /// Read images
    auto [d_images, srcWidth, srcHeight] = read_images(directory);

    // Allocate and initialize image data
    resizeImage(d_images[index], srcWidth, srcHeight, dstWidth, dstHeight);

    for (auto d_image : d_images) {
        handleCudaError(cudaFree(d_image));
    }
    
    return 0;
}
