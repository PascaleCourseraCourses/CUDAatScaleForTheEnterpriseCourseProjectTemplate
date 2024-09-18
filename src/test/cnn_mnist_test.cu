#include <../lib/pooling.h>
#include <../lib/activation.h>
#include <../lib/convolution.h>
#include <../lib/utils.h>

#include <random>


void resizeImageGPU(unsigned char* d_image, unsigned char* d_resized_image, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {

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

}


std::tuple<std::vector<unsigned char*>, int, int> read_images(const fs::path& directory) {
    std::vector<unsigned char*> images;
    int width = 0;
    int height = 0;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            // Read the image in grayscale
            npp::ImageCPU_8u_C1 oHostSrc;

            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);

            if (!img.empty()) {

                width = img.cols;
                height = img.rows;
                size_t img_size = width * height * sizeof(unsigned char);
                unsigned char* d_image = AllocateHostMemory<unsigned char>(img_size, "pinned");
                std::memcpy(d_image, img.data, img_size);
                images.push_back(d_image);

            } else {
                std::cerr << "Failed to load image: " << entry.path() << std::endl;
            }
        }
    }

    return {images, width, height};
}


std::vector<unsigned char*> CopyAllImagestoDevice(std::vector<unsigned char*> images, int srcWidth, int srcHeight){

    std::vector<unsigned char*> d_images;
    
    size_t img_size = srcWidth * srcHeight * sizeof(unsigned char);

    // Allocate memory on device and copy image data
    for (const auto& img : images) {        

        unsigned char* d_image = AllocateDeviceMemory<unsigned char>(img_size);
        CopyToDevice<unsigned char>(img, d_image, img_size);
        d_images.push_back(d_image);
    }

    return d_images;

}


std::tuple<std::string, int, int, int> parseArguments(int argc, char* argv[]) {
    // Initialize default values
    std::string directory = "../data/train/mnist_images";
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


std::tuple<dim3, dim3, size_t> calculateGridAndBlockParams(int outputWidth, int outputHeight, int kernelWidth, int kernelHeight, std::string operation) {
    // Calculate the number of threads per bloc
    int threadsPerBlockX = TILE_WIDTH;
    int threadsPerBlockY = TILE_WIDTH;

    dim3 blockDim(threadsPerBlockX, threadsPerBlockY);
    size_t sharedMemSize;

    // Calculate the size of the shared memory
    if (operation == "convolution"){

        sharedMemSize = (TILE_WIDTH + kernelWidth  - 1) * (TILE_WIDTH + kernelHeight  - 1) * sizeof(float) + kernelHeight * kernelWidth * sizeof(float);

    } else {

        sharedMemSize = (TILE_WIDTH + kernelWidth  - 1) * (TILE_WIDTH + kernelHeight  - 1) * sizeof(float);

    }

    // Calculate the number of blocks per grid in each dimension
    int blocksPerGridX = (outputWidth + TILE_WIDTH - 1) / TILE_WIDTH;
    int blocksPerGridY = (outputHeight + TILE_WIDTH - 1) / TILE_WIDTH;

    dim3 gridDim(blocksPerGridX, blocksPerGridY);

    // Print the calculated parameters
    std::cout << "Threads per block (X): " << threadsPerBlockX << std::endl;
    std::cout << "Threads per block (Y): " << threadsPerBlockY << std::endl;
    std::cout << "Shared memory size (bytes): " << sharedMemSize << std::endl;
    std::cout << "Blocks per grid (X): " << blocksPerGridX << std::endl;
    std::cout << "Blocks per grid (Y): " << blocksPerGridY << std::endl;

    return {blockDim, gridDim, sharedMemSize};
}


__host__ void convertToUnsignedChar(const float* input, unsigned char* output, int size) {

    float minRange = *std::min_element(input, input + size);
    float maxRange = *std::max_element(input, input + size);

    std::cout << minRange << std::endl;
    std::cout << maxRange;

    if (maxRange == minRange) {
        std::fill(output, output + size, 0);  
    } else {
        for (int i = 0; i < size; ++i) {
            unsigned char scaledValue = static_cast<unsigned char>(255.0f * (input[i] - minRange) / (maxRange - minRange));
            output[i] = scaledValue;
        }
    }
}

__host__ void save_image(int outputWidth, int outputHeight, const float* convImage){

    // Calculate size of image
    int output_size = outputWidth * outputHeight;
    size_t conv_size = output_size * sizeof(float);

    // Allocate dynamic memory to host image with flot and unsigned char types
    float* h_conv_image = new float[output_size]; // "new" for malloc
    unsigned char* output = new unsigned char[output_size];

    // Copy image to host
    CopyToHost<float>(h_conv_image, convImage, conv_size);

    // Convert float host image to unsigned char host image (0-255)
    convertToUnsignedChar(h_conv_image, output, output_size);

    // Create an OpenCV matrix for host image to use OpenCV functions
    cv::Mat convMat(outputWidth, outputHeight, CV_8UC1, output);

    // Save the image
    cv::imwrite("./output/output.png", convMat);

    delete[] h_conv_image;
    delete[] output;

}


int main(int argc, char* argv[]) {

    auto[directory, index, dstWidth, dstHeight] = parseArguments(argc, argv);
    
    /// Read images
    auto[h_images, srcWidth, srcHeight] = read_images(directory);

    // Transfer images to device
    auto d_images = CopyAllImagestoDevice(h_images, srcWidth, srcHeight);

    // Resize the images using NPP
    size_t dstSize = dstWidth * dstHeight * sizeof(unsigned char);
    unsigned char* d_resized_image = AllocateDeviceMemory<unsigned char>(dstSize);
    resizeImageGPU(d_images[index], d_resized_image, srcWidth, srcHeight, dstWidth, dstHeight);

    // Initialize convolution paramters
    int kernelHeight = 5, kernelWidth = 5;
    int strideHeight = 2, strideWidth = 2;
    int paddingHeight = 1, paddingWidth = 1;

    // Allocate device memory to convolution image
    int convHeight = (dstHeight + 2 * paddingHeight - kernelHeight) / strideHeight + 1;
    int convWidth = (dstWidth + 2 * paddingWidth - kernelWidth) / strideWidth + 1;
    size_t conv_size = convWidth * convHeight * sizeof(float);
    float* convImage = AllocateDeviceMemory<float>(conv_size);

    // Allocate memory for filter weights
    int filter_num_elements = kernelHeight * kernelWidth;
    size_t filter_size = filter_num_elements * sizeof(float);
    float* d_filterWeights = AllocateDeviceMemory<float>(filter_size);
    float* d_filterGradient = AllocateDeviceMemory<float>(filter_size);

    // Initialize filter weights randomly
    initializeWeights<<<1, filter_num_elements>>>(d_filterWeights, filter_num_elements, 1234ULL, -0.5f, 0.5f);

    // Allocate device memory to pooling image
    int poolHeight = (convHeight + 2 * paddingHeight - kernelHeight) / strideHeight + 1;
    int poolWidth = (convWidth + 2 * paddingWidth - kernelWidth) / strideWidth + 1;
    size_t pool_size = poolWidth * poolHeight * sizeof(float);
    float* poolImage = AllocateDeviceMemory<float>(pool_size);

    // Allocate device memory to activation image
    size_t act_size = poolWidth * poolHeight * sizeof(float);
    float* activeImage = AllocateDeviceMemory<float>(act_size);

    // Calculate threads and blocks required based on image size
    auto[blockDim1, gridDim1, sharedMemSize1] = calculateGridAndBlockParams(convWidth, convHeight, kernelWidth, kernelHeight, "convolution");

    // Run convolution kernel
    convolutionKernelShared<<<gridDim1, blockDim1, sharedMemSize1>>>(d_resized_image, convImage, d_filterWeights, dstWidth, dstHeight, kernelHeight, kernelWidth, strideHeight, strideWidth, paddingHeight, paddingWidth);

    // Calculate threads and blocks required based on image size
    auto[blockDim2, gridDim2, sharedMemSize2] = calculateGridAndBlockParams(poolWidth, poolHeight, kernelWidth, kernelHeight, "pooling");

    // Run max pooling kernel
    MaxPoolingKernelShared<<<gridDim2, blockDim2, sharedMemSize2>>>(convImage, poolImage, 
                                                                convHeight, convWidth, 
                                                                kernelHeight, kernelWidth, 
                                                                strideHeight, strideWidth, 
                                                                paddingHeight, paddingWidth);

    reluKernel<<<gridDim2, blockDim2>>>(poolImage, activeImage, poolWidth, poolHeight);

    // Save output image
    // save_image(convWidth, convHeight, convImage);
    // save_image(poolWidth, poolHeight, poolImage);
    save_image(poolWidth, poolHeight, activeImage);


    // Free allocated device memory
    for (auto d_image : d_images) {
        FreeDevice<unsigned char>(d_image);
    }

    FreeDevice<float>(convImage);
    FreeDevice<float>(poolImage);
    FreeDevice<float>(activeImage);
    FreeDevice<float>(d_filterWeights);
    FreeDevice<float>(d_filterGradient);


    return 0;
}
