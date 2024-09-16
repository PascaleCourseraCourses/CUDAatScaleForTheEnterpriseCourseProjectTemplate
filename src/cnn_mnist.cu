#include <../lib/pooling.h>
#include <../lib/activation.h>
#include <../lib/convolution.h>
#include <../lib/utils.h>

#include <random>
#include <curand_kernel.h>


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


std::tuple<dim3, dim3, size_t> calculateGridAndBlockParams(int imageWidth, int imageHeight) {
    // Calculate the number of threads per block
    int threadsPerBlockX = TILE_WIDTH;
    int threadsPerBlockY = TILE_WIDTH;

    dim3 threads(threadsPerBlockX, threadsPerBlockY);

    // Calculate the size of the shared memory
    size_t sharedMemSize = (TILE_WIDTH + FILTER_WIDTH - 1) * (TILE_WIDTH + FILTER_WIDTH - 1) * sizeof(unsigned char);

    // Calculate the number of blocks per grid in each dimension
    int blocksPerGridX = (imageWidth + TILE_WIDTH - 1) / TILE_WIDTH;
    int blocksPerGridY = (imageHeight + TILE_WIDTH - 1) / TILE_WIDTH;

    dim3 blocks(blocksPerGridX, blocksPerGridY);

    // Print the calculated parameters
    std::cout << "Threads per block (X): " << threadsPerBlockX << std::endl;
    std::cout << "Threads per block (Y): " << threadsPerBlockY << std::endl;
    std::cout << "Shared memory size (bytes): " << sharedMemSize << std::endl;
    std::cout << "Blocks per grid (X): " << blocksPerGridX << std::endl;
    std::cout << "Blocks per grid (Y): " << blocksPerGridY << std::endl;

    return {threads, blocks, sharedMemSize};
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


int main(int argc, char* argv[]) {

    auto[directory, index, dstWidth, dstHeight] = parseArguments(argc, argv);
    

    /// Read images
    auto[h_images, srcWidth, srcHeight] = read_images(directory);

    // Transfer images to device
    auto d_images = CopyAllImagestoDevice(h_images, srcWidth, srcHeight);

    // Allocate memory for filter weights
    int filter_num_elements = FILTER_WIDTH * FILTER_WIDTH;
    size_t filter_size = filter_num_elements * sizeof(float);
    float* d_filterWeights = AllocateDeviceMemory<float>(filter_size);
    float* d_filterGradient = AllocateDeviceMemory<float>(filter_size);

    // Initialize filter weights randomly (or with a method like Xavier initialization)
    initializeWeights<<<1, filter_num_elements>>>(d_filterWeights, filter_num_elements, 1234ULL, -0.5f, 0.5f);

    // Resize the images using NPP
    size_t dstSize = dstWidth * dstHeight * sizeof(unsigned char);
    unsigned char* d_resized_image = AllocateDeviceMemory<unsigned char>(dstSize);
    resizeImageGPU(d_images[index], d_resized_image, srcWidth, srcHeight, dstWidth, dstHeight);

    // Calculate threads and blocks required based on image size
    auto[threads, blocks, sharedMemSize] = calculateGridAndBlockParams(dstWidth, dstHeight);

    // Run convolution kernel
    size_t img_size = dstWidth * dstHeight * sizeof(unsigned char);
    unsigned char* convImage = AllocateDeviceMemory<unsigned char>(img_size);
    unsigned char* poolImage = AllocateDeviceMemory<unsigned char>(img_size);
    unsigned char* activeImage = AllocateDeviceMemory<unsigned char>(img_size);

    convolutionKernel<<<blocks, threads, sharedMemSize>>>(d_resized_image, convImage, dstWidth, dstHeight, d_filterWeights);
    MaxPoolingKernel<<<blocks, threads, sharedMemSize>>>(convImage, poolImage, dstWidth, dstHeight);
    reluKernel<<<blocks, threads>>>(poolImage, activeImage, dstWidth, dstHeight);

    // Copy to host
    unsigned char* h_conv_image = new unsigned char[img_size]; // "new" for malloc
    CopyToHost<unsigned char>(h_conv_image, activeImage, img_size);
    cv::Mat convMat(dstWidth, dstHeight, CV_8UC1, h_conv_image);
    cv::imwrite("./output/output.png", convMat);

    for (auto d_image : d_images) {
        FreeDevice<unsigned char>(d_image);
    }

    FreeDevice<unsigned char>(convImage);
    FreeDevice<unsigned char>(poolImage);
    FreeDevice<unsigned char>(activeImage);
    FreeDevice<float>(d_filterWeights);
    FreeDevice<float>(d_filterGradient);


    return 0;
}
