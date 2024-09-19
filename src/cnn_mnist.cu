#include <../lib/cnnlayer.h>

#include <../lib/utils.h>

#include <random>

std::tuple<std::vector<unsigned char*>, int, int, int> read_images(const fs::path& directory) {
    std::vector<unsigned char*> images;
    int width = 0;
    int height = 0;
    int channels = 0;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            // Read the image in grayscale
            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);

            if (!img.empty()) {

                width = img.cols;
                height = img.rows;
                channels = img.channels();
                size_t img_size = width * height * channels * sizeof(unsigned char);
                unsigned char* d_image = AllocateHostMemory<unsigned char>(img_size, "pinned");
                std::memcpy(d_image, img.data, img_size);
                images.push_back(d_image);

                std::cout << channels;

            } else {
                std::cerr << "Failed to load image: " << entry.path() << std::endl;
            }
        }
    }

    std::cout << height;

    return {images, width, height, channels};
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


__host__ void convertToUnsignedChar(const float* input, unsigned char* output, int size) {

    float minRange = *std::min_element(input, input + size);
    float maxRange = *std::max_element(input, input + size);

    // std::cout << minRange << std::endl;
    // std::cout << maxRange << std::endl;

    if (maxRange == minRange) {
        std::fill(output, output + size, 0);  
    } else {
        for (int i = 0; i < size; ++i) {
            unsigned char scaledValue = static_cast<unsigned char>(255.0f * (input[i] - minRange) / (maxRange - minRange));
            output[i] = scaledValue;
        }
    }
}


__host__ void save_image(int outputWidth, int outputHeight, const float* convImage, int numChannels, int index){

    // Calculate size of image
    int output_size = outputWidth * outputHeight * numChannels;
    size_t conv_size = output_size * sizeof(float);

    // Allocate dynamic memory to host image with flot and unsigned char types
    float* h_conv_image = new float[output_size]; // "new" for malloc
    unsigned char* output = new unsigned char[output_size];

    // Copy image to host
    cudaError_t err = cudaMemcpy(h_conv_image, convImage, conv_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " in File " << __FILE__
                  << " in line " << __LINE__
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // Convert float host image to unsigned char host image (0-255)
    convertToUnsignedChar(h_conv_image, output, output_size);

    // Create an OpenCV matrix for host image to use OpenCV functions
    cv::Mat convMat(outputHeight, outputHeight, CV_MAKETYPE(CV_8U, numChannels), output);

    // Save the image
    std::string outputFileName = "./output/output_" + std::to_string(index) + ".png";
    cv::imwrite(outputFileName, convMat);

    delete[] h_conv_image;
    delete[] output;

}


int main(int argc, char* argv[]) {

    auto[directory, index, dstWidth, dstHeight] = parseArguments(argc, argv);
    
    /// Read images
    auto[h_images, srcWidth, srcHeight, numChannels] = read_images(directory);

    // Initialize convolution paramters
    int filterHeight = 3, filterWidth = 3;
    int strideHeight = 1, strideWidth = 1;
    int paddingHeight = 1, paddingWidth = 1;
    int numFilters = 5;

    // Construct the network
    CNNLayer SimpleCNN(srcHeight, srcWidth, dstHeight, dstWidth, filterHeight, filterWidth,
                    strideHeight, strideWidth, paddingHeight, paddingWidth, numFilters, numChannels);

    int i = 0;
    for (const auto& img : h_images) {     

        SimpleCNN.ForwardPass(img);

        // Get the output (presumably a convolution output)
        auto[poolWidth, poolHeight, outputimage] = SimpleCNN.GetOutput();

        // Save the result (optionally save with a different name for each image)
        save_image(poolWidth, poolHeight, outputimage, 1, i);
        i++;

    }

    return 0;
}
