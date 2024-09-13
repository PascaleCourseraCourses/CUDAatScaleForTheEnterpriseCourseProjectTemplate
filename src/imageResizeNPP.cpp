#include <../lib/utils.h>


cv::Mat resizeImageGPU(unsigned char* d_image, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {

    size_t dstSize_1 = dstWidth * dstHeight * sizeof(unsigned char);

    unsigned char* d_resized_image = AllocateDeviceMemory<unsigned char>(dstSize_1); // Allocate GPU memory for the resized image

    NppiSize srcSize = {srcWidth, srcHeight}; // Source size
    NppiSize dstSize = {dstWidth, dstHeight}; // Destination size
    NppiRect srcRectROI = {0, 0, srcWidth, srcHeight}; // Source ROI
    NppiRect dstRectROI = {0, 0, dstWidth, dstHeight}; // Destination ROI

    int srcStep = srcWidth * sizeof(unsigned char); // Row step for source image
    int dstStep = dstWidth * sizeof(unsigned char); // Row step for destination image

    auto start = std::chrono::high_resolution_clock::now();
    NppStatus status = nppiResize_8u_C1R(
        d_image, srcStep, srcSize, srcRectROI,
        d_resized_image, dstStep, dstSize, dstRectROI,
        NPPI_INTER_LINEAR
    );

    if (status != NPP_SUCCESS) {
        std::cerr << "NPP error: " << status << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();


    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time for GPU: " << elapsed.count() << " seconds" << std::endl;

    // Copy the resized image from device to host
    unsigned char* h_resized_image = new unsigned char[dstSize_1]; // "new" for malloc
    CopyToHost<unsigned char>(h_resized_image, d_resized_image, dstSize_1);

    // Use OpenCV to save the resized image
    cv::Mat resizedMat(dstHeight, dstWidth, CV_8UC1, h_resized_image);
    cv::imwrite("./output/resized_gpu.png", resizedMat);

    FreeDevice<unsigned char>(d_resized_image);

    return resizedMat;
}


cv::Mat resizeImageCPU(const unsigned char* h_image, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {

    cv::Mat dstImage;

    cv::Mat inputImage(srcHeight, srcWidth, CV_8UC1, const_cast<unsigned char*>(h_image));

    // Resize the image
    auto start = std::chrono::high_resolution_clock::now();
    cv::resize(inputImage, dstImage, cv::Size(dstWidth, dstHeight), 0, 0, cv::INTER_LINEAR);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time for CPU: " << elapsed.count() << " seconds" << std::endl;

    // Save the resized image
    cv::imwrite("./output/resized_cpu.png", dstImage);

    return dstImage;

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


double computeMSE(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        std::cerr << "Images must have the same size and type." << std::endl;
        return -1.0;
    }

    cv::Mat diff;
    cv::absdiff(img1, img2, diff); // Compute the absolute difference

    cv::Mat diff_squared;
    cv::multiply(diff, diff, diff_squared); // Square the difference

    cv::Scalar mse = cv::mean(diff_squared); // Compute the mean squared error
    return mse[0];
}

// Function to compute the percentage difference
double computePercentageDifference(const cv::Mat& img1, const cv::Mat& img2) {
    double mse = computeMSE(img1, img2);

    if (mse < 0) {
        return -1.0; // Error in MSE computation
    }

    // Normalize MSE to a percentage
    double max_pixel_value = 255.0; // For 8-bit images
    double max_mse = max_pixel_value * max_pixel_value; // Maximum possible MSE
    double percentage_diff = (mse / max_mse) * 100.0;

    return percentage_diff;
}


int main(int argc, char* argv[]) {

    auto[directory, index, dstWidth, dstHeight] = parseArguments(argc, argv);

    /// Read images
    auto[h_images, srcWidth, srcHeight] = read_images(directory);

    auto d_images = CopyAllImagestoDevice(h_images, srcWidth, srcHeight);

    // Allocate and initialize image data
    cv::Mat resized_gpu_image = resizeImageGPU(d_images[index], srcWidth, srcHeight, dstWidth, dstHeight);
    cv::Mat resized_cpu_image = resizeImageCPU(h_images[index], srcWidth, srcHeight, dstWidth, dstHeight);

    double percentage_diff = computePercentageDifference(resized_gpu_image, resized_cpu_image);

    if (percentage_diff >= 0) {
        std::cout << "Percentage Difference: " << percentage_diff << "%" << std::endl;
    }

    for (auto d_image : d_images) {
        FreeDevice<unsigned char>(d_image);
    }

    return 0;
}
