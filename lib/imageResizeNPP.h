#ifndef IMAGE_RESIZE_NPP_H
#define IMAGE_RESIZE_NPP_H


#include <utils.h>


cv::Mat resizeImageGPU(unsigned char* d_image, int srcWidth, int srcHeight, int dstWidth, int dstHeight);
cv::Mat resizeImageCPU(const unsigned char* h_image, int srcWidth, int srcHeight, int dstWidth, int dstHeight);
std::tuple<std::vector<unsigned char*>, int, int> read_images(const fs::path& directory); 
std::vector<unsigned char*> CopyAllImagestoDevice(std::vector<unsigned char*> images, int srcWidth, int srcHeight);
std::tuple<std::string, int, int, int> parseArguments(int argc, char* argv[]);
double computeMSE(const cv::Mat& img1, const cv::Mat& img2);
double computePercentageDifference(const cv::Mat& img1, const cv::Mat& img2);


#endif // IMAGE_RESIZE_NPP_H