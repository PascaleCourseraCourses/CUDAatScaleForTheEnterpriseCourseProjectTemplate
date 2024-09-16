# Image Resize using NVIDIA NPP and CNN Operations with CUDA

## Overview

This project implements basic operations of a Convolutional Neural Network (CNN), including convolution, max pooling, and ReLU activations, along with image augmentation. The image resizing functionality utilizes the NVIDIA Performance Primitives (NPP) library. Key learning outcomes include building CUDA source files with `make`, understanding various flags, including external libraries, and utilizing shared memory for efficient convolution and pooling operations. Challenges included implementing image padding. The code currently supports grayscale images with a filter size of 3x3 for convolution and pooling, using "Same" padding mode to maintain input and output image sizes.

## Code Organization

- **`bin/`**: Contains binary/executable files that are built automatically or manually.
- **`data/`**: Holds example images from the MNIST database.
- **`lib/`**: Includes libraries not installed via the operating system's package manager.
- **`src/`**: Contains the source files for the project.
- **`Makefile`**: Defines build and run commands for the project. Modify arguments like resized image size within this file.

## Key Concepts

- Performance Strategies
- Image Processing
- NPP Library
- Convolutional Neural Networks

## Supported OSes

- Linux
- Windows

## Dependencies

- [FreeImage](https://freeimage.sourceforge.io/)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)

Ensure correct paths to include and library files for these dependencies are specified in the Makefile.

## Prerequisites

1. Download and install the [CUDA Toolkit 12.5](https://developer.nvidia.com/cuda-downloads) for your platform.
2. Install dependencies as listed in the [Dependencies](#dependencies) section.

## Building the Program

To build the project, use the following command:

```bash
make all
```

## Running the Program
After building the project, you can run the program using the following command:

```bash
make run
```

This command will execute the compiled binary, resizing the input image and apply CNN operations to the resized image, and save the result as output.png in the output/ directory.

If you wish to run the binary directly with custom input/output files, you can use:

```bash
./bin/cnn_mnist.exe -d data/ -i 0 -w 500 -h 500
```

Use `-w` and `-h` flags to determine the width and height of the resized image. Use `-i` flag to determine the index of an image to process among images in directory specified by flag `-d`.

## Cleaning Up
To clean up the compiled binaries and other generated files, run:


```bash
make clean
```

This will remove all files in the bin/ directory.
