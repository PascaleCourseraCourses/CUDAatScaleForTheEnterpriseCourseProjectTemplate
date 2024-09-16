# Define directories
SRC_DIR = ./src
BIN_DIR = ./bin
DATA_DIR = ./data
LIB_DIR = ./lib


# Define the compiler and flags
NVCC = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin/nvcc.exe"

# without ccbin to x64 the compiler gives Access error (MUST FLGAG)
# -I flag is the path to header files
# -L flag is the path to lib files
CXXFLAGS = -std=c++17 \
           -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include" \
           -I"C:/opencv/build/include" \
           -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/Common" \
           -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/Common/UtilNPP" \
           -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/FreeImage/Dist/x64" \
		   -I"$(LIB_DIR)" \
           -ccbin "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/bin/Hostx64/x64"

LDFLAGS = -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/lib/x64" \
          -lcudart -lnppc -lnppif -lnppig -lnppist -lnppisu \
          -L"C:/opencv/build/x64/vc16/lib" \
          -lopencv_world4100 -lopencv_world4100d \
          -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/FreeImage/Dist/x64" \
          -lFreeImage



# Define source files and target executable
SRC = $(SRC_DIR)/cnn_mnist.cu $(SRC_DIR)/convolution.cu $(SRC_DIR)/activation.cu $(SRC_DIR)/pooling.cu 

TARGET = $(BIN_DIR)/cnn_mnist.exe

# Define the default rule
all: $(TARGET)

# Rule for building the target executable
$(TARGET): $(SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Rule for running the application
run: $(TARGET)
	$(TARGET) -d "./data" -i 2 -w 500 -h 500

# Clean up
clean:
	rm -f $(BIN_DIR)/*

# Installation rule (not much to install, but here for completeness)
install:
	@echo "No installation required."

# Help command
help:
	@echo "Available make commands:"
	@echo "  make        - Build the project."
	@echo "  make run    - Run the project."
	@echo "  make clean  - Clean up the build files."
	@echo "  make install- Install the project (if applicable)."
	@echo "  make help   - Display this help message."
