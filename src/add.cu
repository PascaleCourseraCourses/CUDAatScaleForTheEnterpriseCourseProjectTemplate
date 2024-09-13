#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel function to add elements of two arrays
__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int n = 10;
    size_t size = n * sizeof(int);

    // Allocate host memory
    int *h_a = new int[n];
    int *h_b = new int[n];
    int *h_c = new int[n];

    // Initialize host arrays
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and n threads
    add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result:\n";
    for (int i = 0; i < n; ++i) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
