#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float* d_A, float* d_B, float* d_C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += d_A[row * N + k] * d_B[k * N + col];
        }
        d_C[row * N + col] = sum;
    }
}

// Function to fill matrix with random numbers
void fillMatrix(float* matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Function to print the matrix
void printMatrix(const float* matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    srand(time(0));

    int N;
    std::cout << "Enter the size of the matrix: ";
    std::cin >> N;

    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Fill matrices with random numbers
    fillMatrix(h_A, N);
    fillMatrix(h_B, N);

    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the matrix multiplication kernel
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the time taken
    std::cout << "Time taken for matrix multiplication: " << milliseconds << " ms" << std::endl;

    // Print the resulting matrix (optional)
    // printMatrix(h_C, N);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
