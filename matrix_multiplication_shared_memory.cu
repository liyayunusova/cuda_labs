#include <stdio.h>
#include <cuda.h>

#define TILE_WIDTH 16  // Define tile size

// CUDA Kernel for matrix multiplication
__global__ void matrixMulShared(float *A, float *B, float *C, int N) {
    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  // Block index
    int by = blockIdx.y;
    int tx = threadIdx.x;  // Thread index
    int ty = threadIdx.y;

    // Identify the row and column of the C element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0;

    // Loop over the A and B tiles required to compute the C element
    for (int t = 0; t < (N - 1) / TILE_WIDTH + 1; ++t) {
        // Collaborative loading of A and B tiles into shared memory
        if (Row < N && t * TILE_WIDTH + tx < N)
            shared_A[ty][tx] = A[Row * N + t * TILE_WIDTH + tx];
        else
            shared_A[ty][tx] = 0.0;

        if (Col < N && t * TILE_WIDTH + ty < N)
            shared_B[ty][tx] = B[(t * TILE_WIDTH + ty) * N + Col];
        else
            shared_B[ty][tx] = 0.0;

        __syncthreads();

        // Multiply the two matrices together
        for (int i = 0; i < TILE_WIDTH; ++i)
            Cvalue += shared_A[ty][i] * shared_B[i][tx];

        __syncthreads();
    }

    // Write the block sub-matrix to global memory
    if (Row < N && Col < N)
        C[Row * N + Col] = Cvalue;
}

int main() {
    // Matrix size
    int N = 1024;

    // Allocate host memory
    size_t size = N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize matrices A and B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy host memory to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N - 1) / TILE_WIDTH + 1, (N - 1) / TILE_WIDTH + 1);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch kernel
    matrixMulShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to compute matrix multiplication: %f ms\n", elapsedTime);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
