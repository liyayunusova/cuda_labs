#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// Размеры матриц (должны быть кратны 16 для WMMA)
const int M = 16;
const int N = 16;
const int K = 16;

__global__ void matrixMulWarpMMA(const float *a, const float *b, float *c, int M, int N, int K) {
    // Создание матриц WMMA для A, B и C
    wmma::fragment<wmma::matrix_a, 16, 16, 16, float, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, float, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Инициализация C фрагмента
    wmma::fill_fragment(c_frag, 0.0f);

    // Загрузка матриц A и B в фрагменты
    wmma::load_matrix_sync(a_frag, a, K);
    wmma::load_matrix_sync(b_frag, b, K);

    // Умножение матриц
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Сохранение результата в глобальную память
    wmma::store_matrix_sync(c, c_frag, N, wmma::mem_row_major);
}

int main() {
    // Размеры матриц
    int size_a = M * K * sizeof(float);
    int size_b = K * N * sizeof(float);
    int size_c = M * N * sizeof(float);

    // Выделение памяти для матриц
    float *h_a = (float*)malloc(size_a);
    float *h_b = (float*)malloc(size_b);
    float *h_c = (float*)malloc(size_c);

    // Инициализация матриц
    for (int i = 0; i < M * K; i++) h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_b[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    // Запуск ядра
    dim3 threadsPerBlock(32, 8);  // 32 threads per warp, 8 warps per block
    dim3 blocksPerGrid((M + 31) / 32, (N + 31) / 8);

    matrixMulWarpMMA<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, M, N, K);

    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

    // Вывод результата
    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_c[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Освобождение памяти
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
