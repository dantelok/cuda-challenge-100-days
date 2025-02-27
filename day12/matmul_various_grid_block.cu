#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define M 128
#define K 128
#define N 128

// ✅ Naïve Matrix Multiplication
__global__ void matrix_multiplication(double *A, double *X, double *Y, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        double sum = 0.0;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * X[l * n + col];
        }
        Y[row * n + col] = sum;
    }
}

// ✅ Tiled Matrix Multiplication
__global__ void tiled_matrix_multiplication(double *A, double *X, double *Y, int m, int k, int n, int BLOCK_SIZE) {
    extern __shared__ double shared[];
    double* shared_A = shared;
    double* shared_X = shared + BLOCK_SIZE * BLOCK_SIZE;

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    double sum = 0.0;

    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        if (row < M && tile * BLOCK_SIZE + threadIdx.x < K)
            shared_A[threadIdx.y * BLOCK_SIZE + threadIdx.x] = A[row * K + tile * BLOCK_SIZE + threadIdx.x];
        else
            shared_A[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0;

        if (col < N && tile * BLOCK_SIZE + threadIdx.y < K)
            shared_X[threadIdx.y * BLOCK_SIZE + threadIdx.x] = X[(tile * BLOCK_SIZE + threadIdx.y) * N + col];
        else
            shared_X[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum += shared_A[threadIdx.y * BLOCK_SIZE + k] * shared_X[k * BLOCK_SIZE + threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        Y[row * N + col] = sum;
}

// ✅ Timing Function
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void test_matrix_multiplication(int BLOCK_SIZE) {
    double *host_A, *host_X, *host_Y;
    double *device_A, *device_X, *device_Y;

    size_t size_A = M * K * sizeof(double);
    size_t size_X = K * N * sizeof(double);
    size_t size_Y = M * N * sizeof(double);

    host_A = (double*)malloc(size_A);
    host_X = (double*)malloc(size_X);
    host_Y = (double*)malloc(size_Y);

    cudaMalloc(&device_A, size_A);
    cudaMalloc(&device_X, size_X);
    cudaMalloc(&device_Y, size_Y);

    srand(time(NULL));
    for (int i = 0; i < M * K; i++) host_A[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) host_X[i] = (double)rand() / RAND_MAX;

    cudaMemcpy(device_A, host_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(device_X, host_X, size_X, cudaMemcpyHostToDevice);

    double start_time, end_time;

    // ✅ Naïve Matrix Multiplication
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    start_time = get_time();
    matrix_multiplication<<<gridDim, blockDim>>>(device_A, device_X, device_Y, M, K, N);
    cudaDeviceSynchronize();
    end_time = get_time();
    printf("BLOCK_SIZE %d - Naïve MatMul Execution Time: %f ms\n", BLOCK_SIZE, (end_time - start_time) * 1000);

    // ✅ Tiled Matrix Multiplication
    start_time = get_time();
    tiled_matrix_multiplication<<<gridDim, blockDim, 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(double)>>>(
        device_A, device_X, device_Y, M, K, N, BLOCK_SIZE);
    cudaDeviceSynchronize();
    end_time = get_time();
    printf("BLOCK_SIZE %d - Tiled MatMul Execution Time: %f ms\n", BLOCK_SIZE, (end_time - start_time) * 1000);

    cudaMemcpy(host_Y, device_Y, size_Y, cudaMemcpyDeviceToHost);

    free(host_A);
    free(host_X);
    free(host_Y);
    cudaFree(device_A);
    cudaFree(device_X);
    cudaFree(device_Y);
}

int main() {
    printf("Testing Matrix Multiplication with Different Block Sizes:\n");
    int block_sizes[] = {8, 16, 32, 64};
    for (int i = 0; i < 4; i++) {
        test_matrix_multiplication(block_sizes[i]);
    }
    return 0;
}