#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 16
#define TILE_SIZE 16
#define M 256
#define K 256
#define N 256


// A: (M, K); X: (K, N); Y: (M, N)
__global__ void matrix_multiplication(double *A, double *X, double *Y, int m, int k, int n) {

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < m && col < n) {
        double sum = 0.0;
        for (int l=0; l < k; l++) {
            // Output: Y[row, col]
            // A[row, l]: each row has k elements
            // X[l, col]: each row has n elements
            sum += A[row * k + l] * X[l * n + col];;
        }
        Y[row * n + col] = sum;
    }
}

__global__ void matrix_multiplication_tiled(double* A, double* X, double* Y, int m, int k, int n) {
    __shared__ double shared_A[TILE_SIZE][TILE_SIZE+1];
    __shared__ double shared_X[TILE_SIZE][TILE_SIZE+1];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    double sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (row < M && tile * TILE_SIZE + threadIdx.x < K)
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        else
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        
        if (col < N && tile * TILE_SIZE + threadIdx.y < K)
            shared_X[threadIdx.y][threadIdx.x] = X[(tile * TILE_SIZE + threadIdx.y) * N + col];
        else
            shared_X[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += shared_A[threadIdx.y][k] * shared_X[k][threadIdx.x];
        
        __syncthreads();
    }
    
    if (row < M && col < N)
        Y[row * N + col] = sum;
}


void init_matrix(double *matrix, int m, int n) {
    for (int i = 0; i < m * n; i ++) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    double *host_A, *host_X, *host_Y;
    double *device_A, *device_X, *device_Y;

    // Get size of matrix
    size_t size_A = M * K * sizeof(double);
    size_t size_X = K * N * sizeof(double);
    size_t size_Y = M * N * sizeof(double);

    // Memory allocation
    host_A = (double*)malloc(size_A);
    host_X = (double*)malloc(size_X);
    host_Y = (double*)malloc(size_Y);

    cudaMalloc(&device_A, size_A);
    cudaMalloc(&device_X, size_X);
    cudaMalloc(&device_Y, size_Y);

    // init matrix
    srand(time(NULL));
    init_matrix(host_A, M, K);
    init_matrix(host_X, K, N);

    // copy memory 
    cudaMemcpy(device_A, host_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(device_X, host_X, size_X, cudaMemcpyHostToDevice);

    // Run the Kernel
    double start_time, end_time;

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Normal Matrix Multiplication Timing
    start_time = get_time();
    matrix_multiplication<<<gridDim, blockDim>>>(device_A, device_X, device_Y, M, K, N);
    cudaDeviceSynchronize();
    end_time = get_time();
    printf("Before Tiled Execution Time: %f ms\n", (end_time - start_time) * 1000);


    dim3 blockDim_tiled(TILE_SIZE, TILE_SIZE);
    dim3 gridDim_tiled((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Tiled Matrix Multiplication Timing
    start_time = get_time();
    matrix_multiplication_tiled<<<gridDim_tiled, blockDim_tiled>>>(device_A, device_X, device_Y, M, K, N);
    cudaDeviceSynchronize();
    end_time = get_time();
    printf("After Tiled Execution Time: %f ms\n", (end_time - start_time) * 1000);

    // Copy results back to the host
    cudaMemcpy(host_Y, device_Y, size_Y, cudaMemcpyDeviceToHost);

    // Free Memory
    free(host_A);
    free(host_X);
    free(host_Y);
    cudaFree(device_A);
    cudaFree(device_X);
    cudaFree(device_Y);

}