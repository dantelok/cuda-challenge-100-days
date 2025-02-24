#include <stdio.h>
#include <cuda_runtime.h>

#define M 256
#define K 512
#define N 256
#define BLOCK_SIZE 16

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

void init_matrix(double *matrix, int m, int n) {
    for (int i=0; i < m * n; i++){
            matrix[i] = (double)rand() / RAND_MAX;
    }
}

int main() {
    double *host_A, *host_X, *host_Y;
    double *device_A, *device_X, *device_Y;
    
    // Number of threads per block: 16 x 16
    dim3 blockDim (BLOCK_SIZE, BLOCK_SIZE);
    // Get Number of blocks (num_blocks) for 2D matrix
    dim3 gridDim ((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Get size of each variables
    size_t size_A = M * K * sizeof(double);
    size_t size_X = K * N * sizeof(double);
    size_t size_Y = M * N * sizeof(double);

    // Allocate memory to host
    host_A = (double*)malloc(size_A);
    host_X = (double*)malloc(size_X);
    host_Y = (double*)malloc(size_Y);

    //  Allocate memory to device
    cudaMalloc(&device_A, size_A);
    cudaMalloc(&device_X, size_X);
    cudaMalloc(&device_Y, size_Y);

    // init matrix
    init_matrix(host_A, M, K);
    init_matrix(host_X, K, N);

    // copy from the host to the device
    cudaMemcpy(device_A, host_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(device_X, host_X, size_X, cudaMemcpyHostToDevice);

    // Run the Kernels
    matrix_multiplication<<<gridDim, blockDim>>>(device_A, device_X, device_Y, M, K, N);

    // copy results back to the host
    cudaMemcpy(host_Y, device_Y, size_Y, cudaMemcpyDeviceToHost);

    // Verify Results
    for (int i = 0; i < M*N; i++) {
        printf("y[%d] = %f\n", i, host_Y[i]);
    }

    // Free Memory
    free(host_A);
    free(host_X);
    free(host_Y);
    cudaFree(device_A);
    cudaFree(device_X);
    cudaFree(device_Y);
    
    return 0;
}