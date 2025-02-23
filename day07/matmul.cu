#include <stdio.h>
#include <cuda_runtime.h>

#define M 2048
#define N 2048
#define BLOCK_SIZE (16, 16)

// A: (M, N); X: (N, M); Y: (M, M)
_global__ void matrix_multiplication(double *A, double *X, double *Y, int m, int n) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        double sum = 0.0;
        for (int col=0; col < n; col++) {
            sum += A[n * row + col] * X[n * col + row];
        }
        Y[row * m + col] = sum;
    }
}

void init_matrix(double *matrix, int m, int n) {
    for (int i=0; i < m; i++){
        for (int j=0; j < n; j++){
            matrix[i * n + j] = (double)rand() / RAND_MAX;
        }
    }
}

int main() {
    double *host_A, *host_X, *host_Y;
    double *device_A, *device_X, *device_Y;

    // Get num_blocks: 2D
    int num_blocks = ((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Get size of each variables
    size_t size_input = M * N * sizeof(double);
    size_t size_output = M * M * sizeof(double);

    // Allocate memory to host
    host_A = (double*)malloc(size_input);
    host_X = (double*)malloc(size_input);
    host_Y = (double*)malloc(size_output);

    //  Allocate memory to device
    cudaMalloc(&device_A, size_input);
    cudaMalloc(&device_X, size_input);
    cudaMalloc(&device_Y, size_output);

    // init matrix
    init_matrix(host_A, M, N);
    init_matrix(host_X, M, N);

    // copy from the host to the device
    cudaMemcpy(device_A, host_A, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(device_X, host_X, size_input, cudaMemcpyHostToDevice);

    // Run the Kernels
    matrix_multiplication<<<num_blocks, BLOCK_SIZE>>>(device_A, device_X, device_Y, M, N);

    // copy results back to the host
    cudaMemcpy(host_Y, device_Y, size_input, cudaMemcpyDeviceToHost);

    // Verify Results
    for (int i = 0; i < M*N; i++) {
        printf("y[%d] = %f\n", i, host_y[i]);
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