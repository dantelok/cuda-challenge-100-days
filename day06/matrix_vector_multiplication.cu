#include <stdio.h>
#include <cuda_runtime.h>

#define N 2048
#define M 2048
#define BLOCK_SIZE 256

// Matrix A: (m, n) dot vector x: (n, 1) -> vector y: (m, 1)
__global__ void matrix_vector_multiplcation(double *A, double *x, double *y, int m, int n) {

    // threads = row
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < m) {
        double sum = 0.0;
        for (int col = 0; col < n; col++) {
            // the number of elements = no. of elements per row * row + col
            sum += A[n * row + col] * x[col];
        }
        y[row] = sum;
    }
}

void init_vector(double *vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }
}

void init_matrix(double *matrix, int m, int n) {
    for (int i = 0; i < m * n; i++) {
            matrix[i] = (double)rand() / RAND_MAX;
    }
}

int main() {
    double *host_A, *host_x, *host_y;
    double *device_A, *device_x, *device_y;

    // Get num_blocks
    int num_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Get size
    size_t size_A = M * N * sizeof(double);
    size_t size_x = N * sizeof(double);
    size_t size_y = M * sizeof(double);

    // Allocate memory on host
    host_A = (double*)malloc(size_A);
    host_x = (double*)malloc(size_x);
    host_y = (double*)malloc(size_y);

    // Allocate memory on device
    cudaMalloc(&device_A, size_A); 
    cudaMalloc(&device_x, size_x); 
    cudaMalloc(&device_y, size_y); 

    // Init vector
    init_matrix(host_A, M, N);
    init_vector(host_x, N);

    // Copy from host to device
    cudaMemcpy(device_A, host_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(device_x, host_x, size_x, cudaMemcpyHostToDevice);

    // Run the kernel
    matrix_vector_multiplcation<<<num_blocks, BLOCK_SIZE>>>(device_A, device_x, device_y, M, N);
    cudaDeviceSynchronize();

    // Copy results
    cudaMemcpy(host_y, device_y, size_y, cudaMemcpyDeviceToHost);

    // Verify Results
    for (int i = 0; i < M; i++) {
        printf("y[%d] = %f\n", i, host_y[i]);
    }

    // Free Memory
    free(host_A);
    free(host_x);
    free(host_y);
    cudaFree(device_A);
    cudaFree(device_x);
    cudaFree(device_y);
    
    return 0;
}
