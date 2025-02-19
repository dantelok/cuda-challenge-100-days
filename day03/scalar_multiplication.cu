#include <stdio.h>
#include <cuda_runtime.h>

#define N 10000
#define BLOCK_SIZE 256

__global__ void scalar_multiplication(double *a, double *b, double *c, int n) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
        c[id] = (*a) * b[id];
    }
}

void init_scalar(double *scalar) {
    *scalar = (double)rand() / RAND_MAX;
}

void init_vector(double *vector, int n) {
    for (int i=0; i < n; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }
}

int main() {
    double *host_a, *host_b, *host_c, *device_a, *device_b, *device_c;

    // Get size of data type
    size_t size = N * sizeof(double);

    // Allocate memory to the host
    host_a = (double*)malloc(sizeof(double));
    host_b = (double*)malloc(size);
    host_c = (double*)malloc(size);

    // Allocate memory to the device
    cudaMalloc(&device_a, sizeof(double));
    cudaMalloc(&device_b, size);
    cudaMalloc(&device_c, size);

    // Initialize scalar and vector
    init_scalar(host_a);
    init_vector(host_b, N);

    // Copy variables from the host to device
    cudaMemcpy(device_a, host_a, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

    // Calculate num_blocks
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warm up: Calculate the scalar vector multiplication
    for (int i=0; i < 3; i++) {
        scalar_multiplication<<<num_blocks, BLOCK_SIZE>>>(device_a, device_b, device_c, N);
        cudaDeviceSynchronize();
    }

    // Iterate for 100 times
    for (int i=0; i < 100; i++) {
        scalar_multiplication<<<num_blocks, BLOCK_SIZE>>>(device_a, device_b, device_c, N);
        cudaDeviceSynchronize();
    }

    // Copy results from the device back to the host
    cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; i++) {
        printf("c[%d] = %f\n", i, host_c[i]);
    }

    // Free Memory
    free(host_a);
    free(host_b);
    free(host_c);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
}

