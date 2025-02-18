#include <iostream>
#include <cuda_runtime.h>

#define N 10000
#define BLOCK_SIZE 256  // Threads per block

__global__ void vector_addition(double *a, double *b, double *c, int n) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

void init_vector(double *vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = (float)rand() / (float)RAND_MAX;
    }
}

int main() {
    double *host_A, *host_B, *host_C, *device_a, *device_b, *device_c;
    // Allocate the number of bytes
    size_t size = N * sizeof(double);

    // Allocate memory on host (CPU)
    host_A = (double*)malloc(size);
    host_B = (double*)malloc(size);
    host_C = (double*)malloc(size);

    // Allocate memory on device (GPU)
    cudaMalloc(&device_a, size);
    cudaMalloc(&device_b, size);
    cudaMalloc(&device_c, size);

    // Initialize vectors
    srand(time(NULL));
    init_vector(host_A, N);
    init_vector(host_B, N);

    // Copy data from host(CPU) to device(GPU)
    // ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
    // kind: host -> host; host -> device; device -> host; device -> device; 
    cudaMemcpy(device_a, host_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimension
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Single run
    vector_addition<<<num_blocks, BLOCK_SIZE>>>(device_a, device_b, device_c, N);

    // Multiple runs
    for (int i = 0; i < 100; i++) {
        vector_addition<<<num_blocks, BLOCK_SIZE>>>(device_a, device_b, device_c, N);
        cudaDeviceSynchronize();
        // Blocks until the device has completed all preceding requested tasks.
        // returns an error if one of the preceding tasks has failed.
    }

    // Copy results back to the host
    cudaMemcpy(host_C, device_c, size, cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; i++) {
        printf("c[%d] = %f\n", i, host_C[i]);
    }

    // Free Memory
    free(host_A);
    free(host_B);
    free(host_C);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
}
    
