#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 10000
#define BLOCK_SIZE 256

// Warp size default 32

__global__ void vector_addition_conditional(double *a, double *b, double *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Causes warp divergence
    if (id < n) {
        if (a[id] > 0.5) {  
            c[id] = a[id] + b[id];
        } else {
            c[id] = a[id] - b[id];
        }
    }
}

__global__ void vector_addition_no_divergence(double *a, double *b, double *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // No branch, avoids divergence
    if (id < n) {
        // Set if-else before memory access
        // double modifier = 0.0;
        // if (a[id] > 0.5) {
        //     modifier = 1.0;
        // } else {
        //     modifier = -1.0;
        // }
        double modifier = (a[id] > 0.5) ? 1.0 : -1.0;  // Even faster
        // No branch: Remove if-else from the calculation
        c[id] = a[id] + modifier * b[id];
    }
}

void init_vector(double *vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    double *host_A, *host_B, *host_C, *device_a, *device_b, *device_c;
    size_t size = N * sizeof(double);

    // Allocate memory on host
    host_A = (double*)malloc(size);
    host_B = (double*)malloc(size);
    host_C = (double*)malloc(size);

    // Allocate memory on device
    cudaMalloc(&device_a, size);
    cudaMalloc(&device_b, size);
    cudaMalloc(&device_c, size);

    // Initialize vectors
    srand(time(NULL));
    init_vector(host_A, N);
    init_vector(host_B, N);

    // Copy data from host to device
    cudaMemcpy(device_a, host_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double start_time, end_time;

    // Warp-Divergent Kernel Timing
    start_time = get_time();
    vector_addition_conditional<<<num_blocks, BLOCK_SIZE>>>(device_a, device_b, device_c, N);
    cudaDeviceSynchronize();
    end_time = get_time();
    printf("Warp Divergent Execution Time: %f ms\n", (end_time - start_time) * 1000);

    // Warp-Optimized Kernel Timing
    start_time = get_time();
    vector_addition_no_divergence<<<num_blocks, BLOCK_SIZE>>>(device_a, device_b, device_c, N);
    cudaDeviceSynchronize();
    end_time = get_time();
    printf("Optimized Execution Time: %f ms\n", (end_time - start_time) * 1000);

    // Copy results back to host
    cudaMemcpy(host_C, device_c, size, cudaMemcpyDeviceToHost);

    // Free Memory
    free(host_A);
    free(host_B);
    free(host_C);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}