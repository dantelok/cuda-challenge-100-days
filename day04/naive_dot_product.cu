#include <stdio.h>
#include <cuda_runtime.h>

#define N 10000
#define BLOCK_SIZE 256

// partial_results is for storing intermediate result; where a and b is 2 vectors
__global__ void naive_dot_product(double *a, double *b, double *partial_results, int n) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
        // (*c) += a[id] * b[id];  // race condition, might get wrong results
       
        // atomicAdd(c, a[id] * b[id]);  // Fix race condition
        // Because this time we are not overwriting the new value but adding value into it, it must be init to zero before use
        partial_results[id] += a[id] * b[id];
    }
}

void init_vector(double *vector, int n) {
    for (int i=0; i < n; i++) {
        vector[i] = (double) rand() / RAND_MAX;
    }
}

int main() {
    double *host_a, *host_b, *host_partial, *device_a, *device_b, *device_partial;

    // Get size
    size_t size = N * sizeof(double);

    // Allocate memory to the host
    host_a = (double*)malloc(size);
    host_b = (double*)malloc(size);
    host_partial = (double*)malloc(size);

    // Allocate memory to the device
    cudaMalloc(&device_a, size);
    cudaMalloc(&device_b, size);
    cudaMalloc(&device_partial, size);

    // init value
    init_vector(host_a, N);
    init_vector(host_b, N);
    
    // Zero out partial results array before use
    cudaMemset(device_partial, 0, size);

    // Copy from host to device
    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);
    
    // Get grid numa
    int nums_block = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Run the Kernel
    naive_dot_product<<<nums_block, BLOCK_SIZE>>>(device_a, device_b, device_partial, N);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(host_partial, device_partial, size, cudaMemcpyDeviceToHost);

    // Verify Results
    double final_result = 0.0;
    for (int i = 0; i < N; i++) {
        final_result += host_partial[i];  // ✅ Perform final reduction
    }

    // Verify Results
    printf("Final results: %f \n", final_result);

    // Free Memory
    free(host_a);
    free(host_b);
    free(host_partial);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_partial);
}