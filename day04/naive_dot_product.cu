#include <stdio.h>
#include <cuda_runtime.h>

#define N 10000
#define BLOCK_SIZE 256

// c is a scalar; where a and b is 2 vectors
__global__ void naive_dot_product(double *a, double *b, double *c, int n) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
        // (*c) += a[id] * b[id];  // race condition, might get wrong results
        atomicAdd(c, a[id] * b[id]);  // Fix race condition
    }
}

void init_vector(double *vector, int n) {
    for (int i=0; i < n; i++) {
        vector[i] = (double) rand() / RAND_MAX;
    }
}

int main() {
    double *host_a, *host_b, *host_c, *device_a, *device_b, *device_c;

    // Get size
    size_t size = N * sizeof(double);

    // Allocate memory to the host
    host_a = (double*)malloc(size);
    host_b = (double*)malloc(size);
    host_c = (double*)malloc(sizeof(double));

    // Allocate memory to the device
    cudaMalloc(&device_a, size);
    cudaMalloc(&device_b, size);
    cudaMalloc(&device_c, sizeof(double));

    // init value
    init_vector(host_a, N);
    init_vector(host_b, N);
    // *device_c = 0;  // Incorrect memory access: Cannot assign device memory directly from host 

    // Copy from host to device
    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

    // Init c = 0
    double zero = 0.0;
    cudaMemcpy(device_c, &zero, sizeof(double), cudaMemcpyHostToDevice);
    
    // Get grid num
    int nums_block = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Run the Kernel
    naive_dot_product<<<nums_block, BLOCK_SIZE>>>(device_a, device_b, device_c, N);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(host_c, device_c, sizeof(double), cudaMemcpyDeviceToHost);

    // Verify Results
    printf("c: %f \n", *host_c);

    // Free Memory
    free(host_a);
    free(host_b);
    free(host_c);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
}