#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256

__global__ void intra_block_reduction(double *input, double *output, int n) {
    __shared__ double shared_data[BLOCK_SIZE];  // Shared memory for reduction

    int tid = threadIdx.x; // Thread ID within block
    int id = blockIdx.x * blockDim.x + threadIdx.x; // Global index

    // Load data into shared memory
    if (id < n)
        shared_data[tid] = input[id];
    else
        shared_data[tid] = 0.0;  // Handle out-of-bounds case
    __syncthreads();

    // Perform parallel reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();  // Synchronize to ensure all threads finish each step
    }

    // Write the result of this block’s sum to global memory
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}


void init_vector(double *vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }
}

int main() {
    double *host_a, *host_partial_sums, *device_a, *device_partial_sums;
    size_t size = N * sizeof(double);

    // Allocate memory
    host_a = (double*)malloc(size);
    host_partial_sums = (double*)malloc((N / BLOCK_SIZE) * sizeof(double));

    cudaMalloc(&device_a, size);
    cudaMalloc(&device_partial_sums, (N / BLOCK_SIZE) * sizeof(double));

    // Initialize data
    init_vector(host_a, N);
    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);

    // Run kernel
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    intra_block_reduction<<<num_blocks, BLOCK_SIZE>>>(device_a, device_partial_sums, N);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(host_partial_sums, device_partial_sums, (N / BLOCK_SIZE) * sizeof(double), cudaMemcpyDeviceToHost);

    // Sum the partial results on the CPU
    double total_sum = 0.0;
    for (int i = 0; i < num_blocks; i++) {
        total_sum += host_partial_sums[i];
    }

    printf("Final sum: %f\n", total_sum);

    // Cleanup
    free(host_a);
    free(host_partial_sums);
    cudaFree(device_a);
    cudaFree(device_partial_sums);
    
    return 0;
}