#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 8  // Array size
#define BLOCK_SIZE 8  // Threads per block

__global__ void exclusive_prefix_sum(double *input, double *output, int n) {
    __shared__ double shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    if (global_id < n)
        shared_data[tid] = input[global_id];
    else
        shared_data[tid] = 0; // Pad with zero if out of bounds
    __syncthreads();

    // Exclusive scan using upsweep (reduction phase)
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        double temp = 0;
        if (tid >= stride) {
            temp = shared_data[tid - stride];
        }
        __syncthreads();
        shared_data[tid] += temp;
        __syncthreads();
    }

    // Convert to exclusive scan
    if (tid == 0) shared_data[tid] = 0;  // First element is always 0
    __syncthreads();

    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        double temp = shared_data[tid];
        if (tid >= stride) {
            temp = shared_data[tid - stride];
        }
        __syncthreads();
        shared_data[tid] = temp;
        __syncthreads();
    }

    // Store result in global memory
    if (global_id < n)
        output[global_id] = shared_data[tid];
}

void init_vector(double *vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = (double)(rand() % 10);  // Random numbers 0-9
    }
}

void print_vector(double *vector, int n, const char *label) {
    printf("%s: ", label);
    for (int i = 0; i < n; i++) {
        printf("%.0f ", vector[i]);
    }
    printf("\n");
}

int main() {
    double *host_input, *host_output, *device_input, *device_output;
    size_t size = N * sizeof(double);

    // Allocate memory
    host_input = (double*)malloc(size);
    host_output = (double*)malloc(size);
    cudaMalloc(&device_input, size);
    cudaMalloc(&device_output, size);

    // Initialize input vector
    init_vector(host_input, N);
    print_vector(host_input, N, "Input");

    // Copy data to device
    cudaMemcpy(device_input, host_input, size, cudaMemcpyHostToDevice);

    // Run the kernel
    exclusive_prefix_sum<<<1, BLOCK_SIZE>>>(device_input, device_output, N);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(host_output, device_output, size, cudaMemcpyDeviceToHost);

    // Print results
    print_vector(host_output, N, "Exclusive Prefix Sum");

    // Free memory
    free(host_input);
    free(host_output);
    cudaFree(device_input);
    cudaFree(device_output);

    return 0;
}