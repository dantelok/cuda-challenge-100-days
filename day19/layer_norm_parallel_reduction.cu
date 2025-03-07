#include <stdio.h>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256


__global__ void layer_norm_kernel(double *input, double *output, double *gamma, double *beta, int batch_size, int feature_size) {
    __shared__ double batch_sum;
    __shared__ double batch_mean;

    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
        // Step 1: Get batch mean
        if (threadIdx.x == 0)
            batch_sum = 0.0;
        syncthreads();

        AtomicAdd(&batch_sum, input[id]);
        syncthreads();

        batch_mean = batch_sum / batch_size;

        
    }
}


void init_values(double *vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }
}

int main() {
    double h_input[N], h_output[N];
    double *d_input, *d_output;

    cudaMalloc(&d_input, N * sizeof(double));
    cudaMalloc(&d_output, N * sizeof(double));

    init_values(h_input, N);

    cudaMemcpy(d_input, h_input, N * sizeof(double), cudaMemcpyHostToDevice);

    // 1. Try Naive Softmax
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    naive_softmax<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    // Copy back to the host
    cudaMemcpy(h_output, d_output, N * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Naive Softmax Output (first 10):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", h_output[i]);
    }
    printf("\n");

    // 2. Try Log-Sum Exp Softmax
    softmax_logsumexp<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    // Copy back to the host
    cudaMemcpy(h_output, d_output, N * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Log-Sum Exponential Softmax Output (first 10):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", h_output[i]);
    }
    printf("\n");

    // Free Memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}