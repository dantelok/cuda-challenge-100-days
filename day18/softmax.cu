#include <stdio.h>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256

/* log-sum-exp softmax = exp(x_i - max(x)) / sum_j(exp(x_j - max(x))) */

// Naive Softmax 
__global__ void naive_softmax(double *input, double *output, int n) {
    __shared__ double sum_exp;
    
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
        // Step 1: Compute exponential values: exp(x_i - max(x))
        double exp_values = exp(input[id]);

        // Step 2: Get sum of the exponential values
        // The beginning of the row
        if (threadIdx.x == 0) 
            sum_exp = 0.0;
        __syncthreads();
        
        atomicAdd(&sum_exp, exp_values);  // Add all elements inside the block
        __syncthreads();

        // Step 3: Compute softmax
        output[id] = exp_values / sum_exp;
    }
}


// Log-sum Exponential
__global__ void softmax_logsumexp(double *input, double *output, int n) {
    __shared__ double max_exp;
    __shared__ double sum_exp;
    
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    if (id < n) {
        // Step 1: Get max value from the logits (Avoid Overflow)
        double local_max = input[id];

        // Perform a parallel reduction for max inside each block
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                local_max = fmax(local_max, input[id + stride]);
            }
            __syncthreads();
        }

        // The first thread stores max value for the block
        if (tid == 0) 
            max_exp = local_max;
        __syncthreads();

        // Step 2: Compute exponential values: exp(x_i - max(x))
        double exp_values = exp(input[id] - max_exp);

        // Step 3: Get sum of the exponential values
        if (threadIdx.x == 0) 
            sum_exp = 0.0;
        __syncthreads();
        
        atomicAdd(&sum_exp, exp_values);
         __syncthreads();

         // Step 4: Compute softmax
         output[id] = exp_values / sum_exp;
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