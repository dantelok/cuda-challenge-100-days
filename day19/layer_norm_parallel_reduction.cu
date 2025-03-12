#include <stdio.h>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define BATCH_SIZE 64   // Number of independent data samples (rows)
#define FEATURE_SIZE 256 // Number of features per sample (columns)
#define BLOCK_SIZE 256
#define EPSILON 1e-5


__global__ void layer_norm_kernel(double *input, double *output, double *gamma, double *beta, int batch_size, int feature_size) {
    __shared__ double batch_mean;
    __shared__ double batch_var;

    int batch_id = blockIdx.x;
    int feature_id = threadIdx.x;
    int id = batch_id * feature_size + feature_id;
    

    // Step 1: Get batch mean
    double sum = 0.0;

    for (int i = feature_id; i < feature_size; i += blockDim.x) {
        sum += input[batch_id * feature_size + i];
    }

    // Temp shared memory for reduction
    __shared__ double temp[BLOCK_SIZE];
    temp[feature_id] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (feature_id < stride) {
            temp[feature_id] += temp[feature_id + stride];
        }
        __syncthreads();
    }
    
    if (feature_id == 0) {
        batch_mean = temp[0] / feature_size;
    }
    __syncthreads();


    // Step 2: Get batch variance
    double sum_square = 0.0;

    for (int i = feature_id; i < feature_size; i += blockDim.x) {
        double diff = input[batch_id * feature_size + i] - batch_mean;
        sum_square += diff * diff;
    }

    temp[feature_id] = sum_square;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (feature_id < stride) {
            temp[feature_id] += temp[feature_id + stride];
        }
        __syncthreads();
    }

    if (feature_id == 0) {
        batch_var = sqrt(temp[0] / feature_size + EPSILON);
    }
    __syncthreads();

    // Step 3: Normalized
    if (feature_id < feature_size) {
        double normalized = (input[id] - batch_mean) / batch_var;
        output[id] = gamma[feature_id] * normalized + beta[feature_id];
    }

}


void init_values(double *vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }
}

int main() {
    double *h_input, *h_output, *h_gamma, *h_beta;
    double *d_input, *d_output, *d_gamma, *d_beta;

    size_t size = BATCH_SIZE * FEATURE_SIZE * sizeof(double);
    size_t param_size = FEATURE_SIZE * sizeof(double);

    // Allocate host memory
    h_input = (double*)malloc(size);
    h_output = (double*)malloc(size);
    h_gamma = (double*)malloc(param_size);
    h_beta = (double*)malloc(param_size);

    // Initialize data
    init_values(h_input, BATCH_SIZE * FEATURE_SIZE);
    init_values(h_gamma, FEATURE_SIZE);
    init_values(h_beta, FEATURE_SIZE);

    // Allocate device memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_gamma, param_size);
    cudaMalloc(&d_beta, param_size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, param_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, param_size, cudaMemcpyHostToDevice);

    // Launch Layer Normalization Kernel
    dim3 gridDim(BATCH_SIZE);
    dim3 blockDim(BLOCK_SIZE);
    layer_norm_kernel<<<gridDim, blockDim>>>(d_input, d_output, d_gamma, d_beta, BATCH_SIZE, FEATURE_SIZE);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print first row results
    printf("Layer Normalized Output (first 10 values):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", h_output[i]);
    }
    printf("\n");

    // Free Memory
    free(h_input);
    free(h_output);
    free(h_gamma);
    free(h_beta);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    return 0;
}