#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define BATCH_SIZE 64
#define INPUT_DIM 256
#define HIDDEN_DIM 1024
#define OUTPUT_DIM 256
#define BLOCK_SIZE 16

__device__ double gelu(double x) {
    return 0.5 * x * (1.0 + tanhf(0.7978845608 * (x + 0.044715 * x * x * x)));
}

__global__ void transformer_ffn(double *input, double *W1, double *b1, double *W2, double *b2, double *output, int batch_size, int input_dim, int hidden_dim, int output_dim) {

    __shared__ float hidden_values[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < hidden_dim) {
        float sum = 0.0;
        for (int i = 0; i < input_dim; i++) {
            sum = input[row * input_dim + i] * W1[i * hidden_dim + col];
        }
        sum += b1[col];
        hidden_values[threadIdx.y][threadIdx.x] = gelu(sum);
    }
    __syncthreads();

    if (row < batch_size && col < output_dim) {
        float sum = 0.0;
        for (int i = 0; i < hidden_dim; i++) {
            sum = hidden_values[threadIdx.y][i] * W2[i * output_dim + col];
        }
        sum += b2[col];
        output[row * output_dim + col] = sum;
    }
}

void init_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
}

int main() {
    double *h_input, *h_W1, *h_b1, *h_W2, *h_b2, *h_output;
    double *d_input, *d_W1, *d_b1, *d_W2, *d_b2, *d_output;

    size_t size_input = BATCH_SIZE * INPUT_DIM * sizeof(double);
    size_t size_hidden = HIDDEN_DIM * sizeof(double);
    size_t size_W1 = INPUT_DIM * HIDDEN_DIM * sizeof(double);
    size_t size_W2 = HIDDEN_DIM * OUTPUT_DIM * sizeof(double);
    size_t size_output = BATCH_SIZE * OUTPUT_DIM * sizeof(double);

    // Allocate host memory
    h_input = (double*)malloc(size_input);
    h_W1 = (double*)malloc(size_W1);
    h_b1 = (double*)malloc(size_hidden);
    h_W2 = (double*)malloc(size_W2);
    h_b2 = (double*)malloc(size_hidden);
    h_output = (double*)malloc(size_output);

    // Initialize matrices
    init_matrix(h_input, BATCH_SIZE, INPUT_DIM);
    init_matrix(h_W1, INPUT_DIM, HIDDEN_DIM);
    init_matrix(h_b1, 1, HIDDEN_DIM);
    init_matrix(h_W2, HIDDEN_DIM, OUTPUT_DIM);
    init_matrix(h_b2, 1, OUTPUT_DIM);

    // Allocate device memory
    cudaMalloc(&d_input, size_input);
    cudaMalloc(&d_W1, size_W1);
    cudaMalloc(&d_b1, size_hidden);
    cudaMalloc(&d_W2, size_W2);
    cudaMalloc(&d_b2, size_hidden);
    cudaMalloc(&d_output, size_output);

    // Copy data to device
    cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1, size_W1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, size_hidden, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, size_W2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, size_hidden, cudaMemcpyHostToDevice);

    // Define CUDA grid
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((HIDDEN_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE, (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Run CUDA Transformer FFN
    transformer_ffn<<<gridDim, blockDim>>>(d_input, d_W1, d_b1, d_W2, d_b2, d_output,
                                           BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_output, d_output, size_output, cudaMemcpyDeviceToHost);

    // Print first row results
    printf("Transformer FFN Output (first 10 values):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", h_output[i]);
    }
    printf("\n");

    // Free memory
    free(h_input);
    free(h_W1);
    free(h_b1);
    free(h_W2);
    free(h_b2);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_output);

    return 0;
}