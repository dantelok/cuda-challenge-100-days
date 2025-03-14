#include <stdio.h>
#include <cuda_runtime.h>

#define SEQ_LEN 128
#define D_K 64
#define BLOCK_SIZE 256
#define EPSILON 1e-6

__global__ void multihead_attention(double *Q, double *K, double *V, double *output, int seq_len, int d_k) {
    extern __shared__ double shared_mem[]; // Dynamic shared memory

    double* scores = shared_mem;      // Shared memory for attention scores
    double* row_max = scores + seq_len * seq_len;  // Shared memory for row-wise max
    double* row_sum = row_max + seq_len;  // Shared memory for row-wise sum


    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: QK^T
    if (row < seq_len && col < seq_len) {
        double attn_score = 0.0;
        
        // Q: (d_k, seq_len); K^T = (seq_len, d_k)
        // d_k = # columns in Q = # rows in K
        for (int i = 0; i < d_k; i++){
            attn_score += Q[row * d_k + i] * K[col * d_k + i];
        }
        attn_score /= sqrt(d_k);

        scores[row * seq_len + col] =  attn_score;
    }
    __syncthreads();

    // Step 2: Softmax Log-sum exp
    // Step 2a: get max value per row
    if (col == 0 && row < seq_len) {
        double max_value = -INFINITY;

        for (int i = 0; i < seq_len; i++) {
            max_value = fmaxf(max_value, scores[row * seq_len + i]);
        }
        row_max[row] = max_value;
    }
    __syncthreads();

    // Step 2b: exponential value
    if (row < seq_len && col < seq_len) {
        scores[row * seq_len + col] = expf(scores[row * seq_len + col] - row_max[row]);
    }
    __syncthreads();

    // Step 2c: Get row sum of exponential value
    if (col == 0 && row < seq_len) {
        double sum = 0.0;

        for (int i = 0; i < seq_len; i++) {
            sum += scores[row * seq_len + i];
        }
        row_sum[row] = sum + EPSILON;  // Add EPSILON for numerical stability
    }
    __syncthreads();

    // Step 2d: Divided by sum of exponential value
    if (row < seq_len && col < seq_len) {
        scores[row * seq_len + col] /= row_sum[row];
    }
    __syncthreads();

    // Step 3: Softmax logits * V
    if (row < seq_len && col < d_k) {
        double attn_score = 0.0;

        // scores: (seq_len, d_k); V: (d_k, seq_len)
        for (int i = 0; i < seq_len; i++){
            attn_score += scores[row * seq_len + i] * V[col * d_k + i];
        }
        output[row * d_k + col] = attn_score;
    }

}


int main() {
    double *h_Q, *h_K, *h_V, *h_output;
    double *d_Q, *d_K, *d_V, *d_output;

    int size_QK = SEQ_LEN * D_K * sizeof(double);
    int size_output = SEQ_LEN * D_K * sizeof(double);

    h_Q = (double*)malloc(size_QK);
    h_K = (double*)malloc(size_QK);
    h_V = (double*)malloc(size_QK);
    h_output = (double*)malloc(size_output);

    cudaMalloc(&d_Q, size_QK);
    cudaMalloc(&d_K, size_QK);
    cudaMalloc(&d_V, size_QK);
    cudaMalloc(&d_output, size_output);

    init_matrix(h_Q, SEQ_LEN, D_K);
    init_matrix(h_K, SEQ_LEN, D_K);
    init_matrix(h_V, SEQ_LEN, D_K);

    cudaMemcpy(d_Q, h_Q, size_QK, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size_QK, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size_QK, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE, (SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE);

    size_t shared_mem_size = (SEQ_LEN * SEQ_LEN + 2 * SEQ_LEN) * sizeof(double);

    multihead_attention<<<gridDim, blockDim, shared_mem_size>>>(d_Q, d_K, d_V, d_output, SEQ_LEN, D_K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, size_output, cudaMemcpyDeviceToHost);

    printf("Fused Self-Attention Output (first 10 values):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", h_output[i]);
    }
    printf("\n");

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_output);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    return 0;
}