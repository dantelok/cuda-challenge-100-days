#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define BATCH_SIZE 64
#define SEQ_LEN 128
#define EMBED_DIM 256
#define HEADS 8
#define D_K (EMBED_DIM / HEADS)
#define HIDDEN_DIM 1024
#define BLOCK_SIZE 16
#define EPSILON 1e-5


__device__ double gelu(double x) {
    return 0.5 * x * (1.0 + tanhf(0.7978845608 * (x + 0.044715 * x * x * x)));
}

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

__global__ void multihead_attention(double *Q, double *K, double *V, double *output, int seq_len, int d_k) {
    extern __shared__ double shared_mem[]; // Dynamic shared memory

    double* scores = shared_mem;      // Shared memory for attention scores
    double* row_max = scores + seq_len * seq_len;  // Shared memory for row-wise max
    double* row_sum = row_max + seq_len;  // Shared memory for row-wise sum

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: QK^T
    double sum = 0.0;
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
        output[row * d_k + col] = sum;
    }
}

__global__ void transformer_ffn(double *input, double *W1, double *b1, double *W2, double *b2, double *output, int batch_size, int input_dim, int hidden_dim, int output_dim) {

    __shared__ double hidden_values[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < hidden_dim) {
        double sum = 0.0;
        for (int i = 0; i < input_dim; i++) {
            sum = input[row * input_dim + i] * W1[i * hidden_dim + col];
        }
        sum += b1[col];
        hidden_values[threadIdx.y][threadIdx.x] = gelu(sum);
    }
    __syncthreads();

    if (row < batch_size && col < output_dim) {
        double sum = 0.0;
        for (int i = 0; i < hidden_dim; i++) {
            sum = hidden_values[threadIdx.y][i] * W2[i * output_dim + col];
        }
        sum += b2[col];
        output[row * output_dim + col] = sum;
    }
}


int main() {
    double *d_input, *d_output, *d_gamma, *d_beta;
    double *d_Q, *d_K, *d_V, *d_attn_out;
    double *d_W1, *d_b1, *d_W2, *d_b2, *d_ffn_out;

    size_t size_seq = BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(double);
    size_t size_embed = EMBED_DIM * sizeof(double);
    size_t size_ffn = EMBED_DIM * HIDDEN_DIM * sizeof(double);

    cudaMalloc(&d_input, size_seq);
    cudaMalloc(&d_output, size_seq);
    cudaMalloc(&d_gamma, size_embed);
    cudaMalloc(&d_beta, size_embed);
    cudaMalloc(&d_Q, size_seq);
    cudaMalloc(&d_K, size_seq);
    cudaMalloc(&d_V, size_seq);
    cudaMalloc(&d_attn_out, size_seq);
    cudaMalloc(&d_W1, size_ffn);
    cudaMalloc(&d_b1, HIDDEN_DIM * sizeof(double));
    cudaMalloc(&d_W2, size_ffn);
    cudaMalloc(&d_b2, EMBED_DIM * sizeof(double));
    cudaMalloc(&d_ffn_out, size_seq);

    // Run LayerNorm before Attention
    dim3 gridDim1(SEQ_LEN);
    dim3 blockDim1(EMBED_DIM);
    layer_norm_kernel<<<gridDim1, blockDim1>>>(d_input, d_output, d_gamma, d_beta, SEQ_LEN, EMBED_DIM);

    // Run Self-Attention
    dim3 gridDim2((SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE, (SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim2(BLOCK_SIZE, BLOCK_SIZE);
    multihead_attention<<<gridDim2, blockDim2>>>(d_Q, d_K, d_V, d_attn_out, SEQ_LEN, EMBED_DIM);

    // Run Feedforward Network
    dim3 gridDim3((EMBED_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE, (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    transformer_ffn<<<gridDim3, blockDim2>>>(d_attn_out, d_W1, d_b1, d_W2, d_b2, d_ffn_out, BATCH_SIZE, EMBED_DIM, HIDDEN_DIM);

    cudaDeviceSynchronize();

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_attn_out);
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_ffn_out);

    return 0;
}