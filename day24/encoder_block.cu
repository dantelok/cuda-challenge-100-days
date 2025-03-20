#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define BATCH_SIZE 8    // Reduced further
#define SEQ_LEN 16      // Reduced further
#define EMBED_DIM 64    // Reduced further
#define HEADS 4
#define D_K (EMBED_DIM / HEADS) // 16
#define HIDDEN_DIM 128  // Reduced further
#define BLOCK_SIZE 16
#define EPSILON 1e-5

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

__device__ double gelu(double x) {
    return 0.5 * x * (1.0 + tanhf(0.7978845608 * (x + 0.044715 * x * x * x)));
}

__global__ void layer_norm_kernel(double *input, double *output, double *gamma, double *beta, int batch_size, int seq_len, int feature_size) {
    int b = blockIdx.y;
    int s = blockIdx.x;
    int f = threadIdx.x;

    if (b >= batch_size || s >= seq_len || f >= feature_size) return;

    extern __shared__ double temp[];
    int id = b * seq_len * feature_size + s * feature_size + f;
    temp[f] = input[id];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (f < stride && f + stride < feature_size) {
            temp[f] += temp[f + stride];
        }
        __syncthreads();
    }

    double mean = temp[0] / feature_size;
    double diff = input[id] - mean;
    temp[f] = diff * diff;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (f < stride && f + stride < feature_size) {
            temp[f] += temp[f + stride];
        }
        __syncthreads();
    }

    double var = sqrt(temp[0] / feature_size + EPSILON);
    output[id] = gamma[f] * (input[id] - mean) / var + beta[f];
}

__global__ void multihead_attention(double *Q, double *K, double *V, double *output, double *scores, int batch_size, int seq_len, int embed_dim, int heads, int d_k) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int col = threadIdx.x;

    if (b >= batch_size || h >= heads || row >= seq_len) return;

    int head_offset = h * d_k;
    double *Q_head = Q + b * seq_len * embed_dim + head_offset;
    double *K_head = K + b * seq_len * embed_dim + head_offset;
    double *V_head = V + b * seq_len * embed_dim + head_offset;
    double *out_head = output + b * seq_len * embed_dim + head_offset;

    if (col < seq_len) {
        double score = 0.0;
        for (int i = 0; i < d_k; i++) {
            score += Q_head[row * embed_dim + i] * K_head[col * embed_dim + i];
        }
        scores[row * seq_len + col] = score / sqrt((double)d_k);
    }
    __syncthreads();

    if (col == 0) {
        double max_val = -INFINITY;
        for (int i = 0; i < seq_len; i++) {
            max_val = fmax(max_val, scores[row * seq_len + i]);
        }
        double sum_exp = 0.0;
        for (int i = 0; i < seq_len; i++) {
            scores[row * seq_len + i] = exp(scores[row * seq_len + i] - max_val);
            sum_exp += scores[row * seq_len + i];
        }
        for (int i = 0; i < seq_len; i++) {
            scores[row * seq_len + i] /= sum_exp;
        }
    }
    __syncthreads();

    if (col < d_k) {
        double val = 0.0;
        for (int i = 0; i < seq_len; i++) {
            val += scores[row * seq_len + i] * V_head[i * embed_dim + col];
        }
        out_head[row * embed_dim + col] = val;
    }
}

__global__ void transformer_ffn(double *input, double *W1, double *b1, double *W2, double *b2, double *output,
                               int batch_size, int seq_len, int embed_dim, int hidden_dim) {
    int b = blockIdx.z;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || s >= seq_len) return;

    double *input_seq = input + b * seq_len * embed_dim + s * embed_dim;
    double *output_seq = output + b * seq_len * embed_dim + s * embed_dim;

    // Use registers instead of shared memory for simplicity
    if (col < embed_dim) {
        double hidden[HIDDEN_DIM];
        for (int h = 0; h < hidden_dim; h++) {
            double sum = 0.0;
            for (int i = 0; i < embed_dim; i++) {
                sum += input_seq[i] * W1[i * hidden_dim + h];
            }
            hidden[h] = gelu(sum + b1[h]);
        }
        double out = 0.0;
        for (int h = 0; h < hidden_dim; h++) {
            out += hidden[h] * W2[h * embed_dim + col];
        }
        output_seq[col] = out + b2[col];
    }
}

int main() {
    srand(time(NULL)); // Seed for random weights

    double *h_input = (double*)malloc(BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(double));
    double *h_output = (double*)malloc(BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(double));
    double *h_gamma = (double*)malloc(EMBED_DIM * sizeof(double));
    double *h_beta = (double*)malloc(EMBED_DIM * sizeof(double));
    double *h_Q = (double*)malloc(BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(double));
    double *h_K = (double*)malloc(BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(double));
    double *h_V = (double*)malloc(BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(double));
    double *h_W1 = (double*)malloc(EMBED_DIM * HIDDEN_DIM * sizeof(double));
    double *h_b1 = (double*)malloc(HIDDEN_DIM * sizeof(double));
    double *h_W2 = (double*)malloc(HIDDEN_DIM * EMBED_DIM * sizeof(double));
    double *h_b2 = (double*)malloc(EMBED_DIM * sizeof(double));
    double *h_layer_out = (double*)malloc(BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(double));
    double *h_attn_out = (double*)malloc(BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(double));

    // Initialize inputs and weights randomly
    for (int i = 0; i < BATCH_SIZE * SEQ_LEN * EMBED_DIM; i++) {
        h_input[i] = (double)rand() / RAND_MAX; // 0 to 1
        h_Q[i] = (double)rand() / RAND_MAX;
        h_K[i] = (double)rand() / RAND_MAX;
        h_V[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < EMBED_DIM; i++) {
        h_gamma[i] = 1.0 + (double)rand() / RAND_MAX * 0.1; // 1.0 to 1.1
        h_beta[i] = (double)rand() / RAND_MAX * 0.1; // 0 to 0.1
        h_b2[i] = 0.0;
    }
    for (int i = 0; i < HIDDEN_DIM; i++) h_b1[i] = 0.0;
    for (int i = 0; i < EMBED_DIM * HIDDEN_DIM; i++) {
        h_W1[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1; // -0.05 to 0.05
        h_W2[i] = ((double)rand() / RAND_MAX - 0.5) * 0.01; // -0.005 to 0.005
    }

    double *d_input, *d_output, *d_gamma, *d_beta;
    double *d_Q, *d_K, *d_V, *d_attn_out, *d_scores;
    double *d_W1, *d_b1, *d_W2, *d_b2, *d_ffn_out;

    size_t size_seq = BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(double);
    size_t size_embed = EMBED_DIM * sizeof(double);
    size_t size_ffn = EMBED_DIM * HIDDEN_DIM * sizeof(double);

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size_seq));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, size_seq));
    CHECK_CUDA_ERROR(cudaMalloc(&d_gamma, size_embed));
    CHECK_CUDA_ERROR(cudaMalloc(&d_beta, size_embed));
    CHECK_CUDA_ERROR(cudaMalloc(&d_Q, size_seq));
    CHECK_CUDA_ERROR(cudaMalloc(&d_K, size_seq));
    CHECK_CUDA_ERROR(cudaMalloc(&d_V, size_seq));
    CHECK_CUDA_ERROR(cudaMalloc(&d_attn_out, size_seq));
    CHECK_CUDA_ERROR(cudaMalloc(&d_scores, SEQ_LEN * SEQ_LEN * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_W1, size_ffn));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b1, HIDDEN_DIM * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_W2, size_ffn));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b2, size_embed));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ffn_out, size_seq));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size_seq, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_gamma, h_gamma, size_embed, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_beta, h_beta, size_embed, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_Q, h_Q, size_seq, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_K, h_K, size_seq, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, h_V, size_seq, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_W1, h_W1, size_ffn, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b1, h_b1, HIDDEN_DIM * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_W2, h_W2, size_ffn, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b2, h_b2, size_embed, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // LayerNorm
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    dim3 gridDim1(SEQ_LEN, BATCH_SIZE, 1);
    dim3 blockDim1(EMBED_DIM, 1, 1);
    size_t shared_mem_size = EMBED_DIM * sizeof(double);
    layer_norm_kernel<<<gridDim1, blockDim1, shared_mem_size>>>(d_input, d_output, d_gamma, d_beta, BATCH_SIZE, SEQ_LEN, EMBED_DIM);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("LayerNorm time: %f ms\n", milliseconds);
    CHECK_CUDA_ERROR(cudaMemcpy(h_layer_out, d_output, size_seq, cudaMemcpyDeviceToHost));

    // Multi-Head Attention
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    dim3 gridDim2((SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE, HEADS, BATCH_SIZE);
    dim3 blockDim2(BLOCK_SIZE, BLOCK_SIZE, 1);
    multihead_attention<<<gridDim2, blockDim2>>>(d_output, d_K, d_V, d_attn_out, d_scores, BATCH_SIZE, SEQ_LEN, EMBED_DIM, HEADS, D_K);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Attention time: %f ms\n", milliseconds);
    CHECK_CUDA_ERROR(cudaMemcpy(h_attn_out, d_attn_out, size_seq, cudaMemcpyDeviceToHost));

    // Feed-Forward Network
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    dim3 gridDim3((EMBED_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE, (SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE, BATCH_SIZE);
    dim3 blockDim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    transformer_ffn<<<gridDim3, blockDim3>>>(d_attn_out, d_W1, d_b1, d_W2, d_b2, d_ffn_out, BATCH_SIZE, SEQ_LEN, EMBED_DIM, HIDDEN_DIM);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("FFN time: %f ms\n", milliseconds);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_ffn_out, size_seq, cudaMemcpyDeviceToHost));

    // Print outputs
    printf("\nLayerNorm Output (first 16):\n");
    for (int i = 0; i < 16; i++) {
        printf("h_layer_out[%d]: %f\n", i, h_layer_out[i]);
    }
    printf("\nAttention Output (first 16):\n");
    for (int i = 0; i < 16; i++) {
        printf("h_attn_out[%d]: %f\n", i, h_attn_out[i]);
    }
    printf("\nFinal Output (first 16):\n");
    for (int i = 0; i < 16; i++) {
        printf("h_output[%d]: %f\n", i, h_output[i]);
    }

    // Cleanup
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_gamma));
    CHECK_CUDA_ERROR(cudaFree(d_beta));
    CHECK_CUDA_ERROR(cudaFree(d_Q));
    CHECK_CUDA_ERROR(cudaFree(d_K));
    CHECK_CUDA_ERROR(cudaFree(d_V));
    CHECK_CUDA_ERROR(cudaFree(d_attn_out));
    CHECK_CUDA_ERROR(cudaFree(d_scores));
    CHECK_CUDA_ERROR(cudaFree(d_W1));
    CHECK_CUDA_ERROR(cudaFree(d_b1));
    CHECK_CUDA_ERROR(cudaFree(d_W2));
    CHECK_CUDA_ERROR(cudaFree(d_b2));
    CHECK_CUDA_ERROR(cudaFree(d_ffn_out));

    free(h_input); free(h_output); free(h_gamma); free(h_beta);
    free(h_Q); free(h_K); free(h_V); free(h_W1); free(h_b1); free(h_W2); free(h_b2);
    free(h_layer_out); free(h_attn_out);

    return 0;
}