#include <stdio.h>
#include <cuda_runtime.h>

#define SEQ_LEN 128
#define D_K 64
#define BLOCK_SIZE 256

__global__ void multihead_attention(double *Q, double *K, double *V, double *output, int seq_len, int d_k) {
    __shared__ double *scores;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len && col < seq_len) {
        float attn_score = 0.0;

        // Step 1: QK^T
        // d_k = # columns in Q = # rows in K
        for (int i = 0; i < d_k; i++){
            attn_score += Q[row * d_k + i] * K[col * d_k + i];
        }
        attn_score /= sqrt(d_k);

        scores[row * seq_len + col] =  attn_score;
    }
    __syncthreads();

    // Step 2: Softmax Log-sum exp
    // Step 2a: 
    


    // Step 3: Softmax logits * V

}


int main() {
    double *h_q, *h_k, *h_v, *h_attn_score, *h_attn_weights;
    double *d_q, *d_k, *d_v, *d_attn_score, *d_attn_weights;



    return 0;
}