#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>  // WMMA API for Tensor Cores

#define M 256
#define N 128
#define K 64
#define TILE_SIZE 16

// CUDA GEMM Kernel using Tensor Cores
__global__ void tensorcore_gemm(half *A, half *B, float *C, int m, int n, int k) {
    // Declare WMMA fragment types for A, B, and C
    wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> c_frag;

    // Identify warp
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;  // Warp ID in M dimension
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;  // Warp ID in N dimension

    // Ensure we don't go out of bounds
    if (warpM * TILE_SIZE >= m || warpN * TILE_SIZE >= n) return;

    // Initialize accumulator fragment with zeros
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension in chunks of TILE_SIZE
    for (int s = 0; s < k; s += TILE_SIZE) {
        // Load matrix A and B tiles into WMMA fragments
        wmma::load_matrix_sync(a_frag, A + warpM * TILE_SIZE * k + s, k);
        wmma::load_matrix_sync(b_frag, B + s * n + warpN * TILE_SIZE, n);

        // Compute C = A * B using Tensor Cores
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store the result into C
    wmma::store_matrix_sync(C + warpM * TILE_SIZE * n + warpN * TILE_SIZE, c_frag, n, wmma::mem_row_major);
}

void init_matrix(half *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = __float2half((float)(rand()) / RAND_MAX);
    }
}

int main() {
    // Host matrices
    // half = fp16 in CUDA
    half *h_A, *h_B;
    float *h_C;

    size_t size_A = M * K * sizeof(half);
    size_t size_B = K * N * sizeof(half);
    size_t size_C = M * N * sizeof(float);

    h_A = (half*)malloc(size_A);
    h_B = (half*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    init_matrix(h_A, M * K);
    init_matrix(h_B, K * N);

    // Device matrices
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy data to GPU
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);

    // Configure execution grid
    dim3 blockDim(32, 32);  // 32 threads per warp
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    tensorcore_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Print results
    printf("Tensor Core GEMM Output (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", h_C[i]);
    }
    printf("\n");

    // Free memory
    free(h_A); 
    free(h_B); 
    free(h_C);
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);

    return 0;
}