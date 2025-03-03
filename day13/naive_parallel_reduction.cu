#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024
#define BLOCK_SIZE 256

double array_sum_cpu(double *vector, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += vector[i];
    }
    return sum;
}

__global__ void blockwise_parallel_reduction(double *input, double *partial_sums, int n) {
    __shared__ double shared_data[BLOCK_SIZE];

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    shared_data[tid] = (id < n) ? input[id] : 0.0;
    __syncthreads();

    // Perform parallel reduction within block
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Write block's sum to output
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_data[0];
    }
}

__global__ void final_sum_kernel(double *partial_sums, double *output, int num_blocks) {
    __shared__ double final_shared[BLOCK_SIZE];

    int tid = threadIdx.x;
    final_shared[tid] = (tid < num_blocks) ? partial_sums[tid] : 0.0;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            final_shared[tid] += final_shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = final_shared[0];
    }
}

void init_vector(double *vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }
}

double gettime() {
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    return timestamp.tv_sec + timestamp.tv_nsec * 1e-9;
}

int main() {
    double *host_a, *host_partial_sums, *host_c, *device_a, *device_partial_sums, *device_c;
    double cpu_sum;

    size_t size = N * sizeof(double);
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t partial_size = num_blocks * sizeof(double);

    host_a = (double*)malloc(size);
    host_partial_sums = (double*)malloc(partial_size);
    host_c = (double*)malloc(sizeof(double));

    cudaMalloc(&device_a, size);
    cudaMalloc(&device_partial_sums, partial_size);
    cudaMalloc(&device_c, sizeof(double));

    init_vector(host_a, N);

    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);

    double start_time, end_time;

    printf("Start Array Sum with Parallel Reduction...\n");
    start_time = gettime();
    
    // Step 1: Blockwise reduction
    blockwise_parallel_reduction<<<num_blocks, BLOCK_SIZE>>>(device_a, device_partial_sums, N);
    cudaDeviceSynchronize();

    // Step 2: Final reduction
    final_sum_kernel<<<1, BLOCK_SIZE>>>(device_partial_sums, device_c, num_blocks);
    cudaDeviceSynchronize();

    end_time = gettime();
    printf("Parallel Reduction execution time: %f seconds\n", end_time - start_time);

    printf("Start Array Sum with CPU..\n");
    start_time = gettime();
    cpu_sum = array_sum_cpu(host_a, N);
    end_time = gettime();
    printf("CPU execution time: %f seconds\n", end_time - start_time);

    cudaMemcpy(host_c, device_c, sizeof(double), cudaMemcpyDeviceToHost);

    // Verify Results
    printf("CPU Results: %f\n", cpu_sum);
    printf("GPU Results: %f\n", *host_c);

    if (abs(cpu_sum - *host_c) < 1e-6) {
        printf("Answer Matched!\n");
    } else {
        printf("Mismatch in Results!\n");
    }

    // Free memory
    free(host_a);
    free(host_partial_sums);
    free(host_c);
    cudaFree(device_a);
    cudaFree(device_partial_sums);
    cudaFree(device_c);
}