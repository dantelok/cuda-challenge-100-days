#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024
#define BLOCK_SIZE 256

double array_sum_cpu(double *vector, int n) {
    double sum = 0.0;
    for (int i=0; i < n; i++) {
        sum += vector[i];
    }
    return sum;
}

__global__ void naive_array_sum_parallel_reduction(double *input, double *output, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Parallel reduction
    for (int stride = n / 2; stride > 0; stride /= 2) {
        if (id < stride) {
            input[id] += input[id + stride]; // Sum pairs iteratively
        }
        __syncthreads(); 
        // Sync to ensure correct results
    }

    // Store final sum in global memory
    if (id == 0) {
        *output = input[0];
    }

}

void init_vector(double *vector, int n) {
    for (int i=0; i < n; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }
}

double gettime() {
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    return timestamp.tv_sec + timestamp.tv_nsec * 1e-9;
}


int main() {
    double *host_a, *host_c, *device_a, *device_c;
    double cpu_sum;

    size_t size = N * sizeof(double);

    host_a = (double*)malloc(size);
    host_c = (double*)malloc(sizeof(double));

    cudaMalloc(&device_a, size);
    cudaMalloc(&device_c, sizeof(double));

    init_vector(host_a, N);

    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
    
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double start_time, end_time;

    printf("Start Array Sum with Parallel Reduction...\n");
    start_time = gettime();
    naive_array_sum_parallel_reduction<<<num_blocks, BLOCK_SIZE>>>(device_a, device_c, N);
    end_time = gettime();
    printf("Parallel Reduction execution time: %f second\n", end_time - start_time);

    printf("Start Array Sum with CPU..\n");
    start_time = gettime();
    cpu_sum = array_sum_cpu(host_a, N);
    end_time = gettime();
    printf("CPU execution time: %f second\n", end_time - start_time);

    cudaMemcpy(host_c, device_c, sizeof(double), cudaMemcpyDeviceToHost);

    // Verify Results
    printf("CPU Results: %f\n", cpu_sum);
    printf("GPU Results: %f\n", *host_c);

    if (abs(cpu_sum - *host_c) < 1e-6) {
        printf("✅ Answer Matched!\n");
    } else {
        printf("❌ Mismatch in Results!\n");
    }

    // Free memory
    free(host_a);
    free(host_c);
    cudaFree(device_a);
    cudaFree(device_c);
}