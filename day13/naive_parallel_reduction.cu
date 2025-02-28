#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 10000
#define BLOCK_SIZE 256

double array_sum_cpu(double *vector, int n) {
    double sum = 0.0;
    for (int i=0; i < n; i++) {
        sum += vector[i];
    }
    return sum;
}

__global__ void naive_array_sum_parallel_reduction(double *input, double output; int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we are within bounds
    if (id < n / 2) {
        input[id] += input[id + n / 2]; // Sum elements pairwise
    }

    // The first thread of each block writes the final sum
    if (id == 0) {
        *output = input[0];  // Store final sum
    }

}

void init_vector(double *vector, int n) {
    for (int i=0; i < n; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }
}

void get_time() {
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    return timestamp.tv_sec + timestamp.tv_nsec * 1e-9;
}


int main() {
    double *host_vector, *host_result, *device_vector, *device_result;
    double cpu_result;

    size_t size = N * sizeof(double);

    host_a = (double*)malloc(size);
    host_c = (double*)malloc(sizeof(double));

    cudaMalloc(&device_a, size);
    cudaMalloc(&device_c, sizeof(double));

    init_vector(host_a, N);

    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
    
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double start_time, end_time;

    printf("Start Array Sum with Parallel Reduction...")
    start_time = gettime();
    naive_array_sum_parallel_reduction<<<num_blocks, BLOCK_SIZE>>>(device_a, device_c, N);
    end_time = gettime();
    printf("Parallel Reduction execution time: %d second", end_time - start_time);

    printf("Start Array Sum with CPU...")
    start_time = gettime();
    cpu_result = array_sum_cpu(host_a, N);
    end_time = gettime();
    printf("CPU execution time: %d second", end_time - start_time);

    cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

    // Verify Results
    printf("CPU Results: %d", cpu_sum);
    printf("GPU Results: %d", host_c);
    if (cpu_sum == host_c) {
        printf("Answer Matched");
    }

    // Free memory
    free(host_a);
    free(host_c);
    cudaFree(device_a);
    cudaFree(device_c);
}