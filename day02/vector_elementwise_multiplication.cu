#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000
#define BLOCK_SIZE 256
#define EPSILON 1e-6

// CPU matrix multiplication
void vector_elementwise_multiplication_cpu(double *a, double *b, double *c, int n) {
    for (int i = 0; i < n; i++){
        c[i] = a[i] * b[i];
    }
}

__global__ void vector_elementwise_multiplication_gpu(double *a, double *b, double *c, int n) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
        c[id] = a[id] * b[id];
    }
}

void init_vector(double *vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = (float)rand() / (float)RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


int main(){
    double *host_A, *host_B, *host_C_cpu, *host_C_gpu, *device_a, *device_b, *device_c;

    // Allocate the number of bytes
    size_t size = N * sizeof(double);

    // Allocate memory on host (CPU)
    host_A = (double*)malloc(size);
    host_B = (double*)malloc(size);
    host_C_cpu = (double*)malloc(size);
    host_C_gpu = (double*)malloc(size);

    // Allocate memory on device (GPU)
    cudaMalloc(&device_a, size);
    cudaMalloc(&device_b, size);
    cudaMalloc(&device_c, size);

    // Initialize vectors
    init_vector(host_A, N);
    init_vector(host_B, N);

    // Copy data from host(CPU) to device(GPU)
    // ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
    // kind: host -> host; host -> device; device -> host; device -> device; 
    cudaMemcpy(device_a, host_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimension
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warm up run
    for (int i = 0; i < 5; i++) {
        vector_elementwise_multiplication_cpu(host_A, host_B, host_C_cpu, N);
        vector_elementwise_multiplication_gpu<<<num_blocks, BLOCK_SIZE>>>(device_a, device_b, device_c, N);
        cudaDeviceSynchronize();
    }

    // Benchmark
    printf("Implementing CPU runs...\n");
    double time_used_cpu = 0.0;
    for (int i = 0; i < 100; i++) {
        double start_time = get_time();
        vector_elementwise_multiplication_cpu(host_A, host_B, host_C_cpu, N);
        double end_time = get_time();
        time_used_cpu += end_time - start_time;
    }

    printf("Implementing GPU runs...\n");
    double time_used_gpu = 0.0;
    for (int i = 0; i < 100; i++) {
        double start_time = get_time();
        vector_elementwise_multiplication_gpu<<<num_blocks, BLOCK_SIZE>>>(device_a, device_b, device_c, N);
        cudaDeviceSynchronize(); // Ensure the kernel finishes execution before timing
        double end_time = get_time();
        time_used_gpu += end_time - start_time;
    }

    // Print time results
    printf("CPU running time: %f microseconds\n", (time_used_cpu * 1e6f));
    printf("GPU running time: %f microseconds\n", (time_used_gpu * 1e6f));

    // Copy results back to the host
    cudaMemcpy(host_C_gpu, device_c, size, cudaMemcpyDeviceToHost);

    // Verify results
    int match = 1;
    for (int i = 0; i < N; i++) {
        printf("CPU: c[%d] = %f; GPU: c[%d] = %f\n", i, host_C_cpu[i], i, host_C_gpu[i]);
        if (host_C_cpu[i] - host_C_gpu[i] > EPSILON) {
            match = 0;
        }
    }

    // Verify results
    if (match) {
        printf("The results are the same!\n");
    } else {
        printf("Differences found!\n");
    }

    // Free Memory
    free(host_A);
    free(host_B);
    free(host_C_cpu);
    free(host_C_gpu);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
}