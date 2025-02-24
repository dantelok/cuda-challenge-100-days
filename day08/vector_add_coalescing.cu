#include <iostream>
#include <cuda_runtime.h>

#define N 10000
#define BLOCK_SIZE 256  // Threads per block


// Row-major operations are naturally coalesced
__global__ void vector_addition_coalesced(double *a, double *b, double *c, int n) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

// Introduce stride to make vector operations uncoalesced
__global__ void vector_addition_uncoalesced(double *a, double *b, double *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // id is counting from 0 to BLOCK_SIZE = 256
    
    int stride = 32;  // stride needs to be large enough to break coalescing
    // the L1/L2 cache can handle the difference if the stride is too small 

    // thread 0: 0 -> 32 -> 64 -> ...
    // thread 1: 1 -> 33 -> 65 -> ...
    // thread 2: 2 -> 34 -> 66 -> ...
    // ...
    // thread 255: 255 -> 287 -> 319 -> ...
    for (int i = id; i < n; i += stride) {  
        c[i] = a[i] + b[i];  
    }
}

// Try shared memory as well
__global__ void vector_addition_shared(double *a, double *b, double *c, int n) {
    __shared__ double shared_A[BLOCK_SIZE];
    __shared__ double shared_B[BLOCK_SIZE];

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n) {
        shared_A[threadIdx.x] = a[id];
        shared_B[threadIdx.x] = b[id];
    }
    // __syncthreads();  // Ensure all threads finish loading before proceeding

    if (id < n) {
        c[id] = shared_A[threadIdx.x] + shared_B[threadIdx.x];
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

int main() {
    double *host_A, *host_B, *host_C, *device_a, *device_b, *device_c;
    // Allocate the number of bytes
    size_t size = N * sizeof(double);

    // Allocate memory on host (CPU)
    host_A = (double*)malloc(size);
    host_B = (double*)malloc(size);
    host_C = (double*)malloc(size);

    // Allocate memory on device (GPU)
    cudaMalloc(&device_a, size);
    cudaMalloc(&device_b, size);
    cudaMalloc(&device_c, size);

    // Initialize vectors
    srand(time(NULL));
    init_vector(host_A, N);
    init_vector(host_B, N);

    // Copy data from host(CPU) to device(GPU)
    // ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
    // kind: host -> host; host -> device; device -> host; device -> device; 
    cudaMemcpy(device_a, host_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimension
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double start_time, end_time;

    // Coalesced Addition Timing
    start_time = get_time();
    vector_addition_coalesced<<<num_blocks, BLOCK_SIZE>>>(device_a, device_b, device_c, N);
    cudaDeviceSynchronize();
    end_time = get_time();
    printf("Coalesced Execution Time: %f ms\n", (end_time - start_time) * 1000);

    // Uncoalesced Addition Timing
    start_time = get_time();
    vector_addition_uncoalesced<<<num_blocks, BLOCK_SIZE>>>(device_a, device_b, device_c, N);
    cudaDeviceSynchronize();
    end_time = get_time();
    printf("Uncoalesced Execution Time: %f ms\n", (end_time - start_time) * 1000);

    // Shared Memory Optimization Timing
    start_time = get_time();
    vector_addition_shared<<<num_blocks, BLOCK_SIZE>>>(device_a, device_b, device_c, N);
    cudaDeviceSynchronize();
    end_time = get_time();
    printf("Shared Memory Execution Time: %f ms\n", (end_time - start_time) * 1000);

    // Copy results back to the host
    cudaMemcpy(host_C, device_c, size, cudaMemcpyDeviceToHost);

    // Free Memory
    free(host_A);
    free(host_B);
    free(host_C);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
}
