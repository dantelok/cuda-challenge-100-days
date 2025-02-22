#include <stdio.h>
#include <cuda_runtime.h>

#define N 10000
#define BLOCK_SIZE 256


__global__ void dot_product_shared_memory_reduction(double *a, double *b, double *partial_sums, int n) {
    __shared__ double partial_sum[BLOCK_SIZE];

    // id for global memory, containing multiple blocks
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // Thread index for shared memory, only 1 block
    int tid = threadIdx.x;

    if (id < n) {
        partial_sum[tid] = a[id] * b[id];
    } else {
        partial_sum[tid] = 0.0;
    }
    __syncthreads(); // ensures that all threads reach the same point before continuing
    // Used in shared memory reduction, ensuring correct execution order

    // Reduction in shared memory (tree-based reduction)
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads();
    }

    // Write the results back to global memory from shared memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = partial_sum[0];
    }
}

void init_vector(double *vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }
}

int main() {
    double *host_a, *host_b, *host_partial_sum;
    double *device_a, *device_b, *device_partial_sum;

    // Get num_blocks
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Get size
    size_t size = N * sizeof(double);

    // Allocate memory on host
    host_a = (double*)malloc(size);
    host_b = (double*)malloc(size);
    host_partial_sum = (double*)malloc(num_blocks * sizeof(double));

    // Allocate memory on device
    cudaMalloc(&device_a, size); 
    cudaMalloc(&device_b, size); 
    cudaMalloc(&device_partial_sum, num_blocks * sizeof(double)); 

    // Init vector
    init_vector(host_a, N);
    init_vector(host_b, N);

    // Copy from host to device
    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

    // Run the kernel
    dot_product_shared_memory_reduction<<<num_blocks, BLOCK_SIZE>>>(device_a, device_b, device_partial_sum, N);
    cudaDeviceSynchronize();

    // Copy results
    cudaMemcpy(host_partial_sum, device_partial_sum, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);

    // Sum the reduction up in host
    double total_sum = 0.0;
    for (int i = 0; i < num_blocks; i ++) {
        total_sum += host_partial_sum[i];
    }

    // Verify Results
    printf("Total Sum = %f\n", total_sum);

    // Free Memory
    free(host_a);
    free(host_b);
    free(host_partial_sum);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_partial_sum);
    
    return 0;
}
