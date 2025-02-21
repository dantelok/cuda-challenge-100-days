#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dot_product_shared_memory_reduction(int n) {
    __shared__ double a, b, c;

    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
        c[id] += a[id] * b[id];
    }
}