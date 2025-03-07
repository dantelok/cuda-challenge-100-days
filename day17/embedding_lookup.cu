#include <stdio.h>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define EMBED_DIM 3

// CUDA Kernels doesn't support string
__global__ void embedding_lookup(int *input_ids, double *embeddings, double *output, int vocab_size, int embed_dim) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < vocab_size) {
        int word_id = input_ids[id];
        for (int i = 0; i < embed_dim; i++) {
            // Simply copy the value of embeddings to GPU
            output[id * embed_dim + i] = embeddings[word_id * embed_dim + i];
        }
    }
}

// Assign 3-dimension embeddings
void assign_embeddings(double *embeddings, int vocab_size, int embedding_dimension) {

    for (int i=0; i < vocab_size; i++) {
        for (int j=0; j < embedding_dimension; j++) {
            embeddings[i * embedding_dimension + j] = (double)rand() / RAND_MAX;
        }
    }
}

int main() {
    std::vector<std::string> vocabs = {"cat", "dog", "mouse"};
    std::vector<int> input_ids = {0, 1, 2};

    int vocab_size = vocabs.size();
    int embedding_dimension = EMBED_DIM;

    // Allocate and assign embeddings on host
    double *host_embeddings = new double[vocab_size * embedding_dimension];
    assign_embeddings(host_embeddings, vocab_size, embedding_dimension);

    double *device_embeddings, *device_output;
    int *device_input_ids;

    cudaMalloc(&device_embeddings, vocab_size * embedding_dimension * sizeof(double));
    cudaMalloc(&device_output, vocab_size * embedding_dimension * sizeof(double));
    cudaMalloc(&device_input_ids, vocab_size * sizeof(int));

    cudaMemcpy(device_embeddings, host_embeddings, vocab_size * embedding_dimension * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_input_ids, input_ids.data(), vocab_size * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = 1;
    int block_size = vocab_size;

    embedding_lookup<<<num_blocks, block_size>>>(device_input_ids, device_embeddings, device_output, vocab_size, EMBED_DIM);
    cudaDeviceSynchronize();

    double *host_output = new double[vocab_size * embedding_dimension];
    cudaMemcpy(host_output, device_output, vocab_size * embedding_dimension * sizeof(double), cudaMemcpyDeviceToHost);

    // Print embeddings
    printf("GPU Retrieved Embeddings:\n");
    for (int i = 0; i < vocab_size; i++) {
        printf("%s: ", vocabs[i].c_str());
        for (int j = 0; j < embedding_dimension; j++) {
            printf("%.4f ", host_output[i * embedding_dimension + j]);
        }
        printf("\n");
    }

    // Free Memory
    delete[] host_embeddings;
    delete[] host_output;
    cudaFree(device_embeddings);
    cudaFree(device_output);
    cudaFree(device_input_ids);
}