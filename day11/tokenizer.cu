#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include <ctime>
#include <cstring>

#define MAX_WORD_LENGTH 12   // "programming" is the longest word
#define MAX_SENTENCE_WORDS 10 // Maximum words expected in a sentence

// Like a Python Dictionary
std::unordered_map<std::string, int> h_vocab = {
    {"I", 0}, {"love", 1}, {"CUDA", 2}, {"programming", 3}, {"and", 4}, {"deep", 5}, {"learning", 6}
};

// CUDA Constant Memory 
__device__ __constant__ char d_vocab_words[7][MAX_WORD_LENGTH] = {"I", "love", "CUDA", "programming", "and", "deep", "learning"};
__device__ __constant__ int d_vocab_ids[7] = {0, 1, 2, 3, 4, 5, 6};

//CPU Tokenizer
//TODO:: need to change the output back to token_ids
void tokenize_cpu(const std::string &sentence, std::vector<std::string> &words) {
    std::istringstream iss(sentence);
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }
}

__device__ int strcmp_cuda(const char *str1, const char *str2) {
    while (*str1 && (*str1 == *str2)) {
        str1++;
        str2++;
    }
    return *(unsigned char *)str1 - *(unsigned char *)str2;
}

// GPU tokenizer
__global__ void tokenize_gpu(char d_sentence[MAX_SENTENCE_WORDS][MAX_WORD_LENGTH], int *token_ids, int word_count, int vocab_size) {
    int id = threadIdx.x;
    if (id >= word_count) return;

    // **Extract Word**
    char word[MAX_WORD_LENGTH];
    for (int i = 0; i < MAX_WORD_LENGTH; i++) {
        word[i] = d_sentence[id][i];
        if (d_sentence[id][i] == '\0') break;
    }

    // printf("GPU Extracted Word [%d]: %s\n", id, word);

    // Compare with Vocabulary
    int found = -1;
    for (int v = 0; v < vocab_size; v++) {
        if (strcmp_cuda(word, d_vocab_words[v]) == 0) {
            found = d_vocab_ids[v];
            break;
        }
    }

    // printf("GPU Word [%d] matched ID: %d\n", id, found);
    token_ids[id] = found;
}

double get_time() {
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    return timestamp.tv_sec + timestamp.tv_nsec * 1e-9;
}

int main() {
    std::string input = "I love CUDA programming and deep learning";
    std::vector<std::string> words;
    tokenize_cpu(input, words);

    // Get 2D Word Array
    char h_sentence[MAX_SENTENCE_WORDS][MAX_WORD_LENGTH] = {0};
    int word_count = words.size();
    
    for (int i = 0; i < word_count; i++) {
        strncpy(h_sentence[i], words[i].c_str(), MAX_WORD_LENGTH - 1);
        h_sentence[i][MAX_WORD_LENGTH - 1] = '\0';
    }

    // CPU Tokenizer
    printf("CPU Tokenized Output: ");
    for (const auto &w : words) printf("%s ", w.c_str());
    printf("\n");

    // Memory Allocation
    char (*d_sentence)[MAX_WORD_LENGTH];
    int *d_token_ids;

    cudaMalloc(&d_sentence, MAX_SENTENCE_WORDS * MAX_WORD_LENGTH * sizeof(char));
    cudaMalloc(&d_token_ids, word_count * sizeof(int));

    cudaMemcpy(d_sentence, h_sentence, MAX_SENTENCE_WORDS * MAX_WORD_LENGTH * sizeof(char), cudaMemcpyHostToDevice);

    printf("Start GPU Tokenizer...\n");
    start_time = get_time();
    tokenize_gpu<<<1, word_count>>>(d_sentence, d_token_ids, word_count, 7);
    cudaDeviceSynchronize();
    double gpu_time = get_time() - start_time;

    // Copy results back to host
    int *h_token_ids = (int *)malloc(word_count * sizeof(int));
    cudaMemcpy(h_token_ids, d_token_ids, word_count * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify GPU Output
    printf("GPU Tokenized Output: ");
    for (int i = 0; i < word_count; i++) printf("%d ", h_token_ids[i]);
    printf("\nGPU Execution Time: %f ms\n", gpu_time * 1000);

    // Free Memory
    free(h_token_ids);
    cudaFree(d_sentence);
    cudaFree(d_token_ids);

    return 0;
}