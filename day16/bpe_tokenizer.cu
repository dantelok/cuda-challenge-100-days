#include <stdio.h>
#include <map>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define MAX_PAIR_SIZE 50000
#define MAX_VOCAB_SIZE 10000

// text -> (tokenized) -> token -> count tokens
__global__ void count_pairs_gpu(int *tokens, int *pair_counts, int num_tokens, int vocab_size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < num_tokens - 1) {
        // Here
        int pair_id = tokens[id] * vocab_size + tokens[id + 1];
        atomicAdd(&pair_counts[pair_id], 1);
    }
}

// Host function for BPE merge step
void merge_most_frequent_pair(std::vector<int> &tokens, std::unordered_map<int, int> &pair_counts, int &vocab_size) {
    if (pair_counts.empty()) return;
    
    auto max_pair = std::max_element(pair_counts.begin(), pair_counts.end(), 
        [](const auto &a, const auto &b) { return a.second < b.second; });
    
    int pair_to_merge = max_pair->first;
    int token_a = pair_to_merge / vocab_size;
    int token_b = pair_to_merge % vocab_size;
    int new_token = vocab_size++;
    
    std::vector<int> new_tokens;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i < tokens.size() - 1 && tokens[i] == token_a && tokens[i + 1] == token_b) {
            new_tokens.push_back(new_token);
            i++; 
        } else {
            new_tokens.push_back(tokens[i]);
        }
    }
    tokens = new_tokens;
    pair_counts.clear();
}

std::vector<int> tokenize(const std::string &text, std::unordered_map<char, int> &char_vocab) {
    std::vector<int> tokens;
    for (char c : text) {
        // What is it doing here?
        if (char_vocab.find(c) == char_vocab.end()) {
            char_vocab[c] = char_vocab.size();
        }
        tokens.push_back(char_vocab[c]);
    }
    return tokens;
}

int main() {
    std::string text ="Hello Hello World World World";
    std::unordered_map<char, int> char_vocab;
    std::vector<int> tokens = tokenize(text, char_vocab);
    int vocab_size = char_vocab.size();

    // Get the length of array of tokens (characters?)
    int num_tokens = tokens.size();
    int *device_tokens, *device_pair_counts;

    cudaMalloc(&device_tokens, num_tokens * sizeof(int));
    cudaMalloc(&device_pair_counts, MAX_PAIR_SIZE * sizeof(int));
    cudaMemcpy(device_tokens, tokens.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(device_pair_counts, 0, MAX_PAIR_SIZE * sizeof(int));

    int num_blocks = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_pairs_gpu<<<num_blocks, BLOCK_SIZE>>>(device_tokens, device_pair_counts, num_tokens, vocab_size);
    cudaDeviceSynchronize();

    std::vector<int> pair_counts(MAX_PAIR_SIZE);
    cudaMemcpy(pair_counts.data(), device_pair_counts, MAX_PAIR_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    std::unordered_map<int, int> pair_count_map;
    for (int i=0; i < MAX_PAIR_SIZE; i++) {
        if (pair_counts[i] > 0) {
            pair_count_map[i] = pair_counts[i];
        }
    }

    printf("Character to Token Mapping:\n");
    for (const auto &pair : char_vocab) {
        printf("Char: '%c' -> Token ID: %d\n", pair.first, pair.second);
    }

    printf("Tokens before pairing: ");
    for (int t : tokens) 
        printf("%d ", t);
    printf("\n");


    merge_most_frequent_pair(tokens, pair_count_map, vocab_size);

    printf("Final Tokens: ");
    for (int t : tokens) 
        printf("%d ", t);
    printf("\n");

    // Free Memory
    cudaFree(device_tokens);
    cudaFree(device_pair_counts);
    

}
