#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <time.h>

__global__ void count_pairs(char *token, int *pair_counts, int token_index, int vocab_size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < token_index) {

    }
}

__global__ void merge_pairs(std::vector &tokens, std::unordered_map<int, int> &pair_counts, int &vocab_size) {

}

std::vector<int> tokenize(const std::string &text, std::unordered_map<char, int> &char_vocab) {
    std::vector tokens;

    for (char c : text) {
        if char_vocab.find(c) == char_vocab.end() {
            char_vocab[c] = char_vocab.size();
        }
        tokens.push_back(char_vocab[c]);
    }

    return tokens;
}