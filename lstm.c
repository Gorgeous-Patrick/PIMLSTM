#include <stdio.h>
#include <stdlib.h>
#include <alloc.h>
#include <string.h>
#include "mem_manage.h"

// Approximation of exp(x)
float exp_approx(float x) {
    float result = 1.0f + x + (x * x) / 2.0f + (x * x * x) / 6.0f;
    return result > 100.0f ? 100.0f : result; // Prevent overflow
}

// Approximation of sigmoid(x) = 1 / (1 + exp(-x))
float sigmoid(float x) {
    return 1.0f / (1.0f + exp_approx(-x));
}

// Approximation of tanh(x)
float tanh_approx(float x) {
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// Manual array copy function
void array_copy(float *dest, const float *src, int size) {
    for (int i = 0; i < size; i++) {
        dest[i] = src[i];
    }
}

// LSTM forward pass with MRAM-accelerated memory management
void lstm_forward(float *input, Tensor_ptr prev_hidden, Tensor_ptr prev_cell, 
                  Tensor_ptr weights, Tensor_ptr biases, int input_size, int hidden_size,
                  Tensor_ptr output_hidden, Tensor_ptr output_cell, float *wram_buffer) {
    // Workspace memory allocation in WRAM
    float *gates = wram_buffer; // Store [input gate, forget gate, cell gate, output gate] concatenated
    float *new_cell = gates + 4 * hidden_size;

    // Load weights and biases in chunks dynamically
    const int chunk_size = 512; // Size of WRAM chunks to process
    float chunk_buffer[chunk_size];

    // Process weights and biases dynamically without large WRAM allocations
    for (int i = 0; i < 4 * hidden_size; i++) {
        gates[i] = 0.0f; // Initialize gate with bias

        // Load corresponding chunk of weights and biases
        for (int j = 0; j < input_size; j += chunk_size) {
            int load_size = (j + chunk_size > input_size) ? (input_size - j) : chunk_size;
            tensor_load(weights, chunk_buffer, load_size);

            // Compute gate contributions
            for (int k = 0; k < load_size; k++) {
                gates[i] += input[j + k] * chunk_buffer[k];
            }
        }

        for (int j = 0; j < hidden_size; j += chunk_size) {
            int load_size = (j + chunk_size > hidden_size) ? (hidden_size - j) : chunk_size;
            tensor_load(weights, chunk_buffer, load_size);

            // Compute gate contributions
            for (int k = 0; k < load_size; k++) {
                gates[i] += wram_buffer[4 * hidden_size + j + k] * chunk_buffer[k];
            }
        }
    }

    // Apply activations to gates
    for (int i = 0; i < hidden_size; i++) {
        float input_gate = sigmoid(gates[i]);
        float forget_gate = sigmoid(gates[hidden_size + i]);
        float cell_gate = tanh_approx(gates[2 * hidden_size + i]);
        float output_gate = sigmoid(gates[3 * hidden_size + i]);

        // Update cell state
        new_cell[i] = forget_gate * new_cell[i] + input_gate * cell_gate;

        // Compute hidden state
        gates[4 * hidden_size + i] = output_gate * tanh_approx(new_cell[i]);
    }

    // Store updated states back to MRAM
    tensor_store(output_cell, new_cell, hidden_size);
    tensor_store(output_hidden, gates + 4 * hidden_size, hidden_size);
}

// Vocabulary-to-index mapping
int get_vocab_index(char c) {
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c >= 'A' && c <= 'Z') return c - 'A';
    return 26; // Unknown token
}

int main() {
    // LSTM parameters
    const int input_size = 27;  // 26 letters + 1 unknown
    const int hidden_size = 1000;  // Embedding size

    // Text input
    const char *text = "hello, my name is Patrick Li. I am a student from Umich.";
    int text_length = strlen(text);

    // WRAM buffer
    float *wram_buffer = (float *)mem_alloc((8 * hidden_size + hidden_size) * sizeof(float));
    if (!wram_buffer) {
        // perror("Failed to allocate WRAM buffer");
        return -1;
    }

    // Allocate MRAM tensors
    Tensor_ptr weights = tensor_init(4 * hidden_size * (input_size + hidden_size));
    Tensor_ptr biases = tensor_init(4 * hidden_size);
    Tensor_ptr prev_hidden = tensor_init(hidden_size);
    Tensor_ptr prev_cell = tensor_init(hidden_size);
    Tensor_ptr output_hidden = tensor_init(hidden_size);
    Tensor_ptr output_cell = tensor_init(hidden_size);

    // Fill weights and biases with dummy values (in MRAM)
    float temp_wb[4 * hidden_size * (input_size + hidden_size)];
    for (int i = 0; i < 4 * hidden_size * (input_size + hidden_size); i++) {
        temp_wb[i] = 0.1f; // Example value
    }
    tensor_store(weights, temp_wb, 4 * hidden_size * (input_size + hidden_size));

    for (int i = 0; i < 4 * hidden_size; i++) {
        temp_wb[i] = 0.1f; // Example value
    }
    tensor_store(biases, temp_wb, 4 * hidden_size);

    // Initialize previous hidden and cell states (in MRAM)
    for (int i = 0; i < hidden_size; i++) {
        temp_wb[i] = 0.0f;
    }
    tensor_store(prev_hidden, temp_wb, hidden_size);
    tensor_store(prev_cell, temp_wb, hidden_size);

    // Process each character in the text
    for (int t = 0; t < text_length; t++) {
        // Reset input vector
        float input[input_size];
        for (int i = 0; i < input_size; i++) input[i] = 0.0f;

        // One-hot encode the current character
        int char_index = get_vocab_index(text[t]);
        if (char_index < input_size) input[char_index] = 1.0f;

        // Perform forward pass
        lstm_forward(input, prev_hidden, prev_cell, weights, biases, input_size, hidden_size, output_hidden, output_cell, wram_buffer);
    }

    // Load the final hidden state from MRAM to WRAM and print it
    float final_hidden[hidden_size];
    tensor_load(output_hidden, final_hidden, hidden_size);

    printf("Text embedding:\n");
    for (int i = 0; i < hidden_size; i++) {
        printf("%.4f ", final_hidden[i]);
    }
    printf("\n");

    // Free WRAM buffer
    // free(wram_buffer);

    return 0;
}
