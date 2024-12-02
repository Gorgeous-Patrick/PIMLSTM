#include <stdio.h>
#include <stdlib.h>
#include <alloc.h>

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

// LSTM forward pass
void lstm_forward(float *input, float *prev_hidden, float *prev_cell, 
                  float *weights, float *biases, int input_size, int hidden_size,
                  float *output_hidden, float *output_cell, float *workspace) {
    // Workspace memory allocation
    float *gates = workspace; // Store [input gate, forget gate, cell gate, output gate] concatenated
    float *new_cell = gates + 4 * hidden_size;

    // Compute gates: input, forget, cell, output
    for (int i = 0; i < 4 * hidden_size; i++) {
        gates[i] = biases[i];
        for (int j = 0; j < input_size; j++) {
            gates[i] += input[j] * weights[i * input_size + j];
        }
        for (int j = 0; j < hidden_size; j++) {
            gates[i] += prev_hidden[j] * weights[4 * hidden_size * input_size + i * hidden_size + j];
        }
    }

    // Apply activations to gates
    for (int i = 0; i < hidden_size; i++) {
        float input_gate = sigmoid(gates[i]);
        float forget_gate = sigmoid(gates[hidden_size + i]);
        float cell_gate = tanh_approx(gates[2 * hidden_size + i]);
        float output_gate = sigmoid(gates[3 * hidden_size + i]);

        // Update cell state
        new_cell[i] = forget_gate * prev_cell[i] + input_gate * cell_gate;

        // Compute hidden state
        output_hidden[i] = output_gate * tanh_approx(new_cell[i]);
    }

    // Manually copy updated cell state to output_cell
    array_copy(output_cell, new_cell, hidden_size);
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
    const int hidden_size = 50;  // Embedding size

    // Text input
    const char *text = "hello, my name is Patrick Li. I am a student from Umich.";
    int text_length = strlen(text);

    // Allocate memory
    float *memory = (float *)mem_alloc((8 * hidden_size + hidden_size) * sizeof(float));
    if (!memory) {
        // perror("Failed to allocate memory");
        
        return -1;
    }

    // Assign parts of memory
    float *prev_hidden = memory;                       // Hidden state at t-1
    float *prev_cell = prev_hidden + hidden_size;      // Cell state at t-1
    float *workspace = prev_cell + hidden_size;        // Workspace for gates and new cell state

    // Initialize weights, biases, and states
    float weights[4 * hidden_size * (input_size + hidden_size)];
    float biases[4 * hidden_size];
    float input[input_size];
    float output_hidden[hidden_size];
    float output_cell[hidden_size];

    // Fill weights and biases with dummy values
    for (int i = 0; i < 4 * hidden_size * (input_size + hidden_size); i++) {
        weights[i] = 0.1f; // Example value
    }
    for (int i = 0; i < 4 * hidden_size; i++) {
        biases[i] = 0.1f; // Example value
    }

    // Initialize previous hidden and cell states
    for (int i = 0; i < hidden_size; i++) {
        prev_hidden[i] = 0.0f;
        prev_cell[i] = 0.0f;
    }

    // Process each character in the text
    for (int t = 0; t < text_length; t++) {
        // Reset input vector
        for (int i = 0; i < input_size; i++) input[i] = 0.0f;

        // One-hot encode the current character
        int char_index = get_vocab_index(text[t]);
        if (char_index < input_size) input[char_index] = 1.0f;

        // Perform forward pass
        lstm_forward(input, prev_hidden, prev_cell, weights, biases, input_size, hidden_size, prev_hidden, prev_cell, workspace);
    }

    // Use the final hidden state as the embedding
    printf("Text embedding:\n");
    for (int i = 0; i < hidden_size; i++) {
        printf("%.4f ", prev_hidden[i]);
    }
    printf("\n");

    // Free allocated memory
    // free(memory);

    return 0;
}

