#include <stdio.h>
#include "lstm.h"

// Example tokenization and integer encoding for demonstration purposes.
int *tokenize_text(const char *text, int *length) {
    // Tokenize text and convert tokens to integers
    // You would replace this with actual tokenization and word embedding lookup
    *length = strlen(text);
    int * tokens = (int *) mem_alloc(*length * sizeof(int));
    for (int i = 0; i < length; i++) {
        tokens[i] = i;
    }
    return tokens;
}

void get_contextual_embedding(LSTMCell *cell, const char *text, double *embedding) {
    int sequence_length;
    int *token_sequence = tokenize_text(text, &sequence_length);
    printf("%d\n", sequence_length);
    // Iterate over each token and pass it through the LSTM
    for (int t = 0; t < sequence_length; t++) {
        printf("%d\n", token_sequence[t]);
    }

    // Initialize the hidden and cell state to zero for the first token
    double *h_t = mem_alloc(cell->hidden_dim * sizeof(double));
    double *c_t = mem_alloc(cell->hidden_dim * sizeof(double));
    double *h_prev = h_t;
    double *c_prev = c_t;

    // Iterate over each token and pass it through the LSTM
    for (int t = 0; t < sequence_length; t++) {
        

        // Update previous states for the next timestep
        for (int i = 0; i < cell->hidden_dim; i++) {
            h_prev[i] = h_t[i];
            c_prev[i] = c_t[i];
        }

        // free(x_t);  // Free token embedding memory if dynamically allocated
    }

    // Copy final hidden state as the contextual embedding
    for (int i = 0; i < cell->hidden_dim; i++) {
        embedding[i] = h_t[i];
    }

    // Clean up
    // free(h_t);
    // free(c_t);
}


int main() {
    LSTMCell cell = create_lstm_cell(100, 128); // Create an LSTM cell with input dim 100 and hidden dim 128
    double embedding[128]; // Allocate memory for the contextual embedding
    char text[] = "This is a sample sentence."; // Input text
    get_contextual_embedding(&cell, text, embedding); // Get the contextual embedding for the input text
    // printf("Contextual embedding: [");
    // for (int i = 0; i < 128; i++) {
    //     printf("%lf, ", embedding[i]);
    // }
    // printf("]\n");
    mem_reset();
    // printf("%lf", exp(1.0));
    return 0;
}