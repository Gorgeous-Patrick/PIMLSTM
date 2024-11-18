#ifndef LSTM_H
#define LSTM_H
#include <stddef.h>
typedef struct _LSTMMat {
    size_t width, height;
    double * array;
} LSTMMat;

typedef struct _LSTMVec {
    size_t length;
    double * array; 
} LSTMVec;

typedef struct _LSTMCell {
    int hidden_dim, input_dim;
    LSTMVec input_gate, forget_gate, output_gate, cell_state;
} LSTMCell;

void sigmoid_vec(LSTMVec input, LSTMVec output);
void tanh_vec(LSTMVec input, LSTMVec output);
void softmax_vec(LSTMVec input, LSTMVec output);
double dot_product_vec(LSTMVec input1, LSTMVec input2);
void mat_vec_mul(LSTMMat mat, LSTMVec vec, LSTMVec output);
void add_vec(LSTMVec *vec1, LSTMVec *vec2, LSTMVec *output);
LSTMCell create_lstm_cell(size_t input_dim, size_t hidden_dim);
void free_lstm_cell(LSTMCell *cell);

#endif
