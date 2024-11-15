#include "lstm.h"
#include <assert.h>
#include <stdlib.h>

double exp(double x) {
    double a = 1.0, e = 0;
    int invert = x<0;
    x = fabs1(x);
    for (int n = 1; e != e + a ; ++n) {
        e += a;
        a = a * x / n;
    }
    return invert ? 1/e : e;
}


double tanh(double x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

void sigmoid_vec(LSTMVec input, LSTMVec output) {
    assert(input.length == output.length && "Length Mismatch");
    for (size_t i = 0; i < input.length; i++) {
        output.array[i] = 1.0 / (1.0 + exp(-input.array[i]));
    }
}

void tanh_vec(LSTMVec input, LSTMVec output) {
    assert(input.length == output.length && "Length Mismatch");
    for (size_t i = 0; i < input.length; i++) {
        output.array[i] = tanh(input.array[i]);
    }
}

void softmax_vec(LSTMVec input, LSTMVec output) {
    assert(input.length == output.length && "Length Mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < input.length; i++) {
        output.array[i] = exp(input.array[i]);
        sum += output.array[i];
    }
    for (size_t i = 0; i < input.length; i++) {
        output.array[i] /= sum;
    }
}

double dot_product_vec(LSTMVec input1, LSTMVec input2) {
    assert(input1.length == input2.length && "Length Mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < input1.length; i++) {
        sum += input1.array[i] * input2.array[i];
    }
    return sum;
}

void mat_vec_mul(LSTMMat mat, LSTMVec vec, LSTMVec output) {
    assert(mat.width == vec.length && "Width Mismatch");
    assert(mat.height == output.length && "Height Mismatch");
    for (size_t i = 0; i < mat.height; i++) {
        output.array[i] = 0.0;
        for (size_t j = 0; j < mat.width; j++) {
            output.array[i] += mat.array[i * mat.width + j] * vec.array[j];
        }
    }
}

void add_vec(LSTMVec *vec1, LSTMVec *vec2, LSTMVec *output) {
    for (size_t i = 0; i < vec1->length; i++) {
        output->array[i] = vec1->array[i] + vec2->array[i];
    }
}

// Forward pass function for the LSTM cell
void lstm_forward(LSTMCell *cell, LSTMMat Wf, LSTMMat Wi, LSTMMat Wo, LSTMMat Wc, LSTMVec h_prev, LSTMVec x_t, LSTMVec *h_t, LSTMVec *c_t) {
    // Temporary vectors for intermediate calculations
    LSTMVec f_t, i_t, o_t, c_tilde;
    f_t.length = i_t.length = o_t.length = c_tilde.length = cell->input_gate.length;

    // Allocate memory for each gate's result
    f_t.array = malloc(f_t.length * sizeof(double));
    i_t.array = malloc(i_t.length * sizeof(double));
    o_t.array = malloc(o_t.length * sizeof(double));
    c_tilde.array = malloc(c_tilde.length * sizeof(double));

    // Temporary storage for intermediate matrix-vector multiplication results
    LSTMVec h_mul_result = {h_prev.length, malloc(h_prev.length * sizeof(double))};
    LSTMVec x_mul_result = {x_t.length, malloc(x_t.length * sizeof(double))};

    // Forget gate f_t = sigmoid(Wf * h_prev + Wf * x_t)
    mat_vec_mul(Wf, h_prev, f_t);           // f_t = Wf * h_prev
    mat_vec_mul(Wf, x_t, x_mul_result);     // x_mul_result = Wf * x_t
    add_vec(&f_t, &x_mul_result, &f_t);     // f_t += x_mul_result
    sigmoid_vec(f_t, cell->forget_gate);

    // Input gate i_t = sigmoid(Wi * h_prev + Wi * x_t)
    mat_vec_mul(Wi, h_prev, i_t);           // i_t = Wi * h_prev
    mat_vec_mul(Wi, x_t, x_mul_result);     // x_mul_result = Wi * x_t
    add_vec(&i_t, &x_mul_result, &i_t);     // i_t += x_mul_result
    sigmoid_vec(i_t, cell->input_gate);

    // Candidate cell state c~_t = tanh(Wc * h_prev + Wc * x_t)
    mat_vec_mul(Wc, h_prev, c_tilde);       // c_tilde = Wc * h_prev
    mat_vec_mul(Wc, x_t, x_mul_result);     // x_mul_result = Wc * x_t
    add_vec(&c_tilde, &x_mul_result, &c_tilde); // c_tilde += x_mul_result
    tanh_vec(c_tilde, cell->cell_state);

    // Update cell state c_t = f_t * c_prev + i_t * c~_t
    for (size_t i = 0; i < c_t->length; i++) {
        c_t->array[i] = f_t.array[i] * c_t->array[i] + i_t.array[i] * c_tilde.array[i];
    }

    // Output gate o_t = sigmoid(Wo * h_prev + Wo * x_t)
    mat_vec_mul(Wo, h_prev, o_t);           // o_t = Wo * h_prev
    mat_vec_mul(Wo, x_t, x_mul_result);     // x_mul_result = Wo * x_t
    add_vec(&o_t, &x_mul_result, &o_t);     // o_t += x_mul_result
    sigmoid_vec(o_t, cell->output_gate);

    // Compute the new hidden state h_t = o_t * tanh(c_t)
    tanh_vec(*c_t, *h_t);                   // Apply tanh to cell state
    for (size_t i = 0; i < h_t->length; i++) {
        h_t->array[i] *= o_t.array[i];
    }

    // Free temporary vectors
    free(f_t.array);
    free(i_t.array);
    free(o_t.array);
    free(c_tilde.array);
    free(h_mul_result.array);
    free(x_mul_result.array);
}