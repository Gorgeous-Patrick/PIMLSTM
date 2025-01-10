#include <stdio.h>
#include <stdlib.h>
#include <alloc.h>
#include "tensor.h"
double __mram_noinit forget_weight[HIDDEN_SIZE * CHUNK_SIZE];
double __mram_noinit forget_bias[CHUNK_SIZE];

int main() {
    mem_reset();
    Tensor_ptr forget_weight_t = create_tensor(forget_weight, CHUNK_SIZE, HIDDEN_SIZE);
    Vec forget_bias_v = create_vec(forget_bias, CHUNK_SIZE);
    return 0;
}

