#ifndef TENSOR_H
#define TENSOR_H
#include <stdint.h>
#define CHUNK_SIZE 32
typedef struct _Tensor {
    uint64_t width, height;
    uint32_t mram;
} Tensor_ptr;

Tensor_ptr create_tensor(uint64_t, uint64_t);
Tensor_ptr gemv(Tensor_ptr, double * vector)
#endif
