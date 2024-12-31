#ifndef MEM_MANAGE_H
#define MEM_MANAGE_H
#include <mram.h>

#include <stddef.h>
#include <stdint.h>

typedef struct _Tensor_ptr {
    uint32_t mram_ptr;
    float * wram_ptr;
    size_t size;
} Tensor_ptr;

Tensor_ptr create_tensor(size_t);
Tensor_ptr engage_tensor(Tensor_ptr);
float * wram(Tensor_ptr);
Tensor_ptr disengage_tensor(Tensor_ptr);

#endif
