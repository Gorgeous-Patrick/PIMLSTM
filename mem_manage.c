#include "mem_manage.h"
#include <assert.h>

Tensor_ptr create_tensor(size_t size) {
    static uint32_t arena = DPU_MRAM_HEAP_POINTER;
    Tensor_ptr ptr;
    ptr.wram_ptr = NULL;
    ptr.mram_ptr = arena;
    ptr.size = size;
    arena += size * sizeof(float);
    return ptr;
}

Tensor_ptr engage_tensor(Tensor_ptr tensor) {
    if (!tensor.wram_ptr) {
        tensor.wram_ptr = (float *) mem_alloc(tensor.size * sizeof(float));
    }
    // Copy the data from mram to wram
    mram_read((__mram_ptr void const *) tensor.mram_ptr , tensor.wram_ptr, tensor.size * sizeof(float));
    return tensor;
}

float * wram(Tensor_ptr tensor) {
    return tensor.wram_ptr;
}

Tensor_ptr disengage_tensor(Tensor_ptr tensor) {
    mram_write((__mram_ptr void const *) tensor.mram_ptr, tensor.wram_ptr, tensor.size * sizeof(float));
    return tensor;
}

