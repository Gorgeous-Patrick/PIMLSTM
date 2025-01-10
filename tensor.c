#include "tensor.h"
#include <mram.h>

static uint32_t mram_heap_ptr = DPU_MRAM_HEAP_POINTER;

Tensor_ptr create_tensor(uint64_t width, uint64_t height) {
    Tensor_ptr tp;
    tp.width = width;
    tp.height = height;
    tp.mram = mram_heap_ptr;
    mram_heap_ptr += width * height * sizeof(double);
    for (int i = 0; i < width * height; i++) {
        chunk[i] = 0;
    }
    

}
