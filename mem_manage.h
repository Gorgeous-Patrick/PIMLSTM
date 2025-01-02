#ifndef MEM_MANAGE_H
#define MEM_MANAGE_H

#include <stddef.h>
#include <stdint.h>

// Tensor pointer structure
typedef struct {
    uint32_t *mram_addr; // Pointer to the tensor in MRAM
    size_t size;                 // Size of the tensor in elements (not bytes)
} Tensor_ptr;

// Initialize a tensor in MRAM
Tensor_ptr tensor_init(size_t size);

// Load data from MRAM to WRAM
void tensor_load(Tensor_ptr tensor, float *wram_buffer, size_t size);

// Store data from WRAM to MRAM
void tensor_store(Tensor_ptr tensor, const float *wram_buffer, size_t size);



#endif
