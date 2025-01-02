#include "mem_manage.h"
#include <mram.h>
#include <stdint.h>
#include <stdio.h>

// Static pointer to keep track of the current allocation position in MRAM
static uint32_t current_mram_addr = DPU_MRAM_HEAP_POINTER;

// Initialize a tensor in MRAM
Tensor_ptr tensor_init(size_t size) {
    Tensor_ptr tensor;
    tensor.mram_addr = current_mram_addr;
    tensor.size = size;

    // Increment the static MRAM pointer
    current_mram_addr += size;

    // Check for MRAM overflow
    if ((uint32_t)current_mram_addr >= (uint32_t)DPU_MRAM_HEAP_POINTER + 64 * 1024) { // Assuming 64 KB MRAM per DPU
        printf("Error: Not enough MRAM space for tensor allocation.\n");
        tensor.mram_addr = NULL; // Mark as invalid
        tensor.size = 0;
    }

    return tensor;
}

// Load data from MRAM to WRAM
void tensor_load(Tensor_ptr tensor, float *wram_buffer, size_t size) {
    if (size > tensor.size) {
        printf("Error: Requested size exceeds tensor size in MRAM.\n");
        return;
    }

    // Use mram_read to move data from MRAM to WRAM
    mram_read((__mram_ptr void const *)tensor.mram_addr, wram_buffer, size * sizeof(float));
}

// Store data from WRAM to MRAM
void tensor_store(Tensor_ptr tensor, const float *wram_buffer, size_t size) {
    if (size > tensor.size) {
        printf("Error: Data size exceeds tensor size in MRAM.\n");
        return;
    }

    // Use mram_write to move data from WRAM to MRAM
    mram_write(wram_buffer, (__mram_ptr void const *)tensor.mram_addr, size * sizeof(float));
}


