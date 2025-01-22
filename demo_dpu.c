#include <stdio.h>
#include <stdlib.h>
#include <alloc.h>
#include <string.h>
#include <assert.h>

int main() {
    mem_reset();
    Walker walker;

    while (true) {
      // 1. Spin to wait for the start register to be set
      if (walker.page is not on current PE) {
        while (dpu_get_control_register() == 0) {}
        walker = read_walker_from_mram();
      }

      // 2. Read the input data from MRAM
      LSTM model = read_lstm_from_mram();
      Page page = read_page_from_mram();

      // 3. Process the input data
      float * embedding = process_page(model, page);
      
      // 4. Process the Walker
      walker.accumulated += embedding;

      // 5. Write the Walker back to MRAM
      write_walker_to_mram(walker);

      // 6. Tell the CPU that the DPU is done, and upload the next page id
      if (walker.page is not on current PE) {
        dpu_set_control_register(0);
        write_page_id_to_mram(walker.next);
      }
    }
}
