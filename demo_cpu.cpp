#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <vector>
#include <unordered_set>

#ifndef DPU_BINARY
#define DPU_BINARY "./dev"
#endif


std::vector<std::unordered_set<std::string>>> mapping;

void download_data(size_t dpu_id) {
  // Download data from the host to the DPU
  for (auto & page : mapping[dpu_id]) {
    // Download the page to the DPU
    download_page(dpu_id, page);
  }
  // Download LSTM to dpu
  download_lstm(dpu_id);

  // Download the walker to the DPU
  download_walker(dpu_id);
}

void start_running(size_t dpu_id, std::string title) {
  // Set one bit of the DPU's control register to 1
  // to indicate that the DPU is running
}

void initialize_mapping() {
  // Initialize the mapping data structure
}

Walker fetch_walker(size_t dpu_id) {
  // Fetch the walker from the mapping data structure
  return dpu_log_read(dpu_id, stdout)
}

void add_walker(Walker walker, std::string title) {

      // Find the DPU that can handle the request
      for (size_t i = 0; i < mapping.size(); i++) {
        if (mapping[i].find(new_request) != mapping[i].end()) {
          // Add the request to the DPU's queue
          break;
        }
      }
}

void event_loop() {
  // Event loop
  while (true) {
    string new_request = get_new_request();
    if (!new_request.empty()) {
      add_walker(new_request);
    }
    for (size_t dpu_id = 0; dpu_id < mapping.size(); dpu_id++) {
      Walker walker = fetch_walker(dpu_id);
      if (walker.is_done()) {
        // The walker is done, so stop running the DPU
        std::string next = walker.get_next();
        if (!next.empty()) {
          add_walker(next);
        }
        // Run the next walker in the queue
      } 
    }
  }
}

int main(void) {
  size_t dpu_num;
  struct dpu_set_t set, dpu;

  DPU_ASSERT(dpu_alloc(dpu_num, NULL, &set));
  // Load the DPU code to DPU
  DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));
  DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

  // Download data to DPU
  for (size_t dpu_id = 0; dpu_id < dpu_num; dpu_id++) {
    download_data(dpu_id);
  }

  // Start the event loop
  event_loop();


  DPU_FOREACH(set, dpu) {
    DPU_ASSERT(dpu_log_read(dpu, stdout));
  }

  DPU_ASSERT(dpu_free(set));

  return 0;
}