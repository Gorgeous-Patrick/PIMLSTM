node __pim__ Page {
  __shared__ Model lstm = LSTM();
  text: string = "Wikipedia Page Text for Jac";
  title: string = "Jac";
}

edge __pim__ HyperLink {
  rank: float;
  title: str;
}

walker __pim__ WikiWalker on Page { # <-- Running on DPU, All data in Page is loaded on to DPU
  has accumulated: torch.tensor;
  can calculate with Page entry {
    text = here.text;
    # Use the LSTM to calculate the text
    self.accumulated += calculate(self.accumulated, text, 0.5);
    possible = [-:HyperLink:->];
    next = next_page(possible);
    if next is not None {
      visit next;
    }
  }
}


with entry { # <-- Running on CPU
  # 1. Construct the network.
  pages = construct_nodes();
  edges = construct_edges();
  scheduler = Scheduler(pages, edges, dpus);
  dpus = allocate_dpus(scheduler = scheduler);
  while (true) {
    # 2. Initialize the LSTM.
    walker = WikiWalker();
    walker.accumulated = lstm;
    walker.seq = [];

    # 3. Start the walk.
    request = get_request();
    dpus.run(walker, request);
  }
}