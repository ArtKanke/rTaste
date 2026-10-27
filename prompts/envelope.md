Execution environment (the same for every experiment):
- One machine with an A100 GPU (80 GB), 12 CPU cores and 142 GB of memory, for at most 45 minutes from the engineer's first command.
- At most 90 minutes for the implementation in all. The final evaluation runs the whole experiment again on a fresh machine and must finish within 35 minutes, installation and model downloads included.
- Open models sold as 8B or smaller (for example Qwen3-8B or Llama-3.1-8B), run on the GPU. Public datasets and code can be downloaded.
- No hosted model APIs, no human annotation, no web browsing.
- The engineer model that writes the code has at most 60 model calls of its own and $$6 for them. This limits the engineer's work, not your experiment: the experiment may run as many prompts through the open models as fit in the time.

A separate engineer model implements each experiment from the seed paper and the registration. It does not see the proposer's reasoning or the other experiments, so every registration must stand on its own.
