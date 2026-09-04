"""Voice-queried video frame retrieval. Runs on Metal, CUDA, or CPU."""

import os

# Must be set before anything imports torch: PyTorch reads this when the MPS
# backend initializes, and setting it later has no effect. Without it, a VL op
# with no Metal kernel aborts the process (a hard crash, usually reported as
# "leaked semaphore objects") instead of quietly running that op on CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"
