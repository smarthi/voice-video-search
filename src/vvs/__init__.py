"""Voice-queried video frame retrieval. Runs on Metal, CUDA, or CPU."""

import os

# Both of the following must be set before anything imports torch or faiss.
# Python runs this package __init__ before any submodule, which is the only
# reliable place for them.

# PyTorch reads this when the MPS backend initializes; setting it later has no
# effect. Without it, a VL op with no Metal kernel aborts the process (a hard
# crash, usually reported as "leaked semaphore objects") instead of quietly
# running that op on CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# faiss-cpu and torch each bundle their own copy of the LLVM OpenMP runtime
# (faiss/.dylibs/libomp.dylib and torch/lib/libomp.dylib on macOS arm64).
# Whichever initializes second makes OpenMP abort the process, which is why
# FAISS search — an OpenMP-parallel path — could die while single-threaded
# index writes survived. Allowing the duplicate is the upstream-documented
# escape hatch and is what every torch+faiss stack on macOS relies on. It is a
# mitigation, not a cure: the cure is one OpenMP runtime in the process, which
# needs coordinated wheel builds we do not control. Harmless where there is only
# one runtime, as on the Linux manylinux wheels.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
