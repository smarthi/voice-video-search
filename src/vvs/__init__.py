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

# Historically needed because faiss-cpu and torch each bundled their own copy
# of the LLVM OpenMP runtime on macOS arm64, and whichever initialized second
# made OpenMP abort the process. FAISS is gone now (search runs server-side in
# OpenSearch; see retrieve/opensearch_store.py), so that specific collision is
# retired. Left as a harmless setdefault rather than removed: onnxruntime
# (via rapidocr) is still a native library in this process, and this is the
# documented escape hatch if it or anything else ever bundles a duplicate
# runtime. Costs nothing when there is no duplicate to allow.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Native crashes in torch, onnxruntime or the bundled FFmpeg kill the
# process with a signal, so Python raises nothing and the only trace left is the
# resource tracker's "leaked semaphore objects" note at shutdown -- which names
# neither the cause nor the library. faulthandler turns that silence into a C
# traceback on stderr. It only installs signal handlers, so it costs nothing
# until something actually crashes. Set VVS_FAULTHANDLER=0 to opt out.
if (os.environ.get("VVS_FAULTHANDLER") or "1") != "0":
    import faulthandler

    faulthandler.enable()

__version__ = "0.1.0"
