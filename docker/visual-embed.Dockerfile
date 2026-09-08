# Visual-embed service: Qwen3-VL-Embedding-2B behind FastAPI.
#
# Linux only by construction -- pyproject.toml's [tool.uv.sources] gives Linux
# the CUDA torch build (cu130) and everything else the plain one, and a
# container is always Linux at runtime regardless of the host's own OS. Metal
# does not pass through Docker, so this image is CPU or CUDA, never MPS; run
# the service directly on macOS (uv run --extra services uvicorn ...) to use
# Apple Silicon's GPU. For CUDA, run with `docker run --gpus all ...` and the
# NVIDIA Container Toolkit installed on the host -- the image does not bundle
# a CUDA toolkit, only what the torch wheel itself ships.
#
# Build:  docker build -f docker/visual-embed.Dockerfile -t vvs-visual-embed .
# Run:    docker run --gpus all -p 8001:8001 \
#             -v vvs-hf-cache:/root/.cache/huggingface \
#             vvs-visual-embed
# The volume matters: without it, the 5 GB model re-downloads every restart.

FROM python:3.12-slim

# Matches astral's recommended pattern: copy the uv binary itself rather than
# installing it via pip, so this layer never depends on PyPI being reachable.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependency layers first, so an application-code-only change (the common
# case) does not re-resolve or re-download anything.
COPY pyproject.toml uv.lock ./
RUN uv sync --extra services --no-install-project --no-dev

COPY src/ ./src/
COPY README.md ./
RUN uv sync --extra services --no-dev

ENV HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 8001

# Lazy model load on first request, same as running it directly -- add a
# startup probe hitting GET /info if you want the slow first call to happen
# at container-start instead of at first-traffic.
CMD ["uv", "run", "--extra", "services", "uvicorn", "vvs.services.visual_embed_server:app", \
     "--host", "0.0.0.0", "--port", "8001"]
