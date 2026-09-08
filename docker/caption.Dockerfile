# Caption service: Qwen3-VL-2B-Instruct behind FastAPI (scene captions +
# query expansion).
#
# Linux only by construction -- see docker/visual-embed.Dockerfile's header
# for why (same reasoning applies verbatim to this image).
#
# Build:  docker build -f docker/caption.Dockerfile -t vvs-caption .
# Run:    docker run --gpus all -p 8002:8002 \
#             -v vvs-hf-cache:/root/.cache/huggingface \
#             vvs-caption
# The volume matters: without it, the 5 GB model re-downloads every restart.
# It can be the SAME named volume the visual-embed image uses -- both read
# from HF_HOME, and HF's cache is keyed by model id, so sharing it just means
# one download instead of two if both ever pull an overlapping model.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --extra services --no-install-project --no-dev

COPY src/ ./src/
COPY README.md ./
RUN uv sync --extra services --no-dev

ENV HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 8002

CMD ["uv", "run", "--extra", "services", "uvicorn", "vvs.services.caption_server:app", \
     "--host", "0.0.0.0", "--port", "8002"]
