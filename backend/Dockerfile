# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Requirements first: a code change then does not reinstall the world.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# Non-root, owning the two directories the app writes to at runtime.
RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /app/.data /app/.chroma \
    && chown -R agent:agent /app

USER agent

# Build the retrieval index and pull the ONNX embedding model (~80MB) into the
# image now, so the first search_knowledge_base call needs no network. Without
# this the first case a reviewer runs stalls on a model download - the known
# future issue recorded in PROGRESS.md §8.
RUN python -c "from app.rag import retrieval; print('indexed chunks:', retrieval.indexed_chunk_count())"

EXPOSE 8000

# /api/health answers without an API key, so this is a real readiness check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
