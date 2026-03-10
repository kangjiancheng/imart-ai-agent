# ── Base image ─────────────────────────────────────────────────────────────────
# python:3.12.13-slim = Debian slim (~130MB), no dev tools, no test frameworks.
# Keeps the final image as small as possible while staying compatible with
# PyTorch (required by FlagEmbedding / BGE-M3).
FROM python:3.12.13-slim

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
# libgomp1  — OpenMP runtime required by PyTorch for parallel tensor operations.
#             Without it, FlagEmbedding (BGE-M3) crashes at import time.
# curl      — used by the HEALTHCHECK below to poll /health.
# Both are installed in one RUN layer to minimise the number of image layers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements.txt BEFORE source code so this layer is cached independently.
# Docker only re-runs pip install when requirements.txt changes — not on every
# source-code edit — keeping rebuild times fast during development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ─────────────────────────────────────────────────────────
# Copy only src/ — .dockerignore excludes venv/, .env, *.db, model weights,
# tests/, docs/, and other files not needed at runtime.
COPY src/ ./src/

# ── Non-root user ──────────────────────────────────────────────────────────────
# Running as root inside a container is a security risk.
# appuser owns /app so uvicorn can write any temp files it needs.
# /home/appuser/.cache/modelscope is where snapshot_download() caches BGE-M3.
# That directory is mounted as a named volume in docker-compose.yml so the
# model (~2.3 GB) persists across container restarts without re-downloading.
RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p /home/appuser/.cache/modelscope \
    && chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

EXPOSE 8000

# ── Health check ───────────────────────────────────────────────────────────────
# start_period=90s: BGE-M3 takes 30–60s to load into RAM on first start.
#   Docker does NOT count failures during start_period as unhealthy retries.
# interval=30s: poll every 30s after the start_period elapses.
# retries=3: mark the container unhealthy only after 3 consecutive failures.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ─────────────────────────────────────────────────────────────────
# workers=1: BGE-M3 loads ~1 GB of model weights per worker process.
#   Use 1 worker to avoid doubling RAM usage.
#   Scale horizontally via multiple container replicas instead of more workers.
# FastAPI is INTERNAL ONLY — never expose port 8000 to the public internet.
#   Route all traffic through an Nginx reverse proxy or a service mesh.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
