FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching — rebuilds only when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

# FastAPI is INTERNAL ONLY — no public port should be exposed in docker-compose
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
